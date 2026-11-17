"""Frozen vLLM teacher prefills with compact, full-information loop readouts."""
from collections import deque
from contextlib import nullcontext
import copy
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file

from ep_loop.ouro import LoopTrace
from ep_loop.selection import native_ouro_exits
from .actor import VllmActor


def teacher_readout_directory(config, output, required_bytes):
    """Use the configured transient volume; fail before writing a partial batch."""
    location = Path(config['implementation'].get('teacher_trace_dir') or output).expanduser().resolve()
    location.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(location).free
    if free < required_bytes:
        raise RuntimeError(f'Teacher trace volume {location} needs about {required_bytes/2**30:.1f} GiB free; '
                           f'available {free/2**30:.1f} GiB. Set TEACHER_TRACE_DIR to a suitable volume.')
    return tempfile.TemporaryDirectory(prefix='.teacher_readouts_', dir=location)


class VllmTeacher:
    native_exit_reduces_executed_loops=False
    has_coda=False

    def __init__(self,student,config,output,local_rank,observer):
        if config['model']['family']!='ouro' or config['algorithm']['teacher_update']!='frozen':
            raise ValueError('The vLLM teacher currently requires frozen initial Ouro weights')
        # Keep only the INITIAL vocabulary head for exact KL/JS readout in
        # the training process; no duplicate HF teacher backbone is allocated.
        self.model=copy.deepcopy(student.model.lm_head).cpu().eval().requires_grad_(False)
        self.config,self.output,self.observer=config,Path(output),observer
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        self.gpu=visible[local_rank] if visible!=[''] else str(local_rank)
        self.rank=local_rank;self.actor=None;self.pending=deque();self.temporary=None;self.active=None

    def prepare(self,responses,records,update):
        self.clear_batch()
        c=self.config;i=c['implementation'];k=c['algorithm']['teacher_max_loops']
        prompts={record['id']:record['input_ids'] for record in records}
        requests=[dict(input_ids=prompts[ident]+response.token_ids[:-1]) for ident,_,response in responses]
        # BF16 normalized states and gates, with conservative per-file overhead.
        # Reserve enough headroom for all ranks sharing the same transient volume.
        required = (sum(len(r['input_ids'])*k*(self.model.in_features+1)*2 for r in requests)
                    + len(requests)*1024**2)*int(os.environ.get('WORLD_SIZE','1'))
        self.temporary=teacher_readout_directory(c,self.output,required)
        self.observer.event('teacher_trace_storage',update=update+1,path=self.temporary.name,
                            required_headroom_bytes=required)
        with self.observer.stage('vllm_teacher_label_batch',echo=True,update=update+1,
                                 responses=len(requests),teacher_loops=k):
            torch.cuda.empty_cache()
            if self.actor is None:
                self.actor=VllmActor(c['model']['path'],self.gpu,self.output/f'vllm_teacher_rank{self.rank}.log',
                    loops=k,policy='fixed',teacher_trace=True,sleep=True,
                    max_model_len=c['data']['max_prompt_tokens']+c['rollout']['max_new_tokens'],
                    memory_fraction=i.get('vllm_memory_fraction',.45),eager=i.get('vllm_eager',False))
            try:result=self.actor.teacher_trace(requests,self.temporary.name)
            finally:self.actor.request('sleep')
            if result['executed_loops']!=k or len(result['responses'])!=len(responses):
                raise RuntimeError('Teacher readout count or physical depth mismatch')
            for (ident,sample,_),request,trace in zip(responses,requests,result['responses']):
                if trace['input_positions']!=len(request['input_ids']):
                    raise RuntimeError('Teacher did not label exactly the supplied student prefix')
                self.pending.append((ident,sample,trace))
            self.observer.event('teacher_batch_metrics',update=update+1,batch_seconds=result['seconds'],
                teacher_executed_token_loops=sum(t['executed_token_loops'] for t in result['responses']),
                transport='Normalized hidden states and gates; full vocabulary is reconstructed chunkwise')

    def use_response(self,ident,sample):
        expected_ident,expected_sample,trace=self.pending.popleft()
        if (ident,sample)!=(expected_ident,expected_sample):raise RuntimeError('Teacher/student response order mismatch')
        self.active=trace

    def response_trace(self,ids,loops,start,endpoint='fixed',exit_threshold=.9):
        trace=self.active
        if trace is None:raise RuntimeError('Select a precomputed teacher response before the objective')
        if trace['input_positions']!=ids.shape[1] or loops!=self.config['algorithm']['teacher_max_loops']:
            raise RuntimeError('Cached teacher trace does not match student text/depth')
        readout=load_file(trace['path'])
        states=readout['states'][:,start:].to(ids.device)
        gates=readout['gates'][:,start:].to(ids.device)
        Path(trace['path']).unlink()
        self.active=None
        return LoopTrace(tuple(states.unbind(0)),tuple(gates.unbind(0)),
                         executed_token_loops=trace['executed_token_loops'])

    def native_exit_depths(self,trace,threshold):
        return native_ouro_exits(torch.stack(trace.gates),threshold)

    def head(self,states):
        autocast=(torch.autocast('cuda',dtype=torch.bfloat16)
                  if states.device.type=='cuda' and self.config['optimization']['mixed_precision']=='bf16' else nullcontext())
        with autocast:return self.model(states).float()

    def clear_batch(self):
        self.pending.clear();self.active=None
        if self.temporary:self.temporary.cleanup();self.temporary=None

    def close(self):
        try:
            if self.actor:self.actor.close()
        finally:self.clear_batch()
