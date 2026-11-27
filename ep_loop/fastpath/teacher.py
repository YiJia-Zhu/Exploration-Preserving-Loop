"""Frozen/current vLLM teacher prefills with full-information loop readouts."""
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
from .parallel import InferenceGroup


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

    def __init__(self,student,config,output,local_rank,observer,inference_group=None):
        if config['model']['family']!='ouro' or config['algorithm']['teacher_update'] not in {'frozen','current'}:
            raise ValueError('The vLLM teacher requires frozen/current Ouro weights')
        self.student=student
        self.current=config['algorithm']['teacher_update']=='current'
        # D-LoopOPD shares the live head; the objective stops target gradients.
        # Only LoopOPD retains an INITIAL frozen head. Neither allocates a
        # second HF backbone or changes the student's requires_grad flags.
        self.model=(student.model.lm_head if self.current else
                    copy.deepcopy(student.model.lm_head).cpu().eval().requires_grad_(False))
        self.weight_version=None if self.current else 'initial'
        self.config,self.output,self.observer=config,Path(output),observer
        self.max_model_len = config['data']['max_prompt_tokens'] + config['rollout']['max_new_tokens']
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        self.gpu=visible[local_rank] if visible!=[''] else str(local_rank)
        if config['implementation'].get('vllm_tensor_parallel_size',1)>1 and inference_group is None:
            raise ValueError('Paired teacher requires an initialized inference group')
        self.group=inference_group or InferenceGroup(self.gpu)
        self.gpu=self.group.gpu
        self.rank=local_rank;self.actor=None;self.pending=deque();self.temporary=None;self.active=None

    def prepare(self,responses,records,update,*,weights_path=None):
        self.clear_batch()
        self.weight_version=update if self.current else 'initial'
        c=self.config;i=c['implementation'];k=c['algorithm']['teacher_max_loops']
        prompts={record['id']:record['input_ids'] for record in records}
        requests=[dict(input_ids=prompts[ident]+response.token_ids[:-1]) for ident,_,response in responses]
        with self.observer.stage('vllm_teacher_label_batch',echo=True,update=update+1,
                                 responses=len(requests),teacher_loops=k,
                                 weight_version=self.weight_version,
                                 tensor_parallel_size=self.group.size,inference_group_gpus=self.gpu):
            torch.cuda.empty_cache()
            result=self.group.execute(requests,lambda merged:self._label(merged,update,k,weights_path))
            if result['executed_loops']!=k or len(result['responses'])!=len(responses):
                raise RuntimeError('Teacher readout count or physical depth mismatch')
            if result['weight_version']!=self.weight_version:
                raise RuntimeError('Stale teacher batch weights')
            for (ident,sample,_),request,trace in zip(responses,requests,result['responses']):
                if trace['input_positions']!=len(request['input_ids']):
                    raise RuntimeError('Teacher did not label exactly the supplied student prefix')
                self.pending.append((ident,sample,dict(trace,weight_version=self.weight_version)))

    def _label(self,requests,update,k,weights_path=None):
        c=self.config;i=c['implementation']
        # Only the leader owns the directory. Unique response files are handed
        # back to their HF ranks, then consumed/deleted once during the objective.
        replicas=int(os.environ.get('WORLD_SIZE','1'))//self.group.size
        required=(sum(len(r['input_ids'])*k*(self.model.in_features+1)*2 for r in requests)
                  +len(requests)*1024**2)*replicas
        self.temporary=teacher_readout_directory(c,self.output,required)
        self.observer.event('teacher_trace_storage',update=update+1,path=self.temporary.name,
                            required_headroom_bytes=required)
        if self.actor is None:
            self.actor=VllmActor(c['model']['path'],self.gpu,self.output/f'vllm_teacher_rank{self.rank}.log',
                loops=k,policy='fixed',teacher_trace=True,sleep=True,tensor_parallel_size=self.group.size,
                rollout_ngram=False,
                max_model_len=self.max_model_len,
                memory_fraction=i.get('vllm_memory_fraction',.45),eager=i.get('vllm_eager',False),
                python_bin=i.get('vllm_python_bin'),attention_backend=i.get('vllm_attention_backend'),
                **({'max_num_seqs':i['vllm_max_num_seqs']} if i.get('vllm_max_num_seqs') else {}))
            self.observer.event('vllm_teacher_runtime',update=update+1,**getattr(self.actor,'runtime',{}))
        try:
            if self.current:
                from safetensors.torch import save_file
                weights = weights_path or self.output/f'.vllm_teacher_weights_rank{self.rank}.safetensors'
                try:
                    if weights_path is None:
                        save_file(self.student.model.state_dict(), str(weights))
                    acknowledgement=self.actor.request('weights',path=str(weights),version=update)
                    if acknowledgement['weight_version']!=update:
                        raise RuntimeError('Stale teacher weights')
                finally:
                    if weights_path is None:
                        weights.unlink(missing_ok=True)
                self.observer.event('teacher_weights_refreshed',update=update+1,weight_version=update,
                    tensor_parallel_size=self.group.size,inference_group_gpus=self.gpu,
                    readout_head='shared_current_student_head',stop_gradient=True)
            result=self.actor.teacher_trace(requests,self.temporary.name)
        finally:self.actor.request('sleep')
        result=dict(result,weight_version=update if self.current else 'initial')
        self.observer.event('teacher_batch_metrics',update=update+1,batch_seconds=result['seconds'],
            weight_version=result['weight_version'],teacher_update=c['algorithm']['teacher_update'],
            responses=len(requests),tensor_parallel_size=self.group.size,inference_group_gpus=self.gpu,
            teacher_executed_token_loops=sum(t['executed_token_loops'] for t in result['responses']),
            transport='TP leader writes normalized states/gates once; HF ranks reconstruct full vocab chunkwise')
        return result

    def use_response(self,ident,sample):
        expected_ident,expected_sample,trace=self.pending.popleft()
        if (ident,sample)!=(expected_ident,expected_sample):raise RuntimeError('Teacher/student response order mismatch')
        if trace.get('weight_version',self.weight_version)!=self.weight_version:
            raise RuntimeError('Teacher trace belongs to a different weight version')
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
            if self.actor:
                self.actor.close();self.actor=None
        finally:self.clear_batch()
