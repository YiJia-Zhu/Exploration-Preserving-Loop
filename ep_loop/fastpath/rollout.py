"""Generate one unchanged on-policy update while its weights are fixed."""
from collections import deque
from pathlib import Path
import os
import torch
from ep_loop.inference import GeneratedResponse
from .actor import VllmActor
from .inference import generate as hf_generate
from ep_loop.config import student_budget


class RolloutBatcher:
    def __init__(self,student,config,output,local_rank,observer):
        if config['model']['family']!='ouro':raise ValueError('vLLM fastpath currently supports Ouro only')
        self.student,self.config,self.output,self.observer=student,config,Path(output),observer
        self.sampling_device=student.device
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        self.gpu=visible[local_rank] if visible!=[''] else str(local_rank)
        self.actor=None;self.pending=deque()

    def prepare(self,records,indices,rank,world,update,generator):
        c=self.config;rollout=c['rollout'];a=c['algorithm'];i=c['implementation']
        loops=student_budget(c)
        fixed=a.get('student_policy','fixed')=='fixed'
        ordered=[(records[index],sample) for index in indices[rank::world]
                 for sample in range(rollout['responses_per_prompt'])]
        # A persistent rank generator still governs training randomness. vLLM
        # owns independent per-response generators; seed 42 and the mapping are
        # recorded. Cross-framework bitwise sampling equality is not claimed.
        seeds=torch.randint(0,2**63-1,(len(ordered),),generator=generator,
                            device=self.sampling_device,dtype=torch.int64).cpu().tolist()
        requests=[dict(input_ids=record['input_ids'],temperature=rollout['temperature'],
            top_p=rollout['top_p'],max_new_tokens=rollout['max_new_tokens'],seed=seed)
            for (record,sample),seed in zip(ordered,seeds)]
        with self.observer.stage('vllm_on_policy_batch',echo=True,update=update+1,
                responses=len(requests),weight_version=update):
            torch.cuda.empty_cache()
            if self.actor is None:
                self.actor=VllmActor(c['model']['path'],self.gpu,self.output/f'vllm_rank{rank}.log',
                    loops=loops,policy='fixed' if fixed else 'learned',
                    stop_threshold=a.get('student_stop_threshold',.5),native_threshold=a['exit_threshold'],
                    max_model_len=c['data']['max_prompt_tokens']+rollout['max_new_tokens'],
                    memory_fraction=i.get('vllm_memory_fraction',.45),
                    eager=i.get('vllm_eager',False),sleep=True)
            # One transient file, removed after synchronous loading. Never copy
            # the base model directory or leave dense rollout/teacher shards.
            from safetensors.torch import save_file
            weights=self.output/f'.vllm_weights_rank{rank}.safetensors'
            try:
                save_file(self.student.model.state_dict(),str(weights))
                acknowledgement=self.actor.request('weights',path=str(weights),version=update)
                if acknowledgement['weight_version']!=update:raise RuntimeError('Stale rollout weights')
            finally:
                weights.unlink(missing_ok=True)
            try:result=self.actor.generate(requests,logprobs=None)
            finally:self.actor.request('sleep')
            if fixed and result.get('executed_loops')!=loops:
                raise RuntimeError('Fixed-depth actor did not execute the configured student N')
            for (record,sample),seed,response in zip(ordered,seeds,result['responses']):
                token_ids=response['token_ids'];depths=response['exit_depths']
                if fixed and depths!=[loops]*len(token_ids):
                    raise RuntimeError('Fixed-depth actor returned a different readout depth')
                self.pending.append((record['id'],sample,GeneratedResponse(token_ids,depths,
                    bool(token_ids[-1] in self.student.stop_token_ids),response['executed_token_loops'],
                    result['seconds'],[loops]*len(token_ids),0,0,'vllm_fixed_depth' if fixed else 'vllm_fixed_k_gate_readout')))
                self.observer.event('vllm_rollout_ready',update=update+1,id=record['id'],sample=sample,
                    sampling_seed=seed,response_tokens=len(token_ids),weight_version=update)
            self.observer.event('vllm_batch_metrics',update=update+1,batch_seconds=result['seconds'],
                response_tokens=sum(len(r['token_ids']) for r in result['responses']),
                timing='Batch wall time; response.seconds is common submission-to-batch-completion latency')

    def pop(self,context):
        ident,sample,response=self.pending.popleft()
        if (ident,sample)!=(context['id'],context['sample']):raise RuntimeError('On-policy response order mismatch')
        return response

    def close(self):
        if self.actor:self.actor.close()


def generate(backend,*args,**kwargs):
    batcher=getattr(backend,'_ep_rollout_batcher',None)
    if batcher is None:return hf_generate(backend,*args,**kwargs)
    return batcher.pop(kwargs['log_context'])
