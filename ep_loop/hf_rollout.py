"""Pure-HF fixed-depth generation using the same trainable model instance.

The global on-policy update is unchanged. Several responses share an HF
forward; completed rows are removed. KV storage is preallocated, but attention
reads only the populated prefix, avoiding full-cache copies on every token.
"""
from collections import deque
import gc
import time

import torch
from transformers.cache_utils import Cache, StaticLayer

from .inference import GeneratedResponse
from .ouro import evaluation_mode


class PrefixStaticLayer(StaticLayer):
    # Storage is static, but exposed sequence/batch shapes change. This also
    # lets HF use SDPA's causal fast path when no padding mask is needed.
    is_compileable=False

    def __init__(self, max_cache_len):
        super().__init__(max_cache_len=max_cache_len)
        self.visible_length=0
        self.seen_length=0

    def get_mask_sizes(self, cache_position):
        return self.visible_length,0

    def get_seq_length(self):
        return self.seen_length

    def update(self, key_states, value_states, cache_kwargs=None):
        # Ouro's FP32 RoPE promotes K while V remains BF16 under autocast.
        # SDPA casts both to the autocast dtype for attention. Cache that exact
        # effective representation, rather than reallocating FP32 KV per token.
        device_type=key_states.device.type
        if torch.is_autocast_enabled(device_type):
            dtype=torch.get_autocast_dtype(device_type)
            key_states=key_states.to(dtype)
            value_states=value_states.to(dtype)
        keys,values=super().update(key_states,value_states,cache_kwargs)
        self.seen_length=self.visible_length
        return keys[:,:,:self.visible_length],values[:,:,:self.visible_length]


class FixedLoopStaticCache(Cache):
    def __init__(self, layers, max_cache_len):
        super().__init__(layers=[PrefixStaticLayer(max_cache_len) for _ in range(layers)])

    def set_visible_length(self, length):
        for layer in self.layers:layer.visible_length=length

    def batch_select_indices(self, indices):
        for layer in self.layers:
            if layer.is_initialized:
                layer.keys=layer.keys.index_select(0,indices)
                layer.values=layer.values.index_select(0,indices)
                layer.max_batch_size=indices.numel()


def sample_tokens(logits, temperature, top_p, generator):
    scores=logits.float()/temperature
    probabilities=scores.softmax(-1)
    if top_p<1:
        sorted_p,indices=probabilities.sort(dim=-1,descending=True)
        remove=sorted_p.cumsum(-1)-sorted_p>=top_p
        sorted_p=sorted_p.masked_fill(remove,0)
        probabilities=torch.zeros_like(probabilities).scatter_(-1,indices,sorted_p)
    return torch.multinomial(probabilities,1,generator=generator).squeeze(-1)


@torch.no_grad()
def generate_batch(backend, prompts, loops, max_new_tokens, temperature, top_p,
                   generator, observer=None, context=None, cache_capacity=None):
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError('HF batch requires nonempty prompts')
    if loops<1 or max_new_tokens<1 or temperature<=0 or not 0<top_p<=1:
        raise ValueError('Invalid HF sampling budget or temperature')
    device=backend.device
    size,width=len(prompts),max(map(len,prompts))
    unmasked=all(len(prompt)==width for prompt in prompts)
    capacity=cache_capacity or width+max_new_tokens
    if capacity<width+max_new_tokens or capacity>backend.max_context_length:
        raise ValueError('HF cache capacity does not cover the requested context')
    cache=FixedLoopStaticCache(backend.model.config.num_hidden_layers*loops,capacity)
    ids=torch.zeros((size,width),device=device,dtype=torch.long)
    mask=torch.zeros_like(ids)
    for index,prompt in enumerate(prompts):
        ids[index,-len(prompt):]=torch.tensor(prompt,device=device)
        mask[index,-len(prompt):]=1
    token_buffer=torch.empty((size,max_new_tokens),device=device,dtype=torch.long)
    lengths=torch.full((size,),max_new_tokens,device=device,dtype=torch.long)
    active=torch.arange(size,device=device)
    eos=torch.tensor(sorted(backend.stop_token_ids),device=device)
    start=time.perf_counter();last_report=start
    generated=0
    with evaluation_mode(backend.model),backend.generation_context():
        for step in range(max_new_tokens):
            physical_position=0 if step==0 else width+step-1
            visible=width if step==0 else width+step
            cache.set_visible_length(visible)
            cache_position=torch.arange(physical_position,visible,device=device)
            if unmasked:position_ids=cache_position[None]
            else:
                positions=(mask.cumsum(-1)-1).clamp_min(0)
                position_ids=positions if step==0 else positions[:,-1:]
            trace=backend.trace(ids,loops,cache=cache,use_cache=True,
                attention_mask=None if unmasked else mask,position_ids=position_ids,cache_position=cache_position)
            logits=backend.head(trace.states[-1][:,-1])
            sampled=sample_tokens(logits,temperature,top_p,generator)
            token_buffer[active,step]=sampled
            generated+=active.numel()
            keep=(sampled[:,None]!=eos[None]).all(-1)
            if not bool(keep.all()):
                lengths[active[~keep]]=step+1
                if not bool(keep.any()):break
                selected=keep.nonzero(as_tuple=False).flatten()
                active=active.index_select(0,selected)
                sampled=sampled.index_select(0,selected)
                mask=mask.index_select(0,selected)
                cache.batch_select_indices(selected)
            if step+1==max_new_tokens:break
            ids=sampled[:,None]
            if not unmasked:mask=torch.cat((mask,mask.new_ones((mask.shape[0],1))),dim=1)
            now=time.perf_counter()
            if observer and now-last_report>=30:
                observer.event('hf_generation_progress',**(context or {}),generated_tokens=generated,
                    active_responses=active.numel(),decode_steps=step+1,seconds=now-start)
                last_report=now
    if device.type=='cuda':torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    lengths=lengths.cpu().tolist()
    tokens=[token_buffer[index,:length].cpu().tolist() for index,length in enumerate(lengths)]
    result=[]
    for index,length in enumerate(lengths):
        sequence=tokens[index][:length]
        result.append(GeneratedResponse(sequence,[loops]*length,sequence[-1] in backend.stop_token_ids,
            loops*(width+length-1),elapsed,[loops]*length,0,0,'hf_batched_static_prefix'))
    del trace,cache,logits,token_buffer
    return result,elapsed


def move_optimizer_state(optimizer, device):
    """Offload moments only; keep the single model and DDP parameters intact."""
    for state in optimizer.state.values():
        for name,value in state.items():
            if name!='step' and isinstance(value,torch.Tensor):state[name]=value.to(device)
    # torch.save can leave cycles owning the *old* CUDA moment storages after
    # optimizer.state has moved to CPU. Collect after replacing the tensors,
    # before allocating the next full-size rollout KV cache.
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


class HFRolloutBatcher:
    def __init__(self, student, config, observer):
        self.student,self.config,self.observer=student,config,observer
        self.pending=deque()

    def prepare(self, records, indices, rank, world, update, generator):
        if self.pending:raise RuntimeError('Previous HF rollout responses were not consumed')
        rollout=self.config['rollout'];loops=self.config['algorithm']['student_loops']
        ordered=[(records[index],sample) for index in indices[rank::world]
                 for sample in range(rollout['responses_per_prompt'])]
        size=self.config['implementation'].get('hf_rollout_batch_size',1)
        with self.observer.stage('hf_on_policy_batch',echo=True,update=update+1,
                responses=len(ordered),weight_version=update,hf_rollout_batch_size=size):
            for begin in range(0,len(ordered),size):
                chunk=ordered[begin:begin+size]
                with self.observer.stage('hf_generation_batch',update=update+1,batch_index=begin//size,
                        responses=len(chunk)):
                    responses,seconds=generate_batch(self.student,[record['input_ids'] for record,_ in chunk],
                        loops,rollout['max_new_tokens'],rollout['temperature'],rollout['top_p'],generator,
                        self.observer,dict(update=update+1,batch_index=begin//size),
                        cache_capacity=self.config['implementation'].get('hf_static_cache_capacity'))
                for (record,sample),response in zip(chunk,responses):
                    self.pending.append((record['id'],sample,response))
                self.observer.event('hf_batch_metrics',update=update+1,batch_index=begin//size,
                    responses=len(responses),response_tokens=sum(len(r.token_ids) for r in responses),
                    batch_seconds=seconds,weight_version=update,model_instances_per_rank=1)

    def pop(self, ident, sample):
        expected_ident,expected_sample,response=self.pending.popleft()
        if (ident,sample)!=(expected_ident,expected_sample):raise RuntimeError('HF response order mismatch')
        return response
