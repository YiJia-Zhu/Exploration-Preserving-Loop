"""Real Ouro engineering stress: length cap, Adam states, and delayed KV fill.

Forced depths/tokens here measure memory only; production training stays dynamic.
Run with CUDA_VISIBLE_DEVICES=2 and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""
import copy
import gc
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from ep_loop.config import load_config
from ep_loop.objective import DistillationObjective
from ep_loop.ouro import load_ouro
from ep_loop.reproducibility import seed_everything

torch.set_num_threads(4)
seed_everything(42)
rows=[]
started=time.perf_counter()

def memory(stage):
    torch.cuda.synchronize()
    row=dict(stage=stage,seconds=time.perf_counter()-started,
             allocated_gib=torch.cuda.memory_allocated()/2**30,
             reserved_gib=torch.cuda.memory_reserved()/2**30,
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    rows.append(row)
    print(json.dumps(row),flush=True)

config=load_config(ROOT/'configs/e_dynamic_js.json')
student,_=load_ouro(ROOT/'models/Ouro-1.4B-Thinking','cuda','bf16',True)
teacher=type(student)(copy.deepcopy(student.model).eval().requires_grad_(False),'bf16')
student.enable_student_exit()
objective=DistillationObjective(student,teacher,config['algorithm'],128,True)
optimizer=torch.optim.AdamW(student.model.parameters(),lr=1e-6,weight_decay=.01)
student.model.train()
warm=torch.full((1,65),40,device='cuda',dtype=torch.long)
loss,info=objective(warm,1,[4]*64)
loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
del loss,info
gc.collect();torch.cuda.empty_cache()
memory('adam_states_initialized')

# Conservative upper bound: reserve both a full gradient and a separate DDP
# bucket while running no-grad KV tests. Production DDP now shares their storage.
count=sum(parameter.numel() for parameter in student.model.parameters())
gradient_reserve=torch.empty(count,device='cuda',dtype=torch.float32)
ddp_reserve=torch.empty_like(gradient_reserve)
memory('adam_plus_two_fp32_gradient_reserves')
ids=torch.full((1,17408),40,device='cuda',dtype=torch.long)
gate=student.enable_student_exit()
def force(depth):
    calls=[0]
    def forward(hidden):
        calls[0]+=1
        return hidden.new_full((*hidden.shape[:2],1),100. if calls[0]>=depth else -100.)
    return patch.object(gate,'forward',new=forward)

student.model.eval()
with torch.no_grad(),student.generation_context():
    with force(1):
        trace=student.trace(ids[:,:-1],4,use_cache=True,endpoint='learned')
    with force(4):
        trace=student.trace(ids[:,-1:],4,trace.cache,True,'learned')
    assert trace.backfill_token_loops==3*17407
    assert [level.size for level in trace.cache.levels]==[17408]*5
    memory('17408_positions_late_1_to_4_exact_backfill')
del trace
gc.collect();torch.cuda.empty_cache()
with torch.no_grad(),student.generation_context():
    trace=None
    for begin in range(0,17408,512):
        with force(4):
            trace=student.trace(ids[:,begin:begin+512],4,trace.cache if trace else None,True,'learned')
    assert trace.cache.kv.get_seq_length()==17408
    assert all(buffer.storage.shape[2]-buffer.size<=1024 for buffer in trace.cache.kv.keys.values())
    memory('17408_positions_repeated_four_loop_cache_growth')
del trace,gradient_reserve
gc.collect();torch.cuda.empty_cache()
student.model.train()
memory('before_full_length_differentiable_replay')
loss,info=objective(ids,1024,[4]*16384)
assert torch.isfinite(loss)
loss.backward()
assert all(torch.isfinite(parameter.grad).all() for parameter in student.model.parameters() if parameter.grad is not None)
memory('16384_response_tokens_replay_and_backward_with_adam_and_ddp_reserve')
report=dict(status='PASS',max_new_tokens=16384,prompt_tokens=1024,total_tokens=17408,
            mixed_precision='bf16',seed=42,stages=rows,loss=float(loss.detach()),
            scope='Forced depths and constant token IDs test memory/causal execution only, not task quality or sampled training speed.')
(Path(__file__).parent/'gpu_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
