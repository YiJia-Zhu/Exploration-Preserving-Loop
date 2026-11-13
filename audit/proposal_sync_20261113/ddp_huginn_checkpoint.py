"""Audit fixture: DDP plus both checkpoint layers on the official tiny Huginn."""
from pathlib import Path
import sys,json,torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from ep_loop.huginn import tiny_huginn
from ep_loop.objective import DistillationObjective
torch.set_num_threads(2)
torch.manual_seed(42)
dist.init_process_group('gloo')
rank=dist.get_rank()
b=tiny_huginn(root/'models/huginn-0125')
b.model.gradient_checkpointing=True
b.model.train()
algorithm={'student_loops':2,'teacher_max_loops':4,'teacher_update':'current','teacher_endpoint':'native','exit_threshold':5e-4,'stage_selection':'js','js_threshold':0.0,'count_matched_uniform':True}
objective=DistributedDataParallel(DistillationObjective(b,b,algorithm,2,True),find_unused_parameters=True)
loss,_=objective(torch.tensor([[1,20+rank,30,40,50,2]]),3)
loss.backward()
grads=torch.cat([p.grad.flatten() for p in b.model.parameters() if p.grad is not None])
assert torch.isfinite(grads).all()
others=[torch.empty_like(grads) for _ in range(dist.get_world_size())]
dist.all_gather(others,grads)
assert all(torch.equal(grads,g) for g in others)
if rank==0:
 report={'status':'PASS','world_size':2,'native_backbone_checkpointing':True,'readout_checkpointing':True,'all_reduced_gradients_match_exactly':True,'scope':'Random official tiny architecture on CPU, no GPU benchmark'}
 (root/'audit/proposal_sync_20261113/huginn_ddp_checkpoint.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
dist.destroy_process_group()
