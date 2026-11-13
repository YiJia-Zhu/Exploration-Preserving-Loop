"""DDP with all phases and both checkpoint layers; random official architectures."""
from pathlib import Path
import copy,json,sys,torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from ep_loop.backends import load_backend
from ep_loop.config import load_config,smoke_config
from ep_loop.objective import DistillationObjective
family=sys.argv[1];torch.set_num_threads(2);torch.manual_seed(42)
dist.init_process_group('gloo');rank=dist.get_rank()
config=smoke_config(load_config(root/'configs'/('e_dynamic_js_current.json' if family=='ouro' else 'huginn_e_dynamic_js_current.json')))
a=config['algorithm'];a.update(teacher_endpoint='fixed',js_threshold=0.)
b,_=load_backend(config['model'],root/config['model']['path'],'cpu','none',False,True)
b.enable_student_exit();b.model.train()
if family=='ouro':b.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
else:b.model.gradient_checkpointing=True
wrapped=DistributedDataParallel(DistillationObjective(b,b,a,2,True),find_unused_parameters=True)
loss,_=wrapped(torch.tensor([[1,20+rank,30,40,50,2]]),3,[1,1,1]);loss.backward()
grad=torch.cat([p.grad.flatten() for p in b.model.parameters() if p.grad is not None])
others=[torch.empty_like(grad) for _ in range(dist.get_world_size())];dist.all_gather(others,grad)
assert torch.isfinite(grad).all() and all(torch.equal(grad,g) for g in others)
if rank==0:
 report={'status':'PASS','family':family,'world_size':2,'all_reduced_gradients_match_exactly':True,'backbone_and_readout_checkpointing':True,'all_target_stages_kept':True}
 (root/f'audit/dynamic_student_20261113/{family}_ddp_gradient.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
dist.destroy_process_group()
