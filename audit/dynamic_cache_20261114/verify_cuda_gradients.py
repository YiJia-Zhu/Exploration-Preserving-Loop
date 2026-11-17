"""Check that no-grad BF16 cast reuse does not poison shared-model gradients."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from ep_loop.config import load_config,smoke_config
from ep_loop.ouro import tiny_ouro
from ep_loop.huginn import tiny_huginn
from ep_loop.inference import generate
from ep_loop.objective import DistillationObjective
root=Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
results=[]
for family,loader,path in [('ouro',tiny_ouro,'models/Ouro-1.4B-Thinking'),('huginn',tiny_huginn,'models/huginn-0125')]:
 torch.manual_seed(42)
 backend=loader(root/path,'cuda');backend.mixed_precision='bf16';backend.enable_student_exit();backend.model.train()
 cfg=smoke_config(load_config(root/'configs'/('huginn_' if family=='huginn' else '').__add__('e_dynamic_js_current.json')))
 objective=DistillationObjective(backend,backend,cfg['algorithm'],2,True)
 ids=torch.tensor([[1,20,30,40,50,2]],device='cuda')
 def gradients():
  backend.model.zero_grad(set_to_none=True);torch.manual_seed(731)
  loss,_=objective(ids,3,[1,1,1]);loss.backward()
  grads={name:p.grad.clone() for name,p in backend.model.named_parameters() if p.grad is not None}
  assert grads and all(torch.isfinite(g).all() for g in grads.values())
  return loss.detach(),grads
 expected_loss,expected=gradients()
 generate(backend,[1,20,30],4,4,1.,1.,torch.Generator(device='cuda').manual_seed(42),'learned',.5,True)
 assert backend.model.training and not backend._generation_cast_cache
 actual_loss,actual=gradients()
 torch.testing.assert_close(expected_loss,actual_loss,rtol=0,atol=0)
 assert expected.keys()==actual.keys()
 for key in expected:torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
 results.append(dict(family=family,gradient_tensors=len(actual),bf16_shared_model_gradients_exact_after_generation=True))
 print(results[-1],flush=True)
(root/'audit/dynamic_cache_20261114/cuda_gradient_validation.json').write_text(json.dumps(dict(status='PASS',cases=results),indent=2)+'\n')
