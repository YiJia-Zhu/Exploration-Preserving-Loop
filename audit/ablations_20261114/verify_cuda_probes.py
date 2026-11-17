import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from ep_loop.confidence import PromptConfidence
from ep_loop.config import load_config,smoke_config
from ep_loop.data import ByteTokenizer
from ep_loop.huginn import tiny_huginn
from ep_loop.objective import DistillationObjective
from ep_loop.ouro import tiny_ouro
root=Path(__file__).resolve().parents[2];audit=root/'audit/ablations_20261114'
torch.set_num_threads(2);results=[]
for family,loader,directory in [('ouro',tiny_ouro,'models/Ouro-1.4B-Thinking'),('huginn',tiny_huginn,'models/huginn-0125')]:
 torch.manual_seed(42)
 backend=loader(root/directory,'cuda');backend.mixed_precision='bf16';backend.enable_student_exit();backend.model.train()
 config=smoke_config(load_config(root/'configs'/('huginn_' if family=='huginn' else '').__add__('e_dynamic_js_current.json')))
 for signal in ['probe_know','probe_true']:
  config['algorithm'].update(boundary_signal=signal,teacher_endpoint='fixed')
  probe=PromptConfidence(backend,ByteTokenizer(),max_answer_tokens=1)
  ids=torch.tensor([[1,20,30,40]],device='cuda')
  before=torch.cuda.get_rng_state()
  probe.measure(ids[:,:3],2,torch.tensor([4],device='cuda'),signal)
  torch.testing.assert_close(before,torch.cuda.get_rng_state(),rtol=0,atol=0)
  backend.model.zero_grad(set_to_none=True)
  objective=DistillationObjective(backend,backend,config['algorithm'],2,True,confidence_probe=probe)
  loss,info=objective(ids,3,[1]);loss.backward()
  assert torch.isfinite(loss) and info['confidence_probe_token_loops']>0
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in backend.model.parameters())
  results.append(dict(family=family,signal=signal,bf16_current_teacher_gradients_finite=True,cuda_probe_rng_restored_exactly=True))
(audit/'cuda_probe_validation.json').write_text(json.dumps(dict(status='PASS',cases=results,scope='Random tiny CUDA architectures, not pretrained task results'),indent=2)+'\n')
print(json.dumps(results),flush=True)
