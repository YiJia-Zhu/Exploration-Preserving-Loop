"""Short CPU forward integration, no pretrained optimizer or benchmark."""
from pathlib import Path
import gc,json,time,torch
root=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(root))
from ep_loop.backends import load_backend
from ep_loop.config import load_config
reports=[];torch.set_num_threads(4)
for family,name in [('ouro','e_dynamic_js.json'),('huginn','huginn_e_dynamic_js.json')]:
 c=load_config(root/'configs'/name);started=time.perf_counter()
 b,loading=load_backend(c['model'],root/c['model']['path'],'cpu','none',False)
 b.enable_student_exit();b.model.eval().requires_grad_(False)
 ids=torch.tensor([[1,20,30]]) if family=='ouro' else torch.tensor([[65504,123,456]])
 with torch.no_grad():
  torch.manual_seed(95)
  dynamic=b.trace(ids,4,endpoint='learned',exit_threshold=.5)
  actual=int(dynamic.exits[0,-1])
  torch.manual_seed(95)
  fixed=b.trace(ids,actual)
  torch.testing.assert_close(b.head(dynamic.states[actual-1]),b.head(fixed.states[-1]),rtol=0,atol=0)
 reports.append({'family':family,'revision':c['model']['revision'],'device':'cpu','actual_executed_depth':actual,'token_loops':dynamic.executed_token_loops,'official_fixed_prefix_equivalent':True,'seconds':time.perf_counter()-started,'loading':loading})
 print('PASS',family,actual,flush=True)
 del b,dynamic,fixed;gc.collect()
(root/'audit/dynamic_student_20261113/pretrained_forward.json').write_text(json.dumps({'status':'PASS','cases':reports,'scope':'Short CPU forward with base parameters; no training, no benchmark, Huginn student stop head is untrained'},indent=2)+'\n')
