"""Tiny CPU checks only: seeded repeats, resume, and evaluation ordering."""
from pathlib import Path
import json,os,subprocess,torch
from safetensors.torch import load_file
root=Path(__file__).resolve().parents[2];audit=Path(__file__).resolve().parent
python=str(root/'.venv/bin/python')
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'42'}
def call(command,log):
 with (audit/log).open('w') as f:
  result=subprocess.run([python,*command],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:
  raise RuntimeError((audit/log).read_text()[-6000:])
def checkpoint_equal(a,b):
 x,y=load_file(a/'model.safetensors'),load_file(b/'model.safetensors')
 assert x.keys()==y.keys() and all(torch.equal(x[key],y[key]) for key in x)
 return len(x)
def metrics(path):
 return [{k:v for k,v in json.loads(line).items() if 'seconds' not in k} for line in (path/'metrics.jsonl').read_text().splitlines()]
reports=[]
for name in ('e_dynamic_js','e_dynamic_js_current','huginn_e_dynamic_js','huginn_e_dynamic_js_current'):
 folder=root/'experiments/runs'/('20261114_seed42_'+name)
 base=['scripts/train_stage_distillation.py','--config','configs/'+name+'.json','--smoke','--device','cpu']
 call(base+['--output',str(folder)],name+'_full.log')
 repeat=root/'experiments/runs'/('20261114_seed42_'+name+'_repeat')
 call(base+['--output',str(repeat)],name+'_repeat.log')
 count=checkpoint_equal(folder/'checkpoints/update_000002',repeat/'checkpoints/update_000002')
 assert metrics(folder)==metrics(repeat)
 assert (folder/'rollouts_rank0.jsonl').read_bytes()==(repeat/'rollouts_rank0.jsonl').read_bytes()
 resumed=root/'experiments/runs'/('20261114_seed42_'+name+'_resumed')
 call(base+['--output',str(resumed),'--resume',str(folder/'checkpoints/update_000001')],name+'_resume.log')
 checkpoint_equal(folder/'checkpoints/update_000002',resumed/'checkpoints/update_000002')
 assert metrics(folder)[1:]==metrics(resumed)
 assert (folder/'rollouts_rank0.jsonl').read_text().splitlines()[4:]==(resumed/'rollouts_rank0.jsonl').read_text().splitlines()
 state=torch.load(folder/'checkpoints/update_000002/rng_rank0.pt',map_location='cpu',weights_only=True)
 assert {'python_rng','numpy_rng','torch_rng','cuda_rng','data_rng','rollout_rng'} <= state.keys()
 reports.append({'config':name,'checkpoint_tensors_exact':count,'same_seed_repeat_exact':True,'resume_exact':True,'all_rng_streams_saved':True})
 print('PASS repeat + resume',name,flush=True)
(audit/'training_checks.json').write_text(json.dumps({'status':'PASS','cases':reports},indent=2)+'\n')
raw=[{'id':'a','problem':'What is 1+1?','answer':'2'},{'id':'b','problem':'What is 2+1?','answer':'3'}]
for label,rows in [('forward',raw),('reverse',list(reversed(raw)))]:
 (audit/(label+'.jsonl')).write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
evaluation=[]
for family,name in [('ouro','e_dynamic_js'),('huginn','huginn_e_dynamic_js')]:
 checkpoint=root/'experiments/runs'/('20261114_seed42_'+name)/'checkpoints/update_000002'
 outputs=[]
 for order in ('forward','reverse'):
  output=audit/(family+'_'+order+'_eval.jsonl')
  call(['scripts/evaluate_stage_distillation.py','--config','configs/'+name+'.json','--checkpoint',str(checkpoint),'--data',str(audit/(order+'.jsonl')),'--output',str(output),'--smoke','--device','cpu','--samples-per-problem','2'],family+'_'+order+'_eval.log')
  values=[json.loads(line) for line in output.read_text().splitlines()]
  outputs.append({(v['id'],v['sample']):{k:value for k,value in v.items() if 'seconds' not in k} for v in values})
 assert outputs[0]==outputs[1]
 evaluation.append({'family':family,'question_order_independent':True,'samples_compared':4})
 print('PASS evaluation order',family,flush=True)
(audit/'evaluation_checks.json').write_text(json.dumps({'status':'PASS','cases':evaluation},indent=2)+'\n')
