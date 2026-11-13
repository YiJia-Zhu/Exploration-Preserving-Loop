"""Exact dynamic-policy optimizer/RNG resume for both families and teachers."""
from pathlib import Path
import json,os,subprocess,torch
from safetensors.torch import load_file
root=Path(__file__).resolve().parents[2];audit=Path(__file__).resolve().parent
reports=[]
for name in ('e_dynamic_js','e_dynamic_js_current','huginn_e_dynamic_js','huginn_e_dynamic_js_current'):
 baseline=root/'experiments/runs'/('20261113_dynamic_'+name+'_cpu')
 output=root/'experiments/runs'/('20261113_dynamic_'+name+'_resume_cpu')
 command=[str(root/'.venv/bin/python'),'scripts/train_stage_distillation.py','--config','configs/'+name+'.json','--smoke','--device','cpu','--output',str(output)]
 with (audit/(name+'_resume.log')).open('w') as log:
  for extra in (['--stop-after-updates','1'],['--resume',str(output/'checkpoints/update_000001')]):
   result=subprocess.run(command+extra,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:raise RuntimeError('Resume failed: '+name)
 first=load_file(baseline/'checkpoints/update_000002/model.safetensors');second=load_file(output/'checkpoints/update_000002/model.safetensors')
 assert first.keys()==second.keys() and all(torch.equal(first[k],second[k]) for k in first)
 metrics=lambda p:[{k:v for k,v in json.loads(line).items() if 'seconds' not in k} for line in (p/'metrics.jsonl').read_text().splitlines()]
 assert metrics(baseline)==metrics(output)
 assert (baseline/'rollouts_rank0.jsonl').read_text()==(output/'rollouts_rank0.jsonl').read_text()
 reports.append({'config':name,'all_tensors_exact':len(first),'non_timing_metrics_and_rollouts_exact':True})
 print('PASS',name,flush=True)
(audit/'resume.json').write_text(json.dumps({'status':'PASS','cases':reports},indent=2)+'\n')
