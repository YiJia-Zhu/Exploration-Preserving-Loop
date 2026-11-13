"""Check optimizer/RNG resume against already completed fresh CPU runs."""
from pathlib import Path
import json,os,subprocess,torch
from safetensors.torch import load_file
root=Path(__file__).resolve().parents[2];audit=Path(__file__).resolve().parent
reports=[]
for name in ('huginn_d_native_js','huginn_d_current_native_js'):
 baseline=root/'experiments/runs'/('20261113_proposal_'+name+'_cpu')
 output=root/'experiments/runs'/('20261113_proposal_'+name+'_resume_cpu')
 command=[str(root/'.venv/bin/python'),'scripts/train_stage_distillation.py','--config','configs/'+name+'.json','--smoke','--device','cpu','--output',str(output)]
 with (audit/(name+'_resume.log')).open('w') as log:
  for extra in (['--stop-after-updates','1'],['--resume',str(output/'checkpoints/update_000001')]):
   result=subprocess.run(command+extra,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:raise RuntimeError('Resume run failed: '+name)
 original=load_file(baseline/'checkpoints/update_000002/model.safetensors')
 restored=load_file(output/'checkpoints/update_000002/model.safetensors')
 assert original.keys()==restored.keys()
 assert all(torch.equal(original[k],restored[k]) for k in original)
 metrics=lambda p:[{k:v for k,v in json.loads(line).items() if k!='seconds'} for line in (p/'metrics.jsonl').read_text().splitlines()]
 assert metrics(baseline)==metrics(output)
 assert (baseline/'rollouts_rank0.jsonl').read_text()==(output/'rollouts_rank0.jsonl').read_text()
 reports.append({'config':name,'checkpoint_tensors_exact':len(original),'metrics_and_rollouts_exact':True,'teacher_strategy':json.loads((output/'resolved_config.json').read_text())['algorithm']['teacher_update']})
 print('PASS',name,flush=True)
(audit/'huginn_resume.json').write_text(json.dumps({'status':'PASS','cases':reports},indent=2)+'\n')
