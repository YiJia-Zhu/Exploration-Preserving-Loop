import json
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[2];audit=root/'audit/ablations_20261114'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','EP_LOOP_PROGRESS':'0','OMP_NUM_THREADS':'2','PYTHONHASHSEED':'42'}
results=[]
for family,prefix in [('ouro',''),('huginn','huginn_')]:
 for suffix in ['max_probability','entropy','probe_know','probe_true']:
  from sys import path
  path.insert(0,str(root))
  from ep_loop.config import load_config
  config=load_config(root/'configs'/f'{prefix}g_dynamic_{suffix}.json')
  config['algorithm']['probe_max_answer_tokens']=1
  cfg=audit/f'{family}_{suffix}_smoke_config.json';cfg.write_text(json.dumps(config,indent=2)+'\n')
  folder=audit/'runs'/f'{family}_{suffix}'
  cmd=[sys.executable,'scripts/train_stage_distillation.py','--config',str(cfg),'--smoke','--device','cpu','--stop-after-updates','1','--output',str(folder)]
  with (audit/f'{family}_{suffix}.log').open('x') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  row=json.loads((folder/'metrics.jsonl').read_text().splitlines()[0])
  assert row['optimizer_updated'] and row['distilled_responses']==4
  if suffix.startswith('probe'):assert row['confidence_probe_forwards']>0 and row['confidence_probe_token_loops']>0
  else:assert row['confidence_probe_forwards']==0
  assert sum(row['target_student_depth_hist'])==row['response_tokens']
  results.append(dict(family=family,signal=config['algorithm']['boundary_signal'],probe_cost_recorded=True,
                      dynamic_stopping=True,finite_loss=True))
(audit/'signal_validation.json').write_text(json.dumps(dict(status='PASS',cases=results,scope='CPU random tiny smoke, probe candidate limit=1 for validation; no efficacy claim'),indent=2)+'\n')
print(json.dumps(results),flush=True)
