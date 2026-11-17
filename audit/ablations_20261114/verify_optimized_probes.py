import json
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[2];audit=root/'audit/ablations_20261114'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','EP_LOOP_PROGRESS':'0','OMP_NUM_THREADS':'2','PYTHONHASHSEED':'42'}
results=[]
for family in ['ouro','huginn']:
 for signal in ['probe_know','probe_true']:
  cfg=audit/f'{family}_{signal}_smoke_config.json';folder=audit/'runs'/f'optimized_{family}_{signal}'
  cmd=[sys.executable,'scripts/train_stage_distillation.py','--config',str(cfg),'--smoke','--device','cpu','--stop-after-updates','1','--output',str(folder)]
  with (audit/f'optimized_{family}_{signal}.log').open('x') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  row=json.loads((folder/'metrics.jsonl').read_text().splitlines()[0])
  old=json.loads((audit/'runs'/f'{family}_{signal}'/'metrics.jsonl').read_text().splitlines()[0])
  assert row['optimizer_updated'] and row['confidence_probe_forwards']<=old['confidence_probe_forwards']
  assert row['confidence_probe_token_loops']>0 and row['optimizer_updates_so_far']==1
  results.append(dict(family=family,signal=signal,old_probe_forwards=old['confidence_probe_forwards'],optimized_probe_forwards=row['confidence_probe_forwards'],finite_training=True))
(audit/'optimized_probe_validation.json').write_text(json.dumps(dict(status='PASS',cases=results,scope='Final optimized probe smoke on random tiny CPU models'),indent=2)+'\n');print(json.dumps(results),flush=True)
