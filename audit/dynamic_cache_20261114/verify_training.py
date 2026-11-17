"""Tiny CPU training, repeat and optimizer/RNG resume with exact dynamic KV."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from safetensors.torch import load_file

root=Path(__file__).resolve().parents[2]
audit=root/'audit/dynamic_cache_20261114'
results=[]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'42','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
def clean(row):return {key:value for key,value in row.items() if key not in {'seconds','generation_seconds_sum'}}
for family in ('ouro','huginn'):
 for teacher in ('frozen','current'):
  cfg=('huginn_' if family=='huginn' else '')+'e_dynamic_js'+('_current' if teacher=='current' else '')+'.json'
  folders={mode:audit/'runs'/f'{family}_{teacher}_{mode}' for mode in ('full','repeat','resume')}
  for mode,folder in folders.items():
   cmd=[sys.executable,'scripts/train_stage_distillation.py','--config','configs/'+cfg,'--smoke','--device','cpu','--output',str(folder)]
   if mode=='resume':cmd+=['--resume',str(folders['full']/'checkpoints/update_000001')]
   with (audit/f'{family}_{teacher}_{mode}.log').open('x') as log:
    subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  baseline=rows(folders['full']/'metrics.jsonl')
  rollouts=rows(folders['full']/'rollouts_rank0.jsonl')
  assert all(row['cache_policy']=='exact_lazy_depth' for row in rollouts)
  for mode in ('repeat','resume'):
   compare=rows(folders[mode]/'metrics.jsonl')
   assert [clean(row) for row in compare]==[clean(row) for row in baseline[-len(compare):]]
   compare_rollouts=rows(folders[mode]/'rollouts_rank0.jsonl')
   assert compare_rollouts==rollouts[-len(compare_rollouts):]
   expected=load_file(str(folders['full']/'checkpoints/update_000002/model.safetensors'))
   actual=load_file(str(folders[mode]/'checkpoints/update_000002/model.safetensors'))
   assert expected.keys()==actual.keys()
   for key in expected:torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
  result=dict(family=family,teacher=teacher,repeated_metrics_and_weights_exact=True,resumed_metrics_and_weights_exact=True,updates=2)
  results.append(result);print(json.dumps(result),flush=True)
(audit/'training_validation.json').write_text(json.dumps(dict(status='PASS',cases=results),indent=2)+'\n')
