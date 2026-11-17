import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from safetensors.torch import load_file
root=Path(__file__).resolve().parents[2];audit=root/'audit/ablations_20261114'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','EP_LOOP_PROGRESS':'0','OMP_NUM_THREADS':'2','PYTHONHASHSEED':'42'}
script=str(audit/'controlled_train.py');configs={}
for family in ['ouro','huginn']:
 prefix='huginn_' if family=='huginn' else ''
 configs[family]='configs/'+prefix+'e_dynamic_js_student_correct.json'
cases=[]
def run(cmd,label):
 with (audit/(label+'.log')).open('x') as log:subprocess.run([sys.executable,*cmd],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
for family in ['ouro','huginn']:
 for mode in ['all','mixed','none','truncated']:
  folder=audit/'runs'/f'{family}_{mode}'
  run([script,mode,configs[family],str(folder)],f'{family}_{mode}')
  metrics=rows(folder/'metrics.jsonl')
  expected={'all':4,'mixed':2,'none':0,'truncated':0}[mode]
  assert all(row['distilled_responses']==expected and row['optimizer_updated']==bool(expected) for row in metrics)
  if not expected:
   first=load_file(str(folder/'checkpoints/update_000001/model.safetensors'));last=load_file(str(folder/'checkpoints/update_000002/model.safetensors'))
   for name in first:torch.testing.assert_close(first[name],last[name],rtol=0,atol=0)
  cases.append(dict(family=family,case=mode,accepted_per_batch=expected,optimizer_skip_verified=not expected))
 # Stochastic Huginn resumed filtering retains the saved model/RNG state.
 source=audit/'runs'/f'{family}_mixed';dest=audit/'runs'/f'{family}_resumed'
 run([script,'mixed',configs[family],str(dest),str(source/'checkpoints/update_000001')],f'{family}_resumed')
 expected=load_file(str(source/'checkpoints/update_000002/model.safetensors'));actual=load_file(str(dest/'checkpoints/update_000002/model.safetensors'))
 for name in expected:torch.testing.assert_close(expected[name],actual[name],rtol=0,atol=0)
# One rank entirely rejected and one rank accepted; also all ranks rejected.
for mode in ['mixed','none']:
 folder=audit/'runs'/f'ddp_{mode}'
 run(['-m','torch.distributed.run','--standalone','--nproc-per-node=2',script,mode,configs['ouro'],str(folder)],f'ddp_{mode}')
 metrics=rows(folder/'metrics.jsonl');assert all(row['distilled_responses']==(2 if mode=='mixed' else 0) for row in metrics)
 expected=load_file(str(audit/'runs'/f'ouro_{mode}/checkpoints/update_000002/model.safetensors'));actual=load_file(str(folder/'checkpoints/update_000002/model.safetensors'))
 for name in expected:torch.testing.assert_close(expected[name],actual[name],rtol=2e-5,atol=1e-6,msg=name)
report=dict(status='PASS',cases=cases,two_family_filter_resume_exact=True,ddp_empty_rank_global_accepted_mean_matches_single_rank=True,ddp_all_rejected_optimizer_skipped=True,
 scope='Controlled synthetic student outputs with real tiny objectives/backward. Not pretrained correctness rates.')
(audit/'training_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
