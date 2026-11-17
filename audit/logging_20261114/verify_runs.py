"""Validate logging on full/resumed CPU smoke training and saved evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from safetensors.torch import load_file
root=Path(__file__).resolve().parents[2];audit=root/'audit/logging_20261114'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'42','OMP_NUM_THREADS':'2'}
def launch(cmd,label,expected=0):
 with (audit/(label+'.outer.log')).open('x') as f:
  r=subprocess.run([sys.executable,*cmd],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==expected,(label,r.returncode)
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
def events(path):
 rs=rows(path);starts=[r for r in rs if r['event']=='stage_start'];ends=[r for r in rs if r['event']=='stage_end']
 assert {(r['session_id'],r['stage_id']) for r in starts}=={(r['session_id'],r['stage_id']) for r in ends}
 assert all(r['seconds']>=0 for r in ends)
 assert rs[-1]['event']=='run_end'
 return rs
results=[]
for family in ('ouro','huginn'):
 for teacher in ('frozen','current'):
  label=f'{family}_{teacher}';folder=audit/label
  cfg='configs/'+('huginn_' if family=='huginn' else '')+'e_dynamic_js'+('_current' if teacher=='current' else '')+'.json'
  launch(['scripts/train_stage_distillation.py','--config',cfg,'--smoke','--device','cpu','--output',str(folder)],label)
  rs=events(folder/'events_rank0.jsonl');assert rs[-1]['status']=='completed'
  phases={r['stage'] for r in rs if r['event']=='stage_start'}
  assert {'generation','prefill','teacher_forward','stage_planning','student_replay','loss_readout','backward','optimizer_step','checkpoint_save'}<=phases
  assert len([r for r in rs if r['event']=='stage_end' and r['stage']=='generation'])==8
  console=(folder/'console_rank0.log').read_text();assert 'Train updates' in console and 'tok/s' in console and 'rank 0 responses' in console
  previous=root/'audit/dynamic_cache_20261114/runs'/f'{family}_{teacher}_full'
  old=load_file(str(previous/'checkpoints/update_000002/model.safetensors'));new=load_file(str(folder/'checkpoints/update_000002/model.safetensors'))
  assert old.keys()==new.keys()
  for name in old:torch.testing.assert_close(old[name],new[name],rtol=0,atol=0)
  assert rows(folder/'rollouts_rank0.jsonl')==rows(previous/'rollouts_rank0.jsonl')
  results.append(dict(family=family,teacher=teacher,balanced_stages=True,eight_generations_logged=True,weights_and_rollouts_exact_without_logging=True))
# Resume into the SAME directory: logs must append a new session, not overwrite.
first=audit/'train_first'
launch(['scripts/train_stage_distillation.py','--smoke','--device','cpu','--output',str(first),'--resume',str(first/'checkpoints/update_000001')],'resume_same_directory')
rs=events(first/'events_rank0.jsonl');assert len({r['session_id'] for r in rs})==2
assert any(r.get('stage')=='resume_optimizer_rng' for r in rs)
expected=load_file(str(audit/'ouro_frozen/checkpoints/update_000002/model.safetensors'))
actual=load_file(str(first/'checkpoints/update_000002/model.safetensors'))
for name in expected:torch.testing.assert_close(expected[name],actual[name],rtol=0,atol=0)
# Evaluation progress and generation events, using both saved model families.
data=audit/'eval_input.jsonl';data.write_text(json.dumps(dict(id='one',problem='What is 1+1?',answer='2'))+'\n')
for family in ('ouro','huginn'):
 cfg='configs/'+('huginn_' if family=='huginn' else '')+'e_dynamic_js.json'
 out=audit/f'eval_{family}.jsonl'
 launch(['scripts/evaluate_stage_distillation.py','--config',cfg,'--smoke','--device','cpu','--checkpoint',str(audit/f'{family}_frozen/checkpoints/update_000002'),'--data',str(data),'--output',str(out)],f'eval_{family}')
 rs=events(out.with_suffix('.events.jsonl'));assert rs[-1]['status']=='completed'
 assert {'evaluation_mode','evaluation_sample','generation','prefill'}<={r['stage'] for r in rs if r['event']=='stage_start'}
 assert 'Evaluate completions' in out.with_suffix('.console.log').read_text()
# Two CPU distributed ranks: independent complete logs and only rank-0 bars.
ddp=audit/'ddp'
launch(['-m','torch.distributed.run','--standalone','--nproc-per-node=2','scripts/train_stage_distillation.py','--smoke','--device','cpu','--stop-after-updates','1','--output',str(ddp)],'ddp')
for rank in (0,1):
 rs=events(ddp/f'events_rank{rank}.jsonl');assert all(r['rank']==rank for r in rs)
 assert rs[-1]['status']=='completed'
 if rank==1:assert 'Train updates' not in (ddp/f'console_rank{rank}.log').read_text()
# Startup failures still persist the traceback and close the active phase.
failed=audit/'expected_failure'
launch(['scripts/train_stage_distillation.py','--config',str(audit/'missing.json'),'--smoke','--device','cpu','--output',str(failed)],'expected_failure',1)
rs=events(failed/'events_rank0.jsonl');assert rs[-1]['status']=='failed' and 'FileNotFoundError' in rs[-1]['traceback']
report=dict(status='PASS',training=results,same_directory_resume_appends_new_session=True,resumed_weights_exact=True,two_family_evaluation_logged=True,cpu_two_rank_logging_passed=True,startup_failure_traceback_logged=True)
(audit/'run_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
