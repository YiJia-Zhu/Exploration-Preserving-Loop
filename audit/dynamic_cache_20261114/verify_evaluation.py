"""Saved dynamic students use exact KV in evaluation, independent of ordering."""
import json
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[2];audit=root/'audit/dynamic_cache_20261114'
records=[dict(id='first',problem='What is 1+1?',answer='2'),dict(id='second',problem='What is 2+1?',answer='3')]
results=[]
for order,rs in [('forward',records),('reverse',records[::-1])]:
 path=audit/f'eval_{order}_input.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rs))
 for family in ('ouro','huginn'):
  config='configs/'+('huginn_' if family=='huginn' else '')+'e_dynamic_js.json'
  checkpoint=audit/'runs'/f'{family}_frozen_full/checkpoints/update_000002'
  output=audit/f'eval_{family}_{order}.jsonl'
  cmd=[sys.executable,'scripts/evaluate_stage_distillation.py','--config',config,'--checkpoint',str(checkpoint),'--data',str(path),'--output',str(output),'--smoke','--device','cpu']
  with (audit/f'eval_{family}_{order}.log').open('x') as f:
   subprocess.run(cmd,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'42','OMP_NUM_THREADS':'2'},stdout=f,stderr=subprocess.STDOUT,check=True)
for family in ('ouro','huginn'):
 def load(order):
  return {r['id']:{k:v for k,v in r.items() if k!='generation_seconds'} for r in [json.loads(line) for line in (audit/f'eval_{family}_{order}.jsonl').read_text().splitlines()]}
 forward,reverse=load('forward'),load('reverse');assert forward==reverse
 assert all(r['cache_policy']=='exact_lazy_depth' for r in forward.values())
 results.append(dict(family=family,cached_evaluation_passed=True,per_problem_results_exact_after_reordering=True))
(audit/'evaluation_validation.json').write_text(json.dumps(dict(status='PASS',cases=results),indent=2)+'\n')
print(json.dumps(results),flush=True)
