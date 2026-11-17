"""Assemble independently inspectable cache validation evidence; no model runs."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
root=Path(__file__).resolve().parents[2];audit=root/'audit/dynamic_cache_20261114'
def read(name):return json.loads((audit/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
log=(audit/'unit_tests_final.log').read_text();count=int(re.search(r'Ran (\d+) tests',log)[1])
assert count==58 and '\nOK\n' in log and 'FAILED' not in log
training=read('training_validation.json');assert training['status']=='PASS' and len(training['cases'])==4
assert all(c['repeated_metrics_and_weights_exact'] and c['resumed_metrics_and_weights_exact'] for c in training['cases'])
for name in ['cuda_gradient_validation.json','evaluation_validation.json']:
 report=read(name);assert report['status']=='PASS' and len(report['cases'])==2
bench_files=['pretrained_ouro_bf16_cast_cache.json','pretrained_huginn_bf16.json','pretrained_ouro_fp32.json','pretrained_huginn_fp32.json','tiny_benchmark.json']
bench=[];proposal_sha=sha(root/'docs/loop_distillation_research_proposal_zh.md')
for name in bench_files:
 report=read(name);assert report['status']=='PASS' and report['proposal_sha256']==proposal_sha
 for c in report['cases']:
  assert c['validation']=='PASS' and c['max_distribution_js_nats']<=c['js_acceptance_threshold']
  assert c['cached_token_loops'] < c['recompute_token_loops']
 bench.append(dict(file=name,cases=report['cases']))
for family in ('ouro','huginn'):
 from sys import path
 path.insert(0,str(root))
 from ep_loop.config import load_config,validate_config
 config=validate_config(load_config(root/'configs'/('huginn_' if family=='huginn' else '').__add__('e_dynamic_js.json')))
 assert config['seed']==42 and config['rollout']['use_cache'] and config['evaluation']['use_cache']
for name,expected in [('launcher_default.log',True),('launcher_recompute.log',False)]:
 text=(audit/name).read_text();plan=json.JSONDecoder().raw_decode(text)[0]
 assert plan['ours_use_cache'] is expected and plan['seed']==42
sources=[*sorted((root/'ep_loop').glob('*.py')),root/'scripts/benchmark_dynamic_cache.py',root/'scripts/run_pilot.sh',root/'scripts/evaluate_stage_distillation.py']
for source in sources:
 if source.suffix=='.py':compile(source.read_text(),str(source),'exec')
subprocess.run(['git','diff','--check','--','.',':!docs/loop_distillation_research_proposal_zh.md'],cwd=root,check=True)
artifacts=[audit/name for name in bench_files+['unit_tests_final.log','training_validation.json','cuda_gradient_validation.json','evaluation_validation.json','launcher_default.log','launcher_recompute.log']]
report=dict(status='PASS',checked_at=datetime.now(timezone.utc).isoformat(),unit_tests=count,
 default_seed=42,default_student_cache='exact_lazy_depth',sglang_or_vllm_integrated=False,
 authoritative_proposal_modified=False,proposal_sha256=proposal_sha,
 two_family_full_precision_and_bf16_pretrained_checks=True,
 cached_training_repeat_and_resume=training['cases'],
 cuda_shared_model_gradient_validation=read('cuda_gradient_validation.json')['cases'],
 cached_checkpoint_evaluation=read('evaluation_validation.json')['cases'],benchmarks=bench,
 whitespace_check='PASS excluding pre-existing CRLF edits in user-owned authoritative proposal',
 retry_note='Optional Huginn FP32 loading on GPU7 encountered concurrent external GPU allocation; retry on GPU6 passed. Failed attempt log retained.',
 formal_pretrained_training_started=False,task_gain_established=False,
 scope='Exact cache implementation and controlled runtime validation only. FP32 numerical checks and same-initial-state Huginn comparisons passed; BF16 arithmetic varies with forward shape. Timings force exits/output tokens, exclude sampling and loading, and are not a LoopOPD task comparison. Huginn pretrained checks use K4, with K32 fill verified on tiny architecture.',
 artifacts=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in artifacts+sources])
(audit/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in {'artifacts','benchmarks'}},ensure_ascii=False,indent=2))
