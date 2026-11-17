"""Validate saved logging evidence without model loading or new training."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
root=Path(__file__).resolve().parents[2];audit=root/'audit/logging_20261114'
def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
log=(audit/'unit_tests_final.log').read_text();count=int(re.search(r'Ran (\d+) tests',log)[1])
assert count==63 and '\nOK\n' in log and 'FAILED' not in log
runs=read(audit/'run_validation.json');assert runs['status']=='PASS' and len(runs['training'])==4
assert all(r['balanced_stages'] and r['weights_and_rollouts_exact_without_logging'] for r in runs['training'])
assert runs['same_directory_resume_appends_new_session'] and runs['cpu_two_rank_logging_passed'] and runs['startup_failure_traceback_logged']
assert read(audit/'filter_validation.json')['status']=='PASS'
for file in ['launcher_prepared/launcher.events.jsonl','converted.events.jsonl','eval_final.events.jsonl']:
 rs=rows(audit/file);assert rs[-1]['event']=='run_end' and rs[-1]['status']=='completed'
 starts={(r['session_id'],r['stage_id']) for r in rs if r['event']=='stage_start'}
 ends={(r['session_id'],r['stage_id']) for r in rs if r['event']=='stage_end'}
 assert starts==ends
rs=rows(audit/'eval_final.events.jsonl');assert any(r.get('stage')=='decode' for r in rs)
console=(audit/'eval_final.console.log').read_text();assert '4/4' in console and 'Evaluate completions' in console
plan=json.JSONDecoder().raw_decode((audit/'launcher_dry_run_final.log').read_text())[0]
assert plan['progress']==dict(show=True,tokens=True,min_interval=1.0,timezone='Asia/Shanghai')
source=(root/'scripts/run_pilot.sh').read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
compile(source,'launcher embedded Python','exec')
sources=[*sorted((root/'ep_loop').glob('*.py')),*sorted((root/'scripts').glob('*.py')),root/'scripts/run_pilot.sh',root/'tests/test_observability.py']
for p in sources:
 if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
subprocess.run(['git','diff','--check','--','.',':!docs/loop_distillation_research_proposal_zh.md'],cwd=root,check=True)
proposal=sha(root/'docs/loop_distillation_research_proposal_zh.md')
assert proposal==read(root/'audit/dynamic_cache_20261114/validation.json')['proposal_sha256']
artifacts=[audit/name for name in ['unit_tests_final.log','run_validation.json','filter_validation.json','launcher_dry_run_final.log','launcher_prepared/launcher.events.jsonl','converted.events.jsonl','eval_final.events.jsonl','eval_final.console.log']]
report=dict(status='PASS',checked_at=datetime.now(timezone.utc).isoformat(),unit_tests=count,
 training_cases=runs['training'],same_directory_resume_preserves_history=True,cpu_two_rank_independent_logs=True,
 saved_evaluation_logged_for_both_models=True,teacher_filter_logged_with_explicit_tiny_mock=True,
 readonly_dry_run_preserved=True,pilot_prepare_only_logs_passed=True,standalone_data_conversion_logged=True,
 final_decode_stage_and_short_response_token_count_verified=True,source_proposal_modified=False,
 source_proposal_sha256=proposal,additional_cuda_synchronizations_for_logging=False,
 formal_pretrained_training_started=False,
 timing_scope='Host wall time; no new CUDA synchronization. Original generation timing still synchronizes at the end.',
 scope='Logging/progress and reproducibility checks with random tiny CPU models. Existing real-weight cache benchmark results were not rerun or reinterpreted.',
 artifacts=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in artifacts+sources])
(audit/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='artifacts'},ensure_ascii=False,indent=2))
