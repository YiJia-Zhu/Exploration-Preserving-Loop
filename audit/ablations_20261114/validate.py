"""Assemble ablation validation evidence; no pretrained runs or model loads."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
root=Path(__file__).resolve().parents[2];audit=root/'audit/ablations_20261114';sys.path.insert(0,str(root))
from ep_loop.config import load_config,validate_config

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((audit/name).read_text())
log=(audit/'unit_tests_final.log').read_text();count=int(re.search(r'Ran (\d+) tests',log)[1])
assert count==71 and '\nOK\n' in log and 'FAILED' not in log
reports=['training_validation.json','signal_validation.json','optimized_probe_validation.json','cuda_probe_validation.json','counter_validation.json','teacher_pool_validation.json','default_regression_validation.json']
for name in reports:assert read(name)['status']=='PASS'
assert len(read('signal_validation.json')['cases'])==8
assert len(read('optimized_probe_validation.json')['cases'])==4
assert len(read('cuda_probe_validation.json')['cases'])==4
for p in (root/'configs').glob('*.json'):validate_config(load_config(p))
source=(root/'scripts/run_pilot.sh').read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0];compile(source,'launcher embedded Python','exec')
for label,expect in [('default_dry_run_final.log',('js','all','all')),('teacher_probe_dry_run.log',('probe_true','teacher_correct','all')),('random_student_dry_run.log',('js','random_matched','student_correct'))]:
 plan=json.JSONDecoder().raw_decode((audit/label).read_text())[0]
 assert tuple(plan[k] for k in ['boundary_signal','question_filter','response_filter'])==expect
 assert not plan['initial_fixed_references']
raw=(root/'docs/loop_distillation_research_proposal_zh.md').read_bytes();assert b'6.4 ' in raw and raw.count(b'\r\n')==raw.count(b'\n')
subprocess.run(['git','diff','--check','--','.',':!docs/loop_distillation_research_proposal_zh.md'],cwd=root,check=True)
sources=[*sorted((root/'ep_loop').glob('*.py')),*sorted((root/'scripts').glob('*.py')),root/'scripts/run_pilot.sh',root/'tests/test_ablations.py']
for p in sources:
 if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
artifacts=[audit/p for p in reports+['unit_tests_final.log','default_dry_run_final.log','teacher_probe_dry_run.log','random_student_dry_run.log','diagnostic_tiny_final.jsonl','diagnostic_tiny_final.summary.json']]
report=dict(status='PASS',checked_at=datetime.now(timezone.utc).isoformat(),unit_tests=count,
 default_boundary_signal='js',default_question_and_response_filters='all',student_policy='learned per-token stopping',
 correctness_modes=['teacher_correct question pool','random matched-size question pool','student_correct current trajectory'],
 signal_modes=['js','max_probability','normalized_entropy','probe_know','probe_true'],
 main_proposal_updated=True,main_proposal_crlf_preserved=True,main_proposal_sha256=sha(root/'docs/loop_distillation_research_proposal_zh.md'),
 default_js_weights_and_rollouts_unchanged=True,
 zero_acceptance_skips_adamw=True,ddp_global_accepted_response_normalization=True,
 accepted_count_checkpoint_resume_verified=True,probe_rng_isolation_cuda_bf16_verified=True,
 optimized_probe_all_depths_read_from_one_trajectory=True,
 scope='Semantic/configuration verification only: 71 unit tests, controlled synthetic correctness training, tiny CPU/DDP smoke, tiny CUDA BF16 gradients/RNG and frozen teacher pool checks. No pretrained task training, confidence calibration or efficacy result.',
 formal_pretrained_training_started=False,new_method_efficacy_established=False,
 evidence=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in artifacts+sources])
(audit/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='evidence'},ensure_ascii=False,indent=2))
