import json
import os
from pathlib import Path
import subprocess
import sys
from safetensors.torch import load_file
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ep_loop.config import load_config
root=Path(__file__).resolve().parents[2];audit=root/'audit/ablations_20261114'
config=load_config(root/'configs/e_dynamic_js_student_correct.json');config['algorithm']['teacher_update']='current'
cfg=audit/'current_student_correct_config.json';cfg.write_text(json.dumps(config,indent=2)+'\n')
env={**os.environ,'EP_LOOP_PROGRESS':'0','CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'42','OMP_NUM_THREADS':'2'}
for mode in ['none','mixed']:
 folder=audit/'runs'/f'final_current_{mode}'
 cmd=[sys.executable,str(audit/'controlled_train.py'),mode,str(cfg),str(folder)]
 with (audit/f'final_current_{mode}.log').open('x') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 summary=json.loads((folder/'summary.json').read_text());rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
 expected=0 if mode=='none' else 2
 assert summary['attempted_updates']==2 and summary['optimizer_updates']==expected
 assert [r['teacher_weight_version'] for r in rows]==([0,0] if mode=='none' else [0,1])
 state=torch.load(folder/'checkpoints/update_000002/training_state.pt',weights_only=True,map_location='cpu');assert state['optimizer_updates']==expected
folder=audit/'runs'/'final_current_mixed';resumed=audit/'runs'/'final_current_resume'
cmd=[sys.executable,str(audit/'controlled_train.py'),'mixed',str(cfg),str(resumed),str(folder/'checkpoints/update_000001')]
with (audit/'final_current_resume.log').open('x') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
assert json.loads((resumed/'summary.json').read_text())['optimizer_updates']==2
expected=load_file(str(folder/'checkpoints/update_000002/model.safetensors'));actual=load_file(str(resumed/'checkpoints/update_000002/model.safetensors'))
for name in expected:torch.testing.assert_close(expected[name],actual[name],rtol=0,atol=0)
(audit/'counter_validation.json').write_text(json.dumps(dict(status='PASS',attempted_and_effective_updates_separate=True,current_teacher_weight_versions_correct=True,optimizer_count_and_weights_resume_exact=True),indent=2)+'\n')
