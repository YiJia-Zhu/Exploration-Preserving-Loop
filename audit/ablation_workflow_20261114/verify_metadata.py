"""Check shared Python train/eval propagate ablation stage on random tiny models."""
import json
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[2];audit=root/'audit/ablation_workflow_20261114'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','EP_LOOP_PROGRESS':'0','PYTHONHASHSEED':'42','OMP_NUM_THREADS':'2'}
config=audit/'prepared/ouro_threshold_all/configs/ours.json';folder=audit/'metadata_train'
with (audit/'metadata_train.log').open('x') as log:
 subprocess.run([sys.executable,'scripts/train_stage_distillation.py','--config',str(config),'--smoke','--device','cpu','--stop-after-updates','1','--output',str(folder)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for name in ['provenance.json','summary.json','resolved_config.json']:
 assert json.loads((folder/name).read_text())['experiment']['stage']=='ablation'
output=audit/'metadata_eval.jsonl'
with (audit/'metadata_eval.log').open('x') as log:
 subprocess.run([sys.executable,'scripts/evaluate_stage_distillation.py','--config',str(config),'--checkpoint',str(folder/'checkpoints/update_000001'),'--data',str(audit/'source.jsonl'),'--output',str(output),'--smoke','--device','cpu'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
assert json.loads(output.with_suffix('.summary.json').read_text())['experiment']['stage']=='ablation'
(audit/'metadata_validation.json').write_text(json.dumps(dict(status='PASS',train_provenance_stage='ablation',train_summary_stage='ablation',evaluation_summary_stage='ablation',scope='Random tiny CPU model only, generic Python entrypoints retained'),indent=2)+'\n')
