"""Prepare method-ablation configs and verify stage metadata without weights/GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[2];audit=root/'audit/ablation_workflow_20261114'
source=audit/'source.jsonl';source.write_text(''.join(json.dumps(dict(id=str(i),messages=[dict(role='user',content=f'{i}+1?')],answer=str(i+1)))+'\n' for i in range(3)))
base={**os.environ,'SHOW_PROGRESS':'0','DATA_SOURCE':str(source),'SOURCE_EXPECTED_EXAMPLES':'3','TRAIN_EXAMPLES':'2','DEV_EXAMPLES':'1','EVAL_EXAMPLES':'1','PROMPTS_PER_BATCH':'1','RESPONSES_PER_PROMPT':'1','GPUS':'7','OUTPUT_ROOT':str(audit/'prepared')}
results=[]
for family,selection,qfilter in [('ouro','threshold','all'),('ouro','threshold_uniform','all'),('ouro','threshold_endpoint','all'),('huginn','threshold_uniform','teacher_correct')]:
 tag=f'{family}_{selection}_{qfilter}'
 env={**base,'MODEL_FAMILY':family,'STAGE_SELECTION':selection,'QUESTION_FILTER':qfilter,'RUN_TAG':tag,'SPLIT_DIR':str(audit/('split_'+family))}
 with (audit/(tag+'.log')).open('x') as log:
  subprocess.run(['bash','scripts/run_ablation.sh','--prepare-only'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 folder=audit/'prepared'/tag
 plan=json.loads((folder/'launcher_parameters.json').read_text());config=json.loads((folder/'configs/ours.json').read_text())
 assert plan['experiment_stage']=='ablation' and plan['methods']==['ours']
 assert config['experiment']['stage']=='ablation' and config['algorithm']['stage_selection']==selection
 assert config['algorithm']['student_policy']=='learned' and config['algorithm']['student_loops'] is None
 assert config['distillation']['question_filter']==qfilter
 assert not (folder/'ours').exists() and not (folder/'teacher_correctness.jsonl').exists()
 events=[json.loads(line) for line in (folder/'launcher.events.jsonl').read_text().splitlines()]
 assert events[-1]['event']=='run_end' and events[-1]['status']=='completed'
 assert events[0]['name']=='ablation'
 results.append(dict(family=family,stage_selection=selection,question_filter=qfilter,config_stage='ablation',training_started=False))
(audit/'prepare_validation.json').write_text(json.dumps(dict(status='PASS',cases=results),indent=2)+'\n')
print(json.dumps(results),flush=True)
