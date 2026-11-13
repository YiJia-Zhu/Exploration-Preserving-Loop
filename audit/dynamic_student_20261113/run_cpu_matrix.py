"""Run only random tiny CPU models; preserve separate per-run source snapshots."""
from pathlib import Path
import json,os,subprocess
root=Path(__file__).resolve().parents[2];audit=Path(__file__).resolve().parent
paths=[*sorted(root.glob('configs/[cde]_dynamic_*.json')),*sorted(root.glob('configs/huginn_[cde]_dynamic_*.json'))]
paths += [root/'configs'/name for name in ('a_loopopd.json','a_d_loopopd.json','d_native_js.json','huginn_a_loopopd.json','huginn_a_d_loopopd.json','huginn_d_native_js.json')]
for family in ('ouro','huginn'):
 for label,tau in (('all_stages',0.),('one_stage',1.)):
  path=audit/f'{family}_{label}.json'
  path.write_text(json.dumps({'extends':str(root/'configs'/('e_dynamic_js.json' if family=='ouro' else 'huginn_e_dynamic_js.json')),'run_name':f'{family}_{label}','algorithm':{'js_threshold':tau}})+'\n')
  paths.append(path)
cases=[]
for path in paths:
 output=root/'experiments/runs'/('20261113_dynamic_'+path.stem+'_cpu')
 log=audit/(path.stem+'_cpu.log')
 command=[str(root/'.venv/bin/python'),'scripts/train_stage_distillation.py','--config',str(path),'--smoke','--device','cpu','--output',str(output)]
 with log.open('w') as f:
  result=subprocess.run(command,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:
  print(log.read_text()[-8000:],flush=True);raise SystemExit(result.returncode)
 summary=json.loads((output/'summary.json').read_text());assert summary['status']=='completed' and summary['completed_updates']==2
 cases.append({'config':str(path.relative_to(root)),'run':str(output.relative_to(root)),'updates':2})
 print('PASS',path.stem,flush=True)
(audit/'cpu_matrix.json').write_text(json.dumps(cases,indent=2)+'\n')
