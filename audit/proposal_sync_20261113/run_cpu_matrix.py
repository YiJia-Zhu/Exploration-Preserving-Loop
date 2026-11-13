"""Execute small official architectures, never pretrained training or CUDA."""
from pathlib import Path
import json, os, subprocess, sys
root=Path(__file__).resolve().parents[2]
audit=Path(__file__).resolve().parent
configs=sorted(root.glob('configs/[abcd]_*.json'))+sorted(p for p in root.glob('configs/huginn_[abcd]_*.json') if not p.stem.endswith('_s16'))
runs=[]
for path in configs:
 output=root/'experiments/runs'/('20261113_proposal_'+path.stem+'_cpu')
 log=audit/(path.stem+'_cpu.log')
 command=[str(root/'.venv/bin/python'),'scripts/train_stage_distillation.py','--config',str(path),'--smoke','--device','cpu','--output',str(output)]
 with log.open('w') as f:
  result=subprocess.run(command,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:
  print(log.read_text()[-8000:],flush=True);raise SystemExit(result.returncode)
 summary=json.loads((output/'summary.json').read_text())
 assert summary['completed_updates']==2
 runs.append({'config':str(path.relative_to(root)),'run':str(output.relative_to(root)),'updates':2})
 print('PASS',path.stem,flush=True)
(audit/'cpu_matrix.json').write_text(json.dumps(runs,indent=2)+'\n')
