from pathlib import Path
import json,os,subprocess
root=Path(__file__).resolve().parents[2];audit=Path(__file__).resolve().parent
cases=json.loads((audit/'cpu_matrix.json').read_text())
for name in ('default','f_fixed_top_s','huginn_f_fixed_top_s','loopopd_recompute','huginn_loopopd_recompute'):
 out=root/'experiments/runs'/('20261113_dynamic_'+name+'_cpu')
 cmd=[str(root/'.venv/bin/python'),'scripts/train_stage_distillation.py','--smoke','--device','cpu','--output',str(out)]
 if name!='default':cmd+=['--config','configs/'+name+'.json']
 with (audit/(name+'_cpu.log')).open('w') as log:
  r=subprocess.run(cmd,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name)
 cases.append({'config':'configs/'+name+'.json','run':str(out.relative_to(root)),'updates':2})
 print('PASS',name,flush=True)
(audit/'cpu_matrix.json').write_text(json.dumps(cases,indent=2)+'\n')
fixture=audit/'eval_fixture.jsonl';fixture.write_text('\n'.join(json.dumps({'id':str(i),'problem':f'{i}+1?','answer':str(i+1)}) for i in (1,2))+'\n')
reports=[]
for family in ('ouro','huginn'):
 name='e_dynamic_js' if family=='ouro' else 'huginn_e_dynamic_js'
 ckpt=root/'experiments/runs'/('20261113_dynamic_'+name+'_cpu')/'checkpoints/update_000002'
 for mode in ('learned','native','fixed'):
  cfg=('configs/'+name+'.json') if mode!='native' else ('configs/native_base.json' if family=='ouro' else 'configs/huginn_native_base.json')
  output=audit/(family+'_'+mode+'_eval.jsonl')
  cmd=[str(root/'.venv/bin/python'),'scripts/evaluate_stage_distillation.py','--config',cfg,'--data',str(fixture),'--output',str(output),'--smoke','--device','cpu']
  if mode!='native':cmd+=['--checkpoint',str(ckpt)]
  if mode=='fixed':cmd+=['--loops','2']
  with (audit/(family+'_'+mode+'_eval.log')).open('w') as log:
   r=subprocess.run(cmd,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:
   print((audit/(family+'_'+mode+'_eval.log')).read_text()[-6000:]);raise RuntimeError(family+' '+mode)
  summary=json.loads(output.with_suffix('.summary.json').read_text());assert summary['smoke']
  result=next(iter(summary['results'].values()));assert result['problems']==2 and result['completions']==2
  reports.append({'family':family,'policy':mode,'report':str(output.with_suffix('.summary.json').relative_to(root))})
  print('PASS',family,mode,'evaluation',flush=True)
(audit/'evaluation_checks.json').write_text(json.dumps({'status':'PASS','cases':reports},indent=2)+'\n')
