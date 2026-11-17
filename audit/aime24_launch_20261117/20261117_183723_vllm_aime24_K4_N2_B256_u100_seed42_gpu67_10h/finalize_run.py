"""Clean only this run's RAM volume after its owned processes have exited."""
from datetime import datetime
import json,os,shutil,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0,sys.argv[2])
from ep_loop.process_control import experiment_processes,process_record
launch=Path(sys.argv[1]);record=json.loads(launch.read_text());pid=record['pid']
identity=process_record(pid);known={};panel=Path(record['panel'])
def alive(row):
    current=process_record(row['pid'])
    return current and current['start_ticks']==row['start_ticks'] and current['state']!='Z'
while identity and alive(identity):
    for row in experiment_processes(pid,exclude=(os.getpid(),)):known[(row['pid'],row['start_ticks'])]=row
    time.sleep(5)
while any(alive(row) for row in known.values()):time.sleep(5)
trace=Path(record['teacher_trace_dir']);cleaned=False
if trace.exists() and trace.parent==Path('/dev/shm') and trace.name.startswith('ep_loop_'):
    owner=json.loads((trace/'owner.json').read_text())
    if owner['pid']==pid and owner['run_tag'] in trace.name:
        shutil.rmtree(trace);cleaned=True
budget_path=Path(record['time_budget_file'])
budget=json.loads(budget_path.read_text()) if budget_path.exists() else {}
current_path=Path(record['current_group_file'])
current=json.loads(current_path.read_text()) if current_path.exists() else {}
status=('stopped_at_time_budget' if budget.get('state') in {'stopping_at_time_budget','stopped_at_time_budget'} else
        'completed' if current.get('status')=='completed' and current.get('group')==list(record['group_runs'])[-1] else 'exited_before_panel_completion')
checkpoints={group:[str(p.parent) for p in sorted(Path(folder).glob('ours/checkpoints/update_*/checkpoint.complete.json'))]
             for group,folder in record['group_runs'].items()}
result=dict(status=status,finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            ram_volume_cleaned=cleaned,complete_checkpoints=checkpoints,known_owned_processes=len(known))
(panel/'final_status.json').write_text(json.dumps(result,indent=2)+'\n')
for pointer in [launch,Path(sys.argv[2])/'experiments/latest_run.json',Path(sys.argv[2])/'audit/aime24_launch_20261117/latest_launch.json']:
    if pointer.exists():
        current=json.loads(pointer.read_text())
        if current.get('pid')==pid:
            current.update(result);temporary=pointer.with_suffix('.finalize.tmp')
            temporary.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n');temporary.replace(pointer)
