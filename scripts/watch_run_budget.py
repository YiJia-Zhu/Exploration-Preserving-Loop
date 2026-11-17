"""Enforce an experiment wall-time budget without signaling unrelated jobs."""
import argparse
from datetime import datetime,timedelta
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ep_loop.process_control import process_record,terminate_experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--hours',type=float,required=True)
    p.add_argument('--started-monotonic',type=float,required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--run-prefix',required=True)
    a=p.parse_args()
    if not 0<a.hours<float('inf'):p.error('Positive finite hours required')
    identity=process_record(a.pid)
    deadline=a.started_monotonic+a.hours*3600
    zone=ZoneInfo('Asia/Shanghai')
    record=dict(launcher_pid=a.pid,watchdog_pid=os.getpid(),max_runtime_hours=a.hours,
                started_at=datetime.now(zone).isoformat(),
                deadline_local=(datetime.now(zone)+timedelta(seconds=max(0,deadline-time.monotonic()))).isoformat(),
                state='watching',graceful_stop_margin_seconds=15)
    def save():
        temporary=a.panel/'time_budget.tmp'
        temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(a.panel/'time_budget.json')
    save()
    while time.monotonic()<deadline-15:
        current=process_record(a.pid)
        if identity is None or current is None or current['state']=='Z' or current['start_ticks']!=identity['start_ticks']:
            record.update(state='finished_before_deadline',finished_at=datetime.now(zone).isoformat());save();return
        time.sleep(min(5,max(.01,deadline-15-time.monotonic())))
    current=process_record(a.pid)
    if current and identity and current['start_ticks']==identity['start_ticks'] and current['state']!='Z':
        record.update(state='stopping_at_time_budget',stopping_at=datetime.now(zone).isoformat())
        save()
        killed=terminate_experiment(a.pid,grace_seconds=min(8,max(0,deadline-time.monotonic()-2)),exclude=(os.getpid(),))
        record['owned_processes_stopped']=killed
        record['complete_checkpoints']=[str(path.parent) for path in sorted(a.output_root.glob(
            a.run_prefix+'_*/ours/checkpoints/update_*/checkpoint.complete.json'))]
        record.update(state='stopped_at_time_budget',finished_at=datetime.now(zone).isoformat());save()


if __name__=='__main__':main()
