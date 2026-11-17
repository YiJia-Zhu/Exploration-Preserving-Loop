"""Linux process-tree cleanup scoped to one explicitly owned experiment PID."""
import os
from pathlib import Path
import signal
import time


def process_record(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),ppid=int(fields[1]),start_ticks=fields[19],state=fields[0])
    except (OSError,IndexError):
        return None


def experiment_processes(root_pid, exclude=()):
    records={}
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            row=process_record(int(path.name))
            if row:records[row['pid']]=row
    owned={root_pid}
    while True:
        children={pid for pid,row in records.items() if row['ppid'] in owned}
        if children<=owned:break
        owned|=children
    return [records[pid] for pid in owned-set(exclude) if pid in records]


def terminate_experiment(root_pid, grace_seconds=8, exclude=()):
    if root_pid==os.getpid():raise ValueError('Use a separate watchdog to terminate an experiment')
    records=experiment_processes(root_pid,exclude=(*exclude,os.getpid()))
    def alive(row):
        current=process_record(row['pid'])
        return current and current['start_ticks']==row['start_ticks'] and current['state']!='Z'
    def send(row,kind):
        if alive(row):
            try:os.kill(row['pid'],kind)
            except ProcessLookupError:pass
    # Stop the coordinator first so it cannot launch another group, then stop
    # its captured workers (including vLLM actors in separate process groups).
    records.sort(key=lambda row:row['pid']!=root_pid)
    for row in records:send(row,signal.SIGTERM)
    end=time.monotonic()+grace_seconds
    while time.monotonic()<end and any(alive(row) for row in records):time.sleep(.1)
    for row in records:send(row,signal.SIGKILL)
    return records
