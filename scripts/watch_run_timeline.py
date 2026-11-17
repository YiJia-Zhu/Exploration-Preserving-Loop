"""Attach a read-only, timestamped event/heartbeat view to an existing run.

Does not import Torch, inspect GPU tensors, modify existing logs or restart jobs.
Source timestamps and observer timestamps are kept separate.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo


def local_time(value, zone):
    return datetime.fromisoformat(value).astimezone(zone).isoformat(timespec='milliseconds')


class EventTail:
    def __init__(self, path):
        self.path, self.offset, self.pending = path, 0, b''
        self.active, self.last = {}, None

    def read(self):
        if self.path.stat().st_size < self.offset:
            self.offset, self.pending, self.active = 0, b'', {}
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            data=stream.read()
        self.offset += len(data)
        lines=(self.pending+data).split(b'\n')
        self.pending=lines.pop()
        for line in lines:
            if not line.strip():continue
            row=json.loads(line)
            self.last=row
            if row['event'] in {'run_start','run_end'}:self.active.clear()
            elif row['event']=='stage_start':self.active[row['stage_id']]=row
            elif row['event']=='stage_end':self.active.pop(row['stage_id'],None)
            yield row

    def snapshot(self, zone):
        now=datetime.now(timezone.utc)
        context={key:row[key] for row in self.active.values() for key in ('update','id','sample') if key in row}
        stages=[dict(stage=row['stage'],stage_id=row['stage_id'],
                     started_at_local=local_time(row['started_at'],zone),
                     wall_elapsed_seconds=(now-datetime.fromisoformat(row['started_at'])).total_seconds())
                for row in self.active.values()]
        result=dict(source=str(self.path),active_stages=stages,**context)
        if self.last:
            result.update(rank=self.last['rank'],pid=self.last['pid'],
                          last_source_event_at=local_time(self.last['timestamp'],zone),
                          last_source_event=self.last['event'],status=self.last.get('status'))
        # The old rank-0 tqdm has no structured token heartbeat. Quote its last
        # flushed count with the file mtime; do not claim it was sampled now.
        if any(row['stage']=='generation' for row in self.active.values()):
            name=self.path.name
            console=self.path.with_name(name.replace('events_rank','console_rank').replace('.jsonl','.log')) if name.startswith('events_rank') else self.path.with_name(name.replace('.events.jsonl','.console.log'))
            if console.is_file():
                with console.open('rb') as stream:
                    stream.seek(max(0,console.stat().st_size-12000))
                    lines=stream.read().decode(errors='replace').splitlines()
                for line in reversed(lines):
                    match=re.search(r'(\d+)/(\d+).*?(?:tok/s|s/tok)',line)
                    if match:
                        result['last_console_progress']=dict(tokens=int(match[1]),limit=int(match[2]),
                            file_modified_at=datetime.fromtimestamp(console.stat().st_mtime,zone).isoformat(timespec='milliseconds'))
                        break
        return result


def process_identity(pid):
    try:
        value=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return None if value[0]=='Z' else value[19]
    except FileNotFoundError:
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--launcher-pid',type=int)
    parser.add_argument('--output-prefix',type=Path)
    parser.add_argument('--heartbeat-seconds',type=float,default=30)
    parser.add_argument('--poll-seconds',type=float,default=2)
    parser.add_argument('--timezone',default='Asia/Shanghai')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if not args.run.is_dir():parser.error('--run must exist')
    if not args.once and args.launcher_pid is None:parser.error('--launcher-pid is required unless --once')
    if not 0 < args.heartbeat_seconds < float('inf') or not 0 < args.poll_seconds < float('inf'):
        parser.error('heartbeat/poll intervals must be positive and finite')
    zone=ZoneInfo(args.timezone)
    prefix=args.output_prefix or args.run/'timeline.live'
    prefix.parent.mkdir(parents=True,exist_ok=True)
    tails={}
    identity=process_identity(args.launcher_pid) if args.launcher_pid else None
    with Path(str(prefix)+'.jsonl').open('a',buffering=1) as records, Path(str(prefix)+'.log').open('a',buffering=1) as console:
        def emit(kind,**fields):
            now=datetime.now(timezone.utc)
            row=dict(event=kind,timestamp=now.isoformat(timespec='milliseconds'),
                     local_time=now.astimezone(zone).isoformat(timespec='milliseconds'),
                     observer_pid=os.getpid(),**fields)
            records.write(json.dumps(row,ensure_ascii=False)+'\n')
            console.write(f"[{row['local_time']}] {kind} "+json.dumps(fields,ensure_ascii=False,separators=(',',':'))+'\n')
        emit('watch_start',launcher_pid=args.launcher_pid,heartbeat_seconds=args.heartbeat_seconds,
             note='Observer times are current; source event times retain their original meaning.')
        next_heartbeat=0
        while True:
            for path in sorted(args.run.glob('**/*events*.jsonl')):
                new=path not in tails
                tail=tails.setdefault(path,EventTail(path))
                for row in tail.read():
                    if not new and row['event'] in {'stage_start','stage_end','run_start','run_end'}:
                        fields={key:row[key] for key in ('event','stage','stage_id','rank','pid','update','id','sample','status','seconds','response_tokens') if key in row}
                        fields['source_event']=fields.pop('event')
                        fields.update(source=str(path),source_event_at=local_time(row['timestamp'],zone))
                        for key in ('started_at','finished_at'):
                            if key in row:fields[key+'_local']=local_time(row[key],zone)
                        emit('source_event',**fields)
            running=identity is not None and process_identity(args.launcher_pid)==identity
            if time.monotonic()>=next_heartbeat or not running or args.once:
                emit('watch_heartbeat',launcher_alive=running,
                     sources=[tail.snapshot(zone) for tail in tails.values()])
                next_heartbeat=time.monotonic()+args.heartbeat_seconds
            if args.once or not running:break
            time.sleep(args.poll_seconds)
        emit('watch_end',reason='single_snapshot' if args.once else 'launcher_exited')


if __name__=='__main__':
    main()
