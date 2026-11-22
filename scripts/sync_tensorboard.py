"""Backfill/follow metrics for an existing trainer without restarting training."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--follow',action='store_true')
    p.add_argument('--owner-pid',type=int)
    p.add_argument('--poll-seconds',type=float,default=2.)
    args=p.parse_args()
    if args.poll_seconds<=0:p.error('poll-seconds must be positive')
    from ep_loop.tensorboard_logging import ScalarLogger
    from ep_loop.process_control import process_record
    owner=process_record(args.owner_pid) if args.owner_pid else None
    if args.owner_pid and owner is None:p.error('Owner PID is not running')
    metrics=args.run_dir/'metrics.jsonl';position=0;last_identity=None
    writer=ScalarLogger(args.run_dir)
    print(json.dumps(dict(event='tensorboard_sync_start',pid=os.getpid(),run_dir=str(args.run_dir.resolve()),log_dir=str(writer.directory.resolve()))),flush=True)
    try:
        while True:
            if metrics.exists():
                stat=metrics.stat();identity=(stat.st_dev,stat.st_ino)
                if identity!=last_identity or stat.st_size<position:position=0;last_identity=identity
                with metrics.open('rb') as stream:
                    stream.seek(position)
                    while True:
                        line=stream.readline()
                        if not line or not line.endswith(b'\n'):break
                        row=json.loads(line);position=stream.tell()
                        if writer.record(row,walltime=stat.st_mtime):
                            print(json.dumps(dict(event='tensorboard_update',step=row['update'],loss=row.get('mean_response_loss'))),flush=True)
            current=process_record(args.owner_pid) if owner else None
            if not args.follow or (owner and (current is None or current['state']=='Z' or current['start_ticks']!=owner['start_ticks'])):break
            time.sleep(args.poll_seconds)
    finally:writer.close()


if __name__=='__main__':main()
