"""Run only the two user-requested checkpoints, sequentially on GPUs 0 and 1."""
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ep_loop.gpu_resources import require_idle_gpus

AUDIT=Path(__file__).resolve().parent
PLAN=json.loads((AUDIT/'plan.json').read_text())


def now():return datetime.now(timezone.utc).isoformat()


def write(value):
    path=AUDIT/'status.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def interrupt(signum,frame):raise KeyboardInterrupt(f'Received signal {signum}')


def live_lease(lease):
    try:
        fields=Path(f"/proc/{lease['pid']}/stat").read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z' and fields[19]==str(lease['start_ticks'])
    except (FileNotFoundError,ProcessLookupError):return False


def terminate_owned_child(child):
    if child is None or child.poll() is not None:return
    try:owned=psutil.Process(child.pid).children(recursive=True)+[psutil.Process(child.pid)]
    except psutil.NoSuchProcess:return
    for process in owned:
        try:process.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(owned,timeout=15)
    for process in alive:
        try:process.kill()
        except psutil.NoSuchProcess:pass
    child.wait(timeout=15)


def main():
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    report=PLAN|dict(status='starting',pid=os.getpid(),started_at=now(),results={})
    child=None
    write(report)
    try:
        with ExitStack() as stack:
            admission=[];reserved=set()
            registries=[ROOT.parent/'steer_opd/.runtime',
                *[ROOT.parent/'steer_opd/exploration'/name/'.runtime' for name in ('readout_sv','residual_sv')]]
            for registry in registries:
                lock=registry/'gpu_admission.lock'
                if lock.exists():
                    handle=stack.enter_context(lock.open('r'))
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);admission.append(handle)
                path=registry/'gpu_leases.json'
                for lease in json.loads(path.read_text()) if path.exists() else []:
                    if live_lease(lease):reserved.update(lease['uuids'])
            devices=require_idle_gpus(PLAN['gpus'])
            if reserved.intersection(row['uuid'] for row in devices):raise RuntimeError('Selected GPU is leased')
            for gpu in PLAN['gpus']:
                handle=stack.enter_context(open('/tmp/ep-loop-gpu'+gpu+'.lock','a'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for handle in admission:fcntl.flock(handle,fcntl.LOCK_UN)
            report.update(status='running',admission=devices)
            write(report)
            for task in PLAN['checkpoints']:
                report.update(current_step=task['step'],current_checkpoint=task['checkpoint'])
                worker="""import json,sys
from pathlib import Path
from ep_loop.fastpath.evaluation import evaluate_checkpoint
plan=json.loads(Path(sys.argv[1]).read_text());task=plan['checkpoints'][int(sys.argv[2])]
config=json.loads(Path(plan['config']).read_text())
evaluate_checkpoint(config,task['checkpoint'],task['folder'],plan['gpus'],Path.cwd())
"""
                environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES='0,1',
                    CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONHASHSEED='42',CPU_THREADS='16',TOKENIZERS_PARALLELISM='false')
                command=[sys.executable,'-u','-c',worker,str(AUDIT/'plan.json'),str(PLAN['checkpoints'].index(task))]
                with (AUDIT/f"update_{task['step']:06d}.console.log").open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                           start_new_session=True)
                    report.update(worker_pid=child.pid,worker_started_at=now());write(report)
                    while child.poll() is None:
                        report.update(heartbeat_at=now());write(report);time.sleep(5)
                    if child.returncode:raise RuntimeError(f"Step {task['step']} evaluation exited {child.returncode}")
                result=json.loads((Path(task['folder'])/'status.json').read_text())
                if result['status']!='completed':raise RuntimeError('Evaluation did not complete')
                report['results'][str(task['step'])]=result;write(report)
            report.update(status='completed',finished_at=now());write(report)
    except BaseException as error:
        report.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error=f'{type(error).__name__}: {error}',finished_at=now());write(report)
        raise
    finally:
        terminate_owned_child(child)


if __name__=='__main__':main()
