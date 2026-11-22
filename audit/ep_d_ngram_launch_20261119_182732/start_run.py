"""Own exactly GPUs 0/1/4/5 for the requested EP-D-LoopOPD run."""
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time

AUDIT=Path(__file__).resolve().parent
SETUP=json.loads((AUDIT/'setup.json').read_text())
ROOT=Path(SETUP['project'])
sys.path.insert(0,str(ROOT))
from ep_loop.config import validate_config,assets_status
from ep_loop.gpu_resources import require_idle_gpus,gpu_status
from ep_loop.process_control import process_record,experiment_processes,terminate_experiment


def now():return datetime.now().astimezone().isoformat()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
def last_json(path):
    if not path.exists():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-131072))
        for line in reversed(f.read().splitlines()):
            try:return json.loads(line)
            except (ValueError,UnicodeDecodeError):pass
    return None


def main():
    launch_path=AUDIT/'launch.json'
    if launch_path.exists():raise FileExistsError(launch_path)
    info=process_record(os.getpid())
    launch=dict(pid=os.getpid(),start_ticks=int(info['start_ticks']),started_at=now(),status='preflight',
        method='EP-D-LoopOPD',target_updates=100,completed_updates=0,gpus=SETUP['gpus'],gpu_count=4,
        teacher_update='current',teacher_loops=4,student_loops=2,rollout_batch=256,
        prompts_per_batch=32,responses_per_prompt=8,max_new_tokens=16384,seed=42,warmup_steps=10,
        rollout_ngram=True,inference_speculation=False,weight_dtype='bfloat16',kv_dtype='bfloat16',
        model_checkpoints_every=5,retain_all_model_checkpoints=True,resume_save_every=5,
        resume_keep_latest=1,started_from='original_pretrained_weights',
        max_runtime_hours=10,run=SETUP['run'],output=SETUP['training_output'],audit=str(AUDIT),
        external_console=str(AUDIT/'training.console.log'))
    previous=ROOT/'experiments/latest_run.json'
    if previous.exists():shutil.copy2(previous,AUDIT/'previous_latest_run.json')
    def publish():
        launch['last_health_check_at']=now();write(launch_path,launch);write(previous,launch)
    publish()
    child=None;tensorboard=None;leases_added=[];known={};deadline=time.monotonic()+10*3600
    def stop(*_):raise KeyboardInterrupt('User interrupted the training supervisor')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def foreign_processes():
        for row in experiment_processes(os.getpid()):known[row['pid']]=str(row['start_ticks'])
        foreign=[]
        for gpu in gpu_status():
            if gpu['index'] not in SETUP['gpus']:continue
            for proc in gpu['processes']:
                current=process_record(proc['pid'])
                if current and known.get(proc['pid'])!=str(current['start_ticks']):
                    foreign.append(dict(gpu=gpu['index'],**proc))
        return foreign
    try:
        config=json.loads((AUDIT/'production.config.json').read_text());validate_config(config)
        assert config['implementation']['checkpoint_layout']=='split'
        assert config['implementation']['save_every_updates']==config['implementation']['resume_save_every_updates']==5
        assert config['implementation']['keep_latest_checkpoints']==0
        status=assets_status(config,ROOT)
        if not status['ready']:raise RuntimeError(status)
        model_bytes=sum(p.stat().st_size for p in (ROOT/config['model']['path']).glob('*.safetensors'))
        required=model_bytes*(2*20+8+2*4)
        free=shutil.disk_usage(Path(SETUP['training_output']).parent).free
        if free<required:raise RuntimeError(f'Checkpoint volume requires {required/2**30:.1f}GiB, free {free/2**30:.1f}GiB')
        launch.update(checkpoint_required_gib=required/2**30,checkpoint_free_gib=free/2**30)
        sources=[*ROOT.glob('ep_loop/*.py'),*ROOT.glob('ep_loop/fastpath/**/*.py'),*ROOT.glob('scripts/*.py'),
                 *ROOT.glob('scripts/*.sh'),AUDIT/'production.config.json']
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        write(AUDIT/'source_sha256.json',hashes)
        with tarfile.open(AUDIT/'sources.tar.gz','w:gz') as archive:
            for path in sources:archive.add(path,arcname=str(path.relative_to(ROOT)))
        with ExitStack() as stack:
            registries=[ROOT.parent/'steer_opd/.runtime',
                *[ROOT.parent/'steer_opd/exploration'/name/'.runtime' for name in ('readout_sv','residual_sv')]]
            handles=[];reserved=set()
            for registry in registries:
                if not registry.exists():continue
                lock=registry/'gpu_admission.lock'
                handle=stack.enter_context(lock.open('a'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(handle)
                lease_path=registry/'gpu_leases.json'
                leases=json.loads(lease_path.read_text()) if lease_path.exists() else []
                for lease in leases:
                    current=process_record(lease['pid'])
                    if current and current['state']!='Z' and str(current['start_ticks'])==str(lease['start_ticks']):
                        reserved.update(lease['uuids'])
            selected=require_idle_gpus(SETUP['gpus'])
            if reserved.intersection(row['uuid'] for row in selected):raise RuntimeError('A requested GPU is leased by another live job')
            for gpu in SETUP['gpus']:
                fcntl.flock(stack.enter_context(open('/tmp/ep-loop-gpu'+gpu+'.lock','a')),fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Share our allocation with existing cooperative project schedulers.
            for registry in registries:
                if not registry.exists():continue
                path=registry/'gpu_leases.json';leases=json.loads(path.read_text()) if path.exists() else []
                own=dict(pid=os.getpid(),start_ticks=int(info['start_ticks']),uuids=[row['uuid'] for row in selected])
                leases.append(own);write(path,leases);leases_added.append((registry,own))
            for handle in handles:fcntl.flock(handle,fcntl.LOCK_UN)
            write(AUDIT/'gpu_admission.json',dict(checked_at=now(),selected=selected,
                shared_leases=[dict(registry=str(registry),**own) for registry,own in leases_added]))
            environment=os.environ|SETUP['environment']|dict(CUDA_VISIBLE_DEVICES=','.join(SETUP['gpus']),
                CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONHASHSEED='42',PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',NUMEXPR_NUM_THREADS='16',
                TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            for name in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):environment.pop(name,None)
            command=[str(ROOT/'.venv/bin/python'),'-u','-m','torch.distributed.run','--standalone',
                '--nproc-per-node=4',str(ROOT/'scripts/train_fastpath_distillation.py'),
                '--config',str(AUDIT/'production.config.json'),'--output',SETUP['training_output'],'--device','cuda']
            with (AUDIT/'training.console.log').open('wb') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            birth=process_record(child.pid)
            launch.update(status='running',training_pid=child.pid,training_start_ticks=int(birth['start_ticks']),
                training_started_at=now(),command=command,resume_path=str(Path(SETUP['training_output'])/'resume/latest'))
            publish()
            try:
                with socket.socket() as probe:probe.bind(('127.0.0.1',6007))
                with (AUDIT/'tensorboard.console.log').open('wb') as log:
                    tensorboard=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','tensorboard.main',
                        '--logdir',str(Path(SETUP['training_output'])/'tensorboard'),'--host','127.0.0.1','--port','6007'],
                        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        env=environment|dict(CUDA_VISIBLE_DEVICES=''))
                launch.update(tensorboard_pid=tensorboard.pid,tensorboard_url='http://127.0.0.1:6007');publish()
            except OSError as error:launch['tensorboard_server_note']=str(error);publish()
            while child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Reached the retained 10-hour training/evaluation budget')
                foreign=foreign_processes()
                if foreign:raise RuntimeError(f'Unrelated compute entered requested GPUs; stop only this run: {foreign}')
                output=Path(SETUP['training_output'])
                latest=last_json(output/'metrics.jsonl')
                if latest:launch.update(completed_updates=latest['update'],last_metrics=latest)
                events={}
                for rank in range(4):
                    row=last_json(output/f'events_rank{rank}.jsonl')
                    if row:events[str(rank)]=row
                launch['rank_events']=events
                launch['model_checkpoints']=[str(p.parent) for p in sorted((output/'checkpoints').glob('update_*/checkpoint.complete.json'))]
                if (output/'resume/latest/checkpoint.complete.json').exists():
                    launch['latest_resume_update']=json.loads((output/'resume/latest/checkpoint.complete.json').read_text())['completed_updates']
                launch['allocated_gpu_processes']=[row for row in gpu_status() if row['index'] in SETUP['gpus']]
                publish();time.sleep(15)
            if child.returncode:raise RuntimeError(f'Training process exited {child.returncode}; see training.console.log')
            summary=json.loads((Path(SETUP['training_output'])/'summary.json').read_text())
            launch.update(completed_updates=summary['completed_updates'],status='training_completed');publish()
            if summary['completed_updates']==100 and time.monotonic()<deadline:
                for _ in range(30):
                    try:require_idle_gpus(SETUP['gpus']);break
                    except RuntimeError:time.sleep(1)
                checkpoint=Path(SETUP['training_output'])/'checkpoints/update_000100'
                eval_output=Path(SETUP['run'])/'evaluation_final.jsonl'
                eval_command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/evaluate_fastpath.py'),
                    '--config',str(AUDIT/'production.config.json'),'--checkpoint',str(checkpoint),
                    '--data',str(ROOT/'data/aime/aime2024.jsonl'),'--gpus',','.join(SETUP['gpus']),
                    '--output',str(eval_output)]
                with (AUDIT/'evaluation.console.log').open('wb') as log:
                    child=subprocess.Popen(eval_command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                launch.update(status='evaluating',evaluation_pid=child.pid,evaluation_output=str(eval_output));publish()
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Evaluation reached the 10-hour total budget')
                    if foreign_processes():raise RuntimeError('Unrelated compute entered requested GPUs during evaluation')
                    publish();time.sleep(15)
                if child.returncode:raise RuntimeError(f'Final evaluation exited {child.returncode}')
            launch.update(status='completed',finished_at=now());publish()
    except BaseException as error:
        launch.update(status='stopped_at_time_budget' if isinstance(error,TimeoutError) else 'failed_or_interrupted',
                      error=repr(error),finished_at=now());publish()
        raise
    finally:
        if child is not None and child.poll() is None:terminate_experiment(child.pid)
        if tensorboard is not None and tensorboard.poll() is None:terminate_experiment(tensorboard.pid)
        for registry,own in leases_added:
            with (registry/'gpu_admission.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                path=registry/'gpu_leases.json'
                leases=json.loads(path.read_text()) if path.exists() else []
                write(path,[row for row in leases if not (row['pid']==own['pid'] and str(row['start_ticks'])==str(own['start_ticks']))])
        launch['supervisor_cleanup_at']=now();publish()


if __name__=='__main__':main()
