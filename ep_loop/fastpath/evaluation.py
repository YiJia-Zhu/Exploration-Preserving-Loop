"""Evaluate saved fixed-depth students between updates, with ordinary decoding."""
from contextlib import nullcontext
import copy
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch
import torch.distributed as dist

from ep_loop.config import project_relative_path
from ep_loop.reproducibility import capture_rng_state,restore_rng_state


def evaluation_due(config, completed, final_update):
    every=config.get('evaluation',{}).get('every_updates',0)
    return bool(every and (completed%every==0 or completed==final_update))


def evaluation_datasets(config, root):
    settings=config['evaluation']
    datasets=settings.get('datasets')
    if datasets is None:
        path=settings.get('data_path','data/aime/aime2024.jsonl')
        datasets=[dict(name=Path(path).stem,path=path,suite=settings.get('suite'),
                       expected_examples=settings.get('expected_examples'))]
    if not isinstance(datasets,list) or not datasets:
        raise ValueError('evaluation.datasets must be a nonempty list')
    result=[];names=set()
    for spec in datasets:
        if not isinstance(spec,dict):raise ValueError('Each evaluation dataset must be an object')
        name=spec.get('name','')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or name in names:
            raise ValueError('Evaluation dataset names must be distinct safe file names')
        names.add(name)
        path=Path(root)/spec['path']
        result.append(dict(spec,name=name,path=str(path.absolute())))
    return result


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def evaluate_checkpoint(config, checkpoint, folder, devices, root):
    """The rank-zero subprocess owns inference only, never a training actor."""
    from ep_loop.evaluation_data import load_evaluation_records
    checkpoint,folder,root=Path(checkpoint).resolve(),Path(folder),Path(root)
    marker=json.loads((checkpoint/'checkpoint.complete.json').read_text())
    settings=config['evaluation']
    status=dict(status='running',completed_updates=marker['completed_updates'],checkpoint=str(checkpoint),
                speculative_config=None,results={})
    folder.mkdir(parents=True,exist_ok=True)
    _write(folder/'status.json',status)
    environment=os.environ.copy()
    for key in list(environment):
        if key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK',
                   'ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
            environment.pop(key,None)
    environment['CUDA_VISIBLE_DEVICES']=','.join(devices)
    try:
        for dataset in evaluation_datasets(config,root):
            load_evaluation_records(dataset['path'],dataset.get('expected_examples'),dataset.get('suite'))
            task=copy.deepcopy(config)
            task['evaluation'].update(data_path=project_relative_path(dataset['path'],root),suite=dataset.get('suite'),
                expected_examples=dataset.get('expected_examples'),every_updates=0)
            task['evaluation'].pop('datasets',None)
            task['implementation']['vllm_rollout_ngram']=False
            task['evaluation']['samples_per_problem']=dataset.get('samples_per_problem',settings['samples_per_problem'])
            config_path=folder/(dataset['name']+'.config.json')
            output=folder/(dataset['name']+'.responses.jsonl')
            _write(config_path,task)
            command=[sys.executable,'-u',str(root/'scripts/evaluate_fastpath.py'),
                '--config',str(config_path),'--checkpoint',str(checkpoint),'--data',dataset['path'],
                '--gpus',','.join(devices),'--output',str(output)]
            with (folder/(dataset['name']+'.process.log')).open('w') as log:
                subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads(output.with_suffix('.summary.json').read_text())
            if report.get('checkpoint')!=str(checkpoint) or report.get('speculative_config') is not None:
                raise RuntimeError('Evaluation checkpoint or decoding policy does not match the saved model')
            status['results'][dataset['name']]=dict(responses=str(output),
                summary=str(output.with_suffix('.summary.json')),scores=report['results'])
            _write(folder/'status.json',status)
        status['status']='completed';_write(folder/'status.json',status)
        return status
    except BaseException as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        _write(folder/'status.json',status)
        raise


def run_periodic_evaluation(config, output, completed, student, optimizer, batcher, teacher,
                            device, data_rng, generator, rank, world, control_group, root, observer=None):
    """Release all training allocations before one normal evaluation subprocess.

    Rank synchronization uses Gloo so waiting ranks do not reserve more CUDA
    buffers. Errors are propagated to all ranks. RNG snapshots stay in RAM.
    """
    from .residency import move_training_state
    state=capture_rng_state(device)
    data_state=data_rng.getstate();rollout_state=generator.get_state()
    stage=observer.stage('periodic_evaluation',echo=True,update=completed) if observer else nullcontext()
    folder=Path(output)/'evaluation'/f'update_{completed:06d}'
    try:
        with stage:
            error=None
            try:
                if batcher:
                    if batcher.pending:raise RuntimeError('Training rollout responses remain unconsumed')
                    batcher.close()
                if teacher and hasattr(teacher,'close'):teacher.close()
                optimizer.zero_grad(set_to_none=True)
                move_training_state(student.model,optimizer,'cpu')
                if teacher is not None and getattr(teacher,'model',None) is not student.model:
                    teacher.model.to('cpu')
                gc.collect()
                if torch.device(device).type=='cuda':torch.cuda.empty_cache()
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            errors=[error]
            if world>1:
                errors=[None]*world
                dist.all_gather_object(errors,error,group=control_group)
            if any(errors):raise RuntimeError(f'Cannot release training state for evaluation: {errors}')
            result=[None]
            if rank==0:
                try:
                    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
                    if devices==['']:devices=[str(i) for i in range(world)]
                    if len(devices)!=world:raise ValueError('Periodic evaluation requires single-node GPU visibility')
                    report=evaluate_checkpoint(config,Path(output)/'checkpoints'/f'update_{completed:06d}',
                                               folder,devices,root)
                    result[0]=dict(ok=True,report=report)
                except Exception as exc:result[0]=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
            if world>1:dist.broadcast_object_list(result,src=0,group=control_group)
            if not result[0]['ok']:raise RuntimeError('Periodic evaluation failed: '+result[0]['error'])
            if rank==0 and observer:
                observer.event('periodic_evaluation_complete',echo=True,update=completed,
                               report=str(folder/'status.json'),results=result[0]['report']['results'])
            return result[0]['report']
    finally:
        data_rng.setstate(data_state);generator.set_state(rollout_state)
        restore_rng_state(state,device)
