"""Permanent model snapshots, final embedded resume, and legacy rolling resume."""
import json
import os
from pathlib import Path
import shutil

import torch
import torch.distributed as dist

from ep_loop.data import canonical_hash
from ep_loop.reproducibility import capture_rng_state


def checkpoint_due(implementation, completed, final_update):
    model_every=implementation['save_every_updates']
    resume_every=implementation.get('resume_save_every_updates',model_every)
    final=completed==final_update
    if implementation.get('checkpoint_layout','full')=='model_only':
        return completed%model_every==0 or final, final and implementation.get('save_final_resume',False)
    return (completed%model_every==0 or final,
            completed%resume_every==0 or final)


def _write_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def _barrier(world_size):
    if world_size>1:dist.barrier()


def _save_final_resume(model,student,optimizer,data_rng,generator,metadata,rank,world_size):
    """Upgrade the final model snapshot only after every matching state is ready."""
    pending=model/'.resume.pending'
    if rank==0:
        marker=model/'checkpoint.complete.json'
        if not marker.is_file():
            raise ValueError('Final recovery requires a complete model snapshot from the same update')
        current=json.loads(marker.read_text())
        if any(current.get(key)!=value for key,value in metadata.items()):
            raise ValueError('Final model and recovery update/configuration do not match')
        if current.get('resumable'):
            raise FileExistsError(f'Final recovery is already complete: {model}')
        if pending.exists():shutil.rmtree(pending)
        pending.mkdir()
        torch.save(dict(optimizer=optimizer.state_dict(),completed_updates=metadata['completed_updates'],
            world_size=world_size,config_hash=metadata['config_hash'],data_identity=metadata['data_identity']),
            pending/'training_state.pt')
    _barrier(world_size)
    torch.save(dict(data_rng=data_rng.getstate(),rollout_rng=generator.get_state(),
                    **capture_rng_state(student.device)),pending/f'rng_rank{rank}.pt')
    _barrier(world_size)
    if rank==0:
        names=['training_state.pt',*(f'rng_rank{r}.pt' for r in range(world_size))]
        if not all((pending/name).is_file() for name in names):
            raise ValueError('Final recovery is missing optimizer or rank RNG state')
        for name in names:os.replace(pending/name,model/name)
        pending.rmdir()
        _write_json(model/'checkpoint.complete.json',metadata|dict(resumable=True,storage_role='final_resume'))
    _barrier(world_size)


def save_split_checkpoint(output, student, tokenizer, optimizer, completed, config,
                          data_rng, generator, rank, world_size, project_root,
                          save_model=True, save_resume=True, data_identity=None,is_final=False):
    """All ranks participate; publish recovery only after all matching RNG files.

    Model-only layout embeds recovery in the final model directory. Legacy
    split layout hardlinks model files into a rolling recovery directory;
    the old latest stays usable until the new recovery is fully published.
    """
    output=Path(output)
    model=output/'checkpoints'/f'update_{completed:06d}'
    metadata=dict(completed_updates=completed,world_size=world_size,format_version=2,
                  data_identity=data_identity,config_hash=canonical_hash(config))
    final_resume=(config['implementation'].get('checkpoint_layout')=='model_only' and save_resume)
    if final_resume and (not is_final or not config['implementation'].get('save_final_resume',False)):
        raise ValueError('Model-only checkpoints may save recovery state only at the final update')
    if save_resume and not final_resume and not config['implementation']['save_optimizer']:
        raise ValueError('Recovery state saving is disabled for this run')
    if save_model:
        if rank==0:
            if model.exists():
                marker=model/'checkpoint.complete.json'
                current=json.loads(marker.read_text()) if marker.exists() else {}
                if any(current.get(key)!=value for key,value in metadata.items()):
                    raise FileExistsError(f'A different or incomplete model snapshot exists: {model}')
            else:
                pending=model.with_name('.'+model.name+'.pending')
                if pending.exists():shutil.rmtree(pending)
                pending.mkdir(parents=True)
                student.model.save_pretrained(pending,safe_serialization=True)
                source=(Path(project_root)/config['model']['path']).resolve()
                for filename in student.source_filenames:shutil.copy2(source/filename,pending/filename)
                if hasattr(tokenizer,'save_pretrained'):tokenizer.save_pretrained(pending)
                _write_json(pending/'run_config.json',config)
                _write_json(pending/'checkpoint.complete.json',metadata|dict(resumable=False,storage_role='permanent_model'))
                pending.replace(model)
        _barrier(world_size)
    if not save_resume:return
    if final_resume:
        _save_final_resume(model,student,optimizer,data_rng,generator,metadata,rank,world_size)
        return
    resume_root=output/'resume'
    pending=resume_root/f'.update_{completed:06d}.pending'
    recovery=resume_root/f'update_{completed:06d}'
    if rank==0:
        if not (model/'checkpoint.complete.json').is_file():
            raise ValueError('Recovery requires a complete model snapshot from the same update')
        marker=json.loads((model/'checkpoint.complete.json').read_text())
        if any(marker.get(key)!=value for key,value in metadata.items()):
            raise ValueError('Model and recovery update/configuration do not match')
        if recovery.exists():raise FileExistsError(recovery)
        if pending.exists():shutil.rmtree(pending)
        pending.mkdir(parents=True)
        for path in model.rglob('*'):
            if not path.is_file() or path==model/'checkpoint.complete.json':continue
            target=pending/path.relative_to(model)
            target.parent.mkdir(parents=True,exist_ok=True)
            os.link(path,target)
        torch.save(dict(optimizer=optimizer.state_dict(),completed_updates=completed,
            world_size=world_size,config_hash=metadata['config_hash'],data_identity=data_identity),
            pending/'training_state.pt')
    _barrier(world_size)
    torch.save(dict(data_rng=data_rng.getstate(),rollout_rng=generator.get_state(),
                    **capture_rng_state(student.device)),pending/f'rng_rank{rank}.pt')
    _barrier(world_size)
    if rank==0:
        assert all((pending/f'rng_rank{r}.pt').is_file() for r in range(world_size))
        _write_json(pending/'checkpoint.complete.json',metadata|dict(resumable=True,
            storage_role='latest_resume',permanent_model=str(model.resolve())))
        pending.replace(recovery)
        link=resume_root/'.latest.tmp'
        link.unlink(missing_ok=True)
        link.symlink_to(recovery.name,target_is_directory=True)
        os.replace(link,resume_root/'latest')
        for previous in resume_root.glob('update_*'):
            if previous!=recovery and previous.is_dir() and (previous/'checkpoint.complete.json').is_file():
                shutil.rmtree(previous)
    _barrier(world_size)
