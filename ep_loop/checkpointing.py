"""Validate complete training checkpoints without rereading large optimizer files."""
import copy
import json
from pathlib import Path

from .data import canonical_hash


def validate_resume_config(saved, current, completed):
    """The implemented warmup/constant LR depends on step, not total updates."""
    old, new = copy.deepcopy(saved), copy.deepcopy(current)
    old_total = old['optimization'].pop('updates')
    new_total = new['optimization'].pop('updates')
    if new_total < old_total or new_total < completed:
        raise ValueError('Resume may keep or extend the total updates; it cannot reduce the saved target')
    # These affect persistence/observability, never the training objective or RNG.
    for config in (old,new):
        # Absent in historical checkpoints means normal rollout. Enabling
        # speculation changes random-number consumption and must not be hidden.
        config['implementation'].setdefault('vllm_rollout_ngram',False)
        for key in ('save_every_updates','keep_latest_checkpoints','tensorboard','tensorboard_flush_seconds',
                    'checkpoint_layout','resume_save_every_updates','save_optimizer','save_final_resume',
                    'write_rollout_text','write_rollout_records'):
            config['implementation'].pop(key,None)
        # Periodic evaluation restores RNG and does not change the next update.
        for key in ('every_updates','data_path','expected_examples','suite','scope','datasets',
                    'inference_backend','rlt_attention_backend',
                    'rlt_max_num_seqs','rlt_max_num_batched_tokens','rlt_prefill_chunk_size',
                    'rlt_async_scheduling','rlt_prefix_caching'):
            config.get('evaluation',{}).pop(key,None)
    if old != new:
        raise ValueError('Resume requires unchanged model, data, sampling, optimizer and algorithm settings')
    return dict(saved_target_updates=old_total,target_updates=new_total,target_extended=new_total>old_total)


def inspect_resume_checkpoint(folder, config, world_size):
    folder = Path(folder)
    marker = folder/'checkpoint.complete.json'
    if not marker.is_file():
        raise ValueError('Resume requires checkpoint.complete.json; this checkpoint is incomplete')
    info = json.loads(marker.read_text())
    if not info.get('resumable'):
        raise ValueError('Checkpoint does not contain resumable optimizer state')
    if info['world_size'] != world_size:
        raise ValueError('Resume requires the same world size')
    saved = json.loads((folder/'run_config.json').read_text())
    if canonical_hash(saved) != info['config_hash']:
        raise ValueError('Checkpoint configuration does not match its completion marker')
    info.update(validate_resume_config(saved,config,info['completed_updates']))
    for name in ['training_state.pt',*(f'rng_rank{rank}.pt' for rank in range(world_size))]:
        if not (folder/name).is_file():
            raise ValueError(f'Checkpoint is missing required training state: {name}')
    # Read legacy data identity before a same-directory resume rewrites reports.
    info['marker_data_identity'] = info.get('data_identity')
    if not info.get('data_identity'):
        legacy_report = folder.parent.parent/'data_report.json'
        if legacy_report.is_file():
            info['data_identity'] = json.loads(legacy_report.read_text()).get('usable_prompt_hash')
    return info


def validate_resume_state(state, checkpoint_info):
    for key in ('config_hash','world_size','completed_updates'):
        if state.get(key) != checkpoint_info[key]:
            raise ValueError(f'Optimizer state does not match the complete checkpoint: {key}')
    if 'data_identity' in state and state['data_identity'] != checkpoint_info.get('marker_data_identity',checkpoint_info.get('data_identity')):
        raise ValueError('Optimizer state data identity does not match the complete checkpoint')


def validate_resume_data(checkpoint_info, data_report):
    expected = checkpoint_info.get('data_identity') if checkpoint_info else None
    if expected is not None and expected != data_report['usable_prompt_hash']:
        raise ValueError('Training prompt pool or order changed since the checkpoint')


def copy_resume_metrics(checkpoint, output, completed):
    """Carry the small committed scalar prefix when resuming to a new directory."""
    destination = Path(output)/'metrics.jsonl'
    source = Path(checkpoint).parent.parent/'metrics.jsonl'
    if destination.exists() or not source.exists():return
    with source.open() as reader,destination.open('x') as writer:
        for line in reader:
            try:row = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith('\n'):break  # Interrupted, uncommitted final line.
                raise
            if row['update']<=completed:writer.write(line)
