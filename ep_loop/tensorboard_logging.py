"""Small rank-zero scalar logs; JSONL remains the authoritative metrics stream."""
import fcntl
import json
import math
import os
from pathlib import Path


SCALARS = {
    'mean_response_loss':'train/loss',
    'mean_stage_loss':'train/stage_loss',
    'mean_stop_loss':'train/stop_loss',
    'learning_rate':'train/learning_rate',
    'gradient_norm':'train/gradient_norm',
    'response_tokens':'rollout/response_tokens',
    'mean_adjacent_js_nats':'distillation/adjacent_js_nats',
    'mean_actual_student_depth':'compute/mean_student_depth',
    'mean_target_student_depth':'compute/mean_target_depth',
    'seconds':'time/update_before_checkpoint_seconds',
}


def tensorboard_enabled():
    value = os.environ.get('TENSORBOARD','1').lower()
    if value not in {'0','1','false','true'}:
        raise ValueError('TENSORBOARD must be 0/1 or false/true')
    return value in {'1','true'}


class ScalarLogger:
    def __init__(self, run_dir, resumed_at=None, device='cpu'):
        self.directory = Path(run_dir)/'tensorboard'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.lock = (self.directory/'.writer.lock').open('a')
        try: fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f'TensorBoard already has a writer for {self.directory}')
        self.writer = None
        try:
            # Importing/creating an observer must not advance model randomness.
            from .reproducibility import capture_rng_state, restore_rng_state
            rng = capture_rng_state(device)
            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                from torch.utils.tensorboard import SummaryWriter
                events = EventAccumulator(str(self.directory),size_guidance={'scalars':0}).Reload()
                self.steps = ({event.step for event in events.Scalars('train/loss')}
                              if 'train/loss' in events.Tags()['scalars'] else set())
                if resumed_at is not None:self.steps = {step for step in self.steps if step<=resumed_at}
                flush = float(os.environ.get('TENSORBOARD_FLUSH_SECONDS','10'))
                if not math.isfinite(flush) or flush<=0:raise ValueError('TENSORBOARD_FLUSH_SECONDS must be positive')
                self.writer = SummaryWriter(str(self.directory),max_queue=100,flush_secs=flush,
                    purge_step=resumed_at+1 if resumed_at is not None else None)
            finally:restore_rng_state(rng,device)
        except BaseException:
            self.close();raise

    def record(self, row, walltime=None):
        step = int(row['update'])
        if step in self.steps:return False
        values = {tag:row.get(key) for key,tag in SCALARS.items()}
        responses = row.get('responses',0)
        if responses:
            values['rollout/mean_response_tokens'] = row.get('response_tokens',0)/responses
            values['rollout/truncation_rate'] = row.get('truncated_responses',0)/responses
        for tag,value in values.items():
            if value is not None and math.isfinite(float(value)):
                self.writer.add_scalar(tag,float(value),step,walltime=walltime)
        self.writer.flush()
        self.steps.add(step)
        return True

    def backfill(self, metrics, through_step=None):
        metrics = Path(metrics)
        if not metrics.exists():return
        with metrics.open() as stream:
            for line in stream:
                try:row = json.loads(line)
                except json.JSONDecodeError:continue
                if through_step is None or row['update']<=through_step:self.record(row)

    def close(self):
        if self.writer is not None:self.writer.close();self.writer=None
        if not self.lock.closed:self.lock.close()
