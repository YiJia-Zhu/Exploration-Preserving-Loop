"""Resume compatibility and nonintrusive scalar continuity checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ep_loop.checkpointing import (validate_resume_config, inspect_resume_checkpoint,
                                   validate_resume_state, validate_resume_data, copy_resume_metrics)
from ep_loop.config import load_config
from ep_loop.data import canonical_hash
from ep_loop.reproducibility import capture_rng_state
from ep_loop.tensorboard_logging import ScalarLogger

ROOT = Path(__file__).resolve().parents[1]


class CheckpointTensorboardTests(unittest.TestCase):
    def test_extending_target_and_save_cadence_preserves_training_settings(self):
        saved=load_config(ROOT/'configs/vllm_fixed_top_n.json')
        saved['optimization']['updates']=5
        current=copy.deepcopy(saved);current['optimization']['updates']=10
        current['implementation']['save_every_updates']=10
        self.assertTrue(validate_resume_config(saved,current,5)['target_extended'])
        bad=copy.deepcopy(current);bad['optimization']['learning_rate']*=2
        with self.assertRaisesRegex(ValueError,'unchanged'):validate_resume_config(saved,bad,5)
        with self.assertRaisesRegex(ValueError,'cannot reduce'):validate_resume_config(current,saved,5)

    def test_incomplete_world_size_and_data_identity_are_rejected(self):
        config=load_config(ROOT/'configs/vllm_fixed_top_n.json')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'incomplete'):inspect_resume_checkpoint(root,config,2)
            marker=dict(completed_updates=5,world_size=2,config_hash=canonical_hash(config),resumable=True,data_identity='old')
            (root/'checkpoint.complete.json').write_text(json.dumps(marker))
            (root/'run_config.json').write_text(json.dumps(config))
            for name in ['training_state.pt','rng_rank0.pt','rng_rank1.pt']:(root/name).write_bytes(b'test')
            info=inspect_resume_checkpoint(root,config,2)
            with self.assertRaisesRegex(ValueError,'world size'):inspect_resume_checkpoint(root,config,1)
            with self.assertRaisesRegex(ValueError,'pool or order'):validate_resume_data(info,dict(usable_prompt_hash='new'))
            with self.assertRaisesRegex(ValueError,'completed_updates'):validate_resume_state(marker|dict(completed_updates=4),info)

    def test_tensorboard_resumes_at_six_without_duplicates_or_rng_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            before=capture_rng_state('cpu')
            logger=ScalarLogger(directory)
            for step in range(1,6):logger.record(dict(update=step,mean_response_loss=1/step,learning_rate=1e-6))
            logger.close()
            logger=ScalarLogger(directory,resumed_at=5)
            for step in range(1,11):logger.record(dict(update=step,mean_response_loss=1/step,learning_rate=1e-6))
            logger.close()
            after=capture_rng_state('cpu')
            self.assertEqual(before['python_rng'],after['python_rng'])
            self.assertEqual(before['numpy_rng'],after['numpy_rng'])
            torch.testing.assert_close(before['torch_rng'],after['torch_rng'],rtol=0,atol=0)
            events=EventAccumulator(str(Path(directory)/'tensorboard'),size_guidance={'scalars':0}).Reload()
            self.assertEqual([x.step for x in events.Scalars('train/loss')],list(range(1,11)))

    def test_resuming_new_directory_copies_only_committed_scalar_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';destination=root/'destination'
            source.mkdir();destination.mkdir();checkpoint=source/'checkpoints/update_000005'
            checkpoint.mkdir(parents=True)
            (source/'metrics.jsonl').write_text(''.join(json.dumps(dict(update=i,mean_response_loss=.1))+'\n' for i in range(1,8))+'{"update":8')
            copy_resume_metrics(checkpoint,destination,5)
            rows=[json.loads(line) for line in (destination/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['update'] for r in rows],[1,2,3,4,5])


if __name__=='__main__':unittest.main()
