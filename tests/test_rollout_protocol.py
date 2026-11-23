"""Question sampling, held-out isolation and transient teacher storage."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ep_loop.data import sample_rollout_indices, split_training_records


class RolloutProtocolTests(unittest.TestCase):
    def test_all_pool_reserves_disjoint_reproducible_dev_questions(self):
        records = [dict(id=i) for i in range(1000)]
        first = split_training_records(records, None, 64, 64, 42)
        self.assertEqual(first, split_training_records(records, None, 64, 64, 42))
        self.assertEqual(len(first['train.jsonl']), 936)
        self.assertEqual(first['eval.jsonl'], first['dev.jsonl'])
        train = {r['id'] for r in first['train.jsonl']}
        dev = {r['id'] for r in first['dev.jsonl']}
        self.assertFalse(train & dev)
        self.assertEqual(train | dev, set(range(1000)))
        self.assertEqual([r['id'] for r in records], list(range(1000)))

    def test_numeric_subset_keeps_historical_split_and_seed_order(self):
        records = list(range(1000))
        shuffled = records.copy()
        random.Random(42).shuffle(shuffled)
        split = split_training_records(records, 256, 64, 32, 42)
        self.assertEqual(split['train.jsonl'], shuffled[:256])
        self.assertEqual(split['dev.jsonl'], shuffled[256:320])
        self.assertEqual(split['eval.jsonl'], shuffled[256:288])

    def test_batches_revisit_questions_and_resume_same_sampling_stream(self):
        rng = random.Random(42)
        batches = [sample_rollout_indices(rng, 1000, 32) for _ in range(100)]
        self.assertTrue(all(len(set(batch)) == 32 for batch in batches))
        draws = [i for batch in batches for i in batch]
        self.assertEqual(len(draws), 3200)
        self.assertLess(len(set(draws)), len(draws))
        state = rng.getstate()
        expected = sample_rollout_indices(rng, 1000, 32)
        resumed = random.Random()
        resumed.setstate(state)
        self.assertEqual(expected, sample_rollout_indices(resumed, 1000, 32))
        with self.assertRaisesRegex(ValueError, 'distinct questions'):
            sample_rollout_indices(rng, 31, 32)

    def test_teacher_storage_uses_selected_volume_checks_space_and_cleans(self):
        from ep_loop.fastpath.teacher import teacher_readout_directory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = root/'ram_volume'
            config = {'implementation':{'teacher_trace_dir':str(configured)}}
            with teacher_readout_directory(config, root/'output', 1024) as selected:
                path = Path(selected)
                self.assertEqual(path.parent, configured.resolve())
                (path/'transient').write_bytes(b'loop states')
            self.assertFalse(path.exists())
            with patch('ep_loop.fastpath.teacher.shutil.disk_usage', return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(RuntimeError, 'TEACHER_TRACE_DIR'):
                    teacher_readout_directory(config, root/'output', 1024)
            self.assertFalse(list(configured.glob('.teacher_readouts_*')))


if __name__ == '__main__':
    unittest.main()
