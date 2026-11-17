"""Cheap reproducibility checks without deterministic-kernel enforcement."""
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from ep_loop.reproducibility import (capture_rng_state, execution_provenance, rank_seed,
                                    restore_rng_state, sample_seed, seed_everything)

ROOT=Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.original=capture_rng_state('cpu')

    def tearDown(self):
        restore_rng_state(self.original,'cpu')

    def values(self):
        return random.random(),np.random.normal(size=5),torch.rand(5)

    def equal_values(self,first,second):
        self.assertEqual(first[0],second[0])
        np.testing.assert_array_equal(first[1],second[1])
        torch.testing.assert_close(first[2],second[2],rtol=0,atol=0)

    def test_seed_42_repeats_python_numpy_and_torch_without_changing_math_flags(self):
        flags=lambda:(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.benchmark,
                      torch.backends.cudnn.deterministic,torch.backends.cuda.matmul.allow_tf32)
        before=flags()
        seed_everything(42);first=self.values()
        seed_everything(42);second=self.values()
        self.equal_values(first,second)
        self.assertEqual(before,flags())

    def test_checkpoint_rng_is_safe_to_load_and_restores_all_global_streams(self):
        seed_everything(42)
        self.values()
        state=capture_rng_state('cpu')
        expected=self.values()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rng.pt'
            torch.save(state,path)
            seed_everything(111)
            restored=torch.load(path,map_location='cpu',weights_only=True)
        restore_rng_state(restored,'cpu')
        self.equal_values(expected,self.values())

    def test_legacy_rng_checkpoints_remain_accepted(self):
        seed_everything(42)
        state={'torch_rng':torch.get_rng_state(),'cuda_rng':None}
        expected=torch.rand(4)
        restore_rng_state(state,'cpu')
        torch.testing.assert_close(expected,torch.rand(4),rtol=0,atol=0)

    def test_content_seed_is_independent_of_python_hash_seed(self):
        command=[sys.executable,'-c',"from ep_loop.reproducibility import sample_seed; print(sample_seed(42,'question-fingerprint',1))"]
        outputs=[]
        for value in ('1','999'):
            result=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONHASHSEED':value},capture_output=True,text=True,check=True)
            outputs.append(int(result.stdout))
        self.assertEqual(outputs[0],outputs[1])
        self.assertEqual(outputs[0],sample_seed(42,'question-fingerprint',1))
        self.assertNotEqual(outputs[0],sample_seed(42,'question-fingerprint',0))
        self.assertNotEqual(outputs[0],sample_seed(42,'other-question',1))

    def test_rank_streams_share_base_seed_and_have_stable_offsets(self):
        self.assertEqual(rank_seed(42,0),42)
        self.assertEqual(rank_seed(42,1),100045)
        self.assertEqual(len({rank_seed(42,i) for i in range(4)}),4)

    def test_runtime_record_contains_versions_flags_and_current_source_hashes(self):
        record=execution_provenance(ROOT,'cpu',42)
        self.assertEqual(record['base_seed'],42)
        self.assertEqual(record['deterministic_algorithms'],torch.are_deterministic_algorithms_enabled())
        self.assertEqual(record['packages']['numpy'],np.__version__)
        self.assertIn('ep_loop/reproducibility.py',record['source_sha256'])
        self.assertIn('scripts/run_ablation.sh',record['source_sha256'])
        self.assertIsNone(record['gpu'])


if __name__=='__main__':unittest.main()
