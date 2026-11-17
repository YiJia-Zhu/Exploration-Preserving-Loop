"""Exact cached early exits, deeper suffix fill, coda isolation and RNG reuse."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from ep_loop.cache import ExactLoopCache, TensorBuffer
from ep_loop.huginn import tiny_huginn
from ep_loop.inference import generate
from ep_loop.ouro import tiny_ouro

ROOT = Path(__file__).resolve().parents[1]


class DynamicCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.templates = {
            "ouro": tiny_ouro(ROOT / "models/Ouro-1.4B-Thinking"),
            "huginn": tiny_huginn(ROOT / "models/huginn-0125"),
        }

    def backend(self, family):
        template = self.templates[family]
        backend = type(template)(copy.deepcopy(template.model), "none")
        backend.enable_student_exit()
        backend.model.eval()
        return backend

    @staticmethod
    def gate_at(depth):
        calls = [0]
        def forward(hidden):
            calls[0] += 1
            return hidden.new_full((*hidden.shape[:2], 1), 100. if calls[0] >= depth else -100.)
        return forward

    def compare_prefixes(self, family, sliding=False):
        backend = self.backend(family)
        if sliding:
            backbone = backend.model.model
            backbone.config.sliding_window = 3
            backbone.config.layer_types[1] = "sliding_attention"
            backbone.has_sliding_layers = True
            backbone.layers[1].attention_type = "sliding_attention"
            backbone.layers[1].self_attn.sliding_window = 3
        reference = type(backend)(copy.deepcopy(backend.model), "none")
        gate = backend.enable_student_exit()
        ids = torch.tensor([[1, 20, 30, 40, 50]])
        cache, cost = None, 0
        # Includes delayed deeper computation, shallow reuse, and a multi-ID
        # suffix. A multi-token backfill must use a bottom-right causal mask.
        for step, (depth, added) in enumerate(zip([1, 4, 2, 3, 1, 4], [0, 1, 2, 1, 1, 1])):
            if added:
                ids = torch.cat((ids, torch.full((1, added), 60 + step)), dim=1)
            current = ids if cache is None else ids[:, -added:]
            with torch.no_grad(), patch.object(gate, "forward", side_effect=self.gate_at(depth)), \
                    patch.object(reference.enable_student_exit(), "forward", side_effect=self.gate_at(depth)):
                trace = backend.trace(current, 4, cache, True, "learned", .5)
                cache = trace.cache
                if family == "huginn":
                    # Compare the SAME native random initial states, including
                    # history. Redrawing them would compare different models.
                    with patch.object(reference.model, "initialize_state", return_value=cache.levels[0].value()):
                        full = reference.trace(ids, depth)
                else:
                    full = reference.trace(ids, depth)
                self.assertEqual(trace.exits[0, -1].item(), depth)
                for d in range(depth):
                    torch.testing.assert_close(trace.states[d], full.states[d][:, -current.shape[1]:], rtol=3e-4, atol=3e-6)
                    torch.testing.assert_close(trace.gates[d], full.gates[d][:, -current.shape[1]:], rtol=3e-4, atol=3e-6)
                self.assertEqual(trace.executed_token_loops - trace.backfill_token_loops, depth * current.shape[1])
                cost += trace.executed_token_loops
                self.assertEqual(cost, sum(level.size for level in cache.levels[1:]))
                if step == 1:
                    self.assertEqual(trace.backfill_token_loops, 3 * 5)
        self.assertLessEqual(cost, 4 * ids.shape[1])

    def test_ouro_matches_full_prefix_after_arbitrary_depth_changes(self):
        self.compare_prefixes("ouro")

    def test_long_buffer_bounds_slack_and_preserves_values(self):
        buffer=TensorBuffer(1)
        expected=torch.arange(20000).reshape(1,-1,1)
        for start in range(0,20000,83):
            value=buffer.append(expected[:,start:start+83])
            self.assertLessEqual(buffer.storage.shape[1]-buffer.size,1024)
        torch.testing.assert_close(value,expected)

    def test_ouro_chunked_late_backfill_matches_native_all_readouts(self):
        backend=self.backend('ouro')
        ids=torch.tensor([[1]+[20+i%100 for i in range(30)]])
        gate=backend.enable_student_exit()
        with torch.no_grad():
            with patch.object(gate,'forward',side_effect=self.gate_at(1)):
                shallow=backend.trace(ids[:,:-1],4,use_cache=True,endpoint='learned')
            shallow.cache.backfill_chunk_size=3
            with patch.object(gate,'forward',side_effect=self.gate_at(4)):
                deep=backend.trace(ids[:,-1:],4,shallow.cache,True,'learned')
            native=backend.trace(ids,4)
            for actual,expected in zip(deep.states,native.states):
                torch.testing.assert_close(actual,expected[:,-1:],rtol=3e-4,atol=3e-6)
            self.assertEqual(deep.backfill_token_loops,3*30)
            self.assertEqual(deep.executed_token_loops,94)

    def test_ouro_mixed_sliding_attention_and_suffix_masks(self):
        self.compare_prefixes("ouro", sliding=True)

    def test_huginn_matches_full_prefix_with_fixed_native_random_states(self):
        self.compare_prefixes("huginn")

    def test_huginn_initializes_each_position_once_and_separates_coda_depths(self):
        backend = self.backend("huginn")
        gate = backend.enable_student_exit()
        lengths = []
        original = backend.model.initialize_state
        def initialize(embedded):
            lengths.append(embedded.shape[1])
            return original(embedded)
        with torch.no_grad(), patch.object(backend.model, "initialize_state", side_effect=initialize):
            with patch.object(gate, "forward", side_effect=self.gate_at(1)):
                first = backend.trace(torch.tensor([[1, 20, 30]]), 4, use_cache=True, endpoint="learned", exit_threshold=.5)
            with patch.object(gate, "forward", side_effect=self.gate_at(4)):
                second = backend.trace(torch.tensor([[40]]), 4, first.cache, True, "learned", .5)
        self.assertEqual(lengths, [3, 1])
        self.assertEqual(second.backfill_token_loops, 9)
        self.assertEqual(second.coda_token_positions, 13)
        self.assertEqual([kv.get_seq_length(-1) for kv in second.cache.coda], [4] * 4)
        self.assertIsNot(second.cache.coda[0].keys[-1].storage, second.cache.coda[1].keys[-1].storage)

    def test_huginn_can_fill_from_one_to_full_32_loop_limit(self):
        backend = self.backend("huginn")
        gate = backend.enable_student_exit()
        with torch.no_grad():
            gate.bias.fill_(100.)
            first = backend.trace(torch.tensor([[1, 20, 30]]), 32, use_cache=True, endpoint="learned", exit_threshold=.5)
            gate.bias.fill_(-100.)
            final = backend.trace(torch.tensor([[40]]), 32, first.cache, True, "learned", .5)
            with patch.object(backend.model, "initialize_state", return_value=final.cache.levels[0].value()):
                full = backend.trace(torch.tensor([[1, 20, 30, 40]]), 32)
            torch.testing.assert_close(final.states[-1], full.states[-1][:, -1:], rtol=3e-4, atol=3e-6)
        self.assertEqual(final.exits.item(), 32)
        self.assertEqual(final.backfill_token_loops, 31 * 3)
        self.assertEqual(final.executed_token_loops, 32 + 31 * 3)

    def test_generation_counts_backfill_without_inflating_new_token_depths(self):
        for family in ("ouro", "huginn"):
            backend = self.backend(family)
            gate = backend.enable_student_exit()
            calls = [0]
            def forward(hidden):
                calls[0] += 1
                # First token exits at 1; second at 4; third at 1.
                return hidden.new_full((*hidden.shape[:2], 1), -100. if 2 <= calls[0] <= 4 else 100.)
            with patch.object(gate, "forward", side_effect=forward), patch("ep_loop.inference.sample_token", return_value=torch.tensor([40])):
                out = generate(backend, [1, 20, 30], 4, 3, 1., 1., torch.Generator(), "learned", .5, True)
            self.assertEqual(out.exit_depths, [1, 4, 1])
            self.assertEqual(out.execution_depths, [1, 4, 1])
            self.assertEqual(out.executed_token_loops, 3 + 13 + 1)
            self.assertEqual(out.backfill_token_loops, 9)
            self.assertEqual(out.cache_policy, "exact_lazy_depth")

    def test_cached_ouro_sampling_matches_recompute_and_both_families_repeat(self):
        for family in ("ouro", "huginn"):
            backend = self.backend(family)
            results = []
            for cached in ([True, True, False] if family == "ouro" else [True, True]):
                torch.manual_seed(42)
                result = generate(backend, [1, 20, 30], 4, 12, 1., .7,
                    torch.Generator().manual_seed(42), "learned", .5, cached)
                results.append((result.token_ids, result.exit_depths))
            self.assertTrue(all(result == results[0] for result in results))

    def test_cache_rejects_training_gradients_foreign_models_and_changed_depth(self):
        for family in ("ouro", "huginn"):
            backend = self.backend(family)
            ids = torch.tensor([[1, 20, 30]])
            with self.assertRaises(ValueError):
                backend.trace(ids, 4, use_cache=True, endpoint="learned", exit_threshold=.5)
            with torch.no_grad():
                first = backend.trace(ids, 4, use_cache=True, endpoint="learned", exit_threshold=.5)
                for cache, loops in ((first.cache, 3), (ExactLoopCache(self.backend(family).model, 4), 4)):
                    with self.assertRaises(ValueError):
                        backend.trace(ids[:, :1], loops, cache, True, "learned", .5)
                backend.model.train()
                with self.assertRaises(ValueError):
                    backend.trace(ids[:, :1], 4, first.cache, True, "learned", .5)

    def test_generation_restores_training_mode_and_cast_settings_after_failure(self):
        backend = self.backend("ouro")
        backend.model.train()
        with patch("ep_loop.inference.sample_token", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                generate(backend, [1, 20, 30], 4, 3, 1., 1., torch.Generator(), "learned", .5, True)
        self.assertTrue(backend.model.training)
        self.assertFalse(backend._generation_cast_cache)


if __name__ == "__main__":
    unittest.main()
