"""Semantic checks for the physically ragged cache and its differentiable replay."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from ep_loop.config import DEFAULT_CONFIG, load_config, validate_config
from ep_loop.evaluation_data import rad_pass_metrics
from ep_loop.inference import generate
from ep_loop.objective import DistillationObjective
from ep_loop.ouro import OuroBackend, tiny_ouro
from ep_loop.ragged import ReplayRoundKV, replay_trace, sample_training_threshold


ROOT = Path(__file__).resolve().parents[1]


class RadOpdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        cls.template = tiny_ouro(ROOT / 'models/Ouro-1.4B-Thinking')

    def backend(self):
        return OuroBackend(copy.deepcopy(self.template.model), 'none')

    def test_mixed_depth_replay_matches_rollout_with_physical_aliases(self):
        backend = self.backend()
        backend.model.eval()
        seq = torch.tensor([[1, 19, 31, 42, 54, 66, 78]])
        prompt = 3
        exits = [4, 1, 4, 2, 3]
        states = []
        cache = None
        with torch.no_grad():
            for offset, depth in enumerate(exits):
                ids = seq[:, :prompt] if offset == 0 else seq[:, prompt+offset-1:prompt+offset]
                call = 0

                def stop(hidden):
                    nonlocal call
                    call += 1
                    return hidden.new_full((*hidden.shape[:2], 1), 100. if call == depth else -100.)

                with patch.object(backend.model.model.early_exit_gate, 'forward', side_effect=stop):
                    trace = backend.trace(ids, 4, cache, True, 'ragged', .5)
                self.assertEqual(len(trace.states), depth)
                self.assertEqual(trace.backfill_token_loops, 0)
                states.append(trace.states[-1][0, -1])
                cache = trace.cache
            self.assertEqual(cache.physical_slots, (prompt-1)*4+sum(exits))
            self.assertTrue(all(buffer.size == cache.physical_slots for buffer in cache.keys + cache.values))
            # The first decode input stops at 1. All deeper lanes point to its single slot.
            self.assertEqual(len({int(index.value()[prompt]) for index in cache.indices}), 1)
            selected = replay_trace(backend, seq, 4, prompt-1, torch.tensor(exits)).states[0]
            torch.testing.assert_close(selected, torch.stack(states), rtol=3e-4, atol=3e-6)

    def test_ragged_threshold_one_matches_dense_and_counts_prefill(self):
        backend = self.backend()
        expected = [41, 52, 63, 74]
        ids = [1, 19, 31]
        outcomes = {}
        for policy in ('fixed', 'ragged'):
            with patch('ep_loop.inference.sample_token', side_effect=[torch.tensor(t) for t in expected]):
                outcomes[policy] = generate(backend, ids, 4, 4, 1., 1., torch.Generator(),
                                            endpoint=policy, exit_threshold=1.)
        for info in outcomes.values():
            self.assertEqual(info.exit_depths, [4]*4)
            self.assertEqual(info.physical_kv_token_slots, 24)
        with torch.no_grad():
            backend.model.eval()
            ids = torch.tensor([ids+expected[:-1]])
            dense = backend.trace(ids, 4).states[-1][:, -1]
            ragged = backend.trace(ids[:, :3], 4, use_cache=True, endpoint='ragged', exit_threshold=1.)
            for pos in range(3, ids.shape[1]):
                ragged = backend.trace(ids[:, pos:pos+1], 4, ragged.cache, True, 'ragged', 1.)
            torch.testing.assert_close(ragged.states[-1][:, -1], dense, rtol=3e-4, atol=3e-6)

    def test_ragged_changes_depth_without_backfilling(self):
        backend = self.backend()
        backend.model.eval()
        gate = backend.model.model.early_exit_gate
        with torch.no_grad(), patch.object(gate, 'forward', side_effect=lambda x: x.new_full((*x.shape[:2], 1), 100.)):
            prefill = backend.trace(torch.tensor([[1, 19, 31]]), 4, use_cache=True,
                                    endpoint='ragged', exit_threshold=.5)
            shallow = backend.trace(torch.tensor([[42]]), 4, prefill.cache, True, 'ragged', .5)
        with torch.no_grad(), patch.object(gate, 'forward', side_effect=lambda x: x.new_full((*x.shape[:2], 1), -100.)):
            deep = backend.trace(torch.tensor([[54]]), 4, shallow.cache, True, 'ragged', .5)
        self.assertEqual([len(prefill.states), len(shallow.states), len(deep.states)], [4, 1, 4])
        self.assertEqual(deep.backfill_token_loops, 0)
        self.assertEqual(deep.executed_token_loops, 4)

    def test_fixed_budget_timing_does_not_stop_on_eos(self):
        backend = self.backend()
        eos, = backend.stop_token_ids
        for endpoint in ('fixed', 'ragged'):
            for ignore_eos, expected in ((False, [eos]), (True, [eos, 19, eos])):
                with patch('ep_loop.inference.sample_token', side_effect=map(torch.tensor, [eos, 19, eos])):
                    response = generate(backend, [1, 5, 7], 4, 3, 1., 1., torch.Generator(),
                                        endpoint=endpoint, exit_threshold=1., ignore_eos=ignore_eos)
                self.assertEqual(response.token_ids, expected)
                self.assertEqual(response.reached_eos, not ignore_eos)

    def test_reused_kv_gradient_reaches_shallow_projection_unless_detached(self):
        for detach in (False, True):
            old = [torch.ones(1, 1, 2, 1, requires_grad=True) for _ in range(2)]
            current = [torch.full_like(value, 2, requires_grad=True) for value in old]
            cache = ReplayRoundKV(tuple(old), torch.tensor([[[[False], [True]]]]), detach)
            effective = cache.update(*current, 2)
            self.assertEqual([value.flatten().tolist() for value in effective], [[1, 2], [1, 2]])
            sum(value.sum() for value in effective).backward()
            for past, now in zip(old, current, strict=True):
                self.assertEqual(now.grad.flatten().tolist(), [0, 1])
                if detach:
                    self.assertIsNone(past.grad)
                else:
                    self.assertEqual(past.grad.flatten().tolist(), [1, 0])

    def test_checkpointed_objective_matches_gradients_and_freezes_gate(self):
        outputs = []
        for use_checkpoint in (False, True):
            student, teacher = self.backend(), self.backend()
            teacher.model.eval().requires_grad_(False)
            student.model.train()
            if use_checkpoint:
                student.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            a = validate_config(load_config(DEFAULT_CONFIG))['algorithm']
            objective = DistillationObjective(student, teacher, a, 2, use_checkpoint)
            response = torch.tensor([[1, 19, 31, 42, 54, 66, 78, 89]])
            loss, info = objective(response, 3, [4, 1, 4, 2, 3])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(info['stop_loss'].item(), 0.)
            self.assertEqual(info['teacher_executed_loops'], 4)
            self.assertEqual(info['student_executed_token_loops'], 28)
            self.assertIsNone(student.model.model.early_exit_gate.weight.grad)
            self.assertFalse(student.model.model.early_exit_gate.weight.requires_grad)
            self.assertTrue(all(p.grad is None for p in teacher.model.parameters()))
            outputs.append((loss.detach(), [p.grad for p in student.model.parameters() if p.requires_grad]))
        torch.testing.assert_close(outputs[0][0], outputs[1][0])
        for eager, recomputed in zip(outputs[0][1], outputs[1][1], strict=True):
            torch.testing.assert_close(eager, recomputed, rtol=3e-4, atol=3e-6)

    def test_tau_schedule_and_pass_metrics(self):
        tau = load_config(DEFAULT_CONFIG)['algorithm']['rad_tau']
        sample = sample_training_threshold(tau, 42, 'abc', 5)
        self.assertEqual(sample, sample_training_threshold(tau, 42, 'abc', 5))
        self.assertTrue(.15 <= sample <= .4 or sample in tau['anchors'])
        self.assertEqual(sample_training_threshold({**tau, 'fixed': .2}, 42, 'abc', 6), .2)
        self.assertEqual(rad_pass_metrics({'a': [1, 0, 0, 0], 'b': [0]*4}),
                         {'p_at_1': .125, 'p_at_4': .5})
        self.assertIsNone(rad_pass_metrics({'a': [1]})['p_at_4'])

    def test_initial_thresholds_are_shared_by_questions_in_each_update(self):
        tau = load_config(DEFAULT_CONFIG)['algorithm']['rad_tau']
        initial = [.8, .6, .4, .3, .25, .2, .15]
        scheduled = {**tau, 'initial_thresholds': initial}
        for update, threshold in enumerate(initial):
            for fingerprint in ('a', 'b'):
                self.assertEqual(sample_training_threshold(scheduled, 42, fingerprint, update), threshold)
        self.assertEqual(sample_training_threshold(scheduled, 42, 'a', len(initial)),
                         sample_training_threshold(tau, 42, 'a', len(initial)))


if __name__ == '__main__':
    unittest.main()
