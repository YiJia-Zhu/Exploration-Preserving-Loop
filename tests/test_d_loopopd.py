"""Current-teacher versioning and shared-weight, stop-gradient semantics."""
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from ep_loop.config import load_config, validate_config
from ep_loop.fastpath.teacher import VllmTeacher
from ep_loop.inference import GeneratedResponse
from ep_loop.objective import DistillationObjective
from ep_loop.ouro import tiny_ouro

ROOT = Path(__file__).resolve().parents[1]


class Observer:
    def stage(self, *args, **kwargs):
        return nullcontext()

    def event(self, *args, **kwargs):
        pass


class DynamicLoopOPDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.template = tiny_ouro(ROOT / 'models/Ouro-1.4B-Thinking')

    def backend(self):
        return type(self.template)(copy.deepcopy(self.template.model), 'none')

    def actor_type(self, initial, versions, snapshots):
        """CPU transport substitute; compute real recurrent teacher readouts."""
        class Actor:
            runtime = {}

            def __init__(self, *args, **kwargs):
                self.backend = type(initial)(copy.deepcopy(initial.model).eval(), 'none')
                self.loops = kwargs['loops']

            def request(self, op, **kwargs):
                if op == 'weights':
                    weights = load_file(kwargs['path'])
                    self.backend.model.load_state_dict(weights, strict=True)
                    versions.append(kwargs['version'])
                    snapshots.append(weights['lm_head.weight'].clone())
                    return dict(weight_version=kwargs['version'])

            def teacher_trace(self, requests, output):
                responses = []
                for index, request in enumerate(requests):
                    ids = torch.tensor([request['input_ids']])
                    with torch.no_grad():
                        trace = self.backend.response_trace(ids, self.loops, 0)
                    path = Path(output) / f'{index}.safetensors'
                    save_file(dict(states=torch.stack(trace.states).contiguous(),
                                   gates=torch.stack(trace.gates).contiguous()), str(path))
                    responses.append(dict(path=str(path), input_positions=ids.shape[1],
                                          executed_token_loops=trace.executed_token_loops))
                return dict(responses=responses, seconds=.01, executed_loops=self.loops)

            def close(self):
                pass
        return Actor

    def test_current_teacher_matches_shared_model_gradients_across_updates_and_resume(self):
        for filename in ('vllm_fixed_d_loopopd.json', 'vllm_fixed_top_n_current.json'):
            with self.subTest(config=filename):
                self.check_shared_model_gradients(filename)

    def check_shared_model_gradients(self, filename):
        config = validate_config(load_config(ROOT / 'configs' / filename))
        student, reference = self.backend(), self.backend()
        versions, snapshots = [], []
        ids = torch.tensor([[1, 20, 30, 40, 50, 2]])
        response = GeneratedResponse([40, 50, 2], [2]*3, True, 10, .01)
        records = [dict(id='x', input_ids=[1, 20, 30])]
        optimizers = [torch.optim.SGD(b.model.parameters(), lr=.02) for b in (student, reference)]
        with tempfile.TemporaryDirectory() as folder, \
                patch('ep_loop.fastpath.teacher.VllmActor', self.actor_type(student, versions, snapshots)):
            config['implementation']['teacher_trace_dir'] = folder
            teacher = VllmTeacher(student, config, folder, 0, Observer())
            self.assertIs(teacher.model, student.model.lm_head)
            self.assertTrue(all(p.requires_grad for p in student.model.parameters()))
            cached = DistillationObjective(student, teacher, config['algorithm'], 2, True)
            online = DistillationObjective(reference, reference, config['algorithm'], 2, True)
            # A newly created engine must also accept the first resumed version.
            for update in (7, 8):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                teacher.prepare([('x', 0, response)], records, update)
                teacher.use_response('x', 0)
                loss, _ = cached(ids, 3, [2]*3)
                expected, _ = online(ids, 3, [2]*3)
                torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
                loss.backward()
                expected.backward()
                for (name, actual), (_, wanted) in zip(student.model.named_parameters(), reference.model.named_parameters()):
                    if wanted.grad is None:
                        self.assertIsNone(actual.grad, name)
                    else:
                        torch.testing.assert_close(actual.grad, wanted.grad, rtol=2e-4, atol=1e-6, msg=name)
                self.assertGreater(float(student.model.lm_head.weight.grad.norm()), 0)
                self.assertEqual(teacher.weight_version, update)
                self.assertFalse(list(Path(folder).glob('.vllm_teacher_weights*')))
                for optimizer in optimizers:
                    optimizer.step()
                teacher.clear_batch()
            self.assertEqual(versions, [7, 8])
            self.assertGreater(float((snapshots[1]-snapshots[0]).abs().max()), 0)
            teacher.close()
            self.assertFalse(list(Path(folder).glob('.teacher_readouts_*')))

    def test_frozen_teacher_keeps_initial_head_and_never_loads_student_weights(self):
        config = load_config(ROOT / 'configs/vllm_fixed_endpoint.json')
        student = self.backend()
        versions, snapshots = [], []
        response = GeneratedResponse([40, 2], [2]*2, True, 8, .01)
        with tempfile.TemporaryDirectory() as folder, \
                patch('ep_loop.fastpath.teacher.VllmActor', self.actor_type(student, versions, snapshots)):
            config['implementation']['teacher_trace_dir'] = folder
            teacher = VllmTeacher(student, config, folder, 0, Observer())
            initial = teacher.model.weight.clone()
            with torch.no_grad():
                student.model.lm_head.weight.add_(1)
            teacher.prepare([('x', 0, response)], [dict(id='x', input_ids=[1, 20, 30])], 9)
            torch.testing.assert_close(teacher.model.weight, initial, rtol=0, atol=0)
            self.assertEqual(teacher.weight_version, 'initial')
            self.assertEqual(versions, [])
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.model.parameters()))
            self.assertTrue(student.model.lm_head.weight.requires_grad)
            teacher.close()

    def test_stale_weight_acknowledgement_aborts_before_labels_and_cleans_transport(self):
        config = load_config(ROOT / 'configs/vllm_fixed_d_loopopd.json')
        student = self.backend()
        actor = self.actor_type(student, [], [])
        original = actor.request

        def stale(instance, op, **kwargs):
            result = original(instance, op, **kwargs)
            return dict(weight_version=kwargs['version']-1) if op == 'weights' else result

        with tempfile.TemporaryDirectory() as folder, patch.object(actor, 'request', stale), \
                patch('ep_loop.fastpath.teacher.VllmActor', actor):
            config['implementation']['teacher_trace_dir'] = folder
            teacher = VllmTeacher(student, config, folder, 0, Observer())
            response = GeneratedResponse([2], [2], True, 6, .01)
            try:
                with self.assertRaisesRegex(RuntimeError, 'Stale teacher weights'):
                    teacher.prepare([('x', 0, response)], [dict(id='x', input_ids=[1, 20, 30])], 3)
                self.assertFalse(teacher.pending)
                self.assertFalse(list(Path(folder).glob('.vllm_teacher_weights*')))
            finally:
                teacher.close()

    def test_primary_entry_builds_both_teacher_modes_with_matched_stage_controls(self):
        env = os.environ.copy()
        for key in ('OPD_MODE', 'ABLATION_GROUPS', 'EXECUTION_MODE', 'RUN_TAG', 'TEACHER_UPDATE',
                    'TEACHER_BACKEND', 'WARMUP_STEPS', 'GPUS', 'OUTPUT_ROOT'):
            env.pop(key, None)
        result = subprocess.run(['bash', str(ROOT / 'scripts/run_opd_comparison.sh'), '--dry-run'],
            cwd=ROOT, env=env, text=True, capture_output=True, check=True, timeout=60)
        plan, _ = json.JSONDecoder().raw_decode(result.stdout)
        self.assertEqual(plan['opd_modes'], ['loopopd', 'd_loopopd'])
        self.assertEqual(plan['comparison_role'], 'primary_opd')
        self.assertEqual(plan['warmup_steps'], dict(loopopd=0, d_loopopd=10))
        self.assertEqual(len(plan['commands']), 4)
        self.assertEqual(len({row['environment']['RUN_TAG'] for row in plan['commands']}), 4)
        for row in plan['commands']:
            expected = 'current' if row['opd_mode']=='d_loopopd' else 'frozen'
            self.assertEqual(row['environment']['TEACHER_UPDATE'], expected)
            self.assertEqual(row['environment']['EVAL_INITIAL'], '1' if row is plan['commands'][0] else '0')
            if row['stage_selection']=='endpoint':
                self.assertEqual(row['environment']['METHOD'], 'baseline')
                self.assertEqual(plan['teacher_logical_endpoints'][row['group']], 'fixed')


if __name__ == '__main__':
    unittest.main()
