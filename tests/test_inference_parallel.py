"""TP inference sharing must preserve per-rank on-policy samples and labels."""
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ep_loop.config import load_config, validate_config
from ep_loop.fastpath.parallel import create_inference_group, gpu_groups, split_responses

ROOT = Path(__file__).resolve().parents[1]


def paired_worker(rank, directory):
    """Real four-process CPU rendezvous; actors stand in for GPU inference only."""
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from ep_loop.fastpath.rollout import RolloutBatcher
    from ep_loop.fastpath.teacher import VllmTeacher

    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='4', LOCAL_WORLD_SIZE='4',
                      CUDA_VISIBLE_DEVICES='0,1,2,3')
    dist.init_process_group('gloo', init_method='file://'+str(Path(directory)/'rendezvous'),
                            rank=rank, world_size=4, timeout=timedelta(seconds=45))
    actors = []; calls = []; events = []

    class Observer:
        def stage(self, *args, **kwargs): return nullcontext()
        def event(self, event, **kwargs): events.append((event, kwargs))

    class Actor:
        def __init__(self, model, gpu, log, **kwargs):
            self.loops = kwargs['loops']
            assert kwargs['rollout_ngram'] is (not kwargs.get('teacher_trace',False))
            actors.append(dict(gpu=gpu, tp=kwargs['tensor_parallel_size'], loops=self.loops))
            assert kwargs['tensor_parallel_size'] == 2
            assert 'max_num_seqs' not in kwargs

        def request(self, op, **kwargs):
            return dict(weight_version=kwargs['version']) if op == 'weights' else None

        def generate(self, requests, **kwargs):
            calls.append(('rollout', len(requests)))
            return dict(seconds=.1, executed_loops=2, responses=[dict(
                token_ids=[10 + r['seed'] % 97, 2], exit_depths=[2, 2],
                executed_token_loops=2*(len(r['input_ids'])+1)) for r in requests])

        def teacher_trace(self, requests, output):
            calls.append(('teacher', len(requests)))
            result = []
            for index, request in enumerate(requests):
                path = Path(output)/f'{index}.safetensors'
                length = len(request['input_ids'])
                # Encode the actual student prefix in a teacher readout to
                # detect any cross-rank or cross-response label reassignment.
                states = torch.tensor(request['input_ids'], dtype=torch.float32)[None, :, None].repeat(4, 1, 3)
                save_file(dict(states=states, gates=torch.zeros(4, length, 1)), str(path))
                result.append(dict(path=str(path), input_positions=length, executed_token_loops=length*4))
            return dict(seconds=.1, executed_loops=4, responses=result)

        def close(self): pass

    config = load_config(ROOT/'configs/vllm_fixed_top_n.json')
    config['implementation'].update(vllm_tensor_parallel_size=2, teacher_trace_dir=directory)
    group = create_inference_group(config, rank)
    model = torch.nn.Module(); model.lm_head = torch.nn.Linear(3, 9)
    student = SimpleNamespace(model=model, device=torch.device('cpu'), stop_token_ids={2})
    records = [dict(id=str(i), input_ids=[1, 20+i, 30]) for i in range(32)]
    try:
        with patch('ep_loop.fastpath.rollout.VllmActor', Actor), \
             patch('ep_loop.fastpath.teacher.VllmActor', Actor), patch('torch.cuda.empty_cache'):
            batcher = RolloutBatcher(student, config, directory, rank, Observer(), group)
            teacher = VllmTeacher(student, config, directory, rank, Observer(), group)
            generator = torch.Generator().manual_seed(42+rank)
            batcher.prepare(records, list(range(32)), rank, 4, 0, generator)
            expected_seeds = torch.randint(0, 2**63-1, (64,), generator=torch.Generator().manual_seed(42+rank)).tolist()
            assert [row['sampling_seed'] for event, row in events if event == 'vllm_rollout_ready'] == expected_seeds
            teacher.prepare(list(batcher.pending), records, 0)
            consumed = 0
            for record in records[rank::4]:
                for sample in range(8):
                    response = batcher.pop(dict(id=record['id'], sample=sample))
                    assert response.token_ids == [10+expected_seeds[consumed] % 97, 2]
                    teacher.use_response(record['id'], sample)
                    prefix = record['input_ids']+response.token_ids[:-1]
                    trace = teacher.response_trace(torch.tensor([prefix]), 4, 0)
                    assert trace.states[0][:, 0].tolist() == prefix
                    consumed += 1
            # Real DDP also synchronizes before cleanup; followers must have
            # consumed their labels before the leader removes its directory.
            dist.barrier()
            teacher.close(); batcher.close()
            assert not batcher.pending and not teacher.pending
        failed = False
        def failure(requests): raise ValueError('intentional engine error')
        try: group.execute([rank], failure)
        except RuntimeError as error:
            failed = 'intentional engine error' in str(error)
        assert failed
        (Path(directory)/f'rank{rank}.json').write_text(json.dumps(dict(
            rank=rank, actors=actors, calls=calls, consumed=consumed, propagated_error=failed)))
    finally:
        dist.destroy_process_group()


class InferenceParallelTests(unittest.TestCase):
    def test_gpu_groups_and_invalid_layouts(self):
        self.assertEqual(gpu_groups(['4', '5', '0', '1'], 2), ['4,5', '0,1'])
        for devices, size in [([], 2), (['0'], 2), (['0', '1'], 3), (['0'], 0), (['0'], True)]:
            with self.assertRaises(ValueError): gpu_groups(devices, size)

    def test_response_partition_keeps_positional_identity(self):
        rows = [dict(id='same', seed=seed) for seed in [42, 123, 78]]
        parts = split_responses(dict(seconds=1., responses=rows), [2, 1])
        self.assertEqual(parts[0]['responses'], rows[:2])
        self.assertEqual(parts[1]['responses'], rows[2:])
        with self.assertRaises(RuntimeError): split_responses(dict(responses=rows), [1, 1])

    def test_all_tp_workers_must_load_the_same_weights_and_only_leader_writes(self):
        from ep_loop.fastpath.actor import validate_weight_workers, leader_teacher_traces
        rows = [dict(weight_version=5), dict(weight_version=5)]
        self.assertEqual(validate_weight_workers(rows, 5, 2)['workers'], rows)
        for workers in [rows[:1], [dict(weight_version=5), dict(weight_version=4)]]:
            with self.assertRaises(RuntimeError): validate_weight_workers(workers, 5, 2)
        trace = {'request': {'path': 'unique'}}
        self.assertEqual(leader_teacher_traces([trace, {}], 2), trace)
        with self.assertRaises(RuntimeError): leader_teacher_traces([trace, trace], 2)

    def test_teacher_follower_does_not_capture_or_write(self):
        from types import SimpleNamespace
        from ep_loop.fastpath.vllm_worker import OuroWorkerExtension
        worker = OuroWorkerExtension(); worker.rank = 1
        worker.get_model = lambda: SimpleNamespace(model=SimpleNamespace(teacher_trace=True))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'should_not_exist'
            self.assertFalse(worker.ep_begin_teacher_trace(output))
            self.assertEqual(worker.ep_end_teacher_trace(), {})
            self.assertFalse(output.exists())

    def test_configuration_rejects_unsafe_shared_gpu_layout(self):
        for value in [0, -1, True, 1.5]:
            config = load_config(ROOT/'configs/vllm_fixed_top_n.json')
            config['implementation']['vllm_tensor_parallel_size'] = value
            with self.assertRaisesRegex(ValueError, 'vllm_tensor_parallel_size'): validate_config(config)
        config = load_config(ROOT/'configs/vllm_fixed_top_n.json')
        config['implementation']['vllm_tensor_parallel_size'] = 2
        validate_config(config)
        config['implementation']['phase_offload'] = False
        with self.assertRaisesRegex(ValueError, 'phase_offload'): validate_config(config)

    def test_four_cpu_ranks_share_two_engines_and_keep_all_256_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(paired_worker, args=(directory,), nprocs=4, join=True)
            reports = [json.loads((Path(directory)/f'rank{rank}.json').read_text()) for rank in range(4)]
            self.assertEqual(sum(r['consumed'] for r in reports), 256)
            for rank, report in enumerate(reports):
                if rank % 2:
                    self.assertEqual(report['actors'], [])
                    self.assertEqual(report['calls'], [])
                else:
                    self.assertEqual([a['gpu'] for a in report['actors']], [f'{rank},{rank+1}']*2)
                    self.assertEqual(report['calls'], [['rollout', 128], ['teacher', 128]])
            self.assertFalse(list(Path(directory).glob('.teacher_readouts_*')))


if __name__ == '__main__': unittest.main()
