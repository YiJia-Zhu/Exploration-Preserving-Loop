"""Time teacher forward, student replay and backward on one recorded CPU trajectory."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time

from benchmark_runtime import ROOT, digest, write_json


class PhaseTimes:
    def __init__(self):
        self.seconds = defaultdict(float)

    def start(self, name, **fields):
        return name, time.perf_counter()

    def finish(self, span):
        name, started = span
        self.seconds[name] += time.perf_counter() - started

    @contextmanager
    def stage(self, name, **fields):
        span = self.start(name)
        try:
            yield
        finally:
            self.finish(span)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout', type=Path, help='Directory of the matched CPU natural workload')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = json.loads((args.rollout / 'manifest.json').read_text())
    if work['device'] != 'cpu' or work['profile'] != 'natural' or not work['smoke'] or work['loops'] != 4:
        parser.error('Use the four-loop CPU natural smoke benchmark')
    trajectory = args.rollout / 'ragged.json'
    row = json.loads(trajectory.read_text())[0]['batches'][0]['responses'][0]
    job = next(job for job in work['jobs'] if job['id'] == row['id'])
    model = Path(work['model'])
    if any(digest(model / name) != checksum for name, checksum in work['model_sha256'].items()):
        raise RuntimeError('The checkpoint differs from the recorded rollout')
    args.output.mkdir(parents=True, exist_ok=False)

    from ep_loop.config import load_config
    from ep_loop.cpu_runtime import configure_cpu_threads
    from ep_loop.objective import DistillationObjective
    from ep_loop.ouro import load_ouro
    import torch

    threads = configure_cpu_threads()
    student, _ = load_ouro(model, 'cpu', 'bf16', gradient_checkpointing=True)
    teacher, _ = load_ouro(model, 'cpu', 'bf16', gradient_checkpointing=False)
    student.model.to(dtype=torch.bfloat16).train()
    teacher.model.to(dtype=torch.bfloat16).requires_grad_(False).eval()
    base = load_config(ROOT / 'configs/rad_opd.json')['algorithm']
    tokens = torch.tensor([job['input_ids'] + row['token_ids']])
    routes = {
        'fixed4': (dict(base, student_policy='fixed', student_loops=4), None),
        'ragged4': (base, [4] * len(row['token_ids'])),
        'ragged_dynamic': (base, row['exit_depths']),
    }

    def step(algorithm, depths):
        student.model.zero_grad(set_to_none=True)
        phases = PhaseTimes()
        objective = DistillationObjective(student, teacher, algorithm, observer=phases)
        started = time.perf_counter()
        loss, info = objective(tokens, len(job['input_ids']), depths)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss in replay')
        with phases.stage('backward'):
            loss.backward()
        elapsed = time.perf_counter() - started
        return dict(total_seconds=elapsed, phase_seconds=dict(phases.seconds), loss=float(loss.detach()),
                    student_token_loops=info['student_executed_token_loops'],
                    teacher_token_loops=info['teacher_executed_token_loops'])

    report = dict(trajectory=str(trajectory.resolve()), trajectory_sha256=digest(trajectory),
                  id=job['id'], prompt_tokens=len(job['input_ids']), response_tokens=len(row['token_ids']),
                  route=row['exit_depths'], device='cpu', threads=threads, torch_version=torch.__version__,
                  checkpoint_sha256=work['model_sha256']['model.safetensors'], results={})
    for name, (algorithm, depths) in routes.items():
        step(algorithm, depths)
        report['results'][name] = step(algorithm, depths)
        write_json(args.output / 'replay.json', report)
        print(f'{name}: {report["results"][name]["total_seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()
