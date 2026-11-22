"""Profile the BF16 decode hot path and time RLT execution flags."""
import argparse
from collections import Counter
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREVIOUS = OUT.parent / 'rlt_v020_runtime_probe_20261120_151153'
PYTHON = OUT.parent / 'vllm_rlt_v020_release_20261120_144351/benchmark-venv/bin/python'
sys.path.insert(0, str(ROOT))
from ep_loop.gpu_resources import require_idle_gpus

SETTINGS = {
    'baseline': {},
    'single_stream': {'multi_stream': False},
    'static_buffers': {'static_buffers': True},
    'no_uva': {'use_uva': False},
    'trace': {'trace': True},
}


def load(path):
    return json.loads(path.read_text())


def write(path, values):
    path.write_text(json.dumps(values, indent=2) + '\n')


def benchmark(kind, case=None):
    import torch
    from ep_loop.cpu_runtime import configure_cpu_threads
    from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.models import AutoModelForCausalLM, prepare_model
    from vllm_rlt.worker.sampler import Sampler

    configure_cpu_threads()
    workload = load(PREVIOUS / 'workload.json')
    source, revision, config = prepare_model(workload['model'])
    config = replace(config, model_specific_config=replace(config.model_specific_config, total_ut_steps=2))
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, model_config=config, device='cuda:0', dtype=torch.bfloat16,
    )

    def make_engine(settings):
        settings = dict(settings)
        trace = settings.pop('trace', False)
        depths = {job['id']: [2] * 1024 for job in workload['phases']['short']['jobs']}
        return StageEngine(
            model,
            cache_config=CacheConfig(layout='last_exited', block_size=128,
                                     gpu_memory_utilization=.75, enable_prefix_caching=False),
            scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=1024,
                                             prefill_chunk_size=1024),
            attention_backend='triton',
            exit_config=ExitConfig(mode='trace', depths_by_request=depths) if trace else ExitConfig(),
            execution_config=ExecutionConfig(cuda_graphs=True, prefill_uva=True, **settings),
        )

    def admit(engine, phase, cap):
        for job in phase['jobs']:
            engine.add_request(job['id'], job['input_ids'], SamplingParams(
                max_tokens=cap, ignore_eos=True, temperature=1., top_p=.7, seed=job['seed'],
                min_loops=2, max_loops=2,
            ))

    def run(engine, phase, cap):
        torch.cuda.synchronize()
        started = time.perf_counter()
        first = None
        lengths = {}
        final = {}
        admit(engine, phase, cap)
        while engine.has_unfinished_requests():
            for row in engine.step():
                if row.token_ids:
                    lengths[row.request_id] = len(row.token_ids)
                if row.finished:
                    assert len(row.token_ids) == cap and row.exit_depths == [2] * cap
                    final[row.request_id] = row.token_ids
            if first is None and len(lengths) == 8:
                torch.cuda.synchronize()
                first = time.perf_counter(), sum(lengths.values())
        torch.cuda.synchronize()
        end = time.perf_counter()
        assert len(final) == 8 and first is not None
        return dict(seconds=end-started, all_first_s=first[0]-started,
                    remaining_tok_s=(cap*8-first[1])/(end-first[0]), token_ids=final)

    if kind == 'tune':
        engine = make_engine(SETTINGS[case])
        try:
            results = {}
            for phase in ('short', '8192'):
                data = workload['phases'][phase]
                cap = data['generated_tokens']
                run(engine, data, 16)
                results[phase] = [run(engine, data, cap) for _ in range(2)]
            write(OUT/f'tuning_{case}.json', results)
            print(case, {name: round(statistics.median(r['remaining_tok_s'] for r in rows),2)
                         for name, rows in results.items()}, flush=True)
        finally:
            engine.close()
        return

    original_sample = Sampler.sample

    def profiled_sample(self, *args, **kwargs):
        with torch.profiler.record_function('rlt_sampling'):
            return original_sample(self, *args, **kwargs)

    Sampler.sample = profiled_sample
    engine = make_engine({})
    profiles = {}
    try:
        for phase in ('short', '8192', '14336'):
            data = workload['phases'][phase]
            run(engine, data, 32)
            counts = {}

            def step():
                for row in engine.step():
                    if row.token_ids:
                        counts[row.request_id] = len(row.token_ids)

            admit(engine, data, 96)
            while len(counts) < 8:
                step()
            torch.cuda.synchronize()
            before = dict(counts)
            graph = engine.model_runner.graphs
            graphs_before = dict(replay=graph.replays, capture=graph.captures, fallback=graph.fallbacks)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            ) as prof:
                while min(counts[rid] - value for rid, value in before.items()) < 24:
                    step()
                torch.cuda.synchronize()
            committed = sum(counts[rid] - value for rid, value in before.items())
            by_kernel = Counter()
            for evt in prof.events():
                if evt.device_type == torch.autograd.DeviceType.CUDA:
                    by_kernel[evt.name] += evt.self_device_time_total
            total = by_kernel.total()
            attn_time = sum(v for k,v in by_kernel.items() if '_paged_attention_' in k)
            sample_time = next(r.device_time_total for r in prof.key_averages() if r.key == 'rlt_sampling')
            profiles[phase] = dict(
                committed_tokens=committed, sum_gpu_activity_us=total,
                attention_kernel_us=attn_time, sampler_cuda_us=sample_time,
                graph_counts=dict(replay=graph.replays-graphs_before['replay'],
                                  capture=graph.captures-graphs_before['capture'],
                                  fallback=graph.fallbacks-graphs_before['fallback']),
                attention_activity_share=attn_time/total,
                sampler_activity_share=sample_time/total,
                top_kernels=by_kernel.most_common(12),
            )
            print(phase, {k:v for k,v in profiles[phase].items() if k!='top_kernels'}, flush=True)
            while engine.has_unfinished_requests():
                step()
    finally:
        engine.close()
    write(OUT/'profile.json', profiles)


def main():
    gpu = '3'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    env.pop('PYTHONPATH', None)
    with open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_idle_gpus([gpu])
        for case in ('baseline', 'single_stream', 'static_buffers', 'no_uva', 'baseline'):
            subprocess.run([str(PYTHON), '-u', __file__, '--worker', 'tune', '--case', case],
                           env=env, check=True)
            if case == 'baseline' and not (OUT/'tuning_baseline_first.json').exists():
                (OUT/'tuning_baseline.json').replace(OUT/'tuning_baseline_first.json')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('profile', 'tune'))
    parser.add_argument('--case', choices=tuple(SETTINGS))
    args=parser.parse_args()
    if args.worker:
        benchmark(args.worker, args.case)
    else:
        main()
