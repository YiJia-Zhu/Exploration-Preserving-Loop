"""Compare RLT v0.2.0 runtime settings at fixed work on SM120."""
import argparse
import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import statistics
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RELEASE_AUDIT = OUT.parent / 'vllm_rlt_v020_release_20261120_144351'
RELEASE_PYTHON = RELEASE_AUDIT / 'benchmark-venv/bin/python'
sys.path.insert(0, str(ROOT))
from ep_loop.gpu_resources import require_idle_gpus

CASES = {
    'baseline': dict(chunk=128, packed=False, async_scheduling=False, block=16),
    'packed_prefill': dict(chunk=1024, packed=True, async_scheduling=False, block=16),
    'packed_async': dict(chunk=1024, packed=True, async_scheduling=True, block=16),
    'packed_async_block64': dict(chunk=1024, packed=True, async_scheduling=True, block=64),
}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def workload():
    original = read(RELEASE_AUDIT / 'workload.json')
    original_audit = OUT.parent / 'ouro_base_8q_backend_speed_20261120_143308'
    histories = [
        read(p / f'{backend}.json')['full']['answers']
        for p in (RELEASE_AUDIT, original_audit) for backend in ('vllm', 'rlt')
    ]
    jobs = original['jobs']
    completion_ids = {}
    for row in jobs:
        matched = [answer['token_ids'] for answers in histories for answer in answers
                   if answer['id'] == row['id']]
        completion_ids[row['id']] = max(matched, key=len)
    phases = {}
    for context, cap in ((None, 512), (8192, 128), (14336, 128)):
        data = []
        for job in jobs:
            ids = job['input_ids']
            if context:
                suffix = completion_ids[job['id']]
                needed = context - len(ids)
                ids = ids + (suffix * ((needed + len(suffix) - 1) // len(suffix)))[:needed]
            data.append(dict(id=job['id'], input_ids=ids, seed=job['seed']))
        phases['short' if context is None else str(context)] = dict(
            jobs=data, generated_tokens=cap,
        )
    return {'model': original['model'], 'phases': phases}


def worker(name, phases=None, repetitions=2):
    import torch
    from ep_loop.cpu_runtime import configure_cpu_threads
    from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
    from vllm_rlt.attention.backend import create_backend
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.models import AutoModelForCausalLM, prepare_model

    configure_cpu_threads()
    settings = CASES[name]
    spec = read(OUT / 'workload.json')
    source, revision, config = prepare_model(spec['model'])
    cfg = replace(config, model_specific_config=replace(
        config.model_specific_config, total_ut_steps=2,
    ))
    hardware = dict(name=torch.cuda.get_device_name(0),
                    sm=torch.cuda.get_device_capability(0),
                    torch=torch.__version__, cuda=torch.version.cuda)
    try:
        create_backend('flash_attn_4', torch.device('cuda:0'), torch.bfloat16,
                       cfg.model_specific_config.head_dim, 16)
        hardware['explicit_fa4'] = 'selected'
    except (ValueError, ImportError, OSError) as error:
        hardware['explicit_fa4'] = str(error)
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, model_config=cfg, device='cuda:0', dtype=torch.bfloat16,
    )
    is_async = settings['async_scheduling']
    engine = StageEngine(
        model,
        cache_config=CacheConfig(layout='last_exited', block_size=settings['block'],
                                 gpu_memory_utilization=.75, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=settings['chunk'],
                                         prefill_chunk_size=settings['chunk']),
        attention_backend='triton',
        exit_config=ExitConfig(mode='ouro_delayed' if is_async else 'ouro'),
        execution_config=ExecutionConfig(cuda_graphs=True, prefill_uva=settings['packed'],
                                         async_scheduling=is_async, static_buffers=is_async,
                                         pad_to_power_of_two=is_async),
    )

    phase_results = {}

    def run(jobs, cap):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        first = {}
        lengths = {}
        after_all_first = None
        tokens_when_all_first = None
        completed = {}
        for job in jobs:
            engine.add_request(job['id'], job['input_ids'], SamplingParams(
                max_tokens=cap, temperature=1., top_p=.7, seed=job['seed'],
                min_loops=2, max_loops=2, ignore_eos=True,
            ))
        while engine.has_unfinished_requests():
            outputs = engine.step()
            step_time = time.perf_counter()
            for row in outputs:
                request_id = row.request_id
                if row.token_ids:
                    first.setdefault(request_id, step_time)
                    lengths[request_id] = len(row.token_ids)
                if row.finished:
                    if len(row.token_ids) != cap or row.exit_depths != [2] * cap:
                        raise RuntimeError(f'Unexpected length or depth for {request_id}')
                    completed[request_id] = row.token_ids
            if len(first) == len(jobs) and after_all_first is None:
                torch.cuda.synchronize()
                after_all_first = time.perf_counter()
                tokens_when_all_first = sum(lengths.values())
        torch.cuda.synchronize()
        end = time.perf_counter()
        if len(completed) != len(jobs):
            raise RuntimeError('Batch returned an incomplete workload')
        total = sum(map(len, completed.values()))
        elapsed = end - begin
        pure_decode_rate = None
        if after_all_first is not None and tokens_when_all_first < total:
            pure_decode_rate = (total - tokens_when_all_first) / (end - after_all_first)
        return dict(output_tokens=total, seconds=elapsed, output_tok_s=total / elapsed,
                    first_token_mean_s=statistics.mean(t - begin for t in first.values()),
                    all_first_tokens_s=after_all_first - begin if after_all_first is not None else None,
                    decode_after_all_first_tok_s=pure_decode_rate, token_ids=completed)

    try:
        for key, phase in spec['phases'].items():
            if phases is not None and key not in phases:
                continue
            jobs = phase['jobs']
            warm = jobs if key == 'short' else [{**job, 'input_ids': job['input_ids'][:2048]}
                                                for job in jobs]
            run(warm, 16)
            phase_results[key] = [run(jobs, phase['generated_tokens']) for _ in range(repetitions)]
            for measured in phase_results[key]:
                print(key, {k: v for k, v in measured.items() if k != 'token_ids'}, flush=True)
            write(OUT / f'{name}.json', dict(settings=settings, hardware=hardware,
                                              phases=phase_results))
    finally:
        engine.close()


def main():
    write(OUT / 'workload.json', workload())
    gpus = dict(zip(CASES, ('0', '1', '2', '3'), strict=True))
    with ExitStack() as stack:
        for gpu in gpus.values():
            lock = stack.enter_context(open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_idle_gpus(list(gpus.values()))

        def launch(name, gpu):
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
            env.pop('PYTHONPATH', None)
            with (OUT / f'{name}.log').open('w') as log:
                result = subprocess.run([str(RELEASE_PYTHON), '-u', __file__, '--worker', name],
                                        env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            print(name, 'exit', result.returncode, flush=True)
            return result.returncode

        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = {name: executor.submit(launch, name, gpu) for name, gpu in gpus.items()}
            exits = {name: future.result() for name, future in futures.items()}

    summary = dict(gpus=gpus, exits=exits, cases={})
    for name, code in exits.items():
        if code != 0:
            continue
        data = read(OUT / f'{name}.json')
        fields = ('output_tok_s', 'all_first_tokens_s', 'decode_after_all_first_tok_s')
        stats = {}
        for phase, trials in data.pop('phases').items():
            stats[phase] = {field: statistics.median(trial[field] for trial in trials
                                                    if trial[field] is not None)
                            for field in fields if any(trial[field] is not None for trial in trials)}
        summary['cases'][name] = dict(**data, metrics=stats)
    write(OUT / 'comparison.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=list(CASES))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        main()
