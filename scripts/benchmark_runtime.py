"""Run matched Ouro inference workloads against the original checkpoint."""
import argparse
from contextlib import contextmanager, nullcontext
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ep_loop.evaluation_data import ouro_evaluation_group

AIME = ROOT / 'audit/rlt_v020_bf16_vs_vllm_20261120_152301/workload.json'
PREFIXES = ROOT / 'audit/rlt_v020_runtime_probe_20261120_151153/workload.json'
REPEATED = ROOT / 'audit/vllm_native_n16_20261120/workload.json'
CONTEXT = 17408
FIXED_CAPS = {'short': 1024, '8192': 128, '14336': 128, 'repeated': 256}
CASES = {
    'hf': ('fixed', 1.),
    'vllm': ('fixed', 1.),
    'vllm-native': ('native', None),
    'vllm-triton': ('fixed', 1.),
    'vllm-prefix-cache': ('fixed', 1.),
    'rlt': ('fixed', 1.),
    'rlt-block16': ('fixed', 1.),
    'rlt-block128': ('fixed', 1.),
    'rlt-prefix-cache': ('fixed', 1.),
    'ragged-dense': ('ragged', 1.),
    'ragged': ('ragged', None),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=(*FIXED_CAPS, 'natural'))
    parser.add_argument('--case', required=True, nargs='+', choices=CASES)
    parser.add_argument('--loops', type=int, required=True, choices=(2, 4))
    parser.add_argument('--tau', type=float, default=.20)
    parser.add_argument('--gpu', type=int, help='One idle GPU index. Omit for CPU.')
    parser.add_argument('--concurrency', type=int, choices=(1, 8, 64), default=1)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--smoke', action='store_true', help='Explicitly use the smaller validation workload')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 1 if args.profile in ('natural', 'repeated') else 3
    if args.repeats < 1 or not 0 < args.tau < 1 or len(args.case) != len(set(args.case)):
        parser.error('Require positive repetitions, 0 < tau < 1 and unique cases')
    hf_cases = [case for case in args.case if not case.startswith(('vllm', 'rlt'))]
    if args.gpu is None and len(hf_cases) != len(args.case):
        parser.error('vLLM and RLT require a GPU')
    if args.concurrency > 1 and (hf_cases or args.smoke):
        parser.error('Concurrency above 1 requires a full workload and batched engines')
    if args.profile == 'repeated' and (args.concurrency != 64 or args.repeats != 1):
        parser.error('Use concurrency 64 and one timed cold-cache batch per engine startup')
    if args.profile != 'repeated' and (args.concurrency == 64 or any(case.endswith('-prefix-cache') for case in args.case)):
        parser.error('Concurrency 64 and the prefix cache variants belong to the repeated workload')
    if (any(case != 'hf' for case in hf_cases) or 'vllm-native' in args.case) and args.loops != 4:
        parser.error('Use four loops for the RAD cache and policy comparisons')
    if 'ragged' in args.case and args.profile != 'natural':
        parser.error('Dynamic exit belongs to the natural workload')
    return args


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def manifest(args):
    source = json.loads(AIME.read_text())
    jobs = source['jobs']
    if args.profile == 'repeated':
        repeated = json.loads(REPEATED.read_text())
        jobs = [dict(id=f"{job['id']}/{sample}", input_ids=job['input_ids'], seed=job['seed'] + sample)
                for job in repeated['prompts'] for sample in range(repeated['samples_per_problem'])]
        if len(jobs) != args.concurrency:
            raise ValueError('Invalid archived repeated workload')
    if args.profile in ('8192', '14336'):
        archived = json.loads(PREFIXES.read_text())['phases'][args.profile]['jobs']
        by_id = {job['id']: job for job in archived}
        for job in jobs:
            prefix = by_id[job['id']]
            if (prefix['seed'] != job['seed'] or len(prefix['input_ids']) != int(args.profile)
                    or prefix['input_ids'][:len(job['input_ids'])] != job['input_ids']):
                raise ValueError(f"Unexpected archived prefix for {job['id']}")
        jobs = [{**job, 'input_ids': by_id[job['id']]['input_ids']} for job in jobs]
    cap = FIXED_CAPS.get(args.profile, source['max_new_tokens'])
    if args.smoke:
        if args.profile in ('8192', '14336'):
            jobs, cap = jobs[:1], 16
        else:
            cap = 32
    model = (ROOT / 'models/Ouro-1.4B-Thinking').resolve(strict=True)
    if not all(0 < len(job['input_ids']) <= CONTEXT - cap for job in jobs):
        raise ValueError('The workload exceeds the shared context limit')
    return dict(profile=args.profile, smoke=args.smoke, max_new_tokens=cap, max_model_len=CONTEXT,
                jobs=jobs, warm_jobs=(jobs if args.profile.isdigit() else source['warm_jobs'])[:args.concurrency],
                warm_new_tokens=16,
                loops=args.loops, cases={case: dict(endpoint=CASES[case][0],
                    tau=args.tau if CASES[case][1] is None else CASES[case][1],
                    prefix_caching=case.endswith('-prefix-cache')) for case in args.case},
                device='cpu' if args.gpu is None else 'cuda', gpu_index=args.gpu,
                concurrency=args.concurrency, repeats=args.repeats, model=str(model),
                weights_dtype='bfloat16', kv_dtype='bfloat16',
                speculation=False, ignore_eos=args.profile != 'natural',
                temperature=source['temperature'], top_p=source['top_p'],
                source_workload_sha256=digest(AIME),
                prefix_workload_sha256=digest(PREFIXES) if args.profile.isdigit() else None,
                repeated_workload_sha256=digest(REPEATED) if args.profile == 'repeated' else None)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class FirstTokenObserver:
    def __init__(self):
        self.started = time.perf_counter()
        self.first_token = None
        self.last_token = None

    @contextmanager
    def stage(self, name, **fields):
        yield {}

    def update_generation_progress(self, response_tokens, **fields):
        self.last_token = time.perf_counter() - self.started
        if response_tokens == 1:
            self.first_token = self.last_token


def hf_request(backend, work, case, requests):
    import torch
    from ep_loop.inference import generate

    request, = requests
    seed = torch.Generator(device=backend.device).manual_seed(request['seed'])
    policy = work['cases'][case]
    observer = FirstTokenObserver()
    result = generate(backend, request['input_ids'], work['loops'], request['max_new_tokens'],
                      request['temperature'], request['top_p'], seed, endpoint=policy['endpoint'],
                      exit_threshold=policy['tau'], observer=observer, ignore_eos=request['ignore_eos'])
    return [dict(token_ids=result.token_ids, exit_depths=result.exit_depths,
                 executed_token_loops=result.executed_token_loops,
                 backfill_token_loops=result.backfill_token_loops,
                 physical_kv_token_slots=result.physical_kv_token_slots,
                 finish_reason='stop' if result.reached_eos else 'length',
                 request_timing=dict(time_to_first_token_seconds=observer.first_token,
                     first_to_last_token_seconds=observer.last_token - observer.first_token))], {}


def gpu_actor(work, case, output):
    common = dict(loops=work['loops'], max_model_len=CONTEXT, max_num_seqs=work['concurrency'],
                  prefix_caching=work['cases'][case]['prefix_caching'])
    model, gpu = work['model'], work['gpu']['uuid']
    if case.startswith('vllm'):
        from ep_loop.fastpath.actor import VllmActor
        policy = work['cases'][case]
        actor = VllmActor(model, gpu, output / f'{case}.log', policy=policy['endpoint'],
                          native_threshold=policy['tau'], memory_fraction=.75,
                          python_bin=ROOT / 'ep_loop_envs/vllm026/bin/python',
                          attention_backend='TRITON_ATTN' if case == 'vllm-triton' else 'FLASHINFER',
                          **common)
        route = 'native_readout' if case == 'vllm-native' else 'native_fixed_student'
        if not (actor.runtime[route] and actor.runtime['vllm_version'].startswith('0.26.')):
            actor.close()
            raise RuntimeError(f'Expected the vLLM 0.26 Ouro {policy["endpoint"]} route')
        return actor
    from ep_loop.fastpath.rlt_actor import RltActor
    return RltActor.from_release(model, gpu, output / f'{case}.log', max_num_batched_tokens=CONTEXT,
                                 kv_block_size={'rlt-block16': 16, 'rlt-block128': 128}.get(case, 64),
                                 **common)


def engine_request(actor, case, requests):
    if case.startswith('vllm'):
        result = actor.generate(requests, logprobs=None)
        if result['engine_metrics'].get('vllm:num_preemptions', 0):
            raise RuntimeError('Engine preempted sequences within the benchmark batch')
        return result['responses'], dict(engine_seconds=result['seconds'], metrics=result['engine_metrics'])
    rows = [None] * len(requests)
    for index, response, elapsed in actor.iter_responses(requests):
        rows[index] = {**response, 'engine_completion_seconds': elapsed}
    return rows, {}


def batches(jobs, size):
    return [jobs[i:i + size] for i in range(0, len(jobs), size)]


def run_sample(work, jobs, cap, run):
    requests = [dict(input_ids=job['input_ids'], seed=job['seed'], max_new_tokens=cap,
                     temperature=work['temperature'], top_p=work['top_p'],
                     ignore_eos=work['ignore_eos']) for job in jobs]
    started = time.perf_counter()
    rows, runtime = run(requests)
    elapsed = time.perf_counter() - started
    for row, job in zip(rows, jobs, strict=True):
        length = len(row['token_ids'])
        if not 0 < length <= cap or len(row['exit_depths']) != length:
            raise ValueError(f"Invalid output length or exit trace for {job['id']}")
        if row['finish_reason'] not in ('length', 'stop'):
            raise ValueError(f"Unexpected finish reason for {job['id']}")
        if work['ignore_eos'] and row['finish_reason'] != 'length':
            raise ValueError(f"Unexpected stop in the fixed-work profile for {job['id']}")
        if row['finish_reason'] == 'length' and length != cap:
            raise ValueError(f"Output did not reach the requested cap for {job['id']}")
        row.update(id=job['id'], prompt_tokens=len(job['input_ids']), output_tokens=length, seed=job['seed'])
    return dict(client_seconds=elapsed, runtime=runtime, responses=rows)


def run_case(work, case, run, tokenizer, output):
    policy = work['cases'][case]
    run_sample(work, work['warm_jobs'], work['warm_new_tokens'], run)
    records = []
    for repeat in range(work['repeats']):
        measured = []
        jobs = work['jobs'] if repeat % 2 == 0 else list(reversed(work['jobs']))
        # Keep batch scheduler input order fixed in C8; reverse the request
        # order in C1, where each request runs independently.
        if work['concurrency'] > 1:
            jobs = work['jobs']
        for group in batches(jobs, work['concurrency']):
            measured.append(run_sample(work, group, work['max_new_tokens'], run))
        rows = [row for batch in measured for row in batch['responses']]
        for row, job in zip(rows, jobs, strict=True):
            depths = row['exit_depths']
            if depths[0] != work['loops'] or any(not 1 <= depth <= work['loops'] for depth in depths):
                raise ValueError('Invalid prefill or decode depth')
            if policy['tau'] == 1. and any(depth != work['loops'] for depth in depths):
                raise ValueError('Dense control exited early')
            if policy['endpoint'] == 'ragged' and (row['backfill_token_loops'] != 0
                    or row['physical_kv_token_slots'] != row['executed_token_loops']):
                raise ValueError('Ragged KV failed the physical slot invariant')
            if tokenizer is not None:
                from ep_loop.inference import verify_integer_response
                row['text'] = tokenizer.decode(row['token_ids'])
                row.update(verify_integer_response(row['text'], job['answer'], job['messages']))
        elapsed = sum(batch['client_seconds'] for batch in measured)
        tokens = sum(row['output_tokens'] for row in rows)
        decode_depths = [depth for row in rows for depth in row.get('execution_depths', row['exit_depths'])[1:]]
        decode_readouts = [depth for row in rows for depth in row['exit_depths'][1:]]
        ttft = [row['request_timing']['time_to_first_token_seconds'] for row in rows
                if row.get('request_timing', {}).get('time_to_first_token_seconds') is not None]
        tpot = [row['request_timing']['first_to_last_token_seconds'] / (row['output_tokens'] - 1)
                for row in rows if row['output_tokens'] > 1
                and row.get('request_timing', {}).get('first_to_last_token_seconds') is not None]
        slots = [row.get('physical_kv_token_slots') for row in rows]
        lengths = [row['output_tokens'] for row in rows]
        records.append(dict(batches=measured, client_seconds=elapsed, output_tokens=tokens,
            output_tok_s=tokens / elapsed, prompt_tokens=sum(row['prompt_tokens'] for row in rows),
            stopped_at_limit=sum(row['finish_reason'] == 'length' for row in rows),
            min_output_tokens=min(lengths), max_output_tokens=max(lengths),
            correct=sum(row['correct'] for row in rows) if tokenizer is not None else None,
            median_ttft_seconds=statistics.median(ttft) if len(ttft) == len(rows) else None,
            median_tpot_seconds=statistics.median(tpot) if tpot and len(tpot) == sum(n > 1 for n in lengths) else None,
            decode_depth_histogram=[decode_depths.count(depth) for depth in range(1, work['loops'] + 1)],
            mean_decode_depth=statistics.mean(decode_depths) if decode_depths else None,
            mean_decode_readout_depth=statistics.mean(decode_readouts) if decode_readouts else None,
            executed_token_loops=sum(row['executed_token_loops'] for row in rows),
            backfill_token_loops=sum(row.get('backfill_token_loops', 0) for row in rows),
            physical_kv_token_slots=None if None in slots else sum(slots)))
        write_json(output / f'{case}.json', records)
        print(f'{case} {repeat + 1}/{work["repeats"]}: {tokens} tokens / {elapsed:.2f}s', flush=True)
    summary = {key: statistics.median(record[key] for record in records)
               for key in ('client_seconds', 'output_tok_s', 'output_tokens')}
    summary['experiment_group'] = ouro_evaluation_group(policy['endpoint'], work['loops'])
    summary['inference_backend'] = ('vllm' if case.startswith('vllm') else
                                    'vllm-rlt' if case.startswith('rlt') else 'pytorch')
    for key in ('correct', 'stopped_at_limit', 'median_ttft_seconds', 'median_tpot_seconds', 'mean_decode_depth',
                'mean_decode_readout_depth', 'executed_token_loops', 'backfill_token_loops',
                'physical_kv_token_slots'):
        if records[0][key] is not None:
            summary[key] = statistics.median(record[key] for record in records)
    summary['client_seconds_each'] = [record['client_seconds'] for record in records]
    return summary


@contextmanager
def reserve_gpu(args, work):
    if args.gpu is None:
        yield
        return
    import fcntl
    from ep_loop.gpu_resources import require_idle_gpus

    with open(f'/tmp/ep-loop-gpu{args.gpu}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        work['gpu'], = require_idle_gpus([args.gpu])
        work['gpu']['hardware'] = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-gpu=name,pci.bus_id,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = work['gpu']['uuid']
        yield


def main():
    args = parse_args()
    work = manifest(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'manifest.json', work)
    if args.dry_run:
        print(f'Wrote {args.output / "manifest.json"}', flush=True)
        return
    with reserve_gpu(args, work):
        from ep_loop.cpu_runtime import configure_cpu_threads
        import torch

        work['cpu_threads'] = configure_cpu_threads()
        work['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        work['cpu_model'] = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                 if line.startswith('model name'))
        work['torch_version'] = torch.__version__
        work['model_sha256'] = {name: digest(Path(work['model']) / name) for name in (
            'model.safetensors', 'config.json', 'configuration_ouro.py', 'modeling_ouro.py',
            'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')}
        write_json(args.output / 'manifest.json', work)
        tokenizer = None
        if not work['ignore_eos']:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(work['model'], local_files_only=True)
        summary = {}
        backend = None
        for case in args.case:
            if case.startswith(('vllm', 'rlt')):
                if backend is not None:
                    del backend
                    backend = None
                    torch.cuda.empty_cache()
                actor_context = gpu_actor(work, case, args.output)
            else:
                if backend is None:
                    from ep_loop.ouro import load_ouro
                    backend, _ = load_ouro(work['model'], work['device'], 'bf16', gradient_checkpointing=False)
                    backend.model.to(dtype=torch.bfloat16).eval()
                actor_context = nullcontext()
            with actor_context as actor:
                run = (partial(engine_request, actor, case) if actor is not None
                       else partial(hf_request, backend, work, case))
                summary[case] = run_case(work, case, run, tokenizer, args.output)
                if actor is not None:
                    summary[case]['runtime'] = actor.runtime
                write_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
