"""Compare two fixed-depth Ouro backends on the same AIME2024 questions."""
import fcntl
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ep_loop.cpu_runtime import configure_cpu_threads
from ep_loop.data import prompt_ids
from ep_loop.evaluation_data import load_evaluation_records
from ep_loop.fastpath.actor import VllmActor
from ep_loop.fastpath.rlt_actor import RltActor
from ep_loop.gpu_resources import require_idle_gpus
from ep_loop.inference import verify_integer_response
from ep_loop.reproducibility import sample_seed
from transformers import AutoTokenizer

MODEL = (ROOT / 'models/Ouro-1.4B-Thinking').resolve()
VLLM_PYTHON = ROOT / 'ep_loop_envs/vllm026/bin/python'
DEPTH = 2
LIMIT = 16384
CONTEXT = LIMIT + 1024
BASE_SEED = 42
GPUS = {'vllm': '0', 'rlt': '1'}


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def make_job(record, tokenizer):
    return dict(
        id=record['id'], problem=record['problem'], messages=record['messages'],
        answer=record['answer'], problem_sha256=record['problem_sha256'],
        input_ids=prompt_ids(tokenizer, record, True, True),
        seed=sample_seed(BASE_SEED, record['problem_sha256'], 0),
    )


def run_backend(name, gpu, jobs, warm_jobs, tokenizer):
    log = OUT / f'{name}.engine.log'
    start = time.perf_counter()
    if name == 'vllm':
        actor = VllmActor(MODEL, gpu, log, loops=DEPTH, policy='fixed',
                          max_model_len=CONTEXT, memory_fraction=.75,
                          attention_backend='FLASHINFER', python_bin=VLLM_PYTHON)
    else:
        actor = RltActor(MODEL, gpu, log, loops=DEPTH, max_model_len=CONTEXT,
                         attention_backend='triton', prefix_caching=False)

    def run(phase, data, cap):
        requests = [dict(input_ids=job['input_ids'], seed=job['seed'],
                         temperature=1., top_p=.7, max_new_tokens=cap)
                    for job in data]
        rows = {}
        elapsed_by_index = {}
        engine_metrics = {}
        print(f'{name} {phase} begin', flush=True)
        begin = time.perf_counter()
        if name == 'vllm':
            result = actor.generate(requests, logprobs=None)
            rows.update(enumerate(result['responses']))
            seconds = result['seconds']
            engine_metrics = result['engine_metrics']
        else:
            for index, row, elapsed in actor.iter_responses(requests):
                rows[index] = row
                elapsed_by_index[index] = elapsed
                if phase == 'full':
                    print(f'{name} full {len(rows)}/{len(data)} {data[index]["id"]} '
                          f'{len(row["token_ids"])} tokens at {elapsed:.1f}s', flush=True)
            seconds = json.loads(log.read_text().splitlines()[-1])['seconds']
        end_to_end_seconds = time.perf_counter() - begin
        if set(rows) != set(range(len(data))):
            raise RuntimeError(f'{name} {phase} did not return every request')
        answers = []
        for index, job in enumerate(data):
            row = rows[index]
            ids = row['token_ids']
            if not 0 < len(ids) <= cap or row['exit_depths'] != [DEPTH] * len(ids):
                raise RuntimeError(f'{name} {phase} returned an invalid length or depth')
            if row['finish_reason'] not in {'length', 'stop'}:
                raise RuntimeError(f'{name} {phase} has unexpected finish reason {row["finish_reason"]}')
            if row['finish_reason'] == 'length' and len(ids) != cap:
                raise RuntimeError(f'{name} {phase} reported length before reaching the limit')
            answer = dict(id=job['id'], sampling_seed=job['seed'],
                          prompt_tokens=len(job['input_ids']), response_tokens=len(ids),
                          finish_reason=row['finish_reason'], token_ids=ids,
                          request_elapsed_seconds=elapsed_by_index.get(index),
                          request_timing=row.get('request_timing'))
            if phase == 'full':
                answer['text'] = tokenizer.decode(ids)
                answer.update(verify_integer_response(answer['text'], job['answer'], job['messages']))
            answers.append(answer)
        tokens = sum(answer['response_tokens'] for answer in answers)
        report = dict(phase=phase, seconds=seconds, client_seconds=end_to_end_seconds,
                      output_tokens=tokens, tokens_per_second=tokens / seconds,
                      stopped_at_limit=sum(answer['finish_reason'] == 'length' for answer in answers),
                      correct=sum(answer.get('correct', False) for answer in answers),
                      answers=answers, engine_metrics=engine_metrics)
        print(f'{name} {phase} complete {tokens} tokens in {seconds:.2f}s ({tokens / seconds:.2f} tok/s)',
              flush=True)
        return report

    with actor:
        startup = time.perf_counter() - start
        print(f'{name} ready GPU {gpu} in {startup:.1f}s {actor.runtime}', flush=True)
        report = dict(backend=name, gpu=gpu, startup_seconds=startup, runtime=actor.runtime)
        report['warmup'] = run('warmup', warm_jobs, 16)
        report['short'] = run('short', jobs, 1024)
        write_json(OUT / f'{name}.short.json', report)
        report['full'] = run('full', jobs, LIMIT)
        write_json(OUT / f'{name}.json', report)
    return report


def main():
    configure_cpu_threads()
    records, source = load_evaluation_records(ROOT / 'data/aime/aime2024.jsonl', 30, 'aime')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    selection = sorted(random.Random(BASE_SEED).sample(range(len(records)), 8))
    jobs = [make_job(records[index], tokenizer) for index in selection]
    warm_jobs = [make_job(record, tokenizer) for i, record in enumerate(records) if i not in selection][:8]
    if any(len(job['input_ids']) > 1024 for job in jobs + warm_jobs):
        raise RuntimeError('Prompt exceeds 1024 tokens')
    write_json(OUT / 'workload.json', dict(
        model=str(MODEL), source=source, selection_indices=selection, jobs=jobs, warm_jobs=warm_jobs,
        max_new_tokens=LIMIT, short_max_new_tokens=1024, max_model_len=CONTEXT, loops=DEPTH,
        temperature=1., top_p=.7, seed=BASE_SEED, replicas_per_backend=1, gpus=GPUS,
        gpu_memory_utilization=.75, prefix_caching=False, native_eos=True,
        rlt_revision=subprocess.check_output(['git', '-C', ROOT / 'external/vllm-rlt', 'rev-parse', 'HEAD'],
                                             text=True).strip(),
    ))
    with ExitStack() as stack:
        for gpu in GPUS.values():
            lock = stack.enter_context(open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_idle_gpus(list(GPUS.values()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(run_backend, name, gpu, jobs, warm_jobs, tokenizer)
                       for name, gpu in GPUS.items()}
            results = {name: task.result() for name, task in futures.items()}
    summary = {}
    for phase in ('short', 'full'):
        v, r = (results[name][phase] for name in ('vllm', 'rlt'))
        counts_and_timings = ('seconds', 'client_seconds', 'output_tokens', 'tokens_per_second',
                              'stopped_at_limit', 'correct')
        summary[phase] = dict(
            vllm={field: v[field] for field in counts_and_timings},
            rlt={field: r[field] for field in counts_and_timings},
            throughput_ratio_vllm_over_rlt=v['tokens_per_second'] / r['tokens_per_second'],
            batch_speedup_vllm_over_rlt=r['seconds'] / v['seconds'],
            identical_sampled_completions=sum(a['token_ids'] == b['token_ids']
                                               for a, b in zip(v['answers'], r['answers'], strict=True)),
            questions=[dict(id=a['id'], vllm_tokens=a['response_tokens'], rlt_tokens=b['response_tokens'],
                            vllm_finish=a['finish_reason'], rlt_finish=b['finish_reason'],
                            vllm_correct=a.get('correct'), rlt_correct=b.get('correct'))
                       for a, b in zip(v['answers'], r['answers'], strict=True)],
        )
        if [a['id'] for a in v['answers']] != [a['id'] for a in r['answers']]:
            raise RuntimeError('Unmatched question IDs')
    summary['startup_seconds'] = {name: report['startup_seconds'] for name, report in results.items()}
    write_json(OUT / 'comparison.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
