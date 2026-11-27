"""Plan, run or summarize the zero-training BBH frontier in Ragged KV Figure 1(c)."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import fcntl
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from statistics import mean, quantiles
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ep_loop.config import project_relative_path
from ep_loop.evaluation_data import ouro_evaluation_group, rad_pass_metrics
from ep_loop.fastpath.actor import VllmActor
from ep_loop.fastpath.rlt_actor import RltActor, verified_release
from ep_loop.reproducibility import SAMPLE_SEED_SCHEME, sample_seed
from scripts.prepare_ragged_bbh import (
    DATASET, DATASET_REVISION, DEFAULT_FRONTIER_DIR, PER_TASK, PROMPT_REVISION, SEED, TASKS,
)


MODELS = {
    'Ouro-1.4B': ('574fa66cb8bf5abdc979642d01cf2b79b16bfab1',
                  '58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'),
    'Ouro-2.6B': ('1ed04250da1a9936042725d302e81c8fa2ab5abd',
                  '2fdf9805a3510b7dc82959a2db4872ef767a5bdc55520c6a1b54fc3e463c1bba'),
}
MODEL_FILES = ('model.safetensors', 'config.json', 'configuration_ouro.py', 'modeling_ouro.py',
               'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
THRESHOLDS = (.10, .15, .20, .25, .30)
CORE = ('fixed_2', 'fixed_4', *(f'ragged_{tau:.2f}' for tau in THRESHOLDS))
CONTRASTS = (('ragged_0.20', 'ragged_0.30'), ('ragged_0.30', 'fixed_2'), ('ragged_0.30', 'fixed_4'))
LEGACY_CONTRASTS = (('ragged_0.20', 'ragged_0.28'), ('ragged_0.28', 'fixed_2'), ('ragged_0.28', 'fixed_4'))
ROLLOUTS = 4
SAMPLING = dict(max_new_tokens=1024, temperature=1.0, top_p=.7, top_k=0)
GENERATION_SEED = 42
REQUESTS_PER_BATCH = 8
VLLM_PYTHON = ROOT / 'ep_loop_envs/vllm026/bin/python'
VLLM_VERSION = '0.26.0+cu129'
CUDA_TOOLKIT = '/usr/local/cuda-12.8'
VLLM_OPTIONS = dict(policy='fixed', max_model_len=4096, memory_fraction=0.90,
                    attention_backend='FLASHINFER', prefix_caching=False)
RLT_OPTIONS = dict(max_model_len=4096, max_num_batched_tokens=4096,
                   prefill_chunk_size=128, attention_backend='triton', kv_block_size=64,
                   eager=False, async_scheduling=False, prefix_caching=False)
ANSWER = re.compile(r'\bthe answer is[ \t]+([^.\r\n]+)', re.IGNORECASE)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_bbh(path):
    data = path.read_bytes()
    manifest = json.loads(path.with_name('manifest.json').read_text())
    checksum = hashlib.sha256(data).hexdigest()
    if (manifest['output_sha256'] != checksum or manifest['dataset'] != DATASET
            or manifest['dataset_revision'] != DATASET_REVISION
            or manifest['prompt_revision'] != PROMPT_REVISION or manifest['sample_seed'] != SEED):
        raise ValueError(f'BBH data do not match the pinned source manifest: {path}')
    records = [json.loads(line) for line in data.splitlines()]
    if (Counter(row['task'] for row in records) != dict.fromkeys(TASKS, PER_TASK)
            or len({row['id'] for row in records}) != len(records)):
        raise ValueError('Expected five distinct questions per BBH task')
    if any(hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256'] for row in records):
        raise ValueError('A BBH prompt hash does not match its frozen record')
    return records, checksum


def cases():
    fixed = [dict(name=f'fixed_{depth}', endpoint='fixed', backend='vllm',
                  experiment_group=ouro_evaluation_group('fixed', depth),
                  loops=depth, min_loops=depth, threshold=1.0)
             for depth in range(1, 5)]
    ragged = [dict(name=f'ragged_{tau:.2f}', endpoint='ragged', backend='vllm-rlt',
                   experiment_group=ouro_evaluation_group('ragged', 4),
                   loops=4, min_loops=1, threshold=tau)
              for tau in THRESHOLDS]
    return fixed + ragged


def make_plan(args):
    records, data_sha256 = load_bbh(args.data)
    if not 1 <= args.shards <= len(records):
        raise ValueError('Shards must be between one and the number of questions')
    model_dir = args.model_dir or ROOT / 'models' / args.model
    revision, weight_sha256 = MODELS[args.model]
    files = {name: sha256_file(model_dir / name) for name in MODEL_FILES}
    if files['model.safetensors'] != weight_sha256:
        raise ValueError(f'{model_dir} weights differ from the pinned {args.model} revision')
    for name in MODEL_FILES[1:]:
        metadata = model_dir / '.cache/huggingface/download' / f'{name}.metadata'
        if metadata.read_text().splitlines()[0] != revision:
            raise ValueError(f'Unexpected model revision for {model_dir / name}')
    matrix = cases()
    stages = {'core': list(CORE), 'full': [case['name'] for case in matrix]}
    vllm_version = subprocess.check_output(
        [str(VLLM_PYTHON), '-I', '-c', 'from importlib.metadata import version; print(version("vllm"))'],
        text=True).strip()
    if vllm_version != VLLM_VERSION:
        raise ValueError(f'Expected official vLLM {VLLM_VERSION} in {VLLM_PYTHON}, found {vllm_version}')
    release = verified_release()
    for key in ('source', 'python_bin'):
        release[key] = project_relative_path(release[key], ROOT)
    engines = {
        'vllm': dict(version=vllm_version, python_bin=project_relative_path(VLLM_PYTHON, ROOT),
                     cuda_toolkit=CUDA_TOOLKIT, actor_options=VLLM_OPTIONS),
        'vllm-rlt': dict(release=release, actor_options=RLT_OPTIONS, kv_layout='last_exited', rlt_top_k=-1),
    }
    plan = dict(paper='2610.09013v1 Figure 1(c), Appendix E.1', model=args.model,
                model_path=project_relative_path(model_dir, ROOT), model_revision=revision, model_sha256=files,
                dataset=project_relative_path(args.data, ROOT), dataset_sha256=data_sha256, questions=len(records),
                dataset_sample_seed=SEED, questions_per_task=PER_TASK,
                shards=args.shards, samples_per_question=ROLLOUTS, generation_seed=GENERATION_SEED,
                sample_seed_scheme=SAMPLE_SEED_SCHEME, sampling=SAMPLING, cases=matrix, stages=stages,
                dtype='bfloat16', prompt_format='raw 3-shot CoT; no added special tokens',
                engines=engines, requests_per_batch=REQUESTS_PER_BATCH, contrasts=CONTRASTS,
                scorer='Last explicit the answer is … up to period or newline, then exact case-sensitive match',
                completions={name: len(group) * len(records) * ROLLOUTS for name, group in stages.items()})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(plan=str(args.output / 'plan.json'), completions=plan['completions'],
                          questions=len(records), shards=args.shards), indent=2))


def score_answer(text, target):
    matches = ANSWER.findall(text)
    parsed = (matches[-1].strip() or None) if matches else None
    return parsed, parsed == target


def completion_keys(questions, cases_for_stage, rollouts):
    return {(case['name'], row['id'], sample) for case in cases_for_stage
            for row in questions for sample in range(rollouts)}


def initialize_shard(db, plan_hash):
    db.execute('CREATE TABLE IF NOT EXISTS manifest (plan_sha256 TEXT PRIMARY KEY)')
    observed = db.execute('SELECT plan_sha256 FROM manifest').fetchall()
    if observed and observed != [(plan_hash,)]:
        raise ValueError('This shard belongs to a different frozen plan')
    db.execute('INSERT OR IGNORE INTO manifest VALUES (?)', (plan_hash,))
    db.execute('CREATE TABLE IF NOT EXISTS responses (arm TEXT, example TEXT, sample INTEGER, payload TEXT, '
               'PRIMARY KEY (arm, example, sample))')
    db.execute('CREATE TABLE IF NOT EXISTS attempts (runtime TEXT)')
    db.commit()


def plan_asset_path(recorded, project_path):
    """Use the project asset for a missing recorded path, then verify its hash."""
    path = ROOT / recorded
    return path if path.exists() else project_path


def generate_shard(args, plan, records, plan_hash, ordered_cases):
    if not 0 <= args.shard < plan['shards']:
        raise ValueError('Shard index is outside the frozen plan')
    path = args.plan.parent / f'shard_{args.shard:02d}.sqlite'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with closing(sqlite3.connect(path)) as db:
            initialize_shard(db, plan_hash)
            assigned = records[args.shard::plan['shards']]
            expected = completion_keys(assigned, ordered_cases, plan['samples_per_question'])
            done = set(db.execute('SELECT arm, example, sample FROM responses')) & expected
            if done == expected:
                print(f'Shard {args.shard} already complete for {args.stage}')
                return
            with open(f'/tmp/ep-loop-gpu{args.gpu}.lock', 'a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run_pending(args, plan, assigned, ordered_cases, done, len(expected), db)


def run_pending(args, plan, records, ordered_cases, done, total, db):
    from ep_loop.gpu_resources import require_idle_gpus
    from transformers import AutoTokenizer

    model_dir = plan_asset_path(plan['model_path'], ROOT / 'models' / plan['model'])
    if any(sha256_file(model_dir / name) != digest for name, digest in plan['model_sha256'].items()):
        raise ValueError('A model file changed after planning')
    engines = plan['engines'] if 'engines' in plan else {'vllm-rlt': plan['engine']}
    release = verified_release()
    if any(release[key] != engines['vllm-rlt']['release'][key] for key in ('version', 'commit')):
        raise ValueError('The installed RLT release changed after planning')
    if 'vllm' in engines:
        toolkit = Path(engines['vllm']['cuda_toolkit'])
        # Prefer the installed H100 toolkit for FlashInfer JIT.
        if (toolkit / 'bin/nvcc').is_file():
            os.environ.update(CUDA_HOME=str(toolkit), CUDA_PATH=str(toolkit),
                              PATH=os.pathsep.join((str(toolkit / 'bin'), os.environ.get('PATH', os.defpath))))
        os.environ.update(HF_HUB_OFFLINE='1', VLLM_NO_USAGE_STATS='1')
    gpu, = require_idle_gpus([args.gpu])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    max_model_len = min(engine['actor_options']['max_model_len'] for engine in engines.values())
    max_new_tokens = plan['sampling']['max_new_tokens']
    encoded = {}
    for example in records:
        ids = tokenizer.encode(example['prompt'], add_special_tokens=False)
        if not ids or len(ids) + max_new_tokens > max_model_len:
            raise ValueError(f"Frozen prompt exceeds Ouro context: {example['id']}")
        encoded[example['id']] = ids

    completed = len(done)
    groups = groupby(ordered_cases, key=lambda case: (case.get('backend', 'vllm-rlt'), case['loops']))
    for (backend, loops), group in groups:
        pending = [(case, list(pending_jobs(case, records, plan, done, encoded))) for case in group]
        pending = [(case, jobs) for case, jobs in pending if jobs]
        if not pending:
            continue
        engine = engines[backend]
        log_prefix = 'vllm' if backend == 'vllm' else 'rlt'
        log = args.plan.parent / f'shard_{args.shard:02d}_{log_prefix}_prefill{loops}.log'
        options = engine['actor_options']
        if backend == 'vllm':
            actor = VllmActor(model_dir, gpu['uuid'], log, loops=loops,
                              python_bin=ROOT / engine['python_bin'], **options)
        else:
            actor = RltActor.from_release(model_dir, gpu['uuid'], log, loops=loops, **options)
        with actor:
            if backend == 'vllm':
                if (actor.runtime.get('vllm_version') != engine['version']
                        or not actor.runtime.get('native_fixed_student')):
                    raise RuntimeError('Expected the pinned official vLLM fixed-depth engine')
            elif actor.runtime['kv_layout'] != engine['kv_layout']:
                raise RuntimeError('The running RLT KV layout differs from the frozen plan')
            batch_size = plan.get('requests_per_batch') or actor.runtime['max_num_seqs']
            runtime = dict(actor.runtime, backend=backend, gpu_index=args.gpu, gpu_uuid=gpu['uuid'],
                           stage=args.stage, requests_per_batch=batch_size)
            with db:
                db.execute('INSERT INTO attempts VALUES (?)', (json.dumps(runtime),))
            for case, jobs in pending:
                for result in generate_case(actor, tokenizer, case, jobs, batch_size):
                    key = (result['arm'], result['id'], result['sample'])
                    with db:
                        db.execute('INSERT INTO responses VALUES (?, ?, ?, ?)', (*key, json.dumps(result)))
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f'Shard {args.shard} {args.stage}: {completed}/{total}', flush=True)


def pending_jobs(case, records, plan, done, encoded):
    sampling = plan['sampling']
    for example in records:
        for sample in range(plan['samples_per_question']):
            if (case['name'], example['id'], sample) in done:
                continue
            seed = sample_seed(plan['generation_seed'], example['prompt_sha256'], sample)
            request = dict(input_ids=encoded[example['id']], seed=seed, min_loops=case['min_loops'],
                           exit_threshold=case['threshold'], max_new_tokens=sampling['max_new_tokens'],
                           temperature=sampling['temperature'], top_p=sampling['top_p'])
            yield example, sample, request


def generate_case(actor, tokenizer, case, pending, batch_size):
    experiment_group = ouro_evaluation_group(case['endpoint'], case['loops'])
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        for index, response, seconds in actor.iter_responses([request for _, _, request in batch]):
            example, sample, request = batch[index]
            tokens, depths = response['token_ids'], response['exit_depths']
            if (not 0 < len(tokens) <= request['max_new_tokens'] or len(depths) != len(tokens)
                    or depths[0] != case['loops'] or response['finish_reason'] not in ('stop', 'length')
                    or (response['finish_reason'] == 'length' and len(tokens) != request['max_new_tokens'])):
                raise RuntimeError(f"Invalid completion for {case['name']}/{example['id']}/{sample}")
            text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            parsed, correct = score_answer(text, example['target'])
            yield dict(arm=case['name'], id=example['id'], task=example['task'], sample=sample,
                       experiment_group=experiment_group,
                       prompt_sha256=example['prompt_sha256'], prompt_tokens=len(request['input_ids']),
                       seed=request['seed'], target=example['target'], parsed_answer=parsed, correct=correct,
                       text=text, tokens=tokens, exit_depths=depths, reached_eos=response['finish_reason'] == 'stop',
                       seconds=seconds, executed_token_loops=response['executed_token_loops'])


def collect_shards(directory, plan, records, plan_hash, ordered_cases):
    want = {case['name'] for case in ordered_cases}
    for index in range(plan['shards']):
        path = (directory / f'shard_{index:02d}.sqlite').resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Missing shard: {path}')
        with closing(sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)) as db:
            if db.execute('SELECT plan_sha256 FROM manifest').fetchall() != [(plan_hash,)]:
                raise ValueError(f'Shard does not belong to this plan: {path}')
            rows = {(arm, ident, sample): json.loads(payload)
                    for arm, ident, sample, payload in db.execute('SELECT * FROM responses') if arm in want}
        expected = completion_keys(records[index::plan['shards']], ordered_cases, plan['samples_per_question'])
        if rows.keys() != expected:
            raise ValueError(f'Shard {index} has {len(expected - rows.keys())} missing and '
                             f'{len(rows.keys() - expected)} unexpected completions')
        yield from rows.values()


def summarize_case(rows):
    correctness = defaultdict(list)
    depths = Counter()
    for row in rows:
        correctness[row['id']].append(row['correct'])
        depths.update(row['exit_depths'][1:])
    total_decode = sum(depths.values())
    result = dict(**rad_pass_metrics(correctness), questions=len(correctness), completions=len(rows),
                  decode_depth_hist=dict(sorted(depths.items())),
                  mean_decode_depth=sum(depth * count for depth, count in depths.items()) / total_decode if total_decode else None,
                  depth_one_fraction=depths[1] / total_decode if total_decode else None,
                  mean_response_tokens=mean(len(row['tokens']) for row in rows),
                  truncation_rate=mean(not row['reached_eos'] for row in rows),
                  no_answer_rate=mean(row['parsed_answer'] is None for row in rows),
                  executed_token_loops=sum(row['executed_token_loops'] for row in rows),
                  generation_seconds_sum=sum(row['seconds'] for row in rows))
    per_question = {ident: rad_pass_metrics({ident: grades}) for ident, grades in correctness.items()}
    return result, per_question


def paired_effect(left, right, records):
    # Resample matched questions within each of the 27 equally-sized tasks.
    by_task = defaultdict(list)
    for row in records:
        ident = row['id']
        p1 = left[ident]['p_at_1'] - right[ident]['p_at_1']
        p4 = left[ident]['p_at_4'] - right[ident]['p_at_4']
        by_task[row['task']].append((p1, p4, p4 - p1))
    rng = random.Random(2026)
    bootstrap = [[], [], []]
    for _ in range(5000):
        draws = [draw for task in by_task.values() for draw in rng.choices(task, k=len(task))]
        for values, samples in zip(zip(*draws), bootstrap):
            samples.append(mean(values))
    result = {}
    for index, metric in enumerate(('p_at_1', 'p_at_4', 'p_at_4_minus_p_at_1')):
        observed = mean(value[index] for task in by_task.values() for value in task)
        bounds = quantiles(bootstrap[index], n=40, method='inclusive')
        result[metric] = dict(delta=observed, ci95=[bounds[0], bounds[-1]])
    return result


def summarize(args, plan, records, plan_hash, ordered_cases):
    rows_by_arm = defaultdict(list)
    for row in collect_shards(args.plan.parent, plan, records, plan_hash, ordered_cases):
        rows_by_arm[row['arm']].append(row)
    report, per_arm = {}, {}
    for case in ordered_cases:
        name = case['name']
        metrics, per_arm[name] = summarize_case(rows_by_arm[name])
        report[name] = {**case, **metrics, 'experiment_group': ouro_evaluation_group(case['endpoint'], case['loops'])}
    contrasts = {}
    for left, right in plan.get('contrasts', LEGACY_CONTRASTS):
        contrasts[f'{left}_minus_{right}'] = paired_effect(per_arm[left], per_arm[right], records)
    output = args.plan.parent / f'summary_{args.stage}.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(plan_sha256=plan_hash, cases=report, contrasts=contrasts,
                                         bootstrap_samples=5000, bootstrap_seed=2026), indent=2) + '\n')
    temporary.replace(output)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('plan', help='Freeze and validate inputs without loading a model or using CUDA')
    plan.add_argument('--data', type=Path, default=DEFAULT_FRONTIER_DIR / 'eval.jsonl')
    plan.add_argument('--model', choices=MODELS, default='Ouro-1.4B')
    plan.add_argument('--model-dir', type=Path)
    plan.add_argument('--output', type=Path, required=True)
    plan.add_argument('--shards', type=int, default=4)
    run = sub.add_parser('run', help='Generate the selected arms on one GPU and checkpoint every completion')
    run.add_argument('--gpu', type=int, required=True)
    run.add_argument('--shard', type=int, required=True)
    for command in (run, sub.add_parser('summarize', help='Validate coverage and compute metrics on CPU')):
        command.add_argument('--plan', type=Path, required=True)
        command.add_argument('--stage', choices=('core', 'full'), default='core')
    args = parser.parse_args()
    if args.action == 'plan':
        make_plan(args)
        return
    plan_bytes = args.plan.read_bytes()
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    records, data_hash = load_bbh(plan_asset_path(plan['dataset'], DEFAULT_FRONTIER_DIR / 'eval.jsonl'))
    if data_hash != plan['dataset_sha256']:
        raise ValueError('BBH data changed after planning')
    ordered_cases = [case for case in plan['cases'] if case['name'] in plan['stages'][args.stage]]
    action = generate_shard if args.action == 'run' else summarize
    action(args, plan, records, plan_hash, ordered_cases)


if __name__ == '__main__':
    main()
