"""Evaluate a trained policy, RAD-OPD Phase 0, or explicit fixed baselines."""
import argparse
from contextlib import ExitStack, closing
from itertools import chain, groupby
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHASE_ZERO_BACKENDS = {'fixed': 'vllm', 'ragged': 'vllm-rlt', 'native_backfill': 'pytorch'}


def evaluation_engines(modes, pending, args, config, directory, max_model_len, gpu, observer):
    from ep_loop.backends import load_backend
    from ep_loop.observability import model_progress

    def engine_key(mode):
        endpoint, loops, _ = mode
        name = PHASE_ZERO_BACKENDS[endpoint] if args.phase_zero else 'pytorch'
        return name, None if name == 'pytorch' else loops

    for (name, loops), grouped_modes in groupby(modes, key=engine_key):
        group = list(grouped_modes)
        if not any(pending[mode] for mode in group):
            for mode in group:
                yield mode, None, name
            continue
        log = args.output.with_suffix(f'.{name}.{loops}.engine.log')
        with observer.stage('model_load', echo=True, model_path=str(directory),
                            backend=name, device=args.device, loops=loops):
            if name == 'pytorch':
                with model_progress(observer.show_progress):
                    backend, _ = load_backend(config['model'], directory, args.device,
                        config['optimization']['mixed_precision'], False,
                        smoke=args.smoke and args.checkpoint is None)
                backend.model.eval()
            elif name == 'vllm':
                from ep_loop.fastpath.actor import VllmActor
                backend = VllmActor(directory, gpu, log, loops=loops, policy='fixed',
                    max_model_len=max_model_len, memory_fraction=.90,
                    python_bin=ROOT / 'ep_loop_envs/vllm026/bin/python', attention_backend='FLASHINFER')
            else:
                from ep_loop.fastpath.rlt_actor import RltActor
                backend = RltActor.from_release(directory, gpu, log, loops=loops,
                    max_model_len=max_model_len, max_num_batched_tokens=4096,
                    prefill_chunk_size=128, kv_block_size=64)
        if name == 'pytorch':
            if any(mode[0] == 'learned' for mode in group) and getattr(
                    backend.model.config, 'ep_loop_student_policy', None) != 'learned':
                raise ValueError('Checkpoint is not marked as a dynamic student; use its own config or --loops')
            for mode in group:
                yield mode, backend, name
        else:
            with backend:
                if name == 'vllm' and (backend.runtime.get('vllm_version') != '0.26.0+cu129'
                                       or not backend.runtime.get('native_fixed_student')):
                    raise RuntimeError('Phase 0 fixed loops require official vLLM 0.26.0+cu129')
                if name == 'vllm-rlt' and backend.runtime['kv_layout'] != 'last_exited':
                    raise RuntimeError('Phase 0 zero-backfill requires the LAST_EXITED KV layout')
                observer.event('inference_engine', echo=True, backend=name, runtime=backend.runtime)
                for mode in group:
                    yield mode, backend, name


def evaluation_responses(backend, engine, mode, jobs, settings, observer):
    import torch
    from ep_loop.inference import GeneratedResponse, generate
    from ep_loop.reproducibility import seed_everything

    endpoint, loops, threshold = mode
    if engine == 'pytorch':
        for record, sample, seed, ids in jobs:
            seed_everything(seed)
            rng = torch.Generator(device=backend.device).manual_seed(seed)
            response = generate(backend, ids, loops, settings['max_new_tokens'],
                settings['temperature'], settings['top_p'], rng, endpoint=endpoint,
                exit_threshold=threshold, use_cache=settings.get('use_cache', True),
                observer=observer, log_context=dict(id=record['id'], sample=sample), progress=True,
                progress_desc=f"Evaluate {record['id']} #{sample}", progress_position=1)
            yield record, sample, seed, response
        return

    batch_size = min(8, backend.runtime.get('max_num_seqs') or 8)
    limit = settings['max_new_tokens']
    cache_policy = 'rlt_last_exited' if engine == 'vllm-rlt' else 'vllm_fixed_depth'
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        requests = [dict(input_ids=ids, seed=seed, min_loops=1 if endpoint == 'ragged' else loops,
            exit_threshold=threshold, max_new_tokens=limit, temperature=settings['temperature'],
            top_p=settings['top_p']) for _, _, seed, ids in batch]
        with observer.stage('evaluation_batch', echo=True, backend=engine, policy=endpoint,
                            loops=loops, exit_threshold=threshold, start=start, requests=len(batch)):
            for index, output, seconds in backend.iter_responses(requests):
                record, sample, seed, _ = batch[index]
                tokens, depths = output['token_ids'], output['exit_depths']
                finish = output['finish_reason']
                if (not 0 < len(tokens) <= limit or len(tokens) != len(depths) or depths[0] != loops
                        or finish not in ('stop', 'length') or (finish == 'length' and len(tokens) != limit)):
                    raise RuntimeError(f"Invalid {engine} completion for {record['id']}/{sample}")
                yield record, sample, seed, GeneratedResponse(token_ids=tokens, exit_depths=depths,
                    reached_eos=finish == 'stop', executed_token_loops=output['executed_token_loops'],
                    seconds=seconds, execution_depths=depths, cache_policy=cache_policy)


def load_completed_responses(stream, modes, jobs, settings):
    from ep_loop.inference import GeneratedResponse

    completed = {mode: {} for mode in modes}
    expected = {(record['id'], sample): (record['problem_sha256'], seed)
                for record, sample, seed, _ in jobs}
    limit = settings['max_new_tokens']
    budgets = {str(cap) for cap in settings['budget_slices']}
    finished_at = 0
    while line := stream.readline():
        # The writer commits a response with a newline before flushing it.
        if not line.endswith(b'\n'):
            stream.seek(finished_at)
            stream.truncate()
            break
        row = json.loads(line)
        mode = (row['policy'], row['max_loops'], row['exit_threshold'])
        key = (row['id'], row['sample'])
        tokens = row.get('tokens', [])
        if (mode not in completed or key in completed[mode]
                or expected.get(key) != (row['problem_sha256'], row['sampling_seed'])
                or row['inference_backend'] != PHASE_ZERO_BACKENDS[mode[0]]
                or budgets != set(row['budget_correct'])
                or not 0 < len(tokens) == row['response_tokens'] <= limit
                or (not row['reached_eos'] and len(tokens) != limit)
                or len(row['actual_student_exit_depths']) != len(tokens)
                or len(row['actual_execution_depths']) != len(tokens)):
            raise ValueError(f'Incompatible or duplicate saved response for {mode}/{key}')
        completed[mode][key] = GeneratedResponse(token_ids=tokens,
            exit_depths=row['actual_student_exit_depths'], reached_eos=row['reached_eos'],
            executed_token_loops=row['executed_token_loops'], seconds=row['generation_seconds'],
            execution_depths=row['actual_execution_depths'], backfill_token_loops=row['backfill_token_loops'],
            coda_token_positions=row['coda_token_positions'], cache_policy=row['cache_policy'],
            physical_kv_token_slots=row['physical_kv_token_slots'])
        finished_at = stream.tell()
    return completed


def main():
    from ep_loop.config import DEFAULT_CONFIG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--checkpoint', type=Path, help='Required for trained students; optional for native base')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--loops', type=int, action='append', help='Explicit fixed-depth evaluation ablation')
    p.add_argument('--threshold', type=float, action='append', help='RAD-OPD threshold; repeat to scan')
    p.add_argument('--phase-zero', action='store_true', help='Compare base fixed loops (vLLM), backfill and ragged KV (vLLM-RLT)')
    p.add_argument('--skip-backfill', action='store_true', help='Run Phase 0 fixed loops and Ragged KV')
    p.add_argument('--resume', action='store_true', help='Resume the same Phase 0 output recorded with --write-token-ids')
    p.add_argument('--max-new-tokens', type=int, help='Use a smaller generation cap for a short check')
    p.add_argument('--write-token-ids', action='store_true', help='Include token IDs for offline teacher export')
    p.add_argument('--samples-per-problem', type=int)
    p.add_argument('--smoke',action='store_true',help='CPU-only saved/random tiny architecture with ByteTokenizer')
    args = p.parse_args()
    if (args.skip_backfill or args.resume) and not args.phase_zero:
        p.error('--skip-backfill and --resume require --phase-zero')
    if args.resume and not args.write_token_ids:
        p.error('--resume requires --write-token-ids')
    from ep_loop.config import load_config
    from ep_loop.execution import execution_mode, VLLM_EVAL_MODES
    if execution_mode(load_config(args.config)) in VLLM_EVAL_MODES:
        p.error('Use scripts/evaluate_fastpath.py for a vLLM execution mode')
    if args.resume and not args.output.is_file():
        raise FileNotFoundError(args.output)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    from ep_loop.observability import RunLogger
    with RunLogger(args.output.with_suffix('.events.jsonl'), args.output.with_suffix('.console.log'), 'evaluation') as observer:
        with observer.stage('evaluation_run', echo=True, config_path=args.config,
                            checkpoint=str(args.checkpoint) if args.checkpoint else None, data=str(args.data),
                            skip_backfill=args.skip_backfill, resume=args.resume):
            evaluate(args, p, observer)


def evaluate(args, p, observer):
    import torch
    from transformers import AutoTokenizer
    from ep_loop.config import load_config, validate_config, student_budget, smoke_config
    from ep_loop.data import prompt_ids, ByteTokenizer
    from ep_loop.inference import integer_answer, math_metrics, verify_integer_response
    from ep_loop.reproducibility import execution_provenance, sample_seed, seed_everything
    from ep_loop.evaluation_data import load_evaluation_records, benchmark_math_metrics, rad_pass_metrics
    from ep_loop.cpu_runtime import configure_cpu_threads
    observer.event('cpu_runtime', echo=True, num_threads=configure_cpu_threads())
    config = load_config(args.config)
    if args.smoke:
        if args.device != 'cpu':p.error('--smoke requires --device cpu')
        config = smoke_config(config)
    validate_config(config)
    policy = config['evaluation'].get('student_policy',config['algorithm'].get('student_policy', 'fixed'))
    rad_eval = config['algorithm'].get('student_policy') == 'ragged'
    if (args.phase_zero or args.threshold) and (not rad_eval or args.loops):
        p.error('--phase-zero/--threshold require the RAD-OPD config and no --loops override')
    if args.phase_zero and args.checkpoint is not None:
        p.error('--phase-zero evaluates the original model from the config')
    if args.phase_zero and (args.smoke or torch.device(args.device).type != 'cuda'):
        p.error('--phase-zero requires a CUDA device for vLLM and vLLM-RLT')
    if args.checkpoint is None and policy not in {'native','native_adaptive'} and not args.phase_zero:
        p.error('--checkpoint is required for a trained student')
    # Hash the exact bytes being evaluated, once, before a potentially long run.
    data_span = observer.start('evaluation_data_load', echo=True, data=str(args.data))
    records,data_metadata = load_evaluation_records(args.data,
        config['evaluation'].get('expected_examples'),config['evaluation'].get('suite'))
    data_sha256 = data_metadata['sha256']
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Evaluation problems must be nonempty and have unique IDs')
    if any(integer_answer(r['answer']) is None for r in records):
        raise ValueError('Use an external official evaluator for noninteger/symbolic answers')
    observer.finish(data_span, problems=len(records))
    directory = (args.checkpoint or ROOT / config['model']['path']).resolve()
    if args.smoke and args.checkpoint and json.loads((directory/'config.json').read_text()).get('vocab_size') != 260:
        raise ValueError('Smoke evaluation accepts only an explicitly tiny 260-token checkpoint')
    seed_everything(config['seed'])
    with observer.stage('tokenizer_load', echo=True):
        tokenizer = ByteTokenizer() if args.smoke else AutoTokenizer.from_pretrained(directory,local_files_only=True)
    reproducibility = execution_provenance(ROOT, args.device, config['seed'], phase='evaluation')
    settings = dict(config['evaluation'])
    if args.smoke:
        settings.update(max_new_tokens=4,samples_per_problem=1,student_loops=[student_budget(config)])
    if args.max_new_tokens is not None:
        if not 1 <= args.max_new_tokens <= settings['max_new_tokens']:
            p.error('--max-new-tokens must be positive and at most the configured cap')
        settings['max_new_tokens'] = args.max_new_tokens
    if args.samples_per_problem is not None:
        if args.samples_per_problem < 1: p.error('--samples-per-problem must be positive')
        settings['samples_per_problem'] = args.samples_per_problem
    if rad_eval:
        budgets = sorted({min(n, settings['max_new_tokens']) for n in settings['budget_slices']})
        settings.update(budget_slices=budgets, phase_zero=args.phase_zero)
    if rad_eval and not args.loops and (policy == 'ragged' or args.phase_zero):
        thresholds = args.threshold or settings['ragged_thresholds']
        if not all(0 < t <= 1 for t in thresholds) or len(set(thresholds)) != len(thresholds):
            p.error('RAD-OPD thresholds must be distinct and in (0, 1]')
        settings['ragged_thresholds'] = thresholds
        depth = student_budget(config)
        modes = [('ragged', depth, tau) for tau in thresholds]
        if args.phase_zero:
            settings['skip_backfill'] = args.skip_backfill
            modes = [('fixed', k, 1.0) for k in (depth, 2, 1)] + modes
            if not args.skip_backfill:
                modes.extend(('native_backfill', depth, tau) for tau in thresholds)
    else:
        threshold = (config['algorithm'].get('student_stop_threshold', .5) if policy == 'learned'
                     else config['algorithm']['exit_threshold'])
        modes = [('fixed', depth, threshold) for depth in (args.loops or settings['student_loops'])] if args.loops or policy == 'fixed' else [(policy,student_budget(config),threshold)]
    if any(not 1 <= depth <= config['algorithm']['teacher_max_loops'] for _, depth, _ in modes):
        p.error('Requested loop count is outside the configured maximum')
    gpu = None
    if args.phase_zero:
        gpu = 'GPU-' + reproducibility['gpu']['uuid'].removeprefix('GPU-')
        toolkit = Path('/usr/local/cuda-12.8')
        if (toolkit / 'bin/nvcc').is_file():
            os.environ.update(CUDA_HOME=str(toolkit), CUDA_PATH=str(toolkit),
                              PATH=os.pathsep.join((str(toolkit / 'bin'), os.environ.get('PATH', os.defpath))))
        os.environ.update(HF_HUB_OFFLINE='1', VLLM_NO_USAGE_STATS='1')
    jobs = []
    for record in records:
        ids = prompt_ids(tokenizer, record, config['model']['enable_thinking'], config['model']['repair_template_newlines'])
        if not ids or len(ids) > config['data']['max_prompt_tokens']:
            raise ValueError('Evaluation prompt must be nonempty and fit max_prompt_tokens')
        jobs.extend((record, sample, sample_seed(config['seed'], record['problem_sha256'], sample), ids)
                    for sample in range(settings['samples_per_problem']))
    max_model_len = config['data']['max_prompt_tokens'] + settings['max_new_tokens']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    summaries = {}
    with ExitStack() as run:
        f = run.enter_context(args.output.open('r+b' if args.resume else 'xb'))
        if args.phase_zero:
            from fcntl import flock, LOCK_EX, LOCK_NB
            flock(f.fileno(), LOCK_EX | LOCK_NB)
        completed = (load_completed_responses(f, modes, jobs, settings) if args.resume
                     else {mode: {} for mode in modes})
        pending = {mode: [job for job in jobs if (job[0]['id'], job[1]) not in completed[mode]]
                   for mode in modes}
        reused = sum(len(rows) for rows in completed.values())
        total = len(jobs) * len(modes)
        observer.event('evaluation_plan', echo=True, modes=modes, completed=reused, total=total)
        completions = observer.progress(total=total, initial=reused, desc='Evaluate completions',
                                        unit='response', position=0)
        run.callback(completions.close)
        engines = run.enter_context(closing(evaluation_engines(
            modes, pending, args, config, directory, max_model_len, gpu, observer)))
        for mode, backend, engine in engines:
            endpoint, loops, threshold = mode
            mode_span = observer.start('evaluation_mode', echo=True, backend=engine,
                                       policy=endpoint, loops=loops, exit_threshold=threshold)
            correctness = {record['id']: [] for record in records}
            depths, selected_depths, seconds, token_loops = [], [], [], 0
            backfill_token_loops, coda_token_positions = 0, 0
            if rad_eval:
                sliced = {cap: {record['id']: [] for record in records} for cap in budgets}
                output_tokens = truncated = decode_tokens = decode_loops = depth_one = 0
                physical_slots = []
            saved = completed[mode]
            restored = ((record, sample, seed, saved[record['id'], sample])
                        for record, sample, seed, _ in jobs if (record['id'], sample) in saved)
            generated = (evaluation_responses(backend, engine, mode, pending[mode], settings, observer)
                         if pending[mode] else ())
            for record, sample, completion_seed, response in chain(restored, generated):
                text = tokenizer.decode(response.token_ids)
                verdict = verify_integer_response(text, record['answer'], record['messages'])
                correctness[record['id']].append(verdict['correct'])
                selected_depths.extend(response.exit_depths)
                depths.extend(response.execution_depths)
                seconds.append(response.seconds)
                token_loops += response.executed_token_loops
                backfill_token_loops += response.backfill_token_loops
                coda_token_positions += response.coda_token_positions
                extra = {}
                if args.write_token_ids:
                    extra['tokens'] = response.token_ids
                if rad_eval:
                    decode_depths = response.exit_depths[1:]
                    output_tokens += len(response.token_ids)
                    truncated += not response.reached_eos
                    if response.physical_kv_token_slots is not None:
                        physical_slots.append(response.physical_kv_token_slots)
                    decode_tokens += len(decode_depths)
                    decode_loops += sum(decode_depths)
                    depth_one += decode_depths.count(1)
                    slices = {}
                    for cap in budgets:
                        correct = (verdict['correct'] if cap >= len(response.token_ids) else
                                   verify_integer_response(tokenizer.decode(response.token_ids[:cap]),
                                                           record['answer'], record['messages'])['correct'])
                        slices[cap] = correct
                        sliced[cap][record['id']].append(correct)
                    extra.update(exit_threshold=threshold,
                                 physical_kv_token_slots=response.physical_kv_token_slots,
                                 budget_correct=slices)
                if (record['id'], sample) in saved:
                    continue
                row = dict(id=record['id'],problem_sha256=record['problem_sha256'],
                    benchmark=record.get('benchmark','unspecified'),
                    policy=endpoint,max_loops=loops,sample=sample,inference_backend=engine,
                    sampling_seed=completion_seed,text=text,
                    response_tokens=len(response.token_ids),reached_eos=response.reached_eos,
                    actual_student_exit_depths=response.exit_depths,executed_token_loops=response.executed_token_loops,
                    actual_execution_depths=response.execution_depths,
                    cache_policy=response.cache_policy,backfill_token_loops=response.backfill_token_loops,
                    coda_token_positions=response.coda_token_positions,
                    generation_seconds=response.seconds,**verdict,**extra)
                f.write((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))
                f.flush()
                observer.event('evaluation_sample_complete', id=record['id'], sample=sample,
                               policy=endpoint, response_tokens=len(response.token_ids), **verdict)
                completions.update(1)
            ordered = sorted(depths)
            percentile = lambda q: ordered[min(len(ordered)-1,int(q*(len(ordered)-1)))]
            name = f'{endpoint}_{loops}'
            if endpoint in {'ragged', 'native_backfill'}:
                name += f'_tau_{threshold}'
            summaries[name] = dict(**math_metrics(correctness), inference_backend=engine,
                benchmarks=benchmark_math_metrics(records,correctness),
                actual_depth_hist=[depths.count(i) for i in range(1,loops+1)],mean_actual_depth=sum(depths)/len(depths),
                selected_exit_depth_hist=[selected_depths.count(i) for i in range(1,loops+1)],
                actual_depth_p50=percentile(.5),actual_depth_p90=percentile(.9),actual_depth_p99=percentile(.99),
                max_depth_fraction=depths.count(loops)/len(depths),executed_token_loops=token_loops,
                backfill_token_loops=backfill_token_loops,coda_token_positions=coda_token_positions,
                generation_seconds_sum=sum(seconds),mean_response_seconds=sum(seconds)/len(seconds))
            if rad_eval:
                summaries[name].update(
                    **rad_pass_metrics(correctness), exit_threshold=threshold, max_loops=loops,
                    mean_decode_depth=decode_loops/decode_tokens if decode_tokens else None,
                    decode_depth_one_fraction=depth_one/decode_tokens if decode_tokens else None,
                    mean_response_tokens=output_tokens/len(seconds), truncation_rate=truncated/len(seconds),
                    mean_executed_token_loops=token_loops/len(seconds),
                    mean_physical_kv_token_slots=(sum(physical_slots)/len(physical_slots) if physical_slots else None),
                    budget_slices={cap: rad_pass_metrics(scores) for cap, scores in sliced.items()})
            observer.finish(mode_span, metrics=summaries[name])
        report = dict(settings=settings,results=summaries,verifier='prompt_aligned_integer_v2',smoke=args.smoke,
            experiment=config.get('experiment',{}),
            reproducibility=reproducibility,
            observability=observer.metadata(),
            data=data_metadata, resumed_responses=reused,
            scope='Measured generated-token depths and generation latency; excludes external tokenization, validation and model loading. No teacher labels are used at student deployment.')
        args.output.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
