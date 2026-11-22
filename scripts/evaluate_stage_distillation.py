"""Evaluate a trained policy, RAD-OPD Phase 0, or explicit fixed baselines."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    p.add_argument('--phase-zero', action='store_true', help='Compare base dense, static, backfill and ragged KV')
    p.add_argument('--max-new-tokens', type=int, help='Use a smaller generation cap for a short check')
    p.add_argument('--write-token-ids', action='store_true', help='Include token IDs for offline teacher export')
    p.add_argument('--samples-per-problem', type=int)
    p.add_argument('--smoke',action='store_true',help='CPU-only saved/random tiny architecture with ByteTokenizer')
    args = p.parse_args()
    from ep_loop.config import load_config
    from ep_loop.execution import execution_mode, VLLM_EVAL_MODES
    if execution_mode(load_config(args.config)) in VLLM_EVAL_MODES:
        p.error('Use scripts/evaluate_fastpath.py for a vLLM execution mode')
    if args.output.exists():
        raise FileExistsError(args.output)
    from ep_loop.observability import RunLogger
    with RunLogger(args.output.with_suffix('.events.jsonl'), args.output.with_suffix('.console.log'), 'evaluation') as observer:
        with observer.stage('evaluation_run', echo=True, config_path=args.config,
                            checkpoint=str(args.checkpoint) if args.checkpoint else None, data=str(args.data)):
            evaluate(args, p, observer)


def evaluate(args, p, observer):
    import torch
    from transformers import AutoTokenizer
    from ep_loop.config import load_config, validate_config, student_budget, smoke_config
    from ep_loop.data import normalize_record, prompt_ids, ByteTokenizer
    from ep_loop.inference import generate, integer_answer, math_metrics, verify_integer_response
    from ep_loop.backends import load_backend
    from ep_loop.reproducibility import execution_provenance, sample_seed, seed_everything
    from ep_loop.observability import model_progress
    from ep_loop.evaluation_data import load_evaluation_records, benchmark_math_metrics, rad_pass_metrics
    from ep_loop.cpu_runtime import configure_cpu_threads
    observer.event('cpu_runtime', echo=True, num_threads=configure_cpu_threads())
    root = Path(__file__).resolve().parents[1]
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
    directory = args.checkpoint or (root/config['model']['path'])
    if args.smoke and args.checkpoint and json.loads((directory/'config.json').read_text()).get('vocab_size') != 260:
        raise ValueError('Smoke evaluation accepts only an explicitly tiny 260-token checkpoint')
    seed_everything(config['seed'])
    with observer.stage('model_load', echo=True, model_path=str(directory), device=args.device), model_progress(observer.show_progress):
        backend,_ = load_backend(config['model'],directory,args.device,config['optimization']['mixed_precision'],False,
                                smoke=args.smoke and args.checkpoint is None)
    # generate() runs under no_grad. Keep leaf weights eligible for PyTorch's
    # request-scoped autocast cache instead of recasting them every token.
    backend.model.eval()
    if policy == 'learned' and not args.loops:
        if getattr(backend.model.config, 'ep_loop_student_policy', None) != 'learned':
            raise ValueError('Checkpoint is not marked as a dynamic student; use its own config or --loops')
    with observer.stage('tokenizer_load', echo=True):
        tokenizer = ByteTokenizer() if args.smoke else AutoTokenizer.from_pretrained(directory,local_files_only=True)
    reproducibility = execution_provenance(root, backend.device, config['seed'], phase='evaluation')
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
            modes = [('fixed', k, 1.0) for k in (depth, 2, 1)] + [
                ('native_backfill', depth, tau) for tau in thresholds
            ] + modes
    else:
        threshold = (config['algorithm'].get('student_stop_threshold', .5) if policy == 'learned'
                     else config['algorithm']['exit_threshold'])
        modes = [('fixed', depth, threshold) for depth in (args.loops or settings['student_loops'])] if args.loops or policy == 'fixed' else [(policy,student_budget(config),threshold)]
    if any(not 1 <= depth <= config['algorithm']['teacher_max_loops'] for _, depth, _ in modes):
        p.error('Requested loop count is outside the configured maximum')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    summaries = {}
    completions = observer.progress(total=len(records)*settings['samples_per_problem']*len(modes),
        desc='Evaluate completions', unit='response', position=0)
    with args.output.open('x') as f:
        for endpoint,loops,threshold in modes:
            mode_span = observer.start('evaluation_mode', echo=True, policy=endpoint, loops=loops, exit_threshold=threshold)
            correctness,depths,selected_depths,seconds,token_loops = {},[],[],[],0
            backfill_token_loops, coda_token_positions = 0, 0
            if rad_eval:
                sliced = {cap: {record['id']: [] for record in records} for cap in budgets}
                output_tokens = truncated = physical_slots = decode_tokens = decode_loops = depth_one = 0
            for record in records:
                ids = prompt_ids(tokenizer,record,config['model']['enable_thinking'],config['model']['repair_template_newlines'])
                if len(ids)>config['data']['max_prompt_tokens']:
                    raise ValueError('Evaluation prompt exceeds max_prompt_tokens; preserve the full problem and adjust the limit explicitly')
                correctness[record['id']] = []
                for sample in range(settings['samples_per_problem']):
                    sample_span = observer.start('evaluation_sample', id=record['id'], sample=sample, policy=endpoint, loops=loops)
                    completion_seed = sample_seed(config['seed'], record['problem_sha256'], sample)
                    seed_everything(completion_seed)
                    rng = torch.Generator(device=backend.device).manual_seed(completion_seed)
                    response = generate(backend,ids,loops,settings['max_new_tokens'],settings['temperature'],settings['top_p'],rng,
                        endpoint=endpoint,exit_threshold=threshold,
                        use_cache=settings.get('use_cache',True), observer=observer,
                        log_context=dict(id=record['id'],sample=sample), progress=True,
                        progress_desc=f"Evaluate {record['id']} #{sample}", progress_position=1)
                    text = tokenizer.decode(response.token_ids)
                    verdict = verify_integer_response(text,record['answer'],record['messages'])
                    correctness[record['id']].append(verdict['correct'])
                    selected_depths.extend(response.exit_depths)
                    depths.extend(response.execution_depths);seconds.append(response.seconds)
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
                        physical_slots += response.physical_kv_token_slots
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
                    f.write(json.dumps(dict(id=record['id'],problem_sha256=record['problem_sha256'],
                        benchmark=record.get('benchmark','unspecified'),
                        policy=endpoint,max_loops=loops,sample=sample,
                        sampling_seed=completion_seed,text=text,
                        response_tokens=len(response.token_ids),reached_eos=response.reached_eos,
                        actual_student_exit_depths=response.exit_depths,executed_token_loops=response.executed_token_loops,
                        actual_execution_depths=response.execution_depths,
                        cache_policy=response.cache_policy,backfill_token_loops=response.backfill_token_loops,
                        coda_token_positions=response.coda_token_positions,
                        generation_seconds=response.seconds,**verdict,**extra),ensure_ascii=False)+'\n')
                    f.flush()
                    observer.finish(sample_span, response_tokens=len(response.token_ids), **verdict)
                    completions.update(1)
            ordered = sorted(depths)
            percentile = lambda q: ordered[min(len(ordered)-1,int(q*(len(ordered)-1)))]
            name = f'{endpoint}_{loops}'
            if endpoint in {'ragged', 'native_backfill'}:
                name += f'_tau_{threshold}'
            summaries[name] = dict(**math_metrics(correctness),
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
                    mean_physical_kv_token_slots=physical_slots/len(seconds),
                    budget_slices={cap: rad_pass_metrics(scores) for cap, scores in sliced.items()})
            observer.finish(mode_span, metrics=summaries[name])
    completions.close()
    report = dict(settings=settings,results=summaries,verifier='prompt_aligned_integer_v2',smoke=args.smoke,
        experiment=config.get('experiment',{}),
        reproducibility=reproducibility,
        observability=observer.metadata(),
        data=data_metadata,
        scope='Measured generated-token depths and generation latency; excludes external tokenization, validation and model loading. No teacher labels are used at student deployment.')
    args.output.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
