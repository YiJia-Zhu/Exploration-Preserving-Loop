"""Evaluate configured dynamic policy, native base, or explicit fixed budgets."""
import argparse
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
    p.add_argument('--samples-per-problem', type=int)
    p.add_argument('--smoke',action='store_true',help='CPU-only saved/random tiny architecture with ByteTokenizer')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from transformers import AutoTokenizer
    from ep_loop.config import load_config, validate_config, student_budget, smoke_config
    from ep_loop.data import normalize_record, prompt_ids, read_jsonl, ByteTokenizer
    from ep_loop.inference import generate, integer_answer, math_metrics, verify_integer_response
    from ep_loop.backends import load_backend
    root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    if args.smoke:
        if args.device != 'cpu':p.error('--smoke requires --device cpu')
        config = smoke_config(config)
    validate_config(config)
    policy = config['algorithm'].get('student_policy', 'fixed')
    if args.checkpoint is None and policy != 'native':
        p.error('--checkpoint is required for a trained student')
    records = [normalize_record(row,i) for i,row in enumerate(read_jsonl(args.data))]
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Evaluation problems must be nonempty and have unique IDs')
    if any(integer_answer(r['answer']) is None for r in records):
        raise ValueError('Use an external official evaluator for noninteger/symbolic answers')
    directory = args.checkpoint or (root/config['model']['path'])
    if args.smoke and args.checkpoint and json.loads((directory/'config.json').read_text()).get('vocab_size') != 260:
        raise ValueError('Smoke evaluation accepts only an explicitly tiny 260-token checkpoint')
    torch.manual_seed(config['seed'])
    backend,_ = load_backend(config['model'],directory,args.device,config['optimization']['mixed_precision'],False,
                            smoke=args.smoke and args.checkpoint is None)
    backend.model.eval().requires_grad_(False)
    if policy == 'learned' and not args.loops:
        if getattr(backend.model.config, 'ep_loop_student_policy', None) != 'learned':
            raise ValueError('Checkpoint is not marked as a dynamic student; use its own config or --loops')
    tokenizer = ByteTokenizer() if args.smoke else AutoTokenizer.from_pretrained(directory,local_files_only=True)
    settings = dict(config['evaluation'])
    if args.smoke:
        settings.update(max_new_tokens=4,samples_per_problem=1,student_loops=[student_budget(config)])
    if args.samples_per_problem is not None:
        if args.samples_per_problem < 1: p.error('--samples-per-problem must be positive')
        settings['samples_per_problem'] = args.samples_per_problem
    modes = [('fixed',depth) for depth in (args.loops or settings['student_loops'])] if args.loops or policy == 'fixed' else [(policy,student_budget(config))]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    summaries = {}
    with args.output.open('x') as f:
        for endpoint,loops in modes:
            rng = torch.Generator(device=backend.device).manual_seed(config['seed'])
            torch.manual_seed(config['seed'])
            if backend.device.type == 'cuda': torch.cuda.manual_seed_all(config['seed'])
            correctness,depths,selected_depths,seconds,token_loops = {},[],[],[],0
            for record in records:
                ids = prompt_ids(tokenizer,record,config['model']['enable_thinking'],config['model']['repair_template_newlines'])
                correctness[record['id']] = []
                for sample in range(settings['samples_per_problem']):
                    threshold = config['algorithm'].get('student_stop_threshold',.5) if endpoint == 'learned' else config['algorithm']['exit_threshold']
                    response = generate(backend,ids,loops,settings['max_new_tokens'],settings['temperature'],settings['top_p'],rng,
                        endpoint=endpoint,exit_threshold=threshold,
                        use_cache=False if endpoint == 'learned' else settings.get('use_cache',True))
                    text = tokenizer.decode(response.token_ids)
                    verdict = verify_integer_response(text,record['answer'])
                    correctness[record['id']].append(verdict['correct'])
                    selected_depths.extend(response.exit_depths)
                    depths.extend(response.execution_depths);seconds.append(response.seconds)
                    token_loops += response.executed_token_loops
                    f.write(json.dumps(dict(id=record['id'],policy=endpoint,max_loops=loops,sample=sample,text=text,
                        response_tokens=len(response.token_ids),reached_eos=response.reached_eos,
                        actual_student_exit_depths=response.exit_depths,executed_token_loops=response.executed_token_loops,
                        actual_execution_depths=response.execution_depths,
                        generation_seconds=response.seconds,**verdict),ensure_ascii=False)+'\n')
                    f.flush()
            ordered = sorted(depths)
            percentile = lambda q: ordered[min(len(ordered)-1,int(q*(len(ordered)-1)))]
            summaries[f'{endpoint}_{loops}'] = dict(**math_metrics(correctness),
                actual_depth_hist=[depths.count(i) for i in range(1,loops+1)],mean_actual_depth=sum(depths)/len(depths),
                selected_exit_depth_hist=[selected_depths.count(i) for i in range(1,loops+1)],
                actual_depth_p50=percentile(.5),actual_depth_p90=percentile(.9),actual_depth_p99=percentile(.99),
                max_depth_fraction=depths.count(loops)/len(depths),executed_token_loops=token_loops,
                generation_seconds_sum=sum(seconds),mean_response_seconds=sum(seconds)/len(seconds))
    report = dict(settings=settings,results=summaries,verifier='strict_boxed_integer_v1',smoke=args.smoke,
        scope='Measured generated-token depths and generation latency; excludes external tokenization, validation and model loading. No teacher labels are used at student deployment.')
    args.output.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
