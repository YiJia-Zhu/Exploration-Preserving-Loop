"""Time the actual HF/RAD rollout on the same AIME prompts as the vLLM table."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

WORKLOAD = ROOT / 'audit/rlt_v020_bf16_vs_vllm_20261120_152301/workload.json'


class TimingObserver:
    """Keep the production generation path and time prefill/decode without logging."""
    def __init__(self, synchronize):
        self.sync = synchronize
        self.seconds = {}

    @contextmanager
    def stage(self, name, **metadata):
        if name in {'prefill', 'decode'}:
            self.sync()
        started = time.perf_counter()
        yield {}
        if name in {'prefill', 'decode'}:
            self.sync()
            self.seconds[name] = time.perf_counter() - started

    def update_generation_progress(self, **metadata):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mix', action='store_true', help='Use question-level training thresholds instead of fixed tau')
    parser.add_argument('--tau', type=float, default=.20)
    parser.add_argument('--case', action='append',
                        choices=('hf_fixed_2', 'hf_fixed_4', 'exact_dense_4', 'rad_dense_4', 'rad_dynamic'))
    parser.add_argument('--prefix', choices=('8192', '14336'), help='Use the archived fixed-length prefix case')
    parser.add_argument('--questions', type=int, default=8)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or not 0 < args.tau <= 1 or not 1 <= args.questions <= 8:
        parser.error('Invalid output cap or threshold')
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from ep_loop.config import load_config
    from ep_loop.cpu_runtime import configure_cpu_threads
    from ep_loop.inference import generate
    from ep_loop.ouro import load_ouro
    from ep_loop.ragged import sample_training_threshold
    from ep_loop.reproducibility import seed_everything

    threads = configure_cpu_threads()
    work = json.loads(WORKLOAD.read_text())
    config = load_config(ROOT / 'configs/rad_opd.json')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No CUDA devices visible')
    seed_everything(work['seed'])
    backend, _ = load_ouro(work['model'], args.device, 'bf16', gradient_checkpointing=False)
    if args.device == 'cpu':
        backend.model.to(dtype=torch.bfloat16).eval()
    else:
        backend.model.eval()
    synchronize = (lambda: torch.cuda.synchronize(backend.device)) if args.device=='cuda' else lambda: None
    policies = {
        'hf_fixed_2': ('fixed', 2, 1.0),
        'hf_fixed_4': ('fixed', 4, 1.0),
        'rad_dense_4': ('ragged', 4, 1.0),
        'rad_dynamic': ('ragged', 4, args.tau),
    }
    if args.case:
        policies['exact_dense_4'] = ('native_backfill', 4, 1.0)
        policies = {case: policies[case] for case in args.case}
    jobs = work['jobs'][:args.questions]
    if args.prefix:
        prefixes = ROOT / 'audit/rlt_v020_runtime_probe_20261120_151153/workload.json'
        prefix_jobs = json.loads(prefixes.read_text())['phases'][args.prefix]['jobs']
        by_id = {job['id']: job['input_ids'] for job in prefix_jobs}
        jobs = [{**job, 'input_ids': by_id[job['id']]} for job in jobs]
    report = dict(device=args.device, device_name=torch.cuda.get_device_name(backend.device)
                  if args.device=='cuda' else platform.processor(), torch_version=torch.__version__,
                  threads=threads, model=work['model'], dtype='bf16',
                  weight_storage_dtype=str(next(backend.model.parameters()).dtype),
                  questions=len(jobs), request_schedule='one request at a time', prefix=args.prefix,
                  temperature=work['temperature'], top_p=work['top_p'], mix=args.mix,
                  max_new_tokens=args.max_new_tokens, source_workload=str(WORKLOAD), results={})
    (args.output/'settings.json').write_text(json.dumps(report,indent=2)+'\n')

    def run(job, case, cap, warmup=False):
        endpoint, loops, tau = policies[case]
        if case=='rad_dynamic' and args.mix:
            tau = sample_training_threshold(config['algorithm']['rad_tau'], work['seed'],
                                            job['problem_sha256'], 0)
        rng = torch.Generator(device=backend.device).manual_seed(job['seed'])
        timer = TimingObserver(synchronize)
        response = generate(
            backend, job['input_ids'], loops, cap, work['temperature'], work['top_p'], rng,
            endpoint=endpoint, exit_threshold=tau, observer=timer,
        )
        if case.startswith('rad'):
            assert response.backfill_token_loops==0 and response.physical_kv_token_slots==response.executed_token_loops
            assert response.exit_depths[0]==4
        row = dict(case=case, id=job['id'], tau=tau, prompt_tokens=len(job['input_ids']),
                   output_tokens=len(response.token_ids), reached_eos=response.reached_eos,
                   executed_token_loops=response.executed_token_loops,
                   physical_kv_token_slots=response.physical_kv_token_slots,
                   response_seconds=response.seconds, prefill_seconds=timer.seconds['prefill'],
                   decode_seconds=timer.seconds.get('decode',0.), exit_depths=response.exit_depths)
        if not warmup:
            print(json.dumps(row,ensure_ascii=False),flush=True)
        return row

    for case in policies:
        run(work['jobs'][0],case,2,warmup=True)
    with (args.output/'responses.jsonl').open('w') as output:
        by_policy = {name:[] for name in policies}
        for i,job in enumerate(jobs):
            # Reverse order on alternate questions so frequency/temperature drift
            # does not systematically favor the same path.
            order = list(policies) if i%2==0 else list(reversed(policies))
            for case in order:
                row = run(job,case,args.max_new_tokens)
                by_policy[case].append(row)
                output.write(json.dumps(row,ensure_ascii=False)+'\n')
                output.flush()
    for case, rows in by_policy.items():
        tokens = sum(r['output_tokens'] for r in rows)
        elapsed = sum(r['response_seconds'] for r in rows)
        prefill = sum(r['prefill_seconds'] for r in rows)
        decode = sum(r['decode_seconds'] for r in rows)
        depths = [depth for row in rows for depth in row['exit_depths'][1:]]
        report['results'][case] = dict(responses=len(rows), generated_tokens=tokens,
            total_seconds=elapsed, output_tok_s=tokens/elapsed, prefill_seconds=prefill,
            steady_decode_seconds=decode, decode_tok_s=len(depths)/decode if decode else None,
            mean_decode_depth=sum(depths)/len(depths) if depths else None,
            decode_hist=[depths.count(depth) for depth in range(1,5)],
            total_physical_kv_slots=sum(r['physical_kv_token_slots'] for r in rows))
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
