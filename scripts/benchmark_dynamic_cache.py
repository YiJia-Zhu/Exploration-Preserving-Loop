"""Verify exact lazy-depth KV and time controlled continuations (no training).

Gate depths and continuation tokens are forced to compare identical workloads.
Huginn full-prefix references reuse the SAME sampled native initial states.
These are engineering measurements, not task scores or trained-model speedups.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=['ouro', 'huginn', 'both'], default='both')
    p.add_argument('--device', default='cpu')
    p.add_argument('--pretrained', action='store_true', help='Load local real weights; default: random tiny architecture')
    p.add_argument('--mixed-precision', choices=['none', 'bf16'], default='none')
    p.add_argument('--prompt-tokens', type=int, default=64)
    p.add_argument('--new-tokens', type=int, default=16)
    p.add_argument('--loops', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output path; existing measurements are protected')
    if min(args.prompt_tokens, args.new_tokens, args.loops, args.repeats) < 1:
        p.error('Lengths, loops and repeats must be positive')
    import torch
    from ep_loop.backends import load_backend
    from ep_loop.config import load_config
    from ep_loop.inference import generate
    from ep_loop.reproducibility import execution_provenance, seed_everything
    from ep_loop.selection import adjacent_js_from_log_probs
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    results = []
    families = ['ouro', 'huginn'] if args.family == 'both' else [args.family]
    for family in families:
        seed_everything(42)
        config = load_config(root/'configs'/('e_dynamic_js.json' if family == 'ouro' else 'huginn_e_dynamic_js.json'))
        directory = (root/config['model']['path']).resolve()
        backend, _ = load_backend(config['model'], directory, args.device, args.mixed_precision,
                                  False, smoke=not args.pretrained)
        backend.enable_student_exit()
        backend.model.eval()  # generate() uses no_grad and caches BF16 casts.
        device = backend.device
        if args.pretrained:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            text = 'Find the next number in the sequence and explain your reasoning. 1 2 3 4 5 6 7 8 9. '
            tokens = tokenizer.encode(text, add_special_tokens=False)
            prompt = (tokens * (args.prompt_tokens//len(tokens)+1))[:args.prompt_tokens]
            continuation = [t for t in tokenizer.encode(' 1 2 3 4 5', add_special_tokens=False) if t not in backend.stop_token_ids]
        else:
            prompt = [1] + [20 + i % 100 for i in range(args.prompt_tokens - 1)]
            continuation = [40, 50, 60]
        suffix = (continuation * (args.new_tokens//len(continuation)+1))[:args.new_tokens]
        all_ids = torch.tensor([prompt + suffix], device=device)
        if all_ids.shape[1] > backend.max_context_length:
            raise ValueError('Requested benchmark exceeds the model context')
        # A single native initial-state tape defines the stochastic Huginn
        # function being compared. Cached and recomputed paths use its slices.
        initial = None
        if family == 'huginn':
            with torch.no_grad(), backend.autocast():
                embedded, _ = backend.model.embed_inputs(all_ids)
                initial = backend.model.initialize_state(embedded)
            del embedded

        def fixed_init(cached):
            if initial is None:
                return nullcontext()
            seen = [0]
            def initialize(embedded):
                count = embedded.shape[1]
                start = seen[0] if cached else 0
                seen[0] += count
                return initial[:, start:start + count]
            return patch.object(backend.model, 'initialize_state', new=initialize)

        for case in ['shallow', 'mixed']:
            schedule = [1] * args.new_tokens if case == 'shallow' else [
                [1, args.loops, min(2, args.loops), min(3, args.loops)][i % 4] for i in range(args.new_tokens)]
            gate = backend.enable_student_exit()
            native_gate_forward = gate.forward

            def gate_context():
                position, depth = [0], [0]
                def forward(hidden):
                    depth[0] += 1
                    stop = depth[0] >= schedule[position[0]]
                    if stop:
                        position[0] += 1
                        depth[0] = 0
                    return hidden.new_full((*hidden.shape[:2], 1), 100. if stop else -100.)
                return patch.object(gate, 'forward', new=forward)

            max_state_error, max_logit_error, max_js, backfill = 0., 0., 0., 0
            cache = None
            # Independent full-prefix references use the original native
            # fixed-depth path; all intermediate readouts are checked too.
            with torch.no_grad(), gate_context(), fixed_init(True):
                for step, depth in enumerate(schedule):
                    end = len(prompt) + step
                    current = all_ids[:, :end] if cache is None else all_ids[:, end-1:end]
                    trace = backend.trace(current, args.loops, cache, True, 'learned', .5)
                    cache = trace.cache
                    backfill += trace.backfill_token_loops
                    with fixed_init(False), patch.object(gate, 'forward', new=native_gate_forward):
                        full = backend.trace(all_ids[:, :end], depth)
                    for d in range(depth):
                        a, b = trace.states[d][:, -1:], full.states[d][:, -1:]
                        max_state_error = max(max_state_error, (a-b).abs().max().item())
                        la, lb = backend.head(a), backend.head(b)
                        max_logit_error = max(max_logit_error, (la-lb).abs().max().item())
                        js = adjacent_js_from_log_probs(torch.stack((la[0].log_softmax(-1),lb[0].log_softmax(-1))))
                        max_js = max(max_js, js.max().item())
                        if args.mixed_precision == 'none':
                            torch.testing.assert_close(a, b, rtol=5e-4, atol=1e-4)
                    assert trace.exits[0, -1].item() == depth
            tolerance = 5e-4 if args.mixed_precision == 'bf16' else 1e-6
            if not max_js <= tolerance:
                raise AssertionError(f'{family}/{case}: JS={max_js} exceeds {tolerance}')
            del cache, trace, full

            def run(cached):
                tokens = iter(suffix)
                with gate_context(), fixed_init(cached), patch('ep_loop.inference.sample_token',
                        new=lambda *args: torch.tensor([next(tokens)], device=device)):
                    out = generate(backend, prompt, args.loops, args.new_tokens, 1., 1.,
                        torch.Generator(device=device).manual_seed(42), 'learned', .5, cached)
                assert out.token_ids == suffix and out.exit_depths == schedule
                return out

            # Warm BOTH paths, then alternate order to reduce ordering bias.
            run(False)
            run(True)
            times = {False: [], True: []}
            peak = {False: [], True: []}
            last = {}
            for repeat in range(args.repeats):
                for cached in ([False, True] if repeat % 2 == 0 else [True, False]):
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    out = run(cached)
                    times[cached].append(out.seconds)
                    peak[cached].append(torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
                    last[cached] = out
            row = dict(family=family,pretrained=args.pretrained,case=case,loops=args.loops,
                model_directory=str(directory),model_revision=config['model']['revision'],
                prompt_tokens=len(prompt),new_tokens=args.new_tokens,mixed_precision=args.mixed_precision,
                seed=42,repeats=args.repeats,forced_exit_depths=schedule,
                cached_seconds=times[True],recompute_seconds=times[False],
                median_speedup=statistics.median(times[False])/statistics.median(times[True]),
                cached_token_loops=last[True].executed_token_loops,
                recompute_token_loops=last[False].executed_token_loops,
                backfill_token_loops=last[True].backfill_token_loops,
                cached_coda_token_positions=last[True].coda_token_positions,
                cached_peak_allocated_bytes=peak[True],recompute_peak_allocated_bytes=peak[False],
                max_state_absolute_error=max_state_error,max_logit_absolute_error=max_logit_error,
                max_distribution_js_nats=max_js,js_acceptance_threshold=tolerance,
                validation='PASS',huginn_initial_states='native sampled once per position, shared reference tape' if initial is not None else None)
            results.append(row)
            print(json.dumps(row), flush=True)
        provenance = execution_provenance(root, device, 42, phase='cache_benchmark')
        del backend, gate, initial
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report = dict(status='PASS',cases=results,environment=provenance,
        scope='Controlled continuations with real or tiny weights. Stop depths and output tokens are forced; sampling is excluded. Timings include prefill, decode, exact historical suffix fill and LM heads; exclude model loading and validation. Not downstream task performance or a comparison with LoopOPD.',
        proposal_sha256=hashlib.sha256((root/'docs/loop_distillation_research_proposal_zh.md').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
