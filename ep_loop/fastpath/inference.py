"""Independent native KV caches for fixed students and dynamic teachers."""
from dataclasses import dataclass
from contextlib import ExitStack, nullcontext
import re
import time

import torch

from ep_loop.ouro import evaluation_mode


@dataclass
class GeneratedResponse:
    token_ids: list
    exit_depths: list
    reached_eos: bool
    executed_token_loops: int
    seconds: float = 0.
    execution_depths: object = None
    backfill_token_loops: int = 0
    coda_token_positions: int = 0
    cache_policy: str = "none"


def sample_token(logits, temperature, top_p, generator):
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Require temperature > 0 and 0 < top_p <= 1")
    scores = logits.float() / temperature
    if top_p < 1:
        from transformers.generation.logits_process import TopPLogitsWarper
        scores = TopPLogitsWarper(top_p)(None, scores[None, :])[0]
    return torch.multinomial(scores.softmax(-1), 1, generator=generator)


@torch.no_grad()
def generate(backend, input_ids, loops, max_new_tokens, temperature, top_p,
             generator, endpoint="fixed", exit_threshold=0.9, use_cache=True,
             observer=None, log_context=None, progress=False, progress_desc="Generate", progress_position=2):
    fields = dict(log_context or {}, policy=endpoint, max_loops=loops,
                  prompt_tokens=len(input_ids), max_new_tokens=max_new_tokens, use_cache=use_cache)
    context = observer.stage("generation", **fields) if observer else nullcontext({})
    with context as stats:
        bar = observer.progress(total=max_new_tokens, desc=progress_desc, unit="tok",
            position=progress_position, leave=False, tokens=True) if observer and progress else None
        try:
            response = _generate(backend, input_ids, loops, max_new_tokens, temperature, top_p,
                                 generator, endpoint, exit_threshold, use_cache, observer, bar)
        finally:
            if bar is not None:
                # Persist the actual final count even for sub-second responses
                # or EOS before the maximum length, then clear the live bar.
                bar.refresh()
                bar.close()
        stats.update(response_tokens=len(response.token_ids), reached_eos=response.reached_eos,
            generation_seconds=response.seconds, executed_token_loops=response.executed_token_loops,
            backfill_token_loops=response.backfill_token_loops, cache_policy=response.cache_policy,
            mean_exit_depth=sum(response.exit_depths)/len(response.exit_depths))
        return response


def _generate(backend, input_ids, loops, max_new_tokens, temperature, top_p,
              generator, endpoint, exit_threshold, use_cache, observer, bar):
    started = time.perf_counter()
    ids = torch.tensor([input_ids], dtype=torch.long, device=backend.device)
    if max_new_tokens < 1:
        raise ValueError("Maximum response length must be positive")
    if ids.shape[1] + max_new_tokens > backend.max_context_length:
        raise ValueError("Prompt plus maximum response exceeds model context limit")
    cache, response, depths, execution_depths = None, [], [], []
    current = ids
    eos_ids = backend.stop_token_ids
    token_loops, backfill, coda = 0, 0, 0
    decode_stats = None
    with evaluation_mode(backend.model), backend.generation_context(), ExitStack() as phases:
        for step in range(max_new_tokens):
            if observer and step == 1:
                decode_stats = phases.enter_context(observer.stage("decode"))
            # The first forward includes the prompt prefill. Host wall timing
            # adds no CUDA synchronization beyond normal generation.
            with observer.stage("prefill") if observer and step == 0 else nullcontext():
                trace = backend.trace(current, loops, cache=cache, use_cache=use_cache,
                                      endpoint=endpoint, exit_threshold=exit_threshold)
            token_loops += trace.executed_token_loops
            backfill += trace.backfill_token_loops
            coda += trace.coda_token_positions
            execution_depths.append((trace.executed_token_loops - trace.backfill_token_loops) // current.shape[1])
            if endpoint == "learned":
                chosen = len(trace.states)  # Same batch-one exit, already known on host.
            elif endpoint == "native":
                chosen = int(backend.native_exit_depths(trace, exit_threshold)[0,-1])
            elif endpoint == "fixed":
                chosen = loops
            else:
                raise ValueError("Unknown generation endpoint")
            logits = backend.head(trace.states[chosen-1][:,-1:])[0,0]
            token = int(sample_token(logits, temperature, top_p, generator))
            response.append(token)
            depths.append(chosen)
            if observer:
                observer.update_generation_progress(response_tokens=len(response),last_exit_depth=chosen,
                    executed_token_loops=token_loops,backfill_token_loops=backfill)
            if bar is not None and bar.update(1):
                bar.set_postfix(depth=chosen, refresh=False)
            if token in eos_ids:
                break
            cache = trace.cache
            if use_cache:
                current = ids.new_tensor([[token]])
            else:
                ids = torch.cat([ids, ids.new_tensor([[token]])], dim=1)
                current = ids
        if decode_stats is not None:
            decode_stats['response_tokens'] = len(response)-1
    if backend.device.type == "cuda":
        torch.cuda.synchronize(backend.device)
    policy = "exact_lazy_depth" if endpoint == "learned" and use_cache else "native" if use_cache else "none"
    return GeneratedResponse(response, depths, bool(response[-1] in eos_ids), token_loops,
                             time.perf_counter()-started, execution_depths, backfill, coda, policy)


def last_boxed(text):
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(matches):
        start, depth = match.end(), 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
    return None


def integer_answer(value):
    if value is None:
        return None
    cleaned = str(value).strip().replace(",", "").replace(" ", "")
    if len(cleaned) > 1000 or not re.fullmatch(r"[+-]?\d+(?:\.0+)?", cleaned):
        return None
    return int(cleaned.split(".", 1)[0])


def verify_integer_response(text, ground_truth, messages=None):
    # Preserve the public DAPO prompt, which explicitly requests an Answer: line.
    # Other prompts retain the existing strict boxed-answer contract.
    dapo_format = any('Remember to put your answer on its own line after "Answer:"' in m['content']
                      for m in messages or [])
    value = last_boxed(text)
    if dapo_format:
        lines = list(re.finditer(r'(?im)^\s*(?:\*\*)?Answer\s*:(?:\*\*)?\s*([^\r\n]*)', text))
        if lines:
            value = re.sub(r'(?:<\|[^>\r\n]+\|>)+\s*$', '', lines[-1][1]).strip().strip('$').strip()
            if r'\boxed' in value:
                value = last_boxed(value)
    answer = integer_answer(value)
    target = integer_answer(ground_truth)
    if target is None:
        raise ValueError("The bundled verifier supports integer ground truth only")
    return dict(correct=answer is not None and answer == target,
                parsed_integer=answer,
                verifier="dapo_boxed_or_answer_integer_v1" if dapo_format else "strict_boxed_integer_v1")


def math_metrics(correctness_by_problem):
    if not correctness_by_problem or any(not values for values in correctness_by_problem.values()):
        raise ValueError("Every problem needs at least one completion")
    total = sum(len(values) for values in correctness_by_problem.values())
    correct = sum(sum(values) for values in correctness_by_problem.values())
    return dict(avg=correct/total,
                pass_at_n=sum(any(values) for values in correctness_by_problem.values())/len(correctness_by_problem),
                problems=len(correctness_by_problem),completions=total,
                samples_per_problem=sorted({len(values) for values in correctness_by_problem.values()}))
