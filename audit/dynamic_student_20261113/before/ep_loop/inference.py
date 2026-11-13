"""Independent native KV caches for fixed students and dynamic teachers."""
from dataclasses import dataclass
import re

import torch

from .ouro import evaluation_mode


@dataclass
class GeneratedResponse:
    token_ids: list
    exit_depths: list
    reached_eos: bool
    executed_token_loops: int


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
             generator, endpoint="fixed", exit_threshold=0.9, use_cache=True):
    ids = torch.tensor([input_ids], dtype=torch.long, device=backend.device)
    if max_new_tokens < 1:
        raise ValueError("Maximum response length must be positive")
    if ids.shape[1] + max_new_tokens > backend.max_context_length:
        raise ValueError("Prompt plus maximum response exceeds model context limit")
    cache, response, depths = None, [], []
    current = ids
    eos_ids = backend.stop_token_ids
    token_loops = 0
    with evaluation_mode(backend.model):
        for _ in range(max_new_tokens):
            trace = backend.trace(current, loops, cache=cache, use_cache=use_cache,
                                  endpoint=endpoint, exit_threshold=exit_threshold)
            token_loops += trace.executed_token_loops
            if endpoint == "native":
                chosen = int(backend.native_exit_depths(trace, exit_threshold)[0,-1])
            elif endpoint == "fixed":
                chosen = loops
            else:
                raise ValueError("Unknown generation endpoint")
            logits = backend.head(trace.states[chosen-1][:,-1:])[0,0]
            token = int(sample_token(logits, temperature, top_p, generator))
            response.append(token)
            depths.append(chosen)
            if token in eos_ids:
                break
            ids = torch.cat([ids, ids.new_tensor([[token]])], dim=1)
            cache = trace.cache
            current = ids[:,-1:] if use_cache else ids
    return GeneratedResponse(response, depths, bool(response[-1] in eos_ids), token_loops)


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


def verify_integer_response(text, ground_truth):
    answer = integer_answer(last_boxed(text))
    target = integer_answer(ground_truth)
    if target is None:
        raise ValueError("The bundled verifier supports integer ground truth only")
    return dict(correct=answer is not None and answer == target,
                parsed_integer=answer, verifier="strict_boxed_integer_v1")


def math_metrics(correctness_by_problem):
    if not correctness_by_problem or any(not values for values in correctness_by_problem.values()):
        raise ValueError("Every problem needs at least one completion")
    total = sum(len(values) for values in correctness_by_problem.values())
    correct = sum(sum(values) for values in correctness_by_problem.values())
    return dict(avg=correct/total,
                pass_at_n=sum(any(values) for values in correctness_by_problem.values())/len(correctness_by_problem),
                problems=len(correctness_by_problem),completions=total,
                samples_per_problem=sorted({len(values) for values in correctness_by_problem.values()}))
