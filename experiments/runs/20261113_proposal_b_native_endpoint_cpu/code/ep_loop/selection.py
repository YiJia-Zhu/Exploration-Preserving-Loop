"""Per-position native Ouro exits and ordered stage-end supervision.

Depths in the public plan are one-based. A teacher boundary after depth b
means the connection b -> b+1, so the preceding stage ends at b.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class StagePlan:
    teacher_depths: torch.Tensor  # [student_loops, positions], -1 = inactive
    weights: torch.Tensor        # equal weight over active stages at each position
    exits: torch.Tensor          # [positions], one-based
    adjacent_js: torch.Tensor    # [teacher_loops - 1, positions], nats

    @property
    def counts(self):
        return (self.teacher_depths > 0).sum(0)


@torch.no_grad()
def native_ouro_exits(gates, threshold):
    """Replicate the official stick-breaking CDF, including its native dtype.

gates is [K, ..., 1]. The last loop receives all remaining exit probability.
This selects readouts after the full K-loop backbone; it does not stop compute.
"""
    if not 0 < threshold <= 1:
        raise ValueError("Native exit threshold must be in (0, 1]")
    if gates.ndim < 2 or gates.shape[-1] != 1 or gates.shape[0] < 1:
        raise ValueError("Expected gate logits shaped [K, ..., 1]")
    remaining = torch.ones_like(gates[0, ..., 0])
    pdf = []
    for i, gate in enumerate(gates):
        probability = torch.sigmoid(gate[..., 0])
        if i == len(gates) - 1:
            pdf.append(remaining)
        else:
            pdf.append(probability * remaining)
            remaining = remaining * (1 - probability)
    cdf = torch.stack(pdf, dim=-1).cumsum(-1)
    reached = cdf >= threshold
    first = reached.to(torch.int64).argmax(-1) + 1
    return torch.where(reached.any(-1), first, torch.full_like(first, len(gates)))


@torch.no_grad()
def adjacent_js_from_log_probs(log_probs):
    """Standard symmetric JS in natural-log units, over the full vocabulary."""
    if log_probs.ndim != 3:
        raise ValueError("Expected [teacher_loops, positions, vocabulary]")
    if log_probs.shape[0] == 1:
        return log_probs.new_empty((0, log_probs.shape[1]))
    left, right = log_probs[:-1], log_probs[1:]
    midpoint = torch.logaddexp(left, right) - math.log(2)
    return (0.5 * ((left.exp() * (left - midpoint)).sum(-1)
                   + (right.exp() * (right - midpoint)).sum(-1))).clamp_min(0)


@torch.no_grad()
def make_stage_plan(log_probs, exits, student_loops, mode="js",
                    js_threshold=0.01, count_matched_uniform=True):
    """Select <= S stage ends, then map j-th end to ceil(j*S/M).

Uniform controls can use the same JS-derived number M of targets while using
uniform locations. Ties in JS are resolved toward the earlier boundary.
When n_t <= S, every mode uses only the final student checkpoint.
"""
    return make_stage_plan_from_js(adjacent_js_from_log_probs(log_probs), exits,
                                   student_loops, mode, js_threshold, count_matched_uniform)


@torch.no_grad()
def make_stage_plan_from_js(js, exits, student_loops, mode="js",
                            js_threshold=0.01, count_matched_uniform=True):
    """Reuse JS already measured by a native KL-exit teacher's coda readouts."""
    if mode not in {"endpoint", "uniform", "js"}:
        raise ValueError(f"Unknown selection mode: {mode}")
    if student_loops < 1 or not math.isfinite(js_threshold) or js_threshold < 0:
        raise ValueError("S must be positive and JS threshold finite/nonnegative")
    if js.ndim != 2 or not js.is_floating_point() or not torch.isfinite(js).all() or (js < 0).any():
        raise ValueError("Adjacent JS must be finite nonnegative [K-1, positions]")
    k, positions = js.shape[0] + 1, js.shape[1]
    exits = exits.detach().to(device=js.device, dtype=torch.long).reshape(-1)
    if exits.numel() != positions or ((exits < 1) | (exits > k)).any():
        raise ValueError("Exit depths must match positions and lie in [1, K]")
    targets = torch.full((student_loops, positions), -1, device=exits.device, dtype=torch.long)
    if mode == "endpoint" or student_loops == 1 or k == 1:
        targets[-1] = exits
    else:
        edge_depth = torch.arange(2, k + 1, device=exits.device)[:, None]
        eligible = ((edge_depth <= exits[None, :]) & (exits[None, :] > student_loops)
                    & (js > js_threshold))
        scores = js.masked_fill(~eligible, -torch.inf)
        limit = min(student_loops - 1, k - 1)
        order = scores.argsort(dim=0, descending=True, stable=True)[:limit]
        available = scores.gather(0, order).isfinite()
        boundaries = torch.where(available, order + 1, torch.full_like(order, k + 1))
        boundaries = boundaries.sort(dim=0).values
        counts = available.sum(0) + 1
        if mode == "uniform" and not count_matched_uniform:
            counts = torch.where(exits > student_loops,
                                 torch.full_like(exits, student_loops), torch.ones_like(exits))
        columns = torch.arange(positions, device=exits.device)
        for j in range(student_loops):
            active = j < counts
            is_final = j == counts - 1
            if mode == "uniform":
                teacher_end = ((j + 1) * exits + counts - 1) // counts
            elif j < limit:
                teacher_end = boundaries[j]
            else:
                teacher_end = exits
            teacher_end = torch.where(is_final, exits, teacher_end)
            student_slot = ((j + 1) * student_loops + counts - 1) // counts - 1
            targets[student_slot[active], columns[active]] = teacher_end[active]
    active = targets > 0
    weights = active.to(js.dtype) / active.sum(0, keepdim=True)
    return StagePlan(targets, weights, exits, js)


def stage_reverse_kl(student_logits, teacher_log_probs, plan):
    """Per-position mean over selected stages, exact full-vocabulary reverse KL.

The caller averages over response tokens, then averages over responses. Prompt
and padding positions must be excluded before calling this function.
"""
    s, n, vocabulary = student_logits.shape
    if plan.teacher_depths.shape != (s, n) or teacher_log_probs.shape[1:] != (n, vocabulary):
        raise ValueError("Student, teacher, and stage plan shapes do not match")
    logp = student_logits.log_softmax(-1)
    teacher = teacher_log_probs.detach()
    columns = torch.arange(n, device=logp.device)
    losses = logp.new_zeros(n)
    for slot in range(s):
        depth = plan.teacher_depths[slot]
        selected = teacher[(depth - 1).clamp_min(0), columns]
        kl = (logp[slot].exp() * (logp[slot] - selected)).sum(-1)
        losses = losses + plan.weights[slot].to(kl) * kl
    return losses
