"""Frozen/current-teacher distillation with bounded readout memory."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ouro import evaluation_mode
from .selection import StagePlan, make_stage_plan, make_stage_plan_from_js, stage_reverse_kl


class DistillationObjective(nn.Module):
    """One response loss; student/teacher see the same independently recomputed text.

The module registers only the student. A frozen teacher is a separate backend;
a current teacher refers to this same student without gradient on its forward.
All optimizer steps must occur after the whole global rollout batch.
"""
    def __init__(self, student, teacher, algorithm, readout_chunk_size=128,
                 checkpoint_readout=True):
        super().__init__()
        self.student_model = student.model
        self.student = student
        self.teacher = teacher
        self.algorithm = dict(algorithm)
        self.chunk_size = readout_chunk_size
        self.checkpoint_readout = checkpoint_readout
        if self.chunk_size < 1:
            raise ValueError("Readout chunk size must be positive")

    def forward(self, response_ids, prompt_length):
        if response_ids.ndim != 2 or response_ids.shape[0] != 1:
            raise ValueError("Objective expects one unpadded response per microbatch")
        n = response_ids.shape[1] - prompt_length
        if prompt_length < 1 or n < 1:
            raise ValueError("A nonempty prompt and generated response are required")
        a = self.algorithm
        s, k = a["student_loops"], a["teacher_max_loops"]
        inputs, first = response_ids[:, :-1], prompt_length - 1
        with torch.no_grad(), evaluation_mode(self.teacher.model):
            teacher = self.teacher.response_trace(inputs, k, first, a["teacher_endpoint"], a["exit_threshold"])
        student = self.student.response_trace(inputs, s, first)
        if a["teacher_endpoint"] == "native":
            exits = self.teacher.native_exit_depths(teacher, a["exit_threshold"])
        elif a["teacher_endpoint"] == "fixed":
            exits = torch.full((n,), k, device=inputs.device, dtype=torch.long)
        else:
            raise ValueError("Teacher endpoint must be fixed or native")
        loss = student.states[0].new_zeros((), dtype=torch.float32)
        exit_hist = torch.bincount(exits, minlength=k + 1)[1:]
        stage_hist = torch.zeros(s + 1, device=inputs.device, dtype=torch.long)
        mapping_hist = torch.zeros(s, k, device=inputs.device, dtype=torch.long)
        js_sum = torch.zeros((), device=inputs.device)
        js_count = torch.zeros((), device=inputs.device, dtype=torch.long)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            th = torch.stack([h[start:end] for h in teacher.states])
            sh = torch.stack([h[start:end] for h in student.states])
            with torch.no_grad():
                if a["stage_selection"] == "endpoint":
                    targets = torch.full((s, end-start), -1, device=exits.device, dtype=torch.long)
                    targets[-1] = exits[start:end]
                    weights = torch.zeros_like(targets, dtype=torch.float32)
                    weights[-1] = 1
                    plan = StagePlan(targets, weights, exits[start:end], weights.new_zeros((k-1, end-start)))
                elif teacher.adjacent_js is not None:
                    plan = make_stage_plan_from_js(
                        teacher.adjacent_js[:, start:end], exits[start:end], s,
                        a["stage_selection"], a["js_threshold"], a.get("count_matched_uniform", True),
                    )
                else:
                    teacher_logp = self.teacher.head(th).log_softmax(-1)
                    plan = make_stage_plan(
                        teacher_logp, exits[start:end], s, a["stage_selection"],
                        a["js_threshold"], a.get("count_matched_uniform", True),
                    )
                    del teacher_logp
                stage_hist += torch.bincount(plan.counts, minlength=s + 1)
                for slot in range(s):
                    active = plan.teacher_depths[slot] > 0
                    mapping_hist[slot] += torch.bincount(plan.teacher_depths[slot, active] - 1, minlength=k)
                if a["stage_selection"] != "endpoint":
                    valid_edges = torch.arange(2, k + 1, device=exits.device)[:, None] <= exits[None, start:end]
                    js_sum += plan.adjacent_js.masked_select(valid_edges).sum()
                    js_count += valid_edges.sum()
            # Bind plan per chunk: checkpoint recomputation must not capture a
            # mutable loop variable from a later chunk.
            def chunk_loss(student_hidden, teacher_hidden, selected_plan=plan):
                if a["stage_selection"] == "endpoint":
                    columns = torch.arange(teacher_hidden.shape[1], device=teacher_hidden.device)
                    with torch.no_grad():
                        target_hidden = teacher_hidden[selected_plan.exits - 1, columns]
                        target = self.teacher.head(target_hidden).log_softmax(-1)
                    predicted = self.student.head(student_hidden[-1]).log_softmax(-1)
                    return (predicted.exp() * (predicted - target)).sum()
                with torch.no_grad():
                    target = self.teacher.head(teacher_hidden).log_softmax(-1)
                predicted = self.student.head(student_hidden)
                return stage_reverse_kl(predicted, target, selected_plan).sum()
            if self.checkpoint_readout and torch.is_grad_enabled():
                part = checkpoint(chunk_loss, sh, th, use_reentrant=False)
            else:
                part = chunk_loss(sh, th)
            loss = loss + part / n
        metadata = dict(
            response_tokens=n, exit_hist=exit_hist.detach(), stage_hist=stage_hist.detach(),
            mapping_hist=mapping_hist.detach(), js_sum=js_sum.detach(), js_count=js_count.detach(),
            teacher_executed_loops=(None if self.teacher.native_exit_reduces_executed_loops
                                    and a["teacher_endpoint"] == "native" else k),
            student_executed_loops=s,
            teacher_executed_token_loops=teacher.executed_token_loops,
            student_executed_token_loops=student.executed_token_loops,
            teacher_coda_token_positions=teacher.executed_token_loops if self.teacher.has_coda else 0,
            endpoint_only_tokens=int(stage_hist[1].detach()),
            exit_not_deeper_than_student=int((exits <= s).sum().detach()),
        )
        return loss, metadata
