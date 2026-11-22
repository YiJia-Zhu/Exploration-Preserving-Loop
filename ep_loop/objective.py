"""On-policy stage prediction and optional supervised student stopping."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ouro import evaluation_mode
from .confidence import confidence_changes
from .selection import (StagePlan, adjacent_js_from_log_probs, make_dynamic_stage_plan,
                        make_stage_plan, make_stage_plan_from_js, stage_reverse_kl,
                        stage_stop_bce)


class DistillationObjective(nn.Module):
    """Teacher labels come only from the current student TEXT prefix.

    Dynamic replay computes to the largest S_t in the response. This may exceed
    rollout exits and compute unused prefix states; all actual forward work is
    counted. No teacher cache is transplanted into the student.
    """
    def __init__(self, student, teacher, algorithm, readout_chunk_size=128,
                 checkpoint_readout=True, observer=None, confidence_probe=None):
        super().__init__()
        self.student_model = student.model
        self.student, self.teacher = student, teacher
        self.algorithm = dict(algorithm)
        self.observer = observer
        self.confidence_probe = confidence_probe
        self.shared_replay=self.algorithm.get('teacher_reuse_student',False)
        if self.shared_replay and (student is not teacher or self.algorithm.get('student_policy','fixed')!='fixed'):
            raise ValueError('Shared replay requires the same fixed-depth student/teacher backend')
        self.chunk_size, self.checkpoint_readout = readout_chunk_size, checkpoint_readout
        if readout_chunk_size < 1:
            raise ValueError("Readout chunk size must be positive")
        if self.algorithm.get("student_policy", "fixed") == "learned":
            student.enable_student_exit()
        if self.algorithm.get("student_policy") == "ragged":
            student.model.model.early_exit_gate.requires_grad_(False)
            student.model.config.ep_loop_student_policy = "ragged"

    def forward(self, response_ids, prompt_length, rollout_depths=None):
        if response_ids.ndim != 2 or response_ids.shape[0] != 1:
            raise ValueError("Objective expects one unpadded response per microbatch")
        n = response_ids.shape[1] - prompt_length
        if prompt_length < 1 or n < 1:
            raise ValueError("A nonempty prompt and generated response are required")
        a = self.algorithm
        dynamic = a.get("student_policy", "fixed") == "learned"
        ragged = a.get("student_policy") == "ragged"
        k = a["teacher_max_loops"]
        slots = a["student_max_loops"] if dynamic or ragged else a["student_loops"]
        if ragged:
            if rollout_depths is None:
                raise ValueError("RAD-OPD requires the actual rollout depth at each predictor position")
            rollout_depths = torch.as_tensor(rollout_depths, device=response_ids.device, dtype=torch.long).reshape(-1)
            if (rollout_depths.numel() != n or ((rollout_depths < 1) | (rollout_depths > slots)).any()
                    or rollout_depths[0] != slots):
                raise ValueError("RAD-OPD depths must match the response and start with full-depth prefill")
        signal = a.get('boundary_signal','js')
        probe_changes = None
        probe_cost = dict(token_loops=0,coda_token_positions=0,forwards=0,positions=0)
        inputs, first = response_ids[:, :-1], prompt_length-1
        if self.shared_replay:
            with (self.observer.stage('shared_student_teacher_replay',student_loops=slots,teacher_loops=k,response_tokens=n)
                  if self.observer else nullcontext()):
                student,teacher=self.student.shared_response_trace(inputs,slots,k,first)
        else:
            with (self.observer.stage("teacher_forward", max_loops=k, response_tokens=n)
                  if self.observer else nullcontext()), torch.no_grad(), evaluation_mode(self.teacher.model):
                teacher = self.teacher.response_trace(inputs, k, first, a["teacher_endpoint"], a["exit_threshold"])
        with torch.no_grad():
            exits = (teacher.exits if a['teacher_endpoint']=='native_adaptive' else self.teacher.native_exit_depths(teacher, a["exit_threshold"])
                     if a["teacher_endpoint"] == "native" else
                     torch.full((n,), k, device=inputs.device, dtype=torch.long))
        if dynamic and signal=='probe_know':
            if self.confidence_probe is None:
                raise ValueError('Prompt confidence requires tokenizer/probe instrumentation')
            measured, _, probe_cost = self.confidence_probe.measure(inputs, first, exits, signal)
            probe_changes = inputs.new_zeros((k-1,n),dtype=torch.float32)
            probe_changes[:measured.shape[0]] = measured
        stage_hist = torch.zeros(slots+1, device=inputs.device, dtype=torch.long)
        mapping = torch.zeros(slots, k, device=inputs.device, dtype=torch.long)
        js_sum = torch.zeros((), device=inputs.device)
        js_count = torch.zeros((), device=inputs.device, dtype=torch.long)
        signal_sum = torch.zeros((), device=inputs.device)
        target_depths = rollout_depths if ragged else torch.full((n,), slots, device=inputs.device, dtype=torch.long)
        plans = []
        with (self.observer.stage("stage_planning", response_tokens=n)
              if self.observer else nullcontext()), torch.no_grad():
            for start in range(0, n, self.chunk_size):
                end = min(start+self.chunk_size, n)
                selected_exits = exits[start:end]
                if dynamic:
                    need_logits = signal in {'max_probability','normalized_entropy'} or teacher.adjacent_js is None
                    if need_logits:
                        hidden = torch.stack([h[start:end] for h in teacher.states])
                        logq = self.teacher.head(hidden).log_softmax(-1)
                        js = adjacent_js_from_log_probs(logq)
                    else:
                        js = teacher.adjacent_js[:, start:end]
                    if signal == 'js':
                        changes, threshold = js, a['js_threshold']
                    elif probe_changes is not None:
                        changes, threshold = probe_changes[:,start:end], a.get('confidence_threshold',.05)
                    else:
                        changes, _ = confidence_changes(logq,signal)
                        threshold = a.get('confidence_threshold',.05)
                    plan = make_dynamic_stage_plan(changes, selected_exits, a["stage_selection"], threshold)
                    if need_logits:
                        del logq
                    target_depths[start:end] = plan.target_loops
                elif a["stage_selection"] == "endpoint":
                    targets = torch.full((slots, end-start), -1, device=exits.device, dtype=torch.long)
                    if ragged:
                        targets[target_depths[start:end]-1, torch.arange(end-start, device=exits.device)] = selected_exits
                    else:
                        targets[-1] = selected_exits
                    weights = (targets > 0).to(torch.float32)
                    plan = StagePlan(targets, weights, selected_exits, weights.new_zeros((k-1, end-start)))
                elif teacher.adjacent_js is not None:
                    plan = make_stage_plan_from_js(teacher.adjacent_js[:, start:end], selected_exits,
                        slots, a["stage_selection"], a["js_threshold"], a.get("count_matched_uniform", True),
                        a.get('shallow_target_placement','last'))
                else:
                    hidden = torch.stack([h[start:end] for h in teacher.states])
                    logq = self.teacher.head(hidden).log_softmax(-1)
                    plan = make_stage_plan(logq, selected_exits, slots, a["stage_selection"],
                        a["js_threshold"], a.get("count_matched_uniform", True),a.get('shallow_target_placement','last'))
                    del logq
                counts = plan.target_loops if dynamic else plan.counts
                if not dynamic and a.get('shallow_target_placement','last')=='prefix' and a['stage_selection']=='top_n':
                    target_depths[start:end]=((plan.teacher_depths>0)*
                        torch.arange(1,slots+1,device=exits.device)[:,None]).max(0).values
                stage_hist += torch.bincount(counts, minlength=slots+1)
                for slot in range(slots):
                    active = plan.teacher_depths[slot] > 0
                    mapping[slot] += torch.bincount(plan.teacher_depths[slot, active]-1, minlength=k)
                if dynamic or a["stage_selection"] != "endpoint":
                    valid = torch.arange(2, k+1, device=exits.device)[:, None] <= selected_exits[None]
                    js_sum += (js if dynamic else plan.adjacent_js).masked_select(valid).sum()
                    signal_sum += (changes if dynamic else plan.adjacent_js).masked_select(valid).sum()
                    js_count += valid.sum()
                plans.append((start, end, plan))
        replay_loops = int(target_depths.max()) if dynamic else slots
        if not self.shared_replay:
            with (self.observer.stage("student_replay", loops=replay_loops, response_tokens=n)
                  if self.observer else nullcontext()):
                if ragged:
                    from .ragged import replay_trace
                    student = replay_trace(self.student, inputs, replay_loops, first, rollout_depths,
                                           detach_reused=a.get("rad_detach_reused_kv", False))
                else:
                    student = self.student.response_trace(inputs, replay_loops, first)
        readout_span = self.observer.start("loss_readout", response_tokens=n) if self.observer else None
        stage_loss = student.states[0].new_zeros((), dtype=torch.float32)
        for start, end, full_plan in plans:
            plan = StagePlan(full_plan.teacher_depths[:replay_loops], full_plan.weights[:replay_loops],
                             full_plan.exits, full_plan.adjacent_js, full_plan.target_loops)
            th = torch.stack([h[start:end] for h in teacher.states])
            sh = torch.stack([h[start:end] for h in student.states])
            def chunk_loss(student_hidden, teacher_hidden, selected_plan=plan):
                if not dynamic and a["stage_selection"] == "endpoint":
                    columns = torch.arange(teacher_hidden.shape[1], device=teacher_hidden.device)
                    with torch.no_grad():
                        target = self.teacher.head(teacher_hidden[selected_plan.exits-1, columns]).log_softmax(-1)
                    predicted = self.student.head(student_hidden[-1]).log_softmax(-1)
                    return (predicted.exp()*(predicted-target)).sum()
                with torch.no_grad():
                    target = self.teacher.head(teacher_hidden).log_softmax(-1)
                return stage_reverse_kl(self.student.head(student_hidden), target, selected_plan).sum()
            part = (checkpoint(chunk_loss, sh, th, use_reentrant=False)
                    if self.checkpoint_readout and torch.is_grad_enabled() else chunk_loss(sh, th))
            stage_loss = stage_loss+part/n
        stop_loss = stage_loss.new_zeros(())
        if dynamic:
            if len(student.gates) != replay_loops:
                raise RuntimeError("Student stopping head was not included in replay")
            stop_loss = stage_stop_bce(torch.stack(student.gates), target_depths).mean()
        loss = stage_loss+a.get("stop_loss_weight", 1.)*stop_loss
        if self.observer:
            self.observer.finish(readout_span)
        target_hist = torch.bincount(target_depths, minlength=slots+1)[1:]
        confusion = torch.zeros(slots, slots, device=inputs.device, dtype=torch.long)
        extra = error = 0
        if rollout_depths is not None:
            actual = torch.as_tensor(rollout_depths, device=inputs.device, dtype=torch.long).reshape(-1)
            if actual.numel() != n or ((actual < 1) | (actual > slots)).any():
                raise ValueError("Invalid rollout depths for response tokens")
            confusion = torch.bincount((target_depths-1)*slots+actual-1, minlength=slots*slots).reshape(slots, slots)
            extra = int((target_depths-actual).clamp_min(0).sum())
            error = int((target_depths-actual).abs().sum())
        info = dict(response_tokens=n, exit_hist=torch.bincount(exits,minlength=k+1)[1:].detach(),
            stage_hist=stage_hist.detach(), mapping_hist=mapping.detach(), js_sum=js_sum.detach(),
            js_count=js_count.detach(), teacher_executed_loops=(None if a['teacher_endpoint']=='native_adaptive' or
                self.teacher.native_exit_reduces_executed_loops and a["teacher_endpoint"] == "native" else k),
            student_executed_loops=replay_loops,
            teacher_executed_token_loops=teacher.executed_token_loops,
            teacher_backfill_token_loops=teacher.backfill_token_loops,
            teacher_reused_token_loops=teacher.reused_token_loops,
            replay_unique_token_loops=teacher.executed_token_loops+student.executed_token_loops,
            student_executed_token_loops=student.executed_token_loops,
            teacher_coda_token_positions=teacher.executed_token_loops if self.teacher.has_coda else 0,
            endpoint_only_tokens=int(stage_hist[1]),
            exit_not_deeper_than_student=int((exits<=target_depths).sum()),
            stage_loss=stage_loss.detach(), stop_loss=stop_loss.detach(),
            target_depth_hist=target_hist.detach(), target_actual_confusion=confusion.detach(),
            required_response_stage_loops=int(target_depths.sum()),
            replay_additional_required_loops=extra, stop_depth_absolute_error=error,
            uncompressed_tokens=int((target_depths==exits).sum()) if dynamic else 0,
            target_at_max_tokens=int((target_depths==slots).sum()) if dynamic else 0)
        info.update(boundary_signal_sum=signal_sum.detach(),confidence_probe_token_loops=probe_cost['token_loops'],
                    confidence_probe_coda_token_positions=probe_cost['coda_token_positions'],confidence_probe_forwards=probe_cost['forwards'])
        return loss, info
