"""Legacy 2026-11-08 LoRA pilot: LoopOPD vs fixed middle supervision.

This is an objective/engineering pilot, not the complete causal-selection EP.
For the current 2026-11-13 native-exit/JS plan use train_stage_distillation.py.
"""
import argparse
import datetime as dt
from pathlib import Path
import sys
import time
import json
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from ep_loop.runtime import (REVISION, append_json, install_lora, load_model,
    logits_at_loops, make_scenes, metrics, preserve_kl, read_all_loops,
    reverse_kl, sample_rollout, setup, teacher_mode, write_json)


@torch.no_grad()
def evaluate(model, scenes, seed):
    rows = []
    for scene in scenes:
        case = scene["cases"]["two_hop"]
        logits, _ = read_all_loops(model, torch.tensor([case["ids"]], device="cuda"), seed)
        rows.append(dict(id=scene["id"], fingerprint=scene["fingerprint"],
                         loops=[metrics(v, case) for v in logits]))
    summary = dict(n=len(rows), accuracy=[float(np.mean([r["loops"][j]["correct"] for r in rows])) for j in range(4)],
                   mean_entropy=[float(np.mean([r["loops"][j]["full_entropy"] for r in rows])) for j in range(4)],
                   mean_gold_probability=[float(np.mean([r["loops"][j]["gold_probability"] for r in rows])) for j in range(4)])
    return rows, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/Ouro-1.4B-Thinking")
    p.add_argument("--out", required=True)
    p.add_argument("--evaluation-inputs", required=True)
    p.add_argument("--mode", choices=["loopopd", "middle"], required=True)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--aux-weight", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20261109)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "steps.jsonl").exists():
        raise FileExistsError("Use a fresh run directory")
    for rel in ("ep_loop/runtime.py", "scripts/run_training_pilot.py"):
        destination = out / "code" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[1] / rel, destination)
    setup(args.seed)
    started = time.perf_counter()
    write_json(out / "config.json", dict(**vars(args), revision=REVISION,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        adaptation="q_proj/v_proj rank 8 alpha 16 FP32 LoRA; BF16 immutable backbone",
        teacher="adapters disabled; original checkpoint at four loops; no gradient",
        rollout="current two-loop student, temperature 1, no top-k/p, regenerated every update",
        loss="mean full-vocabulary reverse KL on response prefixes; optional fixed T2->S1 candidate+OTHER forward KL on first response prefix",
        limitation="No causal value gate. Not complete EP and not original full-parameter LoopOPD reproduction."))
    model, tokenizer, loading = load_model(args.model_dir)
    evaluation = json.loads(Path(args.evaluation_inputs).read_text())
    def semantic_key(scene):
        return json.dumps({k:scene[k] for k in ("mapping", "colors", "person")}, sort_keys=True)
    seen = {semantic_key(s) for s in evaluation}
    train = []
    for scene in make_scenes(tokenizer, 128, args.seed, exclude=[s["fingerprint"] for s in evaluation]):
        key = semantic_key(scene)
        if key not in seen:
            train.append(scene)
            seen.add(key)
        if len(train) == 32:
            break
    assert len(train) == 32
    assert not ({s["fingerprint"] for s in train} & {s["fingerprint"] for s in evaluation})
    write_json(out / "training_inputs.json", train)
    reference_ids = torch.tensor([train[0]["cases"]["two_hop"]["ids"]], device="cuda")
    with torch.no_grad():
        reference_teacher = logits_at_loops(model, reference_ids, 4, slice(-1,None))[-1].clone()
        reference_student = logits_at_loops(model, reference_ids, 2, slice(-1,None))[-1].clone()
    params = install_lora(model)
    with torch.no_grad():
        initial_student = logits_at_loops(model, reference_ids, 2, slice(-1,None))[-1]
    assert torch.equal(initial_student, reference_student), "Zero adapter changed native forward"
    before_rows, before = evaluate(model, evaluation, args.seed)
    write_json(out / "before.json", dict(summary=before, rows=before_rows))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    order = np.random.default_rng(args.seed).permutation(len(train)).tolist()
    rows = []
    for step in range(args.steps):
        step_start = time.perf_counter()
        case = train[order[step % len(order)]]["cases"]["two_hop"]
        prompt = torch.tensor([case["ids"]], device="cuda")
        full = sample_rollout(model, prompt, args.max_new_tokens, generator)
        n = full.shape[1] - prompt.shape[1]
        assert n > 0
        inputs = full[:,:-1]
        positions = slice(prompt.shape[1]-1, None)
        with teacher_mode(model):
            target = logits_at_loops(model, inputs, 4, positions)
        student = logits_at_loops(model, inputs, 2, positions)
        assert student[-1].shape[1] == n
        endpoint = reverse_kl(student[-1], target[-1])
        auxiliary = preserve_kl(student[0][:,:1], target[1][:,:1], case["candidate_ids"])
        loss = endpoint + (args.aux_weight * auxiliary if args.mode == "middle" else 0)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        assert torch.isfinite(grad_norm) and float(grad_norm) > 0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
        row = dict(step=step, train_index=order[step % len(order)], response_tokens=n,
                   response_ids=full[0,prompt.shape[1]:].tolist(),
                   response_text=tokenizer.decode(full[0,prompt.shape[1]:]),
                   endpoint_kl=float(endpoint.detach()), preserve_kl=float(auxiliary.detach()),
                   total_loss=float(loss.detach()), grad_norm=float(grad_norm),
                   seconds=time.perf_counter()-step_start)
        rows.append(row)
        append_json(out / "steps.jsonl", row)
        print(json.dumps({k:v for k,v in row.items() if k not in ("response_ids","response_text")}), flush=True)
        del student, target, loss, endpoint, auxiliary
    with teacher_mode(model):
        final_teacher = logits_at_loops(model, reference_ids, 4, slice(-1,None))[-1]
    assert torch.equal(reference_teacher, final_teacher), "Frozen teacher changed"
    after_rows, after = evaluate(model, evaluation, args.seed)
    write_json(out / "after.json", dict(summary=after, rows=after_rows))
    adapter = {name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}
    torch.save(adapter, out / "adapter.pt")
    summary = dict(mode=args.mode, steps=args.steps, revision=REVISION,
                   training_examples=len(train), evaluation_examples=len(evaluation),
                   response_tokens=sum(r["response_tokens"] for r in rows),
                   trainable_parameters=sum(p.numel() for p in params),
                   teacher_unchanged_exact=True, initial_adapter_identity_exact=True,
                   finite_nonzero_gradients_all_steps=True, before=before, after=after,
                   accuracy_delta=[b-a for a,b in zip(before["accuracy"],after["accuracy"])],
                   seconds=time.perf_counter()-started,
                   peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                   scope="One seed, 12-step default synthetic pilot; no proof of EP efficacy or necessary exploration")
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
