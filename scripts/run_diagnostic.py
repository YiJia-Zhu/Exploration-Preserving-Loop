"""D0/D1/D3 pilot on independently generated two-hop scenes, not paper replay."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from ep_loop.runtime import (REVISION, Patch, append_json, author_stopped_forward,
    case_masks, depth, load_model, make_scenes, metrics, read_all_loops, setup, write_json)


def paired_interval(differences, seed=20261108):
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(d, (5000, len(d)), replace=True), axis=1)
    return dict(mean=float(d.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist(), n=len(d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/Ouro-1.4B-Thinking")
    p.add_argument("--out", required=True)
    p.add_argument("--n-scenes", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261108)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "rows.jsonl").exists():
        raise FileExistsError("Use a new run directory; observations cannot be overwritten")
    setup(args.seed)
    started = time.perf_counter()
    write_json(out / "config.json", dict(**vars(args), revision=REVISION,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        template="raw paper two-hop completion; no chat template",
        protocol="D0/D1/D3; candidate ranking, not free generation; no training",
        cache_acceptance="KL(full||cached)<0.01 and identical full-vocabulary argmax",
        bootstrap="5000 paired resamples over scenes; exploratory, unadjusted 95% intervals"))
    model, tokenizer, loading = load_model(args.model_dir)
    scenes = make_scenes(tokenizer, args.n_scenes, args.seed)
    write_json(out / "inputs.json", scenes)
    checks = []
    for scene in scenes[:2]:
        case = scene["cases"]["two_hop"]
        ids = torch.tensor([case["ids"]], device="cuda")
        native, _ = read_all_loops(model, ids, args.seed)
        for r in range(1, 5):
            stop = author_stopped_forward(model, ids, r, args.seed)
            check = dict(kind="native_readout_vs_stopped", scene=scene["id"], loop=r,
                         max_abs=float(np.abs(stop - native[r-1]).max()),
                         exact=bool(np.array_equal(stop, native[r-1])))
            checks.append(check)
            assert check["exact"], check
        with Patch(model, case_masks(case), (("Q", "q"),), (1,)):
            identity, _ = read_all_loops(model, ids, args.seed)
        checks.append(dict(kind="query_identity_patch", scene=scene["id"],
                           exact=bool(np.array_equal(native, identity))))
        assert checks[-1]["exact"]
        for r in (2, 4):
            with torch.inference_mode(), depth(model, r):
                full = model(input_ids=ids, use_cache=False, exit_at_step=r-1, logits_to_keep=1).logits[0,-1].float()
                pref = model(input_ids=ids[:,:-1], use_cache=True, exit_at_step=r-1, logits_to_keep=1)
                cached = model(input_ids=ids[:,-1:], past_key_values=pref.past_key_values,
                               use_cache=True, exit_at_step=r-1, logits_to_keep=1).logits[0,-1].float()
                lp, lq = full.log_softmax(-1), cached.log_softmax(-1)
                kl = float((lp.exp() * (lp-lq)).sum())
                same = bool(full.argmax() == cached.argmax())
                checks.append(dict(kind="cached_vs_full", scene=scene["id"], loop=r,
                                   kl=kl, same_top1=same, max_abs=float((full-cached).abs().max())))
                assert kl < 0.01 and same, checks[-1]
    write_json(out / "engineering.json", dict(passed=True, checks=checks, loading=loading))
    rows = []
    queries = []
    for scene in scenes:
        row = dict(id=scene["id"], fingerprint=scene["fingerprint"], native={}, frozen={}, control={})
        for kind, case in scene["cases"].items():
            ids = torch.tensor([case["ids"]], device="cuda")
            with Patch(model, case_masks(case), (), (), capture_loop=3) as capture:
                vec, _ = read_all_loops(model, ids, args.seed)
            row["native"][kind] = [metrics(x, case) for x in vec]
            if kind == "two_hop":
                queries.append({l: {"Q": v["Q"][:,:,-1:,:].cpu()} for l,v in capture.captured.items()})
            for arm, region in (("frozen", "q"), ("control", "cq")):
                with Patch(model, case_masks(case), (("Q", region),), (2,3,4)):
                    mod, _ = read_all_loops(model, ids, args.seed)
                assert np.array_equal(vec[0], mod[0]), "Intervention changed loop 1"
                row[arm][kind] = [metrics(x, case) for x in mod]
        rows.append(row)
        append_json(out / "rows.jsonl", row)
        if len(rows) % 8 == 0:
            print(json.dumps(dict(stage="diagnostic", done=len(rows), seconds=time.perf_counter()-started)), flush=True)
    # Fix donor assignments by length before examining rescue outcomes.
    groups = {}
    for i,s in enumerate(scenes):
        groups.setdefault(len(s["cases"]["two_hop"]["ids"]), []).append(i)
    donors = {i: group[(j+1)%len(group)] for group in groups.values() for j,i in enumerate(group) if len(group)>1}
    rescues = []
    for i, scene in enumerate(scenes):
        case = scene["cases"]["two_hop"]
        ids = torch.tensor([case["ids"]], device="cuda")
        result = dict(id=i, target_1_to_3=not rows[i]["native"]["two_hop"][0]["correct"] and rows[i]["native"]["two_hop"][2]["correct"], donor=donors.get(i))
        for label, source_id in (("same",i),("shuffled",donors.get(i))):
            if source_id is None:
                continue
            source = {l: {"Q": d["Q"].cuda()} for l,d in queries[source_id].items()}
            with Patch(model, case_masks(case), (("Q","q"),), (1,), source=source):
                v = author_stopped_forward(model, ids, 1, args.seed)
            result[label] = metrics(v, case)
        rescues.append(result)
        append_json(out / "rescue.jsonl", result)
    summary = dict(n=args.n_scenes, revision=REVISION, engineering_passed=True,
                   candidate_ranking_only=True, accuracy={}, effects={}, rescue={},
                   seconds=time.perf_counter()-started,
                   peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    for arm in ("native", "frozen", "control"):
        summary["accuracy"][arm] = {kind:[float(np.mean([r[arm][kind][j]["correct"] for r in rows])) for j in range(4)] for kind in ("person_box","box_color","two_hop")}
    for j in (1,2,3):
        for arm in ("frozen", "control"):
            summary["effects"][f"{arm}_minus_native_loop{j+1}"] = paired_interval([int(r[arm]["two_hop"][j]["correct"])-int(r["native"]["two_hop"][j]["correct"]) for r in rows])
        summary["effects"][f"native_loop{j+1}_minus_loop1"] = paired_interval([int(r["native"]["two_hop"][j]["correct"])-int(r["native"]["two_hop"][0]["correct"]) for r in rows])
    targets = [r for r in rescues if r["target_1_to_3"] and "shuffled" in r]
    summary["rescue"] = dict(targets=len(targets),
        same=float(np.mean([r["same"]["correct"] for r in targets])) if targets else None,
        shuffled=float(np.mean([r["shuffled"]["correct"] for r in targets])) if targets else None,
        full_set_same=float(np.mean([r["same"]["correct"] for r in rescues])),
        donor_rule="deterministic non-self rotation within equal token length; all inputs eligible")
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
