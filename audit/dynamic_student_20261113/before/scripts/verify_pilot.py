"""Recompute pilot metrics from observations and optionally replay saved adapters."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "experiments/runs"


def read(p):
    return json.loads(Path(p).read_text())


def lines(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def check_metric(metric, case):
    assert bool(np.argmax(metric["candidate_logits"]) == case["answer_index"]) == metric["correct"]
    probabilities = np.asarray(metric["candidate_probabilities"])
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert abs(float(probabilities.sum())-metric["candidate_mass"]) < 1e-10
    assert abs(float(probabilities[case["answer_index"]])-metric["gold_probability"]) < 1e-10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-checkpoints", action="store_true")
    args = parser.parse_args()
    diagnostic = RUNS / "20261108_diagnostic_v2"
    scenes, rows = read(diagnostic/"inputs.json"), lines(diagnostic/"rows.jsonl")
    summary = read(diagnostic/"summary.json")
    assert len(scenes) == len(rows) == 64
    semantic = lambda s: json.dumps({k:s[k] for k in ("mapping","colors","person")},sort_keys=True)
    assert len({semantic(s) for s in scenes}) == 64
    assert read(diagnostic/"engineering.json")["passed"]
    for arm in ("native","frozen","control"):
        for kind in ("person_box","box_color","two_hop"):
            for scene,row in zip(scenes,rows):
                assert scene["fingerprint"] == row["fingerprint"]
                for metric in row[arm][kind]:
                    check_metric(metric,scene["cases"][kind])
            recomputed = [float(np.mean([r[arm][kind][j]["correct"] for r in rows])) for j in range(4)]
            assert recomputed == summary["accuracy"][arm][kind]
    rescues = lines(diagnostic/"rescue.jsonl")
    targets = [r for r in rescues if r["target_1_to_3"] and "shuffled" in r]
    for result in rescues:
        native = rows[result["id"]]["native"]["two_hop"]
        assert result["target_1_to_3"] == (not native[0]["correct"] and native[2]["correct"])
        if result["donor"] is not None:
            assert result["donor"] != result["id"]
            assert len(scenes[result["donor"]]["cases"]["two_hop"]["ids"]) == len(scenes[result["id"]]["cases"]["two_hop"]["ids"])
        for key in ("same","shuffled"):
            if key in result: check_metric(result[key],scenes[result["id"]]["cases"]["two_hop"])
    assert len(targets) == summary["rescue"]["targets"]
    for arm in ("same","shuffled"):
        assert float(np.mean([r[arm]["correct"] for r in targets])) == summary["rescue"][arm]
    # Paired composition damage, preserving correlations among three questions.
    damage = np.asarray([[int(r["native"][k][3]["correct"])-int(r["frozen"][k][3]["correct"]) for k in ("person_box","box_color","two_hop")] for r in rows])
    rng = np.random.default_rng(20261108)
    sampled = damage[rng.integers(0,64,size=(5000,64))].mean(axis=1)
    damage_stat = sampled[:,2]-np.maximum(sampled[:,0],sampled[:,1])
    means = damage.mean(axis=0)
    composition = dict(mean=float(means[2]-max(means[0],means[1])),ci95=np.quantile(damage_stat,[0.025,0.975]).tolist())
    training = {}
    for mode in ("loopopd","middle"):
        folder = RUNS / f"20261108_{mode}_pilot"
        config, train, steps, result = (read(folder/"config.json"), read(folder/"training_inputs.json"), lines(folder/"steps.jsonl"), read(folder/"summary.json"))
        assert len(steps) == config["steps"] == 12
        assert len({semantic(s) for s in train}) == 32
        assert not ({semantic(s) for s in train} & {semantic(s) for s in scenes})
        assert sum(s["response_tokens"] for s in steps) == result["response_tokens"]
        for step in steps:
            assert len(step["response_ids"]) == step["response_tokens"]
            assert np.isfinite([step["endpoint_kl"],step["preserve_kl"],step["grad_norm"]]).all()
            assert step["grad_norm"] > 0
            expected = step["endpoint_kl"] + (config["aux_weight"]*step["preserve_kl"] if mode=="middle" else 0)
            assert abs(expected-step["total_loss"]) < 1e-6
        for stage in ("before","after"):
            observations = read(folder/f"{stage}.json")
            assert len(observations["rows"]) == len(scenes)
            for scene,row in zip(scenes,observations["rows"]):
                assert row["fingerprint"] == scene["fingerprint"]
                for metric in row["loops"]: check_metric(metric,scene["cases"]["two_hop"])
            accuracy = [float(np.mean([r["loops"][j]["correct"] for r in observations["rows"]])) for j in range(4)]
            assert accuracy == result[stage]["accuracy"]
        assert result["teacher_unchanged_exact"] and result["initial_adapter_identity_exact"]
        training[mode] = dict(steps=len(steps),response_tokens=result["response_tokens"],teacher_unchanged_exact=True,
                              initial_adapter_identity_exact=True,after=result["after"],checkpoint_replay_exact=None)
    assert read(RUNS/"20261108_loopopd_pilot/before.json") == read(RUNS/"20261108_middle_pilot/before.json")
    if args.replay_checkpoints:
        import torch
        from ep_loop.runtime import install_lora, load_model, read_all_loops, setup
        setup(20261109)
        model, _, _ = load_model(ROOT/"models/Ouro-1.4B-Thinking")
        install_lora(model)
        case = scenes[0]["cases"]["two_hop"]
        ids = torch.tensor([case["ids"]],device="cuda")
        for mode in training:
            folder = RUNS/f"20261108_{mode}_pilot"
            state = torch.load(folder/"adapter.pt",map_location="cuda",weights_only=True)
            mismatch = model.load_state_dict(state,strict=False)
            assert not mismatch.unexpected_keys
            assert set(state) == {n for n,p in model.named_parameters() if p.requires_grad}
            vectors,_ = read_all_loops(model,ids,20261109)
            observed = read(folder/"after.json")["rows"][0]["loops"]
            exact = all(np.array_equal(vectors[j][case["candidate_ids"]],np.asarray(observed[j]["candidate_logits"])) for j in range(4))
            assert exact, mode
            training[mode]["checkpoint_replay_exact"] = True
    artifacts = []
    for name in ("20261108_diagnostic_v2","20261108_loopopd_pilot","20261108_middle_pilot"):
        artifacts += [p for p in (RUNS/name).rglob("*") if p.is_file()]
    result = dict(checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),status="PASS",
                  experiments_run=True,training_implementation_included=True,
                  diagnostic_scenes=64,diagnostic_questions=192,engineering_checks=14,
                  diagnostic_metrics_recomputed=True,rescue_targets=len(targets),
                  composition_selective_damage_loop4=composition,training=training,
                  claimed_scope="Local candidate-ranking and objective pilot; not complete EP or original-paper reproduction",
                  artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in sorted(artifacts)])
    (ROOT/"audit/runtime_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="artifacts"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
