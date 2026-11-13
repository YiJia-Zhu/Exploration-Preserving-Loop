"""Audit current saved CPU checks and sources without launching a model."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/proposal_sync_20261113"
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    from ep_loop.config import assets_status, load_config, validate_config
    proposal = ROOT / "docs/loop_distillation_research_proposal_zh.md"
    assert digest(proposal) == "43776668d345ac50ac5136569d9c2a455437090bea38f1915ed8f61e9008be9e"
    log = (AUDIT / "unit_tests.log").read_text()
    count = re.search(r"Ran (\d+) tests", log)
    assert count and int(count[1]) >= 32 and "\nOK\n" in log and "FAILED" not in log
    cases = read(AUDIT / "cpu_matrix.json")
    assert len(cases) == len({case["config"] for case in cases}) == 16
    evidence = list(AUDIT.glob("*.json")) + list(AUDIT.glob("*.log")) + list(AUDIT.glob("*.py"))
    for case in cases:
        path = ROOT / case["run"]
        summary, config = read(path / "summary.json"), read(path / "resolved_config.json")
        validate_config(load_config(ROOT / case["config"]))
        rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        assert summary["smoke"] and summary["status"] == "completed" and len(rows) == summary["completed_updates"] == 2
        a = config["algorithm"]
        for row in rows:
            assert row["responses"] == 4 and row["gradient_norm"] > 0
            assert sum(row["exit_hist"]) == sum(row["stage_count_hist"]) == row["response_tokens"]
            assert sum(row["teacher_depth_by_student_slot"][-1]) == row["response_tokens"]
            assert row["teacher_weight_version"] == (row["update"] - 1 if a["teacher_update"] == "current" else "initial")
            if config["model"]["family"] == "huginn":
                assert row["teacher_coda_token_positions"] == row["teacher_loss_executed_token_loops"]
                assert row["teacher_max_loops"] == 4
                assert row["teacher_executed_loops"] == (None if a["teacher_endpoint"] == "native" else 4)
            else:
                assert row["teacher_executed_loops"] == 4 and row["teacher_coda_token_positions"] == 0
        for name in ("model.safetensors", "training_state.pt", "rng_rank0.pt"):
            assert (path / "checkpoints/update_000002" / name).is_file()
        evidence.extend(path / name for name in ("resolved_config.json", "summary.json", "metrics.jsonl", "provenance.json", "rollouts_rank0.jsonl"))
        evidence.extend((path / "code").rglob("*.py"))
    resume = read(AUDIT / "huginn_resume.json")
    assert resume["status"] == "PASS" and len(resume["cases"]) == 2
    assert all(case["checkpoint_tensors_exact"] == 76 and case["metrics_and_rollouts_exact"] for case in resume["cases"])
    ddp_path = ROOT / "experiments/runs/20261113_proposal_huginn_ddp_cpu"
    ddp = read(ddp_path / "summary.json")
    assert ddp["status"] == "completed" and ddp["world_size"] == ddp["completed_updates"] == 2
    for rank in range(2):
        assert (ddp_path / f"checkpoints/update_000002/rng_rank{rank}.pt").is_file()
    evidence.extend(ddp_path.glob("*.json*"))
    checkpoint_ddp = read(AUDIT / "huginn_ddp_checkpoint.json")
    assert checkpoint_ddp["status"] == "PASS" and checkpoint_ddp["all_reduced_gradients_match_exactly"]
    pretrained = read(AUDIT / "pretrained_huginn.json")
    assert pretrained["status"] == "PASS" and pretrained["pretrained_weights_loaded"] and pretrained["official_fixed_logits_exact"]
    assets = {family: assets_status(validate_config(load_config(ROOT / name)), ROOT) for family, name in (
        ("ouro", "configs/d_current_native_js.json"), ("huginn", "configs/huginn_d_current_native_js.json"))}
    (AUDIT / "assets_preflight.json").write_text(json.dumps(assets, ensure_ascii=False, indent=2) + "\n")
    evidence.append(AUDIT / "assets_preflight.json")
    sources = [*ROOT.glob("ep_loop/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]
    for source in sources:
        compile(source.read_text(), str(source), "exec")
    material = [ROOT / "README.md", *ROOT.glob("docs/*.md"), ROOT / "ref/README.md",
                *ROOT.glob("experiments/*.md"), *ROOT.glob("experiments/*experiment_matrix.csv"),
                *ROOT.glob("configs/*.json")]
    evidence = [p for p in evidence if p.name != "validation.json"]
    report = dict(status="PASS", checked_at=datetime.now(timezone.utc).isoformat(),
        source_proposal_sha256=digest(proposal), authoritative_proposal_modified=False,
        unit_tests_passed=int(count[1]), cpu_training_configurations=cases,
        python_sources_compiled=len(sources), huginn_reference_budgets_tested={"teacher":32,"students":[8,16]},
        huginn_resume_exact=True, huginn_cpu_ddp_world_size=2,
        huginn_backbone_and_readout_checkpoint_ddp_passed=True,
        pretrained_huginn_short_cpu_forward_passed=True,
        paper_benchmark_scores_reproduced=False, new_method_efficacy_established=False,
        formal_training_assets_ready={family:value["ready"] for family,value in assets.items()},
        bod_checkpoint_identity_confirmed=False,
        scope="Native execution/loss semantics; no pretrained training or benchmark efficacy claim. Per-run code snapshots preserve actually executed versions.",
        artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in sorted(set(evidence+sources+material))])
    (AUDIT / "validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"artifacts","cpu_training_configurations"}},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
