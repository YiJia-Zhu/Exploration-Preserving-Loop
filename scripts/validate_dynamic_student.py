"""Audit saved dynamic-student checks and current sources without model execution."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/dynamic_student_20261113"
sys.path.insert(0,str(ROOT))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from ep_loop.config import DEFAULT_CONFIG, assets_status, load_config, validate_config
    proposal = ROOT / "docs/loop_distillation_research_proposal_zh.md"
    assert digest(proposal) == digest(AUDIT / "before/docs/loop_distillation_research_proposal_zh.md")
    assert digest(proposal) == read(ROOT / "experiments/proposal_references_20261113.json")["proposal_sha256"]
    default = validate_config(load_config(DEFAULT_CONFIG))
    assert default["algorithm"]["student_policy"] == "learned" and default["algorithm"]["stage_selection"] == "threshold"
    log = (AUDIT / "unit_tests.log").read_text()
    count = re.search(r"Ran (\d+) tests",log)
    assert count and int(count[1]) >= 43 and "\nOK\n" in log and "FAILED" not in log
    cases = read(AUDIT / "cpu_matrix.json")
    assert len(cases) == 27 and len({row["run"] for row in cases}) == 27
    evidence = [*AUDIT.glob("*.json"),*AUDIT.glob("*.jsonl"),*AUDIT.glob("*.log"),*AUDIT.glob("*.py")]
    for case in cases:
        path = ROOT / case["run"]
        formal = validate_config(load_config(ROOT / case["config"]))
        resolved,summary = read(path / "resolved_config.json"),read(path / "summary.json")
        rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        assert summary["smoke"] and summary["status"] == "completed" and summary["completed_updates"] == len(rows) == 2
        dynamic = formal["algorithm"].get("student_policy", "fixed") == "learned"
        for row in rows:
            tokens = row["response_tokens"]
            assert row["responses"] == 4 and row["gradient_norm"] > 0
            assert sum(row["exit_hist"]) == sum(row["stage_count_hist"]) == tokens
            assert sum(row["target_student_depth_hist"]) == sum(row["actual_student_depth_hist"]) == tokens
            assert sum(map(sum,row["target_actual_depth_confusion"])) == tokens
            assert row["teacher_weight_version"] == (row["update"]-1 if formal["algorithm"]["teacher_update"] == "current" else "initial")
            if dynamic:
                assert row["student_executed_loops"] is None and row["mean_stop_loss"] > 0
                assert row["mean_target_student_depth"] <= row["student_max_loops"]
                count_targets = sum((i+1)*n for i,n in enumerate(row["target_student_depth_hist"]))
                assert abs(row["mean_target_student_depth"]-count_targets/tokens) < 1e-12
                mapping_count = sum(map(sum,row["teacher_depth_by_student_slot"]))
                assert mapping_count == (tokens if formal["algorithm"]["stage_selection"] == "threshold_endpoint" else count_targets)
            else:
                assert row["mean_stop_loss"] == 0 and row["student_executed_loops"] == resolved["algorithm"]["student_loops"]
            if "one_stage" in case["config"]:
                assert row["target_student_depth_hist"][0] == tokens
            if "all_stages" in case["config"]:
                assert row["target_student_depth_hist"] == row["exit_hist"]
        for name in ("model.safetensors","training_state.pt","rng_rank0.pt"):
            assert (path / "checkpoints/update_000002" / name).is_file()
        evidence.extend(path / name for name in ("summary.json","resolved_config.json","metrics.jsonl","provenance.json","rollouts_rank0.jsonl"))
        evidence.extend((path / "code").rglob("*.py"))
    resume = read(AUDIT / "resume.json")
    assert resume["status"] == "PASS" and len(resume["cases"]) == 4
    assert all(row["non_timing_metrics_and_rollouts_exact"] for row in resume["cases"])
    for family in ("ouro","huginn"):
        ddp = read(AUDIT / f"{family}_ddp_gradient.json")
        assert ddp["status"] == "PASS" and ddp["all_reduced_gradients_match_exactly"]
        folder = ROOT / f"experiments/runs/20261113_dynamic_{family}_ddp_cpu"
        summary = read(folder / "summary.json")
        assert summary["status"] == "completed" and summary["world_size"] == summary["completed_updates"] == 2
        evidence.extend(folder.glob("*.json*"))
    forward = read(AUDIT / "pretrained_forward.json")
    assert forward["status"] == "PASS" and len(forward["cases"]) == 2
    assert all(row["official_fixed_prefix_equivalent"] for row in forward["cases"])
    evaluation = read(AUDIT / "evaluation_checks.json")
    assert evaluation["status"] == "PASS" and len(evaluation["cases"]) == 6
    sources = [*ROOT.glob("ep_loop/*.py"),*ROOT.glob("scripts/*.py"),*ROOT.glob("tests/*.py")]
    for path in sources:
        compile(path.read_text(),str(path),"exec")
    assets = {family:assets_status(validate_config(load_config(ROOT / "configs" / name)),ROOT)
              for family,name in (("ouro","e_dynamic_js.json"),("huginn","huginn_e_dynamic_js.json"))}
    (AUDIT / "assets_preflight.json").write_text(json.dumps(assets,ensure_ascii=False,indent=2)+"\n")
    material = [ROOT / "README.md",ROOT / "ref/README.md",*ROOT.glob("docs/*.md"),
                *ROOT.glob("configs/*.json"),*ROOT.glob("experiments/*.md"),
                *ROOT.glob("experiments/*experiment_matrix.csv"),ROOT / "experiments/proposal_references_20261113.json"]
    evidence = [p for p in evidence if p.name != "validation.json"]+[AUDIT / "assets_preflight.json"]
    report = dict(status="PASS",checked_at=datetime.now(timezone.utc).isoformat(),
        source_proposal_sha256=digest(proposal),authoritative_proposal_modified=False,
        default_config="configs/default.json",default_method="E: all JS-threshold stage ends plus learned stopping",
        unit_tests_passed=int(count[1]),python_sources_compiled=len(sources),cpu_training_configurations=cases,
        two_family_four_case_optimizer_rng_resume_exact=True,two_family_cpu_ddp_world_size=2,
        dynamic_checkpoint_gradients_reduced_exactly=True,saved_student_and_native_evaluations_passed=6,
        two_pretrained_short_cpu_forwards_passed=True,paper_benchmark_scores_reproduced=False,
        new_method_efficacy_established=False,bod_checkpoint_identity_confirmed=False,
        formal_assets_ready={family:status["ready"] for family,status in assets.items()},
        scope="Dynamic student execution/loss semantics only. Tiny CPU runs and short pretrained forwards do not prove task gain or speedup. Per-run code snapshots preserve actual executed sources.",
        artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in sorted(set(evidence+sources+material))])
    (AUDIT / "validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"artifacts","cpu_training_configurations"}},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
