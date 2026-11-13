"""Audit completed CPU semantic runs; never starts a model or a training job."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/"audit/implementation_20261113"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    test_log = (AUDIT/"unit_tests.log").read_text()
    count = re.search(r"Ran (\d+) tests",test_log)
    assert count and "\nOK\n" in test_log and "FAILED" not in test_log
    cases = {
        "a_loopopd":"20261113_a_loopopd_cpu_smoke",
        "a_d_loopopd":"20261113_a_d_loopopd_cpu_smoke",
        "b_native_endpoint":"20261113_b_native_endpoint_cpu_smoke",
        "b_current_native_endpoint":"20261113_b_current_native_endpoint_cpu_smoke",
        "c_native_uniform":"20261113_c_native_uniform_cpu_smoke",
        "c_current_native_uniform":"20261113_c_current_native_uniform_cpu_smoke",
        "d_native_js":"20261113_stage_js_cpu_smoke",
        "d_current_native_js":"20261113_current_js_cpu_smoke",
    }
    checked = []
    run_files = []
    for name,folder in cases.items():
        path = ROOT/"experiments/runs"/folder
        summary,config = read(path/"summary.json"),read(path/"resolved_config.json")
        metrics = [json.loads(line) for line in (path/"metrics.jsonl").read_text().splitlines()]
        assert summary["status"] == "completed" and summary["smoke"] is True
        assert summary["completed_updates"] == len(metrics) == 2
        for row in metrics:
            assert row["responses"] == 4
            assert row["teacher_executed_loops"] == 4 and row["student_executed_loops"] == 2
            assert sum(row["exit_hist"]) == sum(row["stage_count_hist"]) == row["response_tokens"]
            assert sum(row["teacher_depth_by_student_slot"][-1]) == row["response_tokens"]
            expected_version = row["update"]-1 if config["algorithm"]["teacher_update"]=="current" else "initial"
            assert row["teacher_weight_version"] == expected_version
            assert row["gradient_norm"] > 0
        tiny = read(path/"checkpoints/update_000002/config.json")
        assert tiny["num_hidden_layers"] == 2 and tiny["hidden_size"] == 32 and tiny["vocab_size"] == 260
        checked.append(dict(config="configs/"+name+".json",run=str(path.relative_to(ROOT)),updates=2,
                            scope="Random tiny official Ouro architecture, CPU only"))
        run_files.extend([path/"summary.json",path/"resolved_config.json",path/"metrics.jsonl",path/"provenance.json"])
    ddp = ROOT/"experiments/runs/20261113_current_js_ddp_cpu_smoke"
    result = read(ddp/"summary.json")
    assert result["world_size"] == 2 and result["completed_updates"] == 2 and result["smoke"]
    for rank in range(2):
        assert (ddp/f"checkpoints/update_000002/rng_rank{rank}.pt").is_file()
    run_files.extend([ddp/"summary.json",ddp/"metrics.jsonl"])
    resume = read(AUDIT/"resume_check.json")
    assert resume["status"] == "PASS" and resume["all_checkpoint_tensors_exact"] and resume["next_update_loss_and_sampling_exact"]
    frozen_resume = read(AUDIT/"frozen_resume_check.json")
    assert frozen_resume["status"] == "PASS" and frozen_resume["all_checkpoint_tensors_exact"]
    checkpoint_ddp = read(AUDIT/"ddp_checkpoint_check.json")
    assert checkpoint_ddp["status"] == "PASS" and checkpoint_ddp["all_reduced_gradients_match_exactly"]
    assets = read(AUDIT/"assets_preflight.json")
    sources = sorted([*ROOT.glob("ep_loop/*.py"),*ROOT.glob("scripts/*.py"),*ROOT.glob("tests/*.py"),*ROOT.glob("configs/*.json")])
    evidence = [AUDIT/"unit_tests.log",AUDIT/"resume_check.json",AUDIT/"frozen_resume_check.json",
                AUDIT/"assets_preflight.json",AUDIT/"paper_alignment.json",AUDIT/"data_preparation_check.json",
                AUDIT/"ddp_checkpoint_check.json",AUDIT/"ddp_checkpoint_smoke.py"]
    report = dict(status="PASS",checked_at=datetime.now(timezone.utc).isoformat(),
        unit_tests_passed=int(count[1]),cpu_training_configurations=checked,
        distributed_cpu_world_size=2,optimizer_rng_checkpoint_resume_exact=True,
        ddp_native_and_readout_checkpointing_combined_passed=True,
        frozen_teacher_resume_keeps_initial_teacher=True,
        source_proposal_sha256=digest(ROOT/"docs/loop_distillation_research_proposal_zh.md"),
        full_training_assets_ready=assets["ready"],paper_benchmark_scores_reproduced=False,
        pretrained_model_performance_tested_in_this_change=False,
        scope="Algorithm/execution semantics only; random CPU models and historical pilot artifacts do not prove new-method efficacy",
        artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in sorted(set(sources+run_files+evidence))])
    (AUDIT/"validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="artifacts"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
