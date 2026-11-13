"""Validate EP materials and recorded pilot artifacts offline; never runs a model."""

from collections import Counter
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "audit/package_validation.json"
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(relative_path):
    path = (ROOT / relative_path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside package: {relative_path}")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_title(text):
    return "".join(re.findall(r"[a-z0-9]+", text.casefold()))


def verify_historical(report):
    for artifact in report["artifacts"]:
        candidates = [inside(prefix + artifact["path"]) for prefix in (
            "", "audit/dynamic_student_20261113/before/", "audit/proposal_sync_20261113/before/")]
        require(any(path.is_file() and digest(path) == artifact["sha256"] for path in candidates),
                f"Historical artifact changed or missing: {artifact['path']}")


def main():
    runtime_path = ROOT / "audit/runtime_validation.json"
    runtime = json.loads(runtime_path.read_text())
    require(runtime["status"] == "PASS", "Pilot verification has not passed")
    require(runtime["experiments_run"] is True, "Missing model execution evidence")
    for artifact in runtime["artifacts"]:
        path = inside(artifact["path"])
        require(path.is_file() and digest(path) == artifact["sha256"],
                f"Pilot artifact changed or missing: {path}")
    historical_path = ROOT / "audit/implementation_20261113/validation.json"
    historical = json.loads(historical_path.read_text())
    verify_historical(historical)
    previous_path = ROOT / "audit/proposal_sync_20261113/validation.json"
    previous = json.loads(previous_path.read_text())
    verify_historical(previous)
    implementation_path = ROOT / "audit/dynamic_student_20261113/validation.json"
    implementation = json.loads(implementation_path.read_text())
    require(implementation["status"] == "PASS", "Current implementation validation failed")
    for artifact in implementation["artifacts"]:
        path = inside(artifact["path"])
        require(path.is_file() and digest(path) == artifact["sha256"],
                f"Implementation artifact changed or missing: {path}")
    manifests = [ROOT / "papers/manifest.json", ROOT / "ref/manifest.json"]
    groups = [json.loads(path.read_text()) for path in manifests]
    papers = [paper for group in groups for paper in group["papers"]]
    require(len(groups[0]["papers"]) == 6, "Expected six core papers")
    require(len(groups[1]["papers"]) == 17, "Expected seventeen related papers")
    require(len({p["version"] for p in papers}) == len(papers), "Duplicate papers")
    require(all(p["status"] == "ready" for p in papers), "Incomplete paper material")
    require(all(group["experiments_run"] is True for group in groups), "Outdated package execution status")
    require(all(group["training_implementation_included"] is True for group in groups), "Outdated implementation status")

    for folder, group in zip(["papers", "ref"], groups):
        actual = {str(p.relative_to(ROOT)) for p in (ROOT / folder).glob("*.pdf")}
        expected = {p["pdf_file"] for p in group["papers"]}
        require(actual == expected, f"Unindexed or missing PDFs in {folder}")

    hash_fields = [
        ("pdf_file", "pdf_sha256"),
        ("source_html", "source_html_sha256"),
        ("source_text", "source_text_sha256"),
        ("source_abstract", "source_abstract_sha256"),
        ("pdf_first_page", "pdf_first_page_sha256"),
    ]
    for paper in papers:
        for file_key, hash_key in hash_fields:
            path = inside(paper[file_key])
            require(path.is_file(), f"Missing {path}")
            require(digest(path) == paper[hash_key], f"Hash mismatch: {path}")
        pdf = inside(paper["pdf_file"])
        require(pdf.read_bytes().startswith(b"%PDF-"), f"Invalid PDF: {pdf}")
        require(pdf.stat().st_size == paper["pdf_bytes"], f"Size mismatch: {pdf}")
        require(paper["pdf_pages"] > 0, f"No pages: {pdf}")
        first_page = inside(paper["pdf_first_page"]).read_text()
        require(paper["version"] in "".join(first_page.split()), f"PDF version mismatch: {pdf}")
        require(normalized_title(paper["title"]) in normalized_title(first_page), f"PDF title mismatch: {pdf}")
        source = inside(paper["source_text"]).read_text()
        require("arXiv:" + paper["version"] in source, f"Source version mismatch: {pdf}")
        for locator in paper["evidence_locators"]:
            lines = [i for i, line in enumerate(source.splitlines(), 1)
                     if locator["marker"].casefold() in line.casefold()]
            require(lines == locator["matching_lines"] and bool(lines),
                    f"Evidence locator mismatch: {paper['version']}: {locator['marker']}")

    docs = sorted((ROOT / "docs").glob("*.md"))
    require({"01_idea.md", "02_experiment_plan.md", "03_paper_notes.md", "04_insights.md",
             "05_related_work.md", "06_loopopd_motivation_evidence.md", "07_pilot_results_20261108.md",
             "08_implementation_and_reproduction.md", "loop_distillation_research_proposal_zh.md"}
            <= {p.name for p in docs}, "Missing current proposal or topic documents")
    markdown = [ROOT / "README.md", *docs, *sorted((ROOT / "experiments").glob("*.md")),
                ROOT / "ref/README.md"]
    local_count, missing, outside = 0, [], []
    indexed_versions = {p["version"] for p in papers}
    references_path = ROOT / "experiments/proposal_references_20261113.json"
    references = json.loads(references_path.read_text())
    require(digest(inside(references["proposal"])) == references["proposal_sha256"],
            "Proposal changed; synchronize reference registration and implementation")
    proposal_versions = {p["version"] for p in references["references"]}
    for path in markdown:
        text = path.read_text()
        require(text.count("```") % 2 == 0, f"Unclosed fence: {path}")
        versions = set(re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}v\d+)", text))
        require(versions <= indexed_versions | proposal_versions, f"Unregistered fixed-version link: {path}")
        for url in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if url.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = (path.parent / url.split("#", 1)[0]).resolve()
            local_count += 1
            if not target.is_relative_to(ROOT):
                outside.append({"file": str(path.relative_to(ROOT)), "target": url})
            elif not target.exists() and target != REPORT:
                missing.append({"file": str(path.relative_to(ROOT)), "target": url})
    require(not missing, f"Missing local links: {missing}")
    require(not outside, f"Links outside package: {outside}")
    # Runtime environments legitimately contain interpreter symlinks. Keep the
    # self-contained evidence check on owned materials, not .venv or downloads.
    for folder in ("papers", "ref", "docs", "experiments", "scripts", "ep_loop", "tests"):
        require(not any(p.is_symlink() for p in (ROOT/folder).rglob("*")),
                f"Owned materials should not depend on symlinks: {folder}")

    matrix = ROOT / "experiments/experiment_matrix.csv"
    with matrix.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        require(all(None not in row for row in rows), "Malformed CSV rows")
    ids = {row["id"] for row in rows}
    require(ids == set("ABCDEF"), "Incomplete current A-F experiment matrix")
    require(len(ids) == len(rows), "Duplicate experiment IDs")
    from ep_loop.config import load_config, validate_config
    for row in rows:
        config = validate_config(load_config(inside(row["config"])))
        for key in ("teacher_update", "stage_selection"):
            require(config["algorithm"][key] == row[key], f"Matrix/config mismatch: {row['id']} {key}")
        policy = config["algorithm"].get("student_policy", "fixed")
        require(policy == row["student_policy"], "Student policy mismatch")
        if policy == "fixed":
            require(config["algorithm"]["student_loops"] == int(row["student_budget"]), "Student budget mismatch")
        else:
            require(config["algorithm"]["student_loops"] is None, "Dynamic student must not specify a fixed budget")
        require(config["algorithm"]["teacher_max_loops"] == int(row["teacher_max_loops"]), "Teacher cap mismatch")
        require(row["status"] in {"cpu_validation", "cpu_verified_fixed_baseline", "inference_baseline"}, "Unverified benchmark status")
        require(inside(row["run_evidence"]).is_file(), f"Missing implementation evidence: {row['id']}")

    huginn_matrix = ROOT / "experiments/huginn_experiment_matrix.csv"
    with huginn_matrix.open(newline="") as stream:
        huginn_rows = list(csv.DictReader(stream))
    require(len(huginn_rows) == 6 and {row["id"] for row in huginn_rows} == ids,
            "Incomplete Huginn A-F matrix")
    for row in huginn_rows:
        config = validate_config(load_config(inside(row["config"])))
        require(config["model"]["family"] == "huginn", "Wrong native backend in Huginn matrix")
        for key in ("teacher_update", "stage_selection"):
            require(config["algorithm"][key] == row[key], f"Huginn matrix mismatch: {row['id']} {key}")
        policy = config["algorithm"].get("student_policy", "fixed")
        require(policy == row["student_policy"], "Huginn student policy mismatch")
        if policy == "fixed":
            require(config["algorithm"]["student_loops"] == int(row["student_budget"]), "Huginn budget mismatch")
        else:
            require(config["algorithm"]["student_max_loops"] == 32, "Huginn dynamic cap mismatch")
        require(config["algorithm"]["teacher_max_loops"] == int(row["teacher_max_loops"]) == 32, "Huginn cap mismatch")
        require(inside(row["run_evidence"]).is_file(), f"Missing Huginn evidence: {row['id']}")

    source_files = sorted({*ROOT.glob("scripts/*.py"), *ROOT.glob("ep_loop/*.py"), *ROOT.glob("tests/*.py")})
    artifacts = [*manifests, matrix, huginn_matrix, historical_path, previous_path, *markdown, *source_files, ROOT/"requirements-pilot.txt",
                 runtime_path, ROOT/"audit/model_download.json", ROOT/"audit/pilot_environment.json",
                 ROOT/"audit/objective_tests.log", references_path, implementation_path,
                 *sorted((ROOT/"configs").glob("*.json")), ROOT/"requirements-stage.txt"]
    result = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(ROOT),
        "papers": len(papers),
        "core_papers": len(groups[0]["papers"]),
        "related_papers": len(groups[1]["papers"]),
        "pdf_hashes_verified": len(papers),
        "pdf_first_page_versions_and_titles_verified": len(papers),
        "primary_source_snapshots_verified": len(papers),
        "experiment_documents": len(docs),
        "markdown_files_checked": len(markdown),
        "local_links_checked": local_count,
        "missing_local_links": missing,
        "links_outside_package": outside,
        "designed_groups": len(rows),
        "groups_by_status": dict(Counter(row["status"] for row in rows)),
        "matrix_configuration_alignment_checked": True,
        "huginn_designed_groups": len(huginn_rows),
        "historical_implementation_artifact_hashes_verified": len(historical["artifacts"]),
        "previous_fixed_budget_artifact_hashes_verified": len(previous["artifacts"]),
        "experiments_run": True,
        "training_implementation_included": True,
        "runtime_artifact_hashes_verified": len(runtime["artifacts"]),
        "current_implementation_artifact_hashes_verified": len(implementation["artifacts"]),
        "current_proposal_benchmark_scores_reproduced": False,
        "artifacts": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in artifacts],
        "scope": "Material/link integrity, recorded pilot and current implementation artifact hashes; no model rerun or benchmark efficacy claim.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
