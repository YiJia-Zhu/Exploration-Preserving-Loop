"""Validate the research package and retained run artifacts without loading a model."""

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


def main():
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
    require(
        {"loop_distillation_research_proposal_zh.md", "runtime_timing.md"}
        <= {p.name for p in docs},
        "Missing proposal or timing report",
    )
    markdown = [ROOT / "README.md", *docs, *sorted((ROOT / "experiments").glob("*.md")),
                ROOT / "ref/README.md"]
    local_count, missing, outside = 0, [], []
    indexed_versions = {p["version"] for p in papers}
    references_path = ROOT / "experiments/proposal_references_20261113.json"
    references = json.loads(references_path.read_text())
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

    from ep_loop.config import load_config, validate_config

    configs = sorted((ROOT / "configs").glob("*.json"))
    for path in configs:
        validate_config(load_config(path))

    timing_path = ROOT / "audit/ep_d_ngram_launch_20261119_182732/phase_timing_10_steps.json"
    run = inside(json.loads(timing_path.read_text())["run"])
    require(run.is_relative_to(ROOT / "ep_loop_runs"), f"Unexpected run directory: {run}")
    for step in (5, 10):
        checkpoint = run / f"checkpoints/update_{step:06d}"
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
        require(all((checkpoint / shard).is_file() for shard in index["weight_map"].values()),
                f"Missing model shard at step {step}")
        with (run / f"evaluation/update_{step:06d}/aime2024.responses.jsonl").open() as stream:
            require(sum(1 for _ in stream) == 480, f"Incomplete AIME2024 results at step {step}")
    resume = run / "resume/update_000010"
    require((resume / "training_state.pt").is_file() and
            all((resume / f"rng_rank{rank}.pt").is_file() for rank in range(4)),
            "Incomplete step-10 resume state")

    result = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "papers": len(papers),
        "core_papers": len(groups[0]["papers"]),
        "related_papers": len(groups[1]["papers"]),
        "experiment_documents": len(docs),
        "markdown_files_checked": len(markdown),
        "local_links_checked": local_count,
        "configs_checked": len(configs),
        "evaluated_checkpoints": [5, 10],
        "responses_per_checkpoint": 480,
        "resume_checkpoint": 10,
        "scope": "Material integrity and artifact presence. No model execution.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
