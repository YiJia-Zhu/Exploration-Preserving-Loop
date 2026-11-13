"""Convert a specified local DAPO release to audited text JSONL; no downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ep_loop.data import normalize_record


def source_rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError("JSON source must be a list of examples")
        yield from value
    else:
        raise ValueError(f"Unsupported source format: {path}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-examples", type=int, default=17398)
    p.add_argument("--deduplicate", action="store_true", help="Explicitly deduplicate identical message lists; conflicting answers are errors")
    p.add_argument("--source-repo", default="unspecified")
    p.add_argument("--source-revision", default="unspecified")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records, seen, source_count = [], {}, 0
    for path in args.input:
        for row in source_rows(path):
            record = normalize_record(row, source_count)
            source_count += 1
            key = record["problem_sha256"]
            if args.deduplicate and key in seen:
                if seen[key]["answer"] != record["answer"]:
                    raise ValueError(f"Conflicting answers for the same problem: {key}")
                continue
            records.append(record)
            seen[key] = record
    if len(records) != args.expected_examples:
        raise ValueError(f"Obtained {len(records)} examples, expected {args.expected_examples}. Source release/subset identity must be resolved explicitly.")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate IDs; resolve them at the source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False)+"\n")
    report = dict(source_repo=args.source_repo, source_revision=args.source_revision,
        sources=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in args.input],
        source_rows=source_count, output_examples=len(records), deduplicate=args.deduplicate,
        output_sha256=sha256(args.output), original_prompt_messages_preserved=True,
        paper_dataset_identity_verified=False,
        scope="Row count and source hashes do not by themselves prove the exact paper training subset")
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
