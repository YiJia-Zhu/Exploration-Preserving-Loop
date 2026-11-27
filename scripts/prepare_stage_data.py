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
    p.add_argument("--expected-examples", type=int, default=17398,
                   help="Expected normalized examples before excluding evaluation overlaps")
    p.add_argument("--deduplicate", action="store_true", help="Explicitly deduplicate identical message lists; conflicting answers are errors")
    p.add_argument("--conflicting-answers", choices=["error", "drop"], default="error",
                   help="drop explicitly excludes every prompt with inconsistent labels; default: error")
    p.add_argument("--source-repo", default="unspecified")
    p.add_argument("--source-revision", default="unspecified")
    p.add_argument("--exclude-evaluation", type=Path,
                   help="Exclude prompts that contain the complete text of an evaluation question")
    args = p.parse_args()
    if args.conflicting_answers == 'drop' and not args.deduplicate:
        p.error('--conflicting-answers drop requires --deduplicate')
    if args.output.exists():
        raise FileExistsError(args.output)
    from ep_loop.observability import RunLogger
    with RunLogger(args.output.with_suffix('.events.jsonl'), args.output.with_suffix('.console.log'), 'prepare_data') as observer:
        with observer.stage('prepare_data_run', echo=True, sources=[str(p) for p in args.input]):
            prepare(args, observer)


def prepare(args, observer):
    records, seen, source_count, conflicts = [], {}, 0, {}
    for path in args.input:
        for row in observer.progress(source_rows(path), desc=f'Convert {path.name}', unit='row', leave=False):
            record = normalize_record(row, source_count)
            source_count += 1
            key = record["problem_sha256"]
            if args.deduplicate and key in seen:
                if seen[key]["answer"] != record["answer"]:
                    if args.conflicting_answers == 'error':
                        raise ValueError(f"Conflicting answers for the same problem: {key}")
                    conflicts.setdefault(key, set()).update([seen[key]['answer'], record['answer']])
                continue
            records.append(record)
            seen[key] = record
    unique_before_conflict_removal = len(records)
    records = [record for record in records if record['problem_sha256'] not in conflicts]
    if len(records) != args.expected_examples:
        raise ValueError(f"Obtained {len(records)} examples, expected {args.expected_examples}. Source release/subset identity must be resolved explicitly.")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate IDs; resolve them at the source")
    overlap = {}
    evaluation_path = getattr(args, 'exclude_evaluation', None)
    if evaluation_path is not None:
        from ep_loop.evaluation_data import exclude_evaluation_overlaps, load_evaluation_records
        evaluation, evaluation_data = load_evaluation_records(evaluation_path)
        records, details = exclude_evaluation_overlaps(records, evaluation)
        overlap = dict(evaluation_dataset=evaluation_data, evaluation_overlaps_dropped=details['removed'],
                       evaluation_overlap_method=details['method'], removed_evaluation_overlaps=details['matches'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False)+"\n")
    with observer.stage('source_hashes', echo=True):
        sources = [dict(path=str(p.resolve()),sha256=sha256(p)) for p in args.input]
    report = dict(source_repo=args.source_repo, source_revision=args.source_revision,
        sources=sources, observability=observer.metadata(),
        source_rows=source_count, output_examples=len(records), deduplicate=args.deduplicate,
        unique_prompts_before_conflict_removal=unique_before_conflict_removal,
        conflicting_answers_policy=args.conflicting_answers,
        conflicting_prompts_dropped=len(conflicts),
        conflicting_labels={key:sorted(values) for key,values in sorted(conflicts.items())},
        output_sha256=sha256(args.output), original_prompt_messages_preserved=True,
        paper_dataset_identity_verified=False,
        **overlap,
        scope="Row count and source hashes do not by themselves prove the exact paper training subset")
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
