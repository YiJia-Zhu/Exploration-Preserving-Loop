"""Prepare pinned AIME 2024/2025/2026 questions; solutions never enter prompts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ep_loop.data import normalize_record
from ep_loop.evaluation_data import load_evaluation_records

# year, repository, revision, file, expected SHA256
SOURCES = [
    (2024,'HuggingFaceH4/aime_2024','2fe88a2f1091d5048c0f36abc874fb997b3dd99a','data/train-00000-of-00001.parquet','26139847601a5037c237d5928b195e7260ca8074cf4f264b794af42847f79ccf'),
    (2025,'opencompass/AIME2025','a6ad95f611d72cf628a80b58bd0432ef6638f958','aime2025-I.jsonl','b91b3c96f05d9635d2a0692b124ebe023c1ff59cb19c074275e6c4b349d0659e'),
    (2025,'opencompass/AIME2025','a6ad95f611d72cf628a80b58bd0432ef6638f958','aime2025-II.jsonl','16a2dcfbbf9db1b11f8a69a3ba5e4cac73e3641b19a37e2307e9c12240bbed5e'),
    (2026,'MathArena/aime_2026','d61f486805306e6e6b5913504c07763a99e1b112','data/train-00000-of-00001.parquet','ae72853349a46c5f08666feb42db2d09fb4ac4fd4768c726b29702df21c7768e'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,default=Path('/local_nvme/data/zhenning/huggingface_datasets'))
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/aime')
    p.add_argument('--years',default='2024,2025,2026')
    p.add_argument('--download',action='store_true',help='Download missing small source files at the pinned revisions')
    args = p.parse_args()
    years = [int(y.strip()) for y in args.years.split(',')]
    if not years or len(years)!=len(set(years)) or not set(years)<={2024,2025,2026}:
        p.error('Select distinct years from 2024,2025,2026')
    records, sources, answer_normalizations = [], [], []
    for year,repo,revision,name,sha in SOURCES:
        if year not in years: continue
        path = args.cache_dir/f'AIME{year}'/revision/Path(name).name
        if not path.exists():
            if not args.download: raise FileNotFoundError(f'{path}; use --download to fetch pinned sources')
            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
            with urllib.request.urlopen(url,timeout=60) as response: raw=response.read()
            if hashlib.sha256(raw).hexdigest()!=sha: raise ValueError(f'Source hash mismatch: {repo}/{name}')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream: stream.write(raw)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha: raise ValueError(f'Cached source hash mismatch: {path}')
        if path.suffix=='.parquet':
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
        else: rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        sources.append(dict(benchmark=f'AIME{year}',repo=repo,revision=revision,file=name,sha256=sha,examples=len(rows)))
        for index,row in enumerate(rows):
            if 'year' in row and int(row['year'])!=year: raise ValueError('AIME source year mismatch')
            question = row.get('problem',row.get('question'))
            if not isinstance(question,str) or not question.strip(): raise ValueError('Empty AIME problem')
            ident = f'AIME{year}_{path.stem}_{index+1:02d}'
            raw_answer = str(row['answer']).strip()
            match = re.fullmatch(r'([0-9]{1,3})(?:\^\\circ)?',raw_answer)
            if match is None: raise ValueError(f'Noninteger AIME ground truth: {ident}: {raw_answer}')
            answer = str(int(match[1]))
            if answer != raw_answer:
                answer_normalizations.append(dict(id=ident,source_answer=raw_answer,answer=answer))
            # Only the problem and boxed-answer instruction enter messages.
            record = normalize_record(dict(id=ident,benchmark=f'AIME{year}',problem=question,answer=answer))
            records.append(record)
    counts = Counter(r['benchmark'] for r in records)
    if counts != {f'AIME{year}':30 for year in years}: raise ValueError(f'Unexpected AIME counts: {counts}')
    if len({r['problem_sha256'] for r in records})!=len(records): raise ValueError('Duplicate AIME problems')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    outputs = {'eval.jsonl':records,**{f'aime{year}.jsonl':[r for r in records if r['benchmark']==f'AIME{year}'] for year in years}}
    hashes = {}
    for name,rows in outputs.items():
        data = ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode()
        path = args.output_dir/name
        if path.exists() and path.read_bytes()!=data: raise FileExistsError(f'Different existing dataset: {path}')
        if not path.exists(): path.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    load_evaluation_records(args.output_dir/'eval.jsonl',30*len(years),'aime')
    report = dict(sources=sources,years=years,examples=len(records),benchmarks=dict(counts),output_sha256=hashes,
                  answer_normalizations=answer_normalizations,
                  prompt='Original problem + step-by-step and boxed-answer instruction; no source solutions',
                  paper_dataset_artifact_identity_verified=False)
    manifest = args.output_dir/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=report: raise FileExistsError(manifest)
    if not manifest.exists(): manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
