"""Audited external evaluation records, per-benchmark metrics and exact overlaps."""
from collections import Counter
from math import comb
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .data import normalize_record


def ouro_evaluation_group(policy, loops):
    if policy == 'fixed':
        return 'Dense 4-Loop' if loops == 4 else f'Static {loops}-Loop'
    return {'native': '原生 Ouro KV', 'ragged': 'Ragged KV'}[policy]


def load_evaluation_records(path, expected_examples=None, suite=None):
    path = Path(path)
    data = path.read_bytes()
    records = [normalize_record(json.loads(line), i) for i, line in enumerate(data.splitlines()) if line.strip()]
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Unique nonempty evaluation records required')
    if expected_examples is not None and len(records) != expected_examples:
        raise ValueError(f'Evaluation contains {len(records)} questions; expected {expected_examples}')
    counts = dict(Counter(r.get('benchmark', 'unspecified') for r in records))
    if suite == 'aime':
        if any(name not in {'AIME2024','AIME2025','AIME2026'} or count != 30 for name,count in counts.items()):
            raise ValueError('AIME evaluation requires all 30 questions of each selected benchmark')
        if any(not str(r['answer']).isdigit() or not 0 <= int(r['answer']) <= 999 for r in records):
            raise ValueError('AIME ground truths must be integers in 0..999')
    return records, dict(path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest(),
                         examples=len(records),benchmarks=counts)


def benchmark_math_metrics(records, correctness):
    from .inference import math_metrics
    groups = {}
    for record in records:
        groups.setdefault(record.get('benchmark','unspecified'), {})[record['id']] = correctness[record['id']]
    return {name:math_metrics(group) for name,group in groups.items()}


def rad_pass_metrics(correctness):
    """Mean per-problem pass@1 and the standard without-replacement pass@4."""
    counts = [(len(samples), sum(samples)) for samples in correctness.values()]
    pass_one = sum(correct/total for total,correct in counts)/len(counts)
    pass_four = (sum(1 - comb(total-correct, 4)/comb(total, 4) for total,correct in counts)/len(counts)
                 if all(total >= 4 for total, _ in counts) else None)
    return dict(p_at_1=pass_one, p_at_4=pass_four)


def exclude_evaluation_overlaps(records, evaluation):
    """Remove complete benchmark problem text contained in a training prompt.

    Whitespace, Unicode compatibility and redundant left/right commands are
    normalized. This checks text identity, not semantic or near-duplicate overlap.
    """
    def normalized(text):
        return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).replace(r'\left','').replace(r'\right','')
    needles = [(r['id'],normalized(r['problem'])) for r in evaluation]
    if any(not text for _,text in needles):
        raise ValueError('Evaluation problem text must be nonempty')
    kept, removed = [], []
    for record in records:
        text = normalized('\n'.join(m['content'] for m in record['messages']))
        matches = [ident for ident,problem in needles if problem in text]
        if matches: removed.append(dict(train_id=record['id'],evaluation_ids=matches))
        else: kept.append(record)
    return kept, dict(method='normalized_problem_substring_v1',removed=len(removed),matches=removed)
