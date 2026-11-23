"""Check Phase 0 plans and scores without loading an inference model."""
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from scripts.reproduce_ragged_bbh import (
    CORE, SAMPLING, cases, collect_shards, generate_case, initialize_shard, paired_effect,
    pending_jobs, score_answer, summarize_case,
)
from ep_loop.fastpath.rlt_server import execution_token_loops


def test_strict_bbh_answer_extraction():
    assert score_answer('The answer is (B).\nThe answer is (A).', '(A)') == ('(A)', True)
    assert score_answer('So the answer is ] ) }', '] ) }') == ('] ) }', True)
    assert score_answer('The answer is true.', 'True') == ('true', False)
    assert score_answer('I select (A).', '(A)') == (None, False)


def test_figure_matrix_contains_core_without_duplicate_arms():
    names = {case['name'] for case in cases()}
    assert len(cases()) == len(names) == 14
    assert len(CORE) == 6 and set(CORE) <= names
    assert all(case['loops'] == 4 for case in cases() if case['endpoint'] == 'ragged')


def test_metrics_use_per_question_pass_and_exclude_prefill_depth():
    rows = [dict(id='q', correct=(sample == 2), exit_depths=[4, 1, 2], tokens=[5, 6, 7],
                 reached_eos=True, parsed_answer='True', executed_token_loops=11, seconds=.2)
            for sample in range(4)]
    report, questions = summarize_case(rows)
    assert questions == {'q': {'p_at_1': .25, 'p_at_4': 1.}}
    assert report['mean_decode_depth'] == 1.5
    assert report['depth_one_fraction'] == .5
    assert report['decode_depth_hist'] == {1: 4, 2: 4}

    effect = paired_effect(questions, {'q': dict(p_at_1=.5, p_at_4=1.)}, [{'id': 'q', 'task': 'a'}])
    assert effect['p_at_1'] == {'delta': -.25, 'ci95': [-.25, -.25]}
    assert effect['p_at_4_minus_p_at_1'] == {'delta': .25, 'ci95': [.25, .25]}


def test_shard_requires_matching_plan_and_complete_coverage(tmp_path):
    plan = dict(shards=1, samples_per_question=4)
    records, selected, plan_hash = [dict(id='q')], [dict(name='fixed_4')], 'plan-a'
    with closing(sqlite3.connect(tmp_path / 'shard_00.sqlite')) as db:
        initialize_shard(db, plan_hash)
        with pytest.raises(ValueError, match='different frozen plan'):
            initialize_shard(db, 'plan-b')
        with pytest.raises(ValueError, match='4 missing'):
            list(collect_shards(tmp_path, plan, records, plan_hash, selected))
        with db:
            db.executemany('INSERT INTO responses VALUES (?, ?, ?, ?)',
                           [('fixed_4', 'q', n, json.dumps(dict(sample=n))) for n in range(4)])
    rows = list(collect_shards(tmp_path, plan, records, plan_hash, selected))
    assert sorted(row['sample'] for row in rows) == list(range(4))


def test_rlt_ragged_requests_preserve_seeds_and_pairing_on_resume():
    fixed, ragged = (next(case for case in cases() if case['name'] == name)
                     for name in ('fixed_4', 'ragged_0.20'))
    plan = dict(sampling=SAMPLING, samples_per_question=4, generation_seed=42)
    rows = [dict(id='q', task='t', prompt_sha256='prompt', target='True')]
    encoded = {'q': [12, 13]}
    fixed_jobs = list(pending_jobs(fixed, rows, plan, set(), encoded))
    jobs = list(pending_jobs(ragged, rows, plan, {('ragged_0.20', 'q', 0)}, encoded))
    assert [request['seed'] for _, _, request in jobs] == [request['seed'] for _, _, request in fixed_jobs[1:]]
    assert all(request['min_loops'] == 1 and request['exit_threshold'] == .20 for _, _, request in jobs)

    class Actor:
        def iter_responses(self, requests):
            for index in reversed(range(len(requests))):
                yield index, dict(token_ids=[7, 0], exit_depths=[4, 1], finish_reason='stop',
                                  executed_token_loops=9), .1

    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: 'The answer is True.')
    results = list(generate_case(Actor(), tokenizer, ragged, jobs, batch_size=2))
    assert [result['sample'] for result in results] == [2, 1, 3]
    assert all(result['correct'] and result['executed_token_loops'] == 9 for result in results)


def test_rlt_counts_full_prefill_and_only_executed_decode_loops():
    output = SimpleNamespace(exit_depths=[4, 1, 3], token_ids=[4, 5, 0], prompt_token_ids=[1, 2])
    dynamic = SimpleNamespace(min_loops=1, exit_threshold=.20)
    assert execution_token_loops(output, dynamic, 4) == 12
    with pytest.raises(RuntimeError, match='prefill or decode'):
        execution_token_loops(output, SimpleNamespace(min_loops=4, exit_threshold=1.0), 4)
    output.exit_depths = [4, 4, 4]
    assert execution_token_loops(output, SimpleNamespace(min_loops=4, exit_threshold=1.0), 4) == 16
    output.exit_depths = [1, 1, 1]
    with pytest.raises(RuntimeError, match='prefill or decode'):
        execution_token_loops(output, dynamic, 4)
