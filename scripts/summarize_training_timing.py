"""Summarize existing rank-local stage clocks without adding overlapping ranks."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    steps = defaultdict(lambda: dict(ranks={}, spans=[], inference_batches={}))
    for path in sorted(directory.glob('events_rank*.jsonl')):
        for line in path.open():
            row = json.loads(line)
            if row.get('event') in {'vllm_batch_metrics', 'teacher_batch_metrics'}:
                step = steps[int(row['update'])]
                rank = step['inference_batches'].setdefault(str(row['rank']), {})
                rank[row['event']] = {key: row[key] for key in
                    ('batch_seconds', 'response_tokens', 'responses', 'generate_calls', 'max_num_seqs', 'teacher_executed_token_loops',
                     'engine_metrics', 'runtime', 'tensor_parallel_size', 'inference_group_gpus') if key in row}
            if row.get('event') != 'stage_end' or 'update' not in row:
                continue
            step = steps[int(row['update'])]
            rank = step['ranks'].setdefault(str(row['rank']), {})
            stage = row['stage']
            rank[stage] = rank.get(stage, 0.) + row['seconds']
            if stage == 'update':
                step['spans'].append((row['started_at'], row['finished_at']))
    metrics = directory / 'metrics.jsonl'
    if metrics.exists():
        for line in metrics.open():
            row = json.loads(line)
            steps[row['update']]['metrics'] = {
                key: row.get(key) for key in
                ('responses', 'response_tokens', 'truncated_responses',
                 'mean_response_loss', 'seconds', 'gradient_norm')}
    for step in steps.values():
        spans = step.pop('spans')
        if spans:
            start = min(datetime.fromisoformat(s) for s, _ in spans)
            end = max(datetime.fromisoformat(e) for _, e in spans)
            step.update(started_at=start.isoformat(), finished_at=end.isoformat(),
                        wall_seconds_including_checkpoint=(end-start).total_seconds())
    return dict(directory=str(directory.resolve()), updates=dict(sorted(steps.items())),
                timing_note='Rank clocks overlap; never add rank0 and rank1. '
                            'response contains objective_forward and backward; '
                            'objective_forward contains teacher_forward, stage_planning, '
                            'student_replay and loss_readout. Backward may include DDP wait. '
                            'The rollout stage contains cold engine initialization on update1.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.directory), ensure_ascii=False, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
