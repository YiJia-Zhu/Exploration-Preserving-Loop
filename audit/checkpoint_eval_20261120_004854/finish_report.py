"""Validate both finished evaluations and write a compact comparison."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[1]
sys.path.insert(0,str(ROOT))
from ep_loop.evaluation_data import load_evaluation_records
from ep_loop.inference import math_metrics
from ep_loop.reproducibility import sample_seed


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def collect(status):
    source,_=load_evaluation_records(status['dataset']['path'],30,'aime')
    records={row['id']:row for row in source};rows=[];paired=None
    for task in status['checkpoints']:
        folder=Path(task['folder']);responses=folder/'aime2024.responses.jsonl'
        answers=[json.loads(line) for line in responses.read_text().splitlines()]
        report=json.loads(responses.with_suffix('.summary.json').read_text())
        assert report['checkpoint']==task['checkpoint'] and report['speculative_config'] is None
        assert len(answers)==480
        seen=set();correctness={ident:[] for ident in records};tokens={ident:[] for ident in records}
        for answer in answers:
            record=records[answer['id']];key=(answer['id'],answer['sample'])
            assert key not in seen and 0<=answer['sample']<16;seen.add(key)
            assert answer['problem']==record['problem'] and answer['messages']==record['messages']
            assert answer['reference_answer']==record['answer'] and isinstance(answer['text'],str) and answer['text']
            assert answer['sampling_seed']==sample_seed(42,record['problem_sha256'],answer['sample'])
            assert answer['max_loops']==2 and 0<answer['response_tokens']<=16384
            assert len(answer['actual_execution_depths'])==answer['response_tokens']
            assert set(answer['actual_execution_depths'])=={2}
            correctness[answer['id']].append(answer['correct']);tokens[answer['id']].append(answer['response_tokens'])
        assert all(len(values)==16 for values in correctness.values())
        if paired is None:paired=seen
        else:assert paired==seen
        scores=report['results']['fixed_2'];recomputed=math_metrics(correctness)
        assert all(scores[key]==value for key,value in recomputed.items())
        assert scores['correct_responses']==sum(answer['correct'] for answer in answers)
        assert scores['response_tokens']==sum(answer['response_tokens'] for answer in answers)
        assert scores['accuracy']==scores['correct_responses']/480
        questions=[json.loads(line) for line in Path(report['questions']).read_text().splitlines()]
        assert len(questions)==30 and {q['id'] for q in questions}==set(records)
        for question in questions:
            record=records[question['id']]
            assert question['problem']==record['problem'] and question['reference_answer']==record['answer']
        per_question=[dict(id=record['id'],problem=record['problem'],reference_answer=record['answer'],
            completions=16,correct=sum(correctness[record['id']]),accuracy=sum(correctness[record['id']])/16,
            mean_response_tokens=sum(tokens[record['id']])/16) for record in source]
        (folder/'aime2024.per_question.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in per_question))
        rows.append(dict(step=task['step'],**scores,seconds=report['wall_seconds'],
                         responses=str(responses),questions=report['questions']))
    write(AUDIT/'comparison.json',dict(status='completed',dataset='AIME2024',samples_per_problem=16,
        max_new_tokens=16384,loops=2,ngram=False,rows=rows))
    table=['# Step 5 / Step 10 评测','',
        'AIME2024，30题×16次；固定N2，BF16权重与KV，普通推理，temperature=1.0、top_p=0.7、上限16384 token。',
        'accuracy 按480条回答统计，等于本轮 Avg@16；Pass@16 表示至少一次答对的题目比例。','',
        '| step | 正确回答 | accuracy / Avg@16 | Pass@16 | 平均输出token | 总输出token | 平均循环数 | 截断数 | 时间(分钟) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f"| {row['step']} | {row['correct_responses']}/480 | {row['accuracy']:.2%} | {row['pass_at_16']:.2%} | "
                     f"{row['mean_response_tokens']:.1f} | {row['response_tokens']} | {row['mean_actual_depth']:.1f} | "
                     f"{row['truncated_responses']} | {row['seconds']/60:.1f} |")
    table.extend(['','每条回答 JSONL 均包含问题、输入 messages、参考答案、模型回答全文、token数、循环次数、随机seed和正确性。'])
    for row in rows:table.append(f"\n- step {row['step']}：[问题与回答]({row['responses']}) · [每题一条]({row['questions']})")
    (AUDIT/'comparison.md').write_text('\n'.join(table)+'\n')
    return rows


if __name__=='__main__':
    try:
        while True:
            status=json.loads((AUDIT/'status.json').read_text())
            if status['status'] in {'completed','failed','stopped'}:break
            time.sleep(20)
        if status['status']!='completed':raise RuntimeError('Evaluation '+status['status'])
        rows=collect(status)
        write(AUDIT/'validation.json',dict(status='passed',verified_at=datetime.now(timezone.utc).isoformat(),
            responses_per_checkpoint=480,questions_per_checkpoint=30,problem_and_answer_text=True,
            identical_prompt_sample_pairs=True,ordinary_inference=True,summary_recomputed=True))
        print(json.dumps(rows,ensure_ascii=False,indent=2),flush=True)
    except BaseException as error:
        write(AUDIT/'validation.json',dict(status='failed',error=f'{type(error).__name__}: {error}'))
        raise
