"""AIME year isolation, 16-sample metrics, leakage exclusion and whole shards."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ep_loop.config import load_config
from ep_loop.data import ByteTokenizer, normalize_record
from ep_loop.evaluation_data import (load_evaluation_records, benchmark_math_metrics,
                                     exclude_evaluation_overlaps)
from ep_loop.inference import verify_integer_response

ROOT = Path(__file__).resolve().parents[1]


def examples():
    return [dict(id=f'{year}-{i}',benchmark=f'AIME{year}',problem=f'{year}/{i}: 1+1?',answer='7')
            for year in (2024,2025,2026) for i in range(30)]


class AimeEvaluationTests(unittest.TestCase):
    def test_complete_year_counts_and_prompt_only_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'eval.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in examples()))
            records,metadata=load_evaluation_records(path,90,'aime')
            self.assertEqual(metadata['benchmarks'],{f'AIME{year}':30 for year in (2024,2025,2026)})
            self.assertEqual(records[0]['messages'],[dict(role='user',content=examples()[0]['problem']+
                '\nPlease reason step by step, and put your final answer within \\boxed{}.')])
            with self.assertRaisesRegex(ValueError,'expected 30'):
                load_evaluation_records(path,30,'aime')
            path.write_text(''.join(json.dumps(row)+'\n' for row in examples()[:-1]))
            with self.assertRaisesRegex(ValueError,'all 30'):
                load_evaluation_records(path,89,'aime')

    def test_exact_problem_overlap_removed_without_answer_filtering(self):
        evaluation=[normalize_record(row) for row in examples()]
        source=[normalize_record(dict(id='leaked',messages=[dict(role='user',content=
            'Solve carefully:\n  '+examples()[0]['problem'].replace(' ', '\n')+'\nAnswer:')],answer='0')),
            normalize_record(dict(id='unrelated',problem='An unrelated training problem',answer='7'))]
        kept,report=exclude_evaluation_overlaps(source,evaluation)
        self.assertEqual([r['id'] for r in kept],['unrelated'])
        self.assertEqual(report['matches'],[dict(train_id='leaked',evaluation_ids=['2024-0'])])
        self.assertEqual(report['removed'],1)

    def test_year_metrics_distinguish_avg16_from_pass16(self):
        records=[normalize_record(row) for row in examples()]
        correctness={r['id']:([True]+[False]*15 if r['benchmark']=='AIME2024' else
                              [True]*16 if r['benchmark']=='AIME2025' else [False]*16) for r in records}
        scores=benchmark_math_metrics(records,correctness)
        self.assertEqual(scores['AIME2024']['avg_at_16'],1/16)
        self.assertEqual(scores['AIME2024']['pass_at_16'],1.)
        self.assertEqual(scores['AIME2025']['avg_at_16'],1.)
        self.assertEqual(scores['AIME2026']['pass_at_16'],0.)
        self.assertTrue(all(s['completions']==480 and s['problems']==30 for s in scores.values()))
        self.assertTrue(verify_integer_response(r'Final \boxed{336^\circ}.','336')['correct'])

    def test_vllm_submits_all_1440_completions_in_two_shards(self):
        from scripts.evaluate_fastpath import main
        submissions=[];seeds=[];tokenizer=ByteTokenizer()
        completion=tokenizer.encode(r'\boxed{7}')+[2]
        class Actor:
            def __init__(self,directory,gpu,log,**kwargs):
                self.gpu=gpu;self.loops=kwargs['loops']
                if kwargs.get('rollout_ngram') is not False:
                    raise AssertionError('Evaluation must disable training n-gram even when the run config enables it')
                if 'max_num_seqs' in kwargs: raise AssertionError('Use vLLM default')
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def generate(self,requests,**kwargs):
                submissions.append((self.gpu,len(requests)))
                seeds.extend(row['seed'] for row in requests)
                return dict(seconds=.1,responses=[dict(token_ids=completion,
                    exit_depths=[self.loops]*len(completion),finish_reason='stop',
                    executed_token_loops=self.loops*(len(r['input_ids'])+len(completion)-1)) for r in requests])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'eval.jsonl';output=root/'result.jsonl';config_path=root/'config.json'
            data.write_text(''.join(json.dumps(row)+'\n' for row in examples()))
            config=load_config(ROOT/'configs/vllm_fixed_top_n.json')
            config['evaluation'].update(suite='aime',expected_examples=90,samples_per_problem=16)
            config_path.write_text(json.dumps(config))
            args=['evaluate_fastpath.py','--config',str(config_path),'--checkpoint',directory,
                  '--data',str(data),'--output',str(output),'--gpus','2,3']
            with patch('sys.argv',args),patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer),\
                    patch('ep_loop.fastpath.actor.VllmActor',Actor),\
                    patch.object(tokenizer,'apply_chat_template',wraps=tokenizer.apply_chat_template) as encode:
                main()
                self.assertEqual(encode.call_count,90)
            report=json.loads(output.with_suffix('.summary.json').read_text())
            scores=report['results']['fixed_2']
            self.assertEqual(scores['completions'],1440)
            self.assertTrue(all(s['avg_at_16']==1. and s['pass_at_16']==1. for s in scores['benchmarks'].values()))
            self.assertEqual(sorted(submissions),[('2',720),('3',720)])
            self.assertEqual(len(set(seeds)),1440)
            responses=[json.loads(line) for line in output.read_text().splitlines()]
            questions=[json.loads(line) for line in output.with_suffix('.questions.jsonl').read_text().splitlines()]
            self.assertEqual(len(responses),1440)
            self.assertEqual(len(questions),90)
            self.assertEqual(report['questions'],str(output.with_suffix('.questions.jsonl').resolve()))
            for row in responses:
                question=questions[row['index']]
                self.assertEqual((row['id'],row['problem'],row['messages'],row['reference_answer']),
                                 (question['id'],question['problem'],question['messages'],question['reference_answer']))
                self.assertTrue(row['text'])


if __name__=='__main__': unittest.main()
