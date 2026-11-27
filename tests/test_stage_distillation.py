"""Semantic checks for the research proposal and official Ouro execution path."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from ep_loop.config import load_config,merge,validate_config
from ep_loop.data import ByteTokenizer,canonical_hash,filter_policy,normalize_record,prepare_training_records,prompt_ids
from ep_loop.inference import last_boxed,math_metrics,verify_integer_response
from ep_loop.objective import DistillationObjective
from ep_loop.ouro import OuroBackend,tiny_ouro
from ep_loop.selection import (adjacent_js_from_log_probs,make_stage_plan,
                               native_ouro_exits,stage_reverse_kl)

ROOT=Path(__file__).resolve().parents[1]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def trajectory(self,values):
        return torch.tensor([[[v,1-v]] for v in values],dtype=torch.float64).log()

    def test_standard_js_bounds_symmetry_and_identity(self):
        lp=self.trajectory([0.99,0.01])
        forward=adjacent_js_from_log_probs(lp)
        torch.testing.assert_close(forward,adjacent_js_from_log_probs(lp.flip(0)))
        self.assertGreater(float(forward),0)
        self.assertLessEqual(float(forward),float(torch.tensor(2.).log()))
        self.assertAlmostEqual(float(adjacent_js_from_log_probs(lp[:1].repeat(2,1,1))),0,places=14)

    def test_proposal_six_loops_boundary_three_to_four(self):
        lp=self.trajectory([.9,.9,.9,.1,.1,.1])
        plan=make_stage_plan(lp,torch.tensor([6]),2,"js",.1)
        self.assertEqual(plan.teacher_depths[:,0].tolist(),[3,6])
        self.assertEqual(plan.weights[:,0].tolist(),[.5,.5])

    def test_terminal_fallback_and_no_boundaries_after_exit(self):
        lp=self.trajectory([.9,.9,.9,.1,.1,.9]).repeat(1,4,1)
        plan=make_stage_plan(lp,torch.tensor([6,3,2,1]),2,"js",.1)
        self.assertEqual(plan.teacher_depths[:,1:].tolist(),[[-1,-1,-1],[3,2,1]])
        none=make_stage_plan(lp,torch.tensor([6,5,4,3]),2,"js",1.0)
        self.assertTrue((none.teacher_depths[0]==-1).all())
        self.assertTrue((none.weights[-1]==1).all())

    def test_largest_boundaries_ties_order_and_student_mapping(self):
        lp=self.trajectory([.9,.1,.1,.9,.9,.9])
        plan=make_stage_plan(lp,torch.tensor([6]),3,"js",.1)
        self.assertEqual(plan.teacher_depths[:,0].tolist(),[1,3,6])
        one=make_stage_plan(self.trajectory([.9,.9,.9,.1,.1,.1]),torch.tensor([6]),4,"js",.1)
        self.assertEqual(one.teacher_depths[:,0].tolist(),[-1,3,-1,6])

    def test_uniform_control_matches_js_target_count_not_locations(self):
        lp=self.trajectory([.9,.1,.1,.1,.1,.1])
        js=make_stage_plan(lp,torch.tensor([6]),2,"js",.1)
        uniform=make_stage_plan(lp,torch.tensor([6]),2,"uniform",.1)
        self.assertEqual(js.counts.tolist(),uniform.counts.tolist())
        self.assertEqual(js.teacher_depths[:,0].tolist(),[1,6])
        self.assertEqual(uniform.teacher_depths[:,0].tolist(),[3,6])

    def test_native_exit_stick_breaking_and_maximum_fallback(self):
        gates=torch.zeros(4,3,1)
        self.assertEqual(native_ouro_exits(gates,.8).tolist(),[3,3,3])
        self.assertEqual(native_ouro_exits(gates,1).tolist(),[4,4,4])
        gates[:,1]=-100
        gates[0,2]=100
        self.assertEqual(native_ouro_exits(gates,.9).tolist(),[4,4,1])

    def test_stage_loss_formula_and_teacher_gradient_isolation(self):
        q=self.trajectory([.9,.9,.1,.1]).requires_grad_()
        p=torch.tensor([[[.7,.3]],[[.4,.6]]],dtype=torch.float64).log().requires_grad_()
        plan=make_stage_plan(q,torch.tensor([4]),2,"js",.1)
        actual=stage_reverse_kl(p,q,plan).mean()
        expected=sum((p[j].softmax(-1)*(p[j].log_softmax(-1)-q[d-1].detach())).sum() for j,d in enumerate([2,4]))/2
        torch.testing.assert_close(actual,expected)
        actual.backward()
        self.assertIsNone(q.grad)
        self.assertGreater(float(p.grad[0].norm()),0)
        self.assertGreater(float(p.grad[1].norm()),0)


class OfficialOuroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        cls.backend=tiny_ouro(ROOT/"models/Ouro-1.4B-Thinking")
        cls.algorithm=load_config(ROOT/"configs/d_current_native_js.json")["algorithm"]

    def test_native_exits_match_official_causallm_readout(self):
        backend=self.backend
        backend.model.eval()
        ids=torch.tensor([[1,20,30,40]])
        with torch.no_grad():
            trace=backend.trace(ids,4)
            logits=torch.stack([backend.head(h) for h in trace.states])
            for threshold in (.5,.9,1.):
                exits=native_ouro_exits(torch.stack(trace.gates),threshold)
                selected=logits[exits-1,torch.zeros_like(exits),torch.arange(ids.shape[1])[None,:]]
                official=backend.model(input_ids=ids,use_cache=False,exit_threshold=threshold).logits
                torch.testing.assert_close(selected,official,rtol=0,atol=0)

    def test_cached_decode_matches_independent_full_prefix(self):
        b=self.backend
        b.model.eval()
        ids=torch.tensor([[1,20,30,40,50]])
        with torch.no_grad():
            for loops in (2,4):
                prefix=b.trace(ids[:,:-1],loops,use_cache=True)
                decoded=b.trace(ids[:,-1:],loops,cache=prefix.cache,use_cache=True)
                full=b.trace(ids,loops)
                torch.testing.assert_close(b.head(decoded.states[-1][:,-1]),b.head(full.states[-1][:,-1]),rtol=1e-5,atol=1e-6)

    def objective(self,backend,checkpointed,chunk):
        algorithm={**self.algorithm,"js_threshold":0.0}
        backend.model.train()
        return DistillationObjective(backend,backend,algorithm,chunk,checkpointed)

    def test_chunk_checkpoint_gradients_match_unchunked_response_mean(self):
        first=OuroBackend(copy.deepcopy(self.backend.model),"none")
        second=OuroBackend(copy.deepcopy(self.backend.model),"none")
        ids=torch.tensor([[1,20,30,40,50,60,70,2]])
        loss_a,info=self.objective(first,True,2)(ids,3)
        loss_b,_=self.objective(second,False,128)(ids,3)
        loss_a.backward();loss_b.backward()
        self.assertEqual(info["response_tokens"],5)
        torch.testing.assert_close(loss_a,loss_b,rtol=1e-5,atol=1e-7)
        for (name,a),(_,b) in zip(first.model.named_parameters(),second.model.named_parameters()):
            if a.grad is None or b.grad is None:
                self.assertIsNone(a.grad,name);self.assertIsNone(b.grad,name)
            else:
                torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=1e-6,msg=name)

    def test_frozen_teacher_stays_initial_current_teacher_refreshes(self):
        student=OuroBackend(copy.deepcopy(self.backend.model),"none")
        teacher=OuroBackend(copy.deepcopy(student.model).requires_grad_(False),"none")
        ids=torch.tensor([[1,20,30,40,50,2]])
        def target(b):
            with torch.no_grad():return b.head(b.trace(ids,4).states[-1]).clone()
        initial=target(teacher)
        objective=DistillationObjective(student,teacher,{**self.algorithm,"stage_selection":"endpoint"},3,True)
        loss,_=objective(ids,3);loss.backward()
        torch.optim.AdamW(student.model.parameters(),lr=.001).step()
        torch.testing.assert_close(initial,target(teacher),rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in teacher.model.parameters()))
        self.assertGreater(float((initial-target(student)).abs().max()),0)

    def test_gradient_checkpointing_backbone_is_compatible(self):
        b=OuroBackend(copy.deepcopy(self.backend.model),"none")
        b.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        loss,_=self.objective(b,True,2)(torch.tensor([[1,20,30,40,50,2]]),3)
        loss.backward()
        self.assertTrue(torch.isfinite(b.model.model.layers[0].self_attn.q_proj.weight.grad).all())

    def test_endpoint_loss_uses_response_prediction_positions_only(self):
        b=OuroBackend(copy.deepcopy(self.backend.model),"none")
        b.model.eval()
        ids=torch.tensor([[1,20,30,40,50,2]])
        with torch.no_grad():
            all_states=b.trace(ids[:,:-1],4).states
            # The prompt ends at input position 2. Positions 2,3,4 predict
            # generated y1,y2,EOS; positions 0,1 must not enter the objective.
            logp=b.head(all_states[1][0,[2,3,4]]).log_softmax(-1)
            logq=b.head(all_states[3][0,[2,3,4]]).log_softmax(-1)
            expected=(logp.exp()*(logp-logq)).sum(-1).mean()
        objective=DistillationObjective(b,b,{**self.algorithm,"teacher_endpoint":"fixed","stage_selection":"endpoint"},2,False)
        loss,info=objective(ids,3)
        torch.testing.assert_close(loss,expected,rtol=1e-5,atol=1e-7)
        self.assertEqual(info["response_tokens"],3)


class DataAndConfigTests(unittest.TestCase):
    def test_native_template_repair_preserves_mathematical_backslashes(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(ROOT/"models/Ouro-1.4B-Thinking",local_files_only=True)
        old=tokenizer.chat_template
        text=r"Keep \n literally and answer in \boxed{1}."
        record={"messages":[{"role":"user","content":text}]}
        decoded=tokenizer.decode(prompt_ids(tokenizer,record,True,True))
        self.assertIn("system\n",decoded)
        self.assertIn(text,decoded)
        self.assertEqual(tokenizer.chat_template,old)

    def test_paper_settings_and_dynamic_warmup(self):
        for name in ("a_loopopd","a_d_loopopd","b_native_endpoint","d_native_js","d_current_native_js"):
            config=validate_config(load_config(ROOT/"configs"/(name+".json")))
            self.assertEqual(config["optimization"]["global_batch_size"],256)
            self.assertEqual(config["rollout"]["max_new_tokens"],31744)
            self.assertEqual(config["optimization"]["learning_rate"],1e-6)
            self.assertEqual(config["optimization"]["warmup_steps"],10 if config["algorithm"]["teacher_update"]=="current" else 0)
        bad=merge(config,{"rollout":{"responses_per_prompt":7}})
        with self.assertRaises(ValueError):validate_config(bad)

    def test_correctness_cache_reuses_only_acceptance_and_matches_prefix_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            raw=[{"id":str(i),"problem":f"What is {i}+1?","answer":str(i+1)} for i in range(2)]
            path=root/"train.jsonl";path.write_text("\n".join(json.dumps(r) for r in raw))
            config=merge(load_config(ROOT/"configs/d_native_js.json"),{"data":{"train_path":str(path),"expected_examples":2,"correctness_cache":"filter.jsonl"}})
            labels=[{"kind":"metadata","policy":filter_policy(config)}]
            for i,r in enumerate(raw):
                normalized=normalize_record(r)
                labels.append({"id":str(i),"problem_sha256":normalized["problem_sha256"],"answer_sha256":canonical_hash(normalized['answer']),"correct":i==0,"unused_exit_depths":[99]})
            (root/"filter.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
            records,report=prepare_training_records(config,ByteTokenizer(),root)
            self.assertEqual(len(records),1);self.assertEqual(report["correctness_rejected"],1)
            self.assertNotIn("unused_exit_depths",records[0])
            labels[1]["problem_sha256"]="wrong"
            (root/"filter.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
            with self.assertRaises(ValueError):prepare_training_records(config,ByteTokenizer(),root)

    def test_integer_verifier_and_avg_pass_are_distinct(self):
        text=r"First \boxed{7}; final \boxed{42}."
        self.assertEqual(last_boxed(text),"42")
        self.assertTrue(verify_integer_response(text,"42")["correct"])
        metrics=math_metrics({"a":[True,False],"b":[False,True]})
        self.assertEqual(metrics["avg"],.5);self.assertEqual(metrics["pass_at_n"],1.)

    def test_dapo_requested_answer_line_and_strict_boxed_default(self):
        messages=[dict(role='user',content='Remember to put your answer on its own line after "Answer:".')]
        for text in ['Answer: 42', 'Reasoning\nAnswer: $42$<|im_end|>', r'Answer: \boxed{42}',
                     r'Final \boxed{42}.', 'Answer: 7\nAnswer: 42']:
            self.assertTrue(verify_integer_response(text,'42',messages)['correct'], text)
        for text in ['There are 42 possibilities.', r'First \boxed{42}.'+'\nAnswer: 7',
                     r'First \boxed{42}.'+'\nAnswer: unknown', 'Answer: 42 or 43']:
            self.assertFalse(verify_integer_response(text,'42',messages)['correct'], text)
        self.assertFalse(verify_integer_response('Answer: 42','42')['correct'])

    def test_data_conflict_policy_drops_entire_prompt_and_default_errors(self):
        from scripts.prepare_stage_data import prepare
        class Observer:
            def progress(self, rows, **kwargs):return rows
            def stage(self, *args, **kwargs):return nullcontext()
            def metadata(self):return {}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.jsonl'
            rows=[dict(id='a',problem='one?',answer='1'),dict(id='b',problem='two?',answer='2'),
                  dict(id='c',problem='one?',answer='3')]
            source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            args=SimpleNamespace(input=[source],output=root/'clean.jsonl',deduplicate=True,
                conflicting_answers='error',expected_examples=1,source_repo='test',source_revision='test')
            with self.assertRaisesRegex(ValueError,'Conflicting answers'):
                prepare(args,Observer())
            self.assertFalse(args.output.exists())
            args.conflicting_answers='drop'
            prepare(args,Observer())
            clean=[json.loads(line) for line in args.output.read_text().splitlines()]
            self.assertEqual([row['id'] for row in clean],['b'])
            manifest=json.loads(args.output.with_suffix('.manifest.json').read_text())
            self.assertEqual(manifest['conflicting_prompts_dropped'],1)
            self.assertEqual(manifest['unique_prompts_before_conflict_removal'],2)

    def test_overlong_prompts_do_not_require_unused_correctness_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            raw=[{"id":"short","problem":"1+1?","answer":"2"},
                 {"id":"long","problem":"x"*1000,"answer":"1"}]
            path=root/"train.jsonl";path.write_text("\n".join(json.dumps(r) for r in raw))
            config=merge(load_config(ROOT/"configs/d_native_js.json"),{"data":{"train_path":str(path),"expected_examples":2,"max_prompt_tokens":128,"correctness_cache":"labels.jsonl"}})
            labels=[{"kind":"metadata","policy":filter_policy(config)},
                    {"id":"short","problem_sha256":normalize_record(raw[0])["problem_sha256"],"answer_sha256":canonical_hash(normalize_record(raw[0])['answer']),"correct":True}]
            (root/"labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
            records,report=prepare_training_records(config,ByteTokenizer(),root)
            self.assertEqual([r["id"] for r in records],["short"])
            self.assertEqual(report["overlong_prompts_dropped"],1)

    def test_dapo_parquet_keeps_repeated_rows_and_original_answers(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'train.parquet'
            prompt = [{'role': 'user', 'content': '1+1?'}]
            rows = [dict(prompt=prompt, reward_model={'ground_truth': answer}, extra_info={'index': 'same'})
                    for answer in ('2', '2', '3')]
            pq.write_table(pa.Table.from_pylist(rows), path)
            config = merge(load_config(ROOT / 'configs/rad_opd.json'), {
                'data': {'train_path': path.name, 'expected_examples': len(rows)},
            })
            records, report = prepare_training_records(config, ByteTokenizer(), root)
            self.assertEqual([r['messages'] for r in records], [prompt] * 3)
            self.assertEqual([r['answer'] for r in records], ['2', '2', '3'])
            self.assertEqual([r['id'] for r in records], ['0', '1', '2'])
            self.assertEqual(report['usable_examples'], 3)


if __name__=="__main__":
    unittest.main()
