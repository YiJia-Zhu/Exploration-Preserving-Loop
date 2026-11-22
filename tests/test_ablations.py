"""Confidence boundary signals and independent correctness ablations."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from ep_loop.confidence import PromptConfidence, confidence_changes
from ep_loop.config import load_config,merge,smoke_config,validate_config
from ep_loop.data import ByteTokenizer,canonical_hash,filter_policy,normalize_record,prepare_training_records
from ep_loop.huginn import tiny_huginn
from ep_loop.objective import DistillationObjective
from ep_loop.ouro import tiny_ouro
from ep_loop.selection import adjacent_js_from_log_probs,make_dynamic_stage_plan

ROOT=Path(__file__).resolve().parents[1]


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.templates={'ouro':tiny_ouro(ROOT/'models/Ouro-1.4B-Thinking'),
                       'huginn':tiny_huginn(ROOT/'models/huginn-0125')}

    def test_identity_switch_detected_by_js_but_scalar_confidence_unchanged(self):
        logp=torch.tensor([[[.9,.1]],[[.1,.9]]]).log()
        self.assertGreater(adjacent_js_from_log_probs(logp).item(),0)
        for signal in ['max_probability','normalized_entropy']:
            changes,_=confidence_changes(logp,signal)
            torch.testing.assert_close(changes,torch.zeros_like(changes),rtol=0,atol=0)

    def test_confidence_keeps_rises_falls_strict_boundaries_and_native_endpoint(self):
        probabilities=torch.tensor([[[.5,.5]],[[.8,.2]],[[.6,.4]],[[.9,.1]]])
        changes,confidence=confidence_changes(probabilities.log(),'max_probability')
        torch.testing.assert_close(changes[:,0],torch.tensor([.3,.2,.3]))
        plan=make_dynamic_stage_plan(changes,torch.tensor([3]),js_threshold=.15)
        self.assertEqual(plan.target_loops.item(),3)
        self.assertEqual(plan.teacher_depths[:3,0].tolist(),[1,2,3])
        self.assertEqual(make_dynamic_stage_plan(changes,torch.tensor([1]),js_threshold=0).target_loops.item(),1)

    def test_direct_signals_train_differentiably_without_teacher_gradients(self):
        for family in ['ouro','huginn']:
            template=self.templates[family]
            for signal in ['js','max_probability','normalized_entropy']:
                student=type(template)(copy.deepcopy(template.model),'none')
                teacher=type(template)(copy.deepcopy(template.model).eval().requires_grad_(False),'none')
                algorithm=smoke_config(load_config(ROOT/'configs'/('huginn_' if family=='huginn' else '').__add__('e_dynamic_js.json')))['algorithm']
                algorithm.update(boundary_signal=signal,confidence_threshold=0.)
                objective=DistillationObjective(student,teacher,algorithm,2,True)
                loss,info=objective(torch.tensor([[1,20,30,40,50,2]]),3,[1,1,1])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(student.enable_student_exit().weight.grad)
                self.assertTrue(all(p.grad is None for p in teacher.model.parameters()))
                self.assertEqual(info['confidence_probe_token_loops'],0)

    def test_probe_isolated_rng_unmodified_prefix_and_costs_measured(self):
        template=self.templates['huginn']
        backend=type(template)(copy.deepcopy(template.model),'none')
        backend.model.train()
        instrument=PromptConfidence(backend,ByteTokenizer(),seed=42)
        ids=torch.tensor([[1,20,30]])
        original=backend.trace
        captured=[]
        def traced(current,*args,**kwargs):
            captured.append(current.clone())
            return original(current,*args,**kwargs)
        for signal in ['probe_know']:
            before=torch.get_rng_state()
            with patch.object(backend,'trace',side_effect=traced):
                changes,confidence,cost=instrument.measure(ids,2,torch.tensor([2]),signal)
            torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
            self.assertTrue(backend.model.training)
            self.assertEqual(confidence.shape,(2,1))
            self.assertEqual(changes.shape,(1,1))
            self.assertTrue(((confidence>=0)&(confidence<=1)).all())
            self.assertGreater(cost['forwards'],0)
            self.assertGreater(cost['token_loops'],0)
            self.assertEqual(cost['forwards'],2)
            self.assertTrue(all(torch.equal(row[:,:3],ids) for row in captured))
            self.assertTrue(all(p.grad is None for p in backend.model.parameters()))

    def test_probe_labels_train_without_replacing_original_teacher_targets(self):
        template=self.templates['ouro']
        for signal in ['probe_know']:
            student=type(template)(copy.deepcopy(template.model),'none')
            teacher=type(template)(copy.deepcopy(template.model).requires_grad_(False),'none')
            algorithm=smoke_config(load_config(ROOT/'configs/e_dynamic_js.json'))['algorithm']
            algorithm.update(boundary_signal=signal,teacher_endpoint='fixed')
            instrument=PromptConfidence(teacher,ByteTokenizer())
            objective=DistillationObjective(student,teacher,algorithm,2,False,confidence_probe=instrument)
            loss,info=objective(torch.tensor([[1,20,30,40]]),3,[1])
            loss.backward()
            self.assertGreater(info['confidence_probe_token_loops'],0)
            self.assertEqual(info['teacher_executed_token_loops'],3*4)
            self.assertTrue(all(p.grad is None for p in teacher.model.parameters()))

    def test_teacher_filter_identity_duplicates_and_random_count_matched(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            raw=[dict(id=str(i),problem=f'{i}+1?',answer=str(i+1)) for i in range(6)]
            (root/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
            config=merge(load_config(ROOT/'configs/e_dynamic_js.json'),dict(data=dict(train_path='train.jsonl',expected_examples=6,correctness_cache='filter.jsonl')))
            records=[normalize_record(r) for r in raw]
            labels=[dict(kind='metadata',policy=filter_policy(config))]+[
                dict(id=r['id'],problem_sha256=r['problem_sha256'],answer_sha256=canonical_hash(r['answer']),correct=i<2) for i,r in enumerate(records)]
            def save():(root/'filter.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            save()
            kept,report=prepare_training_records(config,ByteTokenizer(),root)
            self.assertEqual([r['id'] for r in kept],['0','1'])
            matched=merge(config,dict(distillation=dict(question_filter='random_matched')))
            random_kept,_=prepare_training_records(matched,ByteTokenizer(),root)
            self.assertEqual(len(random_kept),2)
            self.assertEqual([r['id'] for r in random_kept],[r['id'] for r in prepare_training_records(matched,ByteTokenizer(),root)[0]])
            labels[1]['answer_sha256']='changed';save()
            with self.assertRaisesRegex(ValueError,'ground truth changed'):prepare_training_records(config,ByteTokenizer(),root)
            labels[1]['answer_sha256']=canonical_hash(records[0]['answer']);labels.append(labels[1]);save()
            with self.assertRaisesRegex(ValueError,'duplicate'):prepare_training_records(config,ByteTokenizer(),root)

    def test_ablation_configs_remain_dynamic_and_invalid_combinations_rejected(self):
        for name in ('e_dynamic_js.json', 'huginn_e_dynamic_js.json'):
            base = load_config(ROOT / 'configs' / name)
            for signal in ('max_probability', 'normalized_entropy', 'probe_know'):
                with self.subTest(config=name, signal=signal):
                    variant = merge(base, {'algorithm': {
                        'boundary_signal': signal, 'confidence_threshold': 0.05,
                    }})
                    config = validate_config(variant)
                    self.assertIsNone(config['algorithm']['student_loops'])
                    self.assertEqual(config['algorithm']['student_policy'], 'learned')
            filtered = merge(base, {
                'distillation': {'question_filter': 'teacher_correct', 'require_eos': True},
                'data': {'correctness_cache': 'labels.jsonl'},
            })
            validate_config(filtered)
        config=load_config(ROOT/'configs/e_dynamic_js.json')
        for change in [dict(algorithm=dict(boundary_signal='other')),dict(algorithm=dict(boundary_signal='probe_true')),dict(distillation=dict(response_filter='student_correct')),
                       dict(distillation=dict(question_filter='teacher_correct')),dict(algorithm=dict(confidence_threshold=-1,boundary_signal='probe_know'))]:
            with self.assertRaises(ValueError):validate_config(merge(config,change))


if __name__=='__main__':unittest.main()
