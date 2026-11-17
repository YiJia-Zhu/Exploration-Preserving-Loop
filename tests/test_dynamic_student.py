"""Threshold-only stages, learned stopping, genuine early execution and replay."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from ep_loop.config import DEFAULT_CONFIG, load_config, merge, smoke_config, validate_config
from ep_loop.huginn import HuginnBackend, load_huginn, tiny_huginn
from ep_loop.inference import generate
from ep_loop.objective import DistillationObjective
from ep_loop.ouro import OuroBackend, tiny_ouro
from ep_loop.selection import make_dynamic_stage_plan, stage_stop_bce

ROOT=Path(__file__).resolve().parents[1]


class DynamicSelectionTests(unittest.TestCase):
    def test_proposal_tokens_a_b_c_keep_every_threshold_boundary(self):
        js=torch.tensor([[.01,.01,.01],[.02,.20,.02],[.30,.01,.03],[.01,.18,.01],[.02,.01,.02]])
        plan=make_dynamic_stage_plan(js,torch.tensor([6,6,6]),'threshold',.05)
        self.assertEqual(plan.target_loops.tolist(),[2,3,1])
        self.assertEqual(plan.teacher_depths[:3].tolist(),[[3,2,6],[6,4,-1],[-1,6,-1]])
        self.assertEqual(plan.weights[:,0].tolist(),[.5,.5,0,0,0,0])

    def test_no_top_cap_no_old_shallow_fallback_and_strict_threshold(self):
        plan=make_dynamic_stage_plan(torch.full((5,3),.2),torch.tensor([6,2,1]),js_threshold=.1)
        self.assertEqual(plan.target_loops.tolist(),[6,2,1])
        self.assertEqual(plan.teacher_depths[:,0].tolist(),[1,2,3,4,5,6])
        self.assertEqual(plan.teacher_depths[:2,1].tolist(),[1,2])
        equal=make_dynamic_stage_plan(torch.full((5,1),.1),torch.tensor([6]),js_threshold=.1)
        self.assertEqual(equal.target_loops.item(),1)
        self.assertEqual(equal.teacher_depths[0].item(),6)

    def test_all_dynamic_ablations_share_target_depths_and_stop_labels(self):
        js=torch.tensor([[.3],[.01],[.01],[.01],[.01]])
        plans=[make_dynamic_stage_plan(js,torch.tensor([6]),mode,.05) for mode in ('threshold','threshold_uniform','threshold_endpoint')]
        self.assertEqual([p.target_loops.item() for p in plans],[2,2,2])
        self.assertEqual(plans[0].teacher_depths[:2,0].tolist(),[1,6])
        self.assertEqual(plans[1].teacher_depths[:2,0].tolist(),[3,6])
        self.assertEqual(plans[2].teacher_depths[:2,0].tolist(),[-1,6])
        self.assertEqual(plans[2].weights[:2,0].tolist(),[0.,1.])

    def test_stop_bce_normalizes_per_token_and_masks_later_stages(self):
        gates=torch.zeros(4,3,1,requires_grad=True)
        target=torch.tensor([1,2,4])
        loss=stage_stop_bce(gates,target)
        torch.testing.assert_close(loss,torch.full((3,),torch.tensor(2.).log()))
        loss.mean().backward()
        self.assertLess(gates.grad[0,0,0],0)
        self.assertGreater(gates.grad[0,1,0],0)
        self.assertLess(gates.grad[1,1,0],0)
        self.assertEqual(gates.grad[2,1,0],0)
        self.assertTrue(torch.count_nonzero(gates.grad[1:,0])==0)

    def test_cli_default_selects_new_main_method_and_old_baseline_is_fixed(self):
        current=validate_config(load_config(DEFAULT_CONFIG))
        self.assertEqual(current['algorithm']['student_policy'],'learned')
        self.assertEqual(current['algorithm']['stage_selection'],'threshold')
        self.assertIsNone(current['algorithm']['student_loops'])
        self.assertEqual(current['algorithm']['student_max_loops'],4)
        baseline=validate_config(load_config(ROOT/'configs/a_loopopd.json'))
        self.assertEqual(baseline['algorithm'].get('student_policy','fixed'),'fixed')
        result=subprocess.run([sys.executable,'scripts/train_stage_distillation.py','--show-config'],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout)['algorithm']['stage_selection'],'threshold')
        self.assertTrue(current['rollout']['use_cache'])
        self.assertTrue(current['evaluation']['use_cache'])
        validate_config(merge(current,{'rollout':{'use_cache':False}}))


class DynamicBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.ouro=tiny_ouro(ROOT/'models/Ouro-1.4B-Thinking')
        cls.huginn=tiny_huginn(ROOT/'models/huginn-0125')

    def test_ouro_local_stop_executes_one_or_all_loops_and_matches_native(self):
        b=OuroBackend(copy.deepcopy(self.ouro.model),'none')
        b.model.eval()
        ids=torch.tensor([[1,20,30]])
        calls=[]
        hook=b.model.model.layers[0].register_forward_hook(lambda *args:calls.append(1))
        with torch.no_grad():
            gate=b.enable_student_exit()
            gate.weight.zero_();gate.bias.fill_(100.)
            early=b.trace(ids,4,endpoint='learned',exit_threshold=.5)
            self.assertEqual(len(calls),1)
            self.assertEqual(early.exits[0,-1].item(),1)
            self.assertEqual(early.executed_token_loops,3)
            expected=b.trace(ids,1).states[0]
            torch.testing.assert_close(early.states[0],expected,rtol=0,atol=0)
            calls.clear();gate.bias.fill_(-100.)
            deep=b.trace(ids,4,endpoint='learned',exit_threshold=.5)
            self.assertEqual(len(calls),4)
            self.assertEqual(deep.exits[0,-1].item(),4)
            expected=b.trace(ids,4).states[-1]
            torch.testing.assert_close(deep.states[-1],expected,rtol=0,atol=0)
        hook.remove()

    def test_generation_recomputes_prefix_when_next_token_needs_deeper_loops(self):
        b=OuroBackend(copy.deepcopy(self.ouro.model),'none')
        gate=b.enable_student_exit()
        def stop(hidden):
            value=100. if hidden.shape[1]==3 else -100.
            return hidden.new_full((*hidden.shape[:2],1),value)
        with patch.object(gate,'forward',side_effect=stop),patch('ep_loop.inference.sample_token',return_value=torch.tensor([40])):
            out=generate(b,[1,20,30],4,2,1.,1.,torch.Generator(),endpoint='learned',exit_threshold=.5,use_cache=False)
        self.assertEqual(out.exit_depths,[1,4])
        self.assertEqual(out.execution_depths,[1,4])
        self.assertEqual(out.executed_token_loops,3+4*4)

    def test_huginn_new_head_stops_without_teacher_and_uses_native_coda(self):
        b=HuginnBackend(copy.deepcopy(self.huginn.model),'none')
        b.model.eval()
        gate=b.enable_student_exit()
        ids=torch.tensor([[1,20,30]])
        with torch.no_grad():
            gate.bias.fill_(100.)
            torch.manual_seed(71)
            out=b.trace(ids,4,endpoint='learned',exit_threshold=.5)
            self.assertEqual(out.exits[0,-1].item(),1)
            self.assertEqual(out.executed_token_loops,3)
            torch.manual_seed(71)
            fixed=b.trace(ids,1)
            torch.testing.assert_close(out.states[0],fixed.states[0],rtol=0,atol=0)
            gate.bias.fill_(-100.)
            out=b.trace(ids,4,endpoint='learned',exit_threshold=.5)
            self.assertEqual(out.exits[0,-1].item(),4)

    def objective_case(self,family,checkpointed=False):
        template=self.ouro if family=='ouro' else self.huginn
        student=type(template)(copy.deepcopy(template.model),'none')
        teacher=type(template)(copy.deepcopy(template.model).requires_grad_(False),'none')
        path=ROOT/'configs'/('e_dynamic_js.json' if family=='ouro' else 'huginn_e_dynamic_js.json')
        algorithm=smoke_config(load_config(path))['algorithm']
        algorithm.update(js_threshold=0.,teacher_endpoint='fixed')
        student.enable_student_exit()
        if family=='ouro' and checkpointed:
            student.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        elif family=='huginn':student.model.gradient_checkpointing=checkpointed
        student.model.train()
        with torch.no_grad():student.model.lm_head.weight.add_(.003*torch.randn_like(student.model.lm_head.weight))
        objective=DistillationObjective(student,teacher,algorithm,2,checkpointed)
        torch.manual_seed(731)
        loss,info=objective(torch.tensor([[1,20,30,40,50,2]]),3,[1,1,1])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(info['stop_loss'].item(),0)
        self.assertGreater(info['replay_additional_required_loops'],0)
        self.assertTrue(torch.isfinite(student.enable_student_exit().weight.grad).all())
        self.assertGreater(student.enable_student_exit().weight.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in teacher.model.parameters()))
        return student,loss.detach()

    def test_both_families_train_stage_and_stop_with_teacher_isolated(self):
        for family in ('ouro','huginn'):self.objective_case(family,True)

    def test_dynamic_chunk_and_backbone_checkpoint_match_all_gradients(self):
        for family in ('ouro','huginn'):
            torch.manual_seed(943)
            first,loss_a=self.objective_case(family,True)
            torch.manual_seed(943)
            second,loss_b=self.objective_case(family,False)
            torch.testing.assert_close(loss_a,loss_b,rtol=1e-5,atol=1e-7)
            for (name,a),(_,b) in zip(first.model.named_parameters(),second.model.named_parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=4e-4,atol=1e-6,msg=name)

    def test_huginn_exit_head_full_checkpoint_reloads(self):
        b=HuginnBackend(copy.deepcopy(self.huginn.model),'none')
        b.enable_student_exit()
        with torch.no_grad():b.model.ep_loop_exit_head.weight.fill_(.123)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            b.model.save_pretrained(folder,max_shard_size='40KB')
            for name in b.source_filenames:shutil.copy2(ROOT/'models/huginn-0125'/name,folder/name)
            restored,_=load_huginn(folder,'cpu','none',False)
            self.assertEqual(restored.model.config.ep_loop_student_policy,'learned')
            for name,value in b.model.state_dict().items():
                torch.testing.assert_close(value,restored.model.state_dict()[name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
