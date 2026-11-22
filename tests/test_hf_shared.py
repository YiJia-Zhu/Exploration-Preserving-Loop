"""Fused shallow/deep replay and pure-HF batched generation correctness."""
import copy
import gc
import io
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

import torch

from ep_loop.config import load_config, merge, validate_config
from ep_loop.hf_rollout import generate_batch, move_optimizer_state, PrefixStaticLayer
from ep_loop.objective import DistillationObjective
from ep_loop.ouro import tiny_ouro

ROOT=Path(__file__).resolve().parents[1]


class SharedHFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.template=tiny_ouro(ROOT/'models/Ouro-1.4B-Thinking')

    def backend(self):
        return type(self.template)(copy.deepcopy(self.template.model),'none')

    def test_continuation_states_match_full_teacher_without_teacher_gradients(self):
        backend=self.backend();backend.model.train()
        ids=torch.tensor([[1,20,30,40,50]])
        student,teacher=backend.shared_response_trace(ids,2,4,2)
        backend.model.eval()
        with torch.no_grad():expected=backend.response_trace(ids,4,2)
        for actual,wanted in zip(teacher.states,expected.states):
            torch.testing.assert_close(actual,wanted,rtol=1e-6,atol=1e-7)
            self.assertFalse(actual.requires_grad)
        for actual,wanted in zip(teacher.gates,expected.gates):
            torch.testing.assert_close(actual,wanted,rtol=1e-6,atol=1e-7)
        self.assertTrue(all(state.requires_grad for state in student.states))
        self.assertEqual(student.executed_token_loops,10)
        self.assertEqual(teacher.executed_token_loops,10)
        self.assertEqual(teacher.reused_token_loops,10)

    def test_shared_loss_gradients_match_separate_same_model_teacher(self):
        for checkpointed in (False,True):
            for selection,endpoint,placement in [('endpoint','fixed','last'),('top_n','native','prefix'),
                                                  ('top_n','native','last'),('uniform_n','fixed','last')]:
                with self.subTest(checkpointed=checkpointed,selection=selection,endpoint=endpoint,placement=placement):
                    shared,separate=self.backend(),self.backend()
                    for backend in (shared,separate):
                        backend.model.train()
                        if checkpointed:
                            backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                    algorithm=load_config(ROOT/'configs/hf_shared_ep_d_loopopd.json')['algorithm']
                    algorithm.update(stage_selection=selection,teacher_endpoint=endpoint,shallow_target_placement=placement)
                    fused=DistillationObjective(shared,shared,algorithm,2,checkpointed)
                    reference=DistillationObjective(separate,separate,{**algorithm,'teacher_reuse_student':False},2,checkpointed)
                    ids=torch.tensor([[1,20,30,40,50,2]])
                    loss,info=fused(ids,3,[2,2,2]);expected,wanted=reference(ids,3,[2,2,2])
                    torch.testing.assert_close(loss,expected,rtol=1e-6,atol=1e-7)
                    self.assertEqual(info['mapping_hist'].tolist(),wanted['mapping_hist'].tolist())
                    self.assertEqual(info['replay_unique_token_loops'],20)
                    self.assertEqual(wanted['replay_unique_token_loops'],30)
                    loss.backward();expected.backward()
                    for (name,actual),(_,target) in zip(shared.model.named_parameters(),separate.model.named_parameters()):
                        if target.grad is None:self.assertIsNone(actual.grad,name)
                        else:torch.testing.assert_close(actual.grad,target.grad,rtol=2e-4,atol=1e-6,msg=name)

    def test_forward_visits_each_loop_once_and_backprop_stops_at_student_depth(self):
        backend=self.backend();backend.model.train()
        calls=[]
        original=backend.model.model.layers[0].forward
        def forward(*args,**kwargs):
            calls.append((kwargs['current_ut'],torch.is_grad_enabled()))
            return original(*args,**kwargs)
        with patch.object(backend.model.model.layers[0],'forward',forward):
            student,teacher=backend.shared_response_trace(torch.tensor([[1,20,30,40]]),2,4,2)
            backend.head(student.states[-1]).square().mean().backward()
        self.assertEqual(calls,[(0,True),(1,True),(2,False),(3,False)])
        self.assertTrue(all(state.grad_fn is None for state in teacher.states))

    def test_batched_logits_match_unpadded_prefixes_through_eos_row_removal(self):
        for prompts in ([[1,20,30],[1,50]],[[1,20,30],[1,20,30]]):
            with self.subTest(prompts=prompts):self.check_batched_logits(prompts)

    def check_batched_logits(self,prompts):
        backend,reference=self.backend(),self.backend()
        reference.model.eval()
        histories=[[],[]];active=[0,1];script=[[40,2],[50],[2]];call=[0]
        def sample(logits,*args):
            current=script[call[0]];call[0]+=1
            self.assertEqual(logits.shape[0],len(active))
            for row,index in enumerate(active):
                ids=torch.tensor([prompts[index]+histories[index]])
                with torch.no_grad():
                    trace=reference.trace(ids,2,use_cache=False)
                    expected=reference.head(trace.states[-1][:,-1])[0]
                torch.testing.assert_close(logits[row],expected,rtol=2e-5,atol=2e-6)
                histories[index].append(current[row])
            active[:]=[index for index,token in zip(active,current) if token!=2]
            return torch.tensor(current,device=logits.device)
        with patch('ep_loop.hf_rollout.sample_tokens',sample):
            responses,_=generate_batch(backend,prompts,2,4,1.,1.,torch.Generator().manual_seed(42))
        self.assertEqual([r.token_ids for r in responses],[[40,50,2],[2]])
        self.assertTrue(all(r.reached_eos for r in responses))
        self.assertTrue(backend.model.training)
        self.assertTrue(all(p.grad is None for p in backend.model.parameters()))
        algorithm=load_config(ROOT/'configs/hf_shared_ep_d_loopopd.json')['algorithm']
        loss,_=DistillationObjective(backend,backend,algorithm,2,True)(torch.tensor([[1,20,30,40,50,2]]),3)
        loss.backward()
        self.assertGreater(float(backend.model.lm_head.weight.grad.norm()),0)

    def test_shared_execution_rejects_frozen_dynamic_and_unshared_profiles(self):
        config=validate_config(load_config(ROOT/'configs/hf_shared_ep_d_loopopd.json'))
        changes=[dict(algorithm=dict(teacher_update='frozen')),
                 dict(algorithm=dict(teacher_reuse_student=False)),
                 dict(algorithm=dict(student_policy='learned')),
                 dict(implementation=dict(hf_rollout_batch_size=0)),
                 dict(implementation=dict(phase_offload=True)),dict(rollout=dict(use_cache=False))]
        for override in changes:
            with self.subTest(override=override),self.assertRaises(ValueError):
                validate_config(merge(config,override))

    def test_mixed_rope_key_and_value_cache_matches_autocast_sdpa(self):
        layer=PrefixStaticLayer(12);layer.visible_length=3
        query=torch.randn(2,4,1,8)
        keys=torch.randn(2,4,3,8,dtype=torch.float32)
        values=torch.randn(2,4,3,8,dtype=torch.bfloat16)
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            expected=torch.nn.functional.scaled_dot_product_attention(query,keys,values)
            cached_keys,cached_values=layer.update(keys,values,dict(cache_position=torch.arange(3)))
            actual=torch.nn.functional.scaled_dot_product_attention(query,cached_keys,cached_values)
        self.assertEqual(cached_keys.dtype,torch.bfloat16)
        self.assertEqual(cached_values.dtype,torch.bfloat16)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_offload_releases_checkpoint_serialization_storage_cycles(self):
        # Reproduce the real torch.save cycle without allocating a large GPU
        # optimizer: moving CPU moments to meta also replaces their storages.
        gc.collect()
        enabled=gc.isenabled()
        gc.disable()
        try:
            parameter=torch.nn.Parameter(torch.ones(4))
            optimizer=torch.optim.AdamW([parameter])
            step=torch.tensor(1.)
            optimizer.state[parameter]=dict(step=step,exp_avg=torch.ones(4),exp_avg_sq=torch.ones(4))
            old_storage=weakref.ref(optimizer.state[parameter]['exp_avg'].untyped_storage())
            torch.save(optimizer.state_dict(),io.BytesIO())
            self.assertIsNotNone(old_storage())
            move_optimizer_state(optimizer,'meta')
            self.assertIsNone(old_storage())
            self.assertIs(optimizer.state[parameter]['step'],step)
            self.assertEqual(parameter.device.type,'cpu')
        finally:
            gc.collect()
            if enabled:gc.enable()


if __name__=='__main__':unittest.main()
