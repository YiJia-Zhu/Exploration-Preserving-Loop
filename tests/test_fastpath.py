"""Readout policy preservation and the only allowed top-20 approximation."""
import copy
from pathlib import Path
import unittest
import torch
from ep_loop.fastpath.sparse import TopKDistribution,adjacent_js,stage_reverse_kl
from ep_loop.fastpath.readout import select_readout
from ep_loop.selection import adjacent_js_from_log_probs,make_dynamic_stage_plan,native_ouro_exits


class FastpathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)

    def test_sparse_distribution_keeps_mass_and_nonzero_tail(self):
        logits=torch.randn(4,7,101)
        sparse=TopKDistribution.from_logits(logits)
        dense=sparse.dense_log_probs()
        torch.testing.assert_close(dense.exp().sum(-1),torch.ones(4,7),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(dense.gather(-1,sparse.token_ids.long()),logits.log_softmax(-1).gather(-1,sparse.token_ids.long()))
        self.assertEqual(sparse.token_ids.shape[-1],20)

    def test_sparse_js_matches_dense_reconstructed_distribution(self):
        for vocabulary in [13,23,101]:
            sparse=TopKDistribution.from_logits(torch.randn(4,7,vocabulary))
            torch.testing.assert_close(adjacent_js(sparse),adjacent_js_from_log_probs(sparse.dense_log_probs()),atol=2e-7,rtol=2e-5)

    def test_analytic_reverse_kl_and_gradient_match_dense_reconstruction(self):
        teacher=TopKDistribution.from_logits(torch.randn(4,9,73))
        plan=make_dynamic_stage_plan(adjacent_js(teacher),torch.tensor([1,2,3,4,4,3,2,1,4]),js_threshold=.02)
        original=torch.randn(4,9,73)
        sparse_input=original.clone().requires_grad_()
        dense_input=original.clone().requires_grad_()
        from ep_loop.selection import stage_reverse_kl as dense_loss
        actual=stage_reverse_kl(sparse_input,teacher,plan).mean()
        expected=dense_loss(dense_input,teacher.dense_log_probs(),plan).mean()
        actual.backward();expected.backward()
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(sparse_input.grad,dense_input.grad,atol=1e-7,rtol=2e-5)

    def test_fixed_compute_readout_matches_first_local_stop_and_native_cdf(self):
        states=[torch.randn(5,8) for _ in range(4)]
        gates=[torch.tensor([[-20.],[-20.],[20.],[-20.],[0.]]) for _ in range(4)]
        gates[1][1]=20.;gates[2][0]=20.
        selected,depths=select_readout(states,gates,'learned')
        self.assertEqual(depths.tolist(),[3,2,1,4,1])
        for i,d in enumerate(depths):torch.testing.assert_close(selected[i],states[int(d)-1][i])
        _,native=select_readout(states,gates,'native')
        torch.testing.assert_close(native,native_ouro_exits(torch.stack(gates),.9))

    def test_depth_trace_excludes_partial_prefill_tokens_discarded_by_vllm(self):
        from types import SimpleNamespace
        from ep_loop.fastpath.vllm_worker import OuroWorkerExtension
        worker=OuroWorkerExtension();model=SimpleNamespace()
        worker.get_model=lambda:model
        worker.model_runner=SimpleNamespace(input_batch=SimpleNamespace(req_ids=['a','b']),
            discard_request_indices=SimpleNamespace(np=torch.tensor([1])),num_discarded_requests=1)
        worker.ep_start_trace()
        model._depth_callback(torch.tensor([[2.,.1,.2],[4.,9.,9.]]))
        worker.model_runner.input_batch.req_ids=['b']
        worker.model_runner.num_discarded_requests=0
        model._depth_callback(torch.tensor([[3.,.3,.4]]))
        self.assertEqual(worker._ep_trace_counts,{'a':1,'b':1})
        result=worker.ep_finish_trace()
        self.assertEqual(result['a']['depths'],[2])
        self.assertEqual(result['b']['depths'],[3])
        torch.testing.assert_close(torch.tensor(result['b']['gate_logits']),torch.tensor([[.3,.4]]))
        self.assertIsNone(model._depth_callback)

    def test_native_readout_tracks_first_tokens_in_mixed_and_chunked_batches(self):
        from types import SimpleNamespace
        from ep_loop.fastpath.vllm_worker import OuroWorkerExtension

        worker = OuroWorkerExtension()
        model = SimpleNamespace()
        worker.get_model = lambda: model

        def sample(_hidden, batch, _grammar):
            depths = torch.where(model._ep_prefill_readout_mask, 4., 2.)
            model._depth_callback(torch.stack((depths, torch.zeros_like(depths)), dim=-1))
            return None, batch.sampled, None

        worker.model_runner = SimpleNamespace(sample=sample, req_states=SimpleNamespace(
            prompt_len=SimpleNamespace(gpu=torch.tensor([5, 3]))))
        worker.ep_begin_native_readout()
        for ids, slots, positions, logits, sampled in (
            (['a', 'b'], [1, 0], [0, 1, 2, 0, 1], [2, 4], [1, 0]),
            (['b', 'a'], [0, 1], [2, 3, 4, 3], [2, 3], [1, 1]),
            (['b'], [0], [5], [0], [1]),
        ):
            batch = SimpleNamespace(req_ids=ids, idx_mapping=torch.tensor(slots),
                                    positions=torch.tensor(positions), logits_indices=torch.tensor(logits),
                                    sampled=torch.tensor(sampled))
            worker.model_runner.sample(None, batch, None)
        self.assertEqual(worker.ep_end_native_readout(), {
            'a': {0: [4., 0.], 1: [2., 0.]}, 'b': {0: [4., 0.], 1: [2., 0.]},
        })
        self.assertIs(worker.model_runner.sample, sample)
        self.assertIsNone(model._ep_prefill_readout_mask)

    def test_full_vocab_limit_preserves_original_objective_stop_labels_and_gradients(self):
        from ep_loop.config import load_config
        from ep_loop.ouro import tiny_ouro
        from ep_loop.objective import DistillationObjective as Original
        from ep_loop.fastpath.objective import DistillationObjective as Sparse
        root=Path(__file__).resolve().parents[1]
        template=tiny_ouro(root/'models/Ouro-1.4B-Thinking')
        algorithm=load_config(root/'configs/e_dynamic_js.json')['algorithm']
        results=[]
        for objective_type in [Original,Sparse]:
            student=type(template)(copy.deepcopy(template.model),'none')
            teacher=type(template)(copy.deepcopy(template.model).eval().requires_grad_(False),'none')
            objective=objective_type(student,teacher,algorithm,2,True)
            if isinstance(objective,Sparse):objective.top_k=260
            loss,info=objective(torch.tensor([[1,20,30,40,50,2]]),3,[1,3,4]);loss.backward()
            results.append((loss.detach(),info,{k:p.grad.clone() for k,p in student.model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0],results[1][0],atol=2e-6,rtol=2e-6)
        for key in ['target_depth_hist','target_actual_confusion','mapping_hist','stage_hist']:
            torch.testing.assert_close(results[0][1][key],results[1][1][key])
        for key,value in results[0][2].items():
            torch.testing.assert_close(value,results[1][2][key],atol=2e-6,rtol=2e-4)

    def test_hf_cached_fastpath_keeps_exact_tokens_depths_and_random_stream(self):
        from ep_loop.ouro import tiny_ouro
        from ep_loop.inference import generate as original
        from ep_loop.fastpath.inference import generate as optimized
        backend=tiny_ouro(Path(__file__).resolve().parents[1]/'models/Ouro-1.4B-Thinking')
        backend.enable_student_exit()
        for policy,loops in [('learned',4),('native',4),('fixed',2)]:
            outputs=[];states=[]
            for function in [original,optimized]:
                generator=torch.Generator().manual_seed(42)
                response=function(backend,[1,20,30],loops,12,1.,.7,generator,policy,.5,True)
                outputs.append((response.token_ids,response.exit_depths,response.executed_token_loops))
                states.append(generator.get_state())
            self.assertEqual(outputs[0],outputs[1])
            torch.testing.assert_close(states[0],states[1],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
