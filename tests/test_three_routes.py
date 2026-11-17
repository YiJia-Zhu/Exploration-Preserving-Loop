"""Research-route semantics: exact budgets, real CDF exits and hybrid deployment."""
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock

import torch
from ep_loop.config import load_config, validate_config
from ep_loop.data import ByteTokenizer
from ep_loop.objective import DistillationObjective
from ep_loop.ouro import tiny_ouro
from ep_loop.selection import make_stage_plan_from_js

ROOT=Path(__file__).resolve().parents[1]


class ThreeRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.template=tiny_ouro(ROOT/'models/Ouro-1.4B-Thinking')

    def backend(self):
        return type(self.template)(copy.deepcopy(self.template.model),'none')

    def test_fixed_top_n_keeps_exactly_n_and_endpoint_even_at_zero_js(self):
        js=torch.zeros(3,5)
        for n in [1,2,3,4]:
            plan=make_stage_plan_from_js(js,torch.full((5,),4),n,'top_n',999)
            self.assertEqual(plan.counts.tolist(),[n]*5)
            self.assertEqual(plan.teacher_depths[-1].tolist(),[4]*5)
            self.assertEqual(plan.teacher_depths[:,0].tolist(),list(range(1,n))+[4])
            torch.testing.assert_close(plan.weights.sum(0),torch.ones(5))

    def test_fixed_top_n_uses_rank_not_threshold_and_keeps_temporal_order(self):
        js=torch.tensor([[.03],[.01],[.02]])
        plan=make_stage_plan_from_js(js,torch.tensor([4]),3,'top_n',.9)
        self.assertEqual(plan.teacher_depths[:,0].tolist(),[1,3,4])
        uniform=make_stage_plan_from_js(js,torch.tensor([4]),2,'uniform_n')
        self.assertEqual(uniform.teacher_depths[:,0].tolist(),[2,4])
        native=make_stage_plan_from_js(js,torch.tensor([3]),2,'top_n')
        self.assertEqual(native.teacher_depths[:,0].tolist(),[1,3])
        with self.assertRaisesRegex(ValueError,'fixed teacher endpoint'):
            make_stage_plan_from_js(js,torch.tensor([3]),2,'uniform_n')

    def test_top_n_masks_after_native_exit_and_supports_shallow_teacher(self):
        js=torch.tensor([[.01,.01,.01],[.1,.1,.1],[.9,.9,.9]])
        exits=torch.tensor([1,2,4])
        prefix=make_stage_plan_from_js(js,exits,2,'top_n',shallow_target_placement='prefix')
        self.assertEqual(prefix.teacher_depths.tolist(),[[1,1,3],[-1,2,4]])
        self.assertEqual(prefix.counts.tolist(),[1,2,2])
        torch.testing.assert_close(prefix.weights.sum(0),torch.ones(3))
        last=make_stage_plan_from_js(js,exits,2,'top_n',shallow_target_placement='last')
        self.assertEqual(last.teacher_depths[:,0].tolist(),[-1,1])

    def test_prefilled_teacher_readouts_preserve_full_objective_and_gradients(self):
        from collections import deque
        from safetensors.torch import save_file
        from ep_loop.fastpath.teacher import VllmTeacher
        algorithm=load_config(ROOT/'configs/vllm_fixed_top_n.json')['algorithm']
        ids=torch.tensor([[1,20,30,40,50,2]])
        teacher=self.backend();teacher.model.eval().requires_grad_(False)
        with torch.no_grad():trace=teacher.response_trace(ids[:,:-1],4,0)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);file=folder/'trace.safetensors'
            save_file(dict(states=torch.stack(trace.states),gates=torch.stack(trace.gates)),str(file))
            students=[self.backend(),self.backend()]
            frozen=VllmTeacher(students[1],load_config(ROOT/'configs/vllm_fixed_top_n.json'),folder,0,None)
            frozen.pending=deque([('example',0,dict(path=str(file),input_positions=5,executed_token_loops=20))])
            frozen.use_response('example',0)
            results=[]
            for student,source in zip(students,[teacher,frozen]):
                loss,info=DistillationObjective(student,source,algorithm,2,True)(ids,3,[2,2,2])
                loss.backward()
                results.append((loss.detach(),{n:p.grad for n,p in student.model.named_parameters()}))
            torch.testing.assert_close(results[0][0],results[1][0])
            for name,gradient in results[0][1].items():
                if gradient is None:self.assertIsNone(results[1][1][name])
                else:torch.testing.assert_close(gradient,results[1][1][name])
            self.assertFalse(file.exists())
            self.assertTrue(all(p.grad is None for p in frozen.model.parameters()))
            frozen.close()

    def test_gate_endpoint_one_supervises_only_student_loop_one(self):
        student=self.backend();teacher=self.backend();teacher.model.eval().requires_grad_(False)
        with torch.no_grad():teacher.model.model.early_exit_gate.bias.fill_(100.)
        algorithm=load_config(ROOT/'configs/vllm_fixed_top_n.json')['algorithm']
        loss,info=DistillationObjective(student,teacher,algorithm,2,True)(torch.tensor([[1,20,30,40,50,2]]),3,[2,2,2])
        loss.backward()
        self.assertEqual(info['mapping_hist'].tolist(),[[3,0,0,0],[0,0,0,0]])
        self.assertEqual(info['exit_hist'].tolist(),[3,0,0,0])
        self.assertEqual(info['student_executed_loops'],2)
        self.assertEqual(float(info['stop_loss']),0.)

    def test_teacher_prefill_capture_reassembles_chunks_and_counts_recomputation(self):
        from types import SimpleNamespace
        from safetensors.torch import load_file
        from ep_loop.fastpath.vllm_worker import OuroWorkerExtension
        model=SimpleNamespace(model=SimpleNamespace(teacher_trace=True,total_ut_steps=4),config=SimpleNamespace(hidden_size=2))
        original=lambda hidden,tokens:{}
        runner=SimpleNamespace(_get_prompt_logprobs_dict=original,
            input_batch=SimpleNamespace(req_ids=['a'],req_id_to_index={'a':0}),
            query_start_loc=SimpleNamespace(np=[0]),requests={'a':SimpleNamespace(num_computed_tokens=0)})
        worker=OuroWorkerExtension();worker.get_model=lambda:model;worker.model_runner=runner
        with tempfile.TemporaryDirectory() as directory:
            worker.ep_begin_teacher_trace(directory)
            first=torch.arange(24,dtype=torch.float32).reshape(2,12)
            runner._get_prompt_logprobs_dict(first,{'a':2})
            runner.requests['a'].num_computed_tokens=2
            runner._get_prompt_logprobs_dict(torch.full((1,12),30.),{'a':1})
            runner.requests['a'].num_computed_tokens=0
            runner._get_prompt_logprobs_dict(torch.full((1,12),50.),{'a':1})
            result=worker.ep_end_teacher_trace()['a'];saved=load_file(result['path'])
            self.assertIs(runner._get_prompt_logprobs_dict,original)
            self.assertEqual(result['input_positions'],3)
            self.assertEqual(result['executed_token_loops'],16)
            self.assertEqual(saved['states'].shape,(4,3,2))
            self.assertTrue((saved['states'][:,0]==50).all())
            self.assertTrue((saved['states'][:,2]==30).all())

    def test_native_cdf_stops_after_two_loops_not_after_first_partial_trace(self):
        backend=self.backend();backend.model.eval()
        ids=torch.tensor([[1,20,30]])
        gate=backend.model.model.early_exit_gate
        calls=[]
        hook=backend.model.model.layers[0].register_forward_hook(lambda *args:calls.append(1))
        try:
            with torch.no_grad(),patch.object(gate,'forward',side_effect=lambda hidden:hidden.new_zeros((*hidden.shape[:2],1))):
                trace=backend.trace(ids,4,use_cache=True,endpoint='native_adaptive',exit_threshold=.7)
        finally:hook.remove()
        self.assertEqual(trace.exits[0,-1].item(),2)
        self.assertEqual(len(calls),2)
        self.assertEqual(trace.executed_token_loops,6)

    def test_adaptive_teacher_labels_match_full_prefix_with_late_backfill(self):
        backend=self.backend();backend.model.eval()
        ids=torch.tensor([[1,20,30,40,50,60]])
        schedule=[1,4,2,3];position=[0];depth=[0]
        def gate(hidden):
            depth[0]+=1
            stop=depth[0]>=schedule[position[0]]
            if stop:position[0]+=1;depth[0]=0
            return hidden.new_full((*hidden.shape[:2],1),100. if stop else -100.)
        with torch.no_grad():
            with patch.object(backend.model.model.early_exit_gate,'forward',side_effect=gate):
                adaptive=backend.response_trace(ids,4,2,'native_adaptive',.7)
            full=backend.response_trace(ids,4,2)
        self.assertEqual(adaptive.exits.tolist(),schedule)
        self.assertEqual(adaptive.executed_token_loops,22)
        self.assertEqual(adaptive.backfill_token_loops,10)
        for token,visited in enumerate(schedule):
            for loop in range(visited):
                torch.testing.assert_close(adaptive.states[loop][token],full.states[loop][token],rtol=3e-4,atol=3e-6)

    def test_true_adaptive_teacher_keeps_only_one_stage_and_stays_frozen(self):
        student=self.backend();teacher=self.backend()
        teacher.model.eval().requires_grad_(False)
        with torch.no_grad():teacher.model.model.early_exit_gate.bias.fill_(100.)
        algorithm=load_config(ROOT/'configs/hf_adaptive.json')['algorithm']
        objective=DistillationObjective(student,teacher,algorithm,2,True)
        loss,info=objective(torch.tensor([[1,20,30,40,50,2]]),3,[1,1,1])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(info['target_depth_hist'].tolist(),[3,0,0,0])
        self.assertEqual(info['teacher_executed_token_loops'],5)
        self.assertIsNone(info['teacher_executed_loops'])
        self.assertTrue(all(parameter.grad is None for parameter in teacher.model.parameters()))

    def test_fixed_objective_has_n_targets_and_no_stop_bce(self):
        student=self.backend();teacher=self.backend()
        teacher.model.eval().requires_grad_(False)
        algorithm=load_config(ROOT/'configs/vllm_fixed_uniform.json')['algorithm']
        objective=DistillationObjective(student,teacher,algorithm,2,True)
        loss,info=objective(torch.tensor([[1,20,30,40,50,2]]),3,[2,2,2])
        loss.backward()
        self.assertEqual(info['stage_hist'].tolist(),[0,0,3])
        self.assertEqual(info['mapping_hist'].tolist(),[[0,3,0,0],[0,0,0,3]])
        self.assertEqual(float(info['stop_loss']),0.)
        self.assertIsNone(student.model.model.early_exit_gate.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.model.parameters()))

    def test_fixed_rollout_actor_really_receives_student_n_not_teacher_k(self):
        from ep_loop.fastpath.rollout import RolloutBatcher
        captures=[]
        class Actor:
            def __init__(self,*args,**kwargs):self.loops=kwargs['loops'];captures.append(kwargs)
            def request(self,op,**kwargs):return {'weight_version':kwargs['version']} if op=='weights' else None
            def generate(self,requests,**kwargs):
                return dict(seconds=.1,executed_loops=self.loops,responses=[dict(token_ids=[40,2],exit_depths=[self.loops]*2,
                    executed_token_loops=self.loops*(len(row['input_ids'])+1)) for row in requests])
            def close(self):pass
        class Observer:
            def stage(self,*args,**kwargs):return nullcontext()
            def event(self,*args,**kwargs):pass
        config=load_config(ROOT/'configs/vllm_fixed_top_n.json')
        config['rollout']['responses_per_prompt']=1
        with tempfile.TemporaryDirectory() as folder,patch('ep_loop.fastpath.rollout.VllmActor',Actor),patch('torch.cuda.empty_cache'):
            batcher=RolloutBatcher(self.backend(),config,folder,0,Observer())
            batcher.prepare([dict(id='x',input_ids=[1,20,30])],[0],0,1,0,torch.Generator().manual_seed(42))
            response=batcher.pop(dict(id='x',sample=0))
            self.assertEqual(captures[0]['loops'],2)
            self.assertEqual(captures[0]['policy'],'fixed')
            self.assertNotIn('max_num_seqs',captures[0])
            self.assertEqual(response.execution_depths,[2,2])
            self.assertEqual(response.cache_policy,'vllm_fixed_depth')
            self.assertFalse(list(Path(folder).glob('.vllm_weights*')))

    def test_paper_batch_submits_all_128_responses_per_rank_together(self):
        from ep_loop.fastpath.rollout import RolloutBatcher
        submissions=[]
        class Actor:
            def __init__(self,*args,**kwargs): self.loops=kwargs['loops']
            def request(self,op,**kwargs):
                return {'weight_version':kwargs['version']} if op=='weights' else None
            def generate(self,requests,**kwargs):
                submissions.append(len(requests))
                return dict(seconds=.1,executed_loops=self.loops,responses=[dict(
                    token_ids=[40,2],exit_depths=[self.loops]*2,
                    executed_token_loops=self.loops*(len(row['input_ids'])+1)) for row in requests])
            def close(self): pass
        class Observer:
            def stage(self,*args,**kwargs): return nullcontext()
            def event(self,*args,**kwargs): pass
        config=load_config(ROOT/'configs/vllm_fixed_top_n.json')
        self.assertEqual(config['rollout']['prompts_per_batch'],32)
        self.assertEqual(config['rollout']['responses_per_prompt'],8)
        records=[dict(id=str(i),input_ids=[1,20,30]) for i in range(32)]
        consumed=0
        with tempfile.TemporaryDirectory() as directory,patch('ep_loop.fastpath.rollout.VllmActor',Actor),patch('torch.cuda.empty_cache'):
            for rank in range(2):
                batcher=RolloutBatcher(self.backend(),config,directory,rank,Observer())
                batcher.prepare(records,list(range(32)),rank,2,0,torch.Generator().manual_seed(42+rank))
                for record in records[rank::2]:
                    for sample in range(8):
                        batcher.pop(dict(id=record['id'],sample=sample));consumed+=1
                self.assertFalse(batcher.pending)
                batcher.close()
        self.assertEqual(submissions,[128,128])
        self.assertEqual(consumed,256)

    def test_hybrid_evaluates_one_checkpoint_at_all_fixed_budgets(self):
        from scripts.evaluate_fastpath import main
        captures=[];submissions=[]
        class Actor:
            def __init__(self,directory,gpu,log,**kwargs):
                self.loops=kwargs['loops'];self.gpu=gpu
                if 'max_num_seqs' in kwargs:raise AssertionError('Use the vLLM scheduler default')
                captures.append((str(directory),kwargs['loops'],kwargs['policy']))
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def generate(self,requests,**kwargs):
                submissions.append((self.gpu,self.loops,len(requests)))
                return dict(seconds=.1,responses=[dict(token_ids=[2],exit_depths=[self.loops],finish_reason='stop',
                    executed_token_loops=self.loops*len(row['input_ids'])) for row in requests])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data.jsonl';output=root/'result.jsonl'
            data.write_text(''.join(json.dumps(dict(id=str(i),problem='1+1?',answer='2'))+'\n' for i in range(10)))
            args=['evaluate_fastpath.py','--config',str(ROOT/'configs/hybrid.json'),'--checkpoint',directory,
                  '--data',str(data),'--output',str(output),'--gpus','2,3','--samples-per-problem','1']
            with patch('sys.argv',args),patch('transformers.AutoTokenizer.from_pretrained',return_value=ByteTokenizer()),\
                    patch('ep_loop.fastpath.actor.VllmActor',Actor):
                main()
            results=[json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(results),40)
            self.assertEqual({row['max_loops'] for row in results},{1,2,3,4})
            self.assertTrue(all(row['policy']=='fixed' and row['actual_execution_depths']==[row['max_loops']] for row in results))
            self.assertEqual({row[0] for row in captures},{directory})
            self.assertTrue(all(row[2]=='fixed' for row in captures))
            self.assertEqual(sorted(submissions),[(gpu,depth,5) for gpu in ['2','3'] for depth in [1,2,3,4]])

    def test_vllm_actor_does_not_inherit_incompatible_hf_allocator(self):
        from ep_loop.fastpath.actor import VllmActor
        process=MagicMock()
        process.stdout=iter(['{"ready":true}\n'])
        process.poll.return_value=0
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,
                {'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','RANK':'1','WORLD_SIZE':'2'}),\
                patch('subprocess.Popen',return_value=process) as spawn:
            actor=VllmActor('unused','3',Path(directory)/'actor.log',loops=2,policy='fixed',sleep=True)
            child=spawn.call_args.kwargs['env']
            self.assertNotIn('--max-num-seqs',spawn.call_args.args[0])
            self.assertNotIn('PYTORCH_CUDA_ALLOC_CONF',child)
            self.assertNotIn('RANK',child)
            self.assertEqual(child['CUDA_VISIBLE_DEVICES'],'3')
            self.assertEqual(os.environ['PYTORCH_CUDA_ALLOC_CONF'],'expandable_segments:True')
            actor.close()

    def test_route_configs_validate_without_teacher_distribution_approximation(self):
        for name in ['hf_adaptive','hybrid','vllm_fixed_top_n','vllm_fixed_uniform','vllm_fixed_endpoint']:
            config=validate_config(load_config(ROOT/'configs'/f'{name}.json'))
            self.assertEqual(config['algorithm']['teacher_distribution'],'full_vocab')
            self.assertIsNone(config['algorithm']['teacher_top_k'])


if __name__=='__main__':unittest.main()
