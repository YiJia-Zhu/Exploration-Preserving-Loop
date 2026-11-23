"""Runtime separation and fixed native decode protocol, without GPU execution."""
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class VllmRuntimeTests(unittest.TestCase):
    def test_teacher_trace_mapping_keeps_sample_identity_with_v026_internal_ids(self):
        from ep_loop.fastpath.actor import align_teacher_traces
        self.assertEqual(align_teacher_traces({'1-123456ab':'b','0-abcdef01':'a'},['0','1']),['a','b'])
        self.assertEqual(align_teacher_traces({'0':'a'},['0']),['a'])
        for traces,ids in [({'0-abcdef01':'a','0-abcdef02':'b'},['0']),
                           ({'0-abcdef01':'a'},['1']),
                           ({'0':'a','1':'b'},['0'])]:
            with self.assertRaises(RuntimeError):align_teacher_traces(traces,ids)

    def test_small_cuda_reservation_is_not_an_idle_gpu(self):
        from ep_loop.gpu_resources import require_idle_gpus
        row=dict(index='2',memory_used_MiB=345,processes=[dict(pid=123,name='reservation')])
        with patch('ep_loop.gpu_resources.gpu_status',return_value=[row]):
            with self.assertRaisesRegex(RuntimeError,'reserved'):require_idle_gpus(['2'])

    def test_driver_only_overhead_does_not_require_zero_gpu_utilization(self):
        from ep_loop.gpu_resources import require_idle_gpus
        row=dict(index='2',memory_used_MiB=3,processes=[])
        with patch('ep_loop.gpu_resources.gpu_status',return_value=[row]):
            self.assertEqual(require_idle_gpus(['2']),[row])

    def test_actor_uses_the_isolated_interpreter_and_cleans_trainer_rank(self):
        from ep_loop.fastpath.actor import VllmActor
        process=SimpleNamespace(stdout=io.StringIO('{"ready":true,"vllm_version":"0.26.0"}\n'),
                                poll=lambda:0)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('ep_loop.fastpath.actor.subprocess.Popen',return_value=process) as spawn, \
             patch.dict('os.environ',{'PATH':'/usr/bin','RANK':'1','WORLD_SIZE':'2','MASTER_PORT':'12345'}):
            with VllmActor('unused','3',Path(tmp)/'engine.log',policy='fixed',
                           python_bin='/isolated/python',attention_backend='FLASHINFER') as actor:
                self.assertEqual(actor.runtime['vllm_version'],'0.26.0')
            command=spawn.call_args.args[0]
            self.assertEqual(command[0],'/isolated/python')
            self.assertEqual(command[-2:],['--attention-backend','FLASHINFER'])
            env=spawn.call_args.kwargs['env']
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'3')
            self.assertEqual(env['PATH'].split(os.pathsep), ['/isolated', '/usr/bin'])
            self.assertFalse({'RANK','WORLD_SIZE','MASTER_PORT'} & set(env))

    def test_modern_native_student_keeps_requested_depth_and_batch_submission(self):
        self.check_native_submission(None)

    def test_engine_concurrency_limit_keeps_one_generate_call_for_all_requests(self):
        self.check_native_submission(2)

    def test_native_tp2_still_submits_the_whole_batch_once(self):
        self.check_native_submission(None,tensor_parallel_size=2)

    def test_training_ngram_preserves_native_depth_and_batch_submission(self):
        self.check_native_submission(None,rollout_ngram=True)

    def test_concurrency_configuration_rejects_invalid_counts(self):
        from ep_loop.config import load_config,validate_config
        for value in (-1,True,1.5):
            config=load_config(ROOT/'configs/vllm_fixed_top_n.json')
            config['implementation']['vllm_max_num_seqs']=value
            with self.assertRaisesRegex(ValueError,'vllm_max_num_seqs'):validate_config(config)
        for value in (0,64,128):
            config=load_config(ROOT/'configs/vllm_fixed_top_n.json')
            config['implementation']['vllm_max_num_seqs']=value
            validate_config(config)

    def check_native_submission(self,maximum,tensor_parallel_size=1,rollout_ngram=False):
        captures = []
        rpc = []

        class Engine:
            def __init__(self, **kwargs):
                captures.append(kwargs)

            def generate(self, prompts, sampling, use_tqdm):
                self.requests = prompts
                self.params = sampling
                captures.append((prompts, sampling))
                return [SimpleNamespace(request_id=str(i), prompt_token_ids=p['prompt_token_ids'],
                    outputs=[SimpleNamespace(token_ids=[10, 11], logprobs=None, finish_reason='length')])
                    for i, p in enumerate(prompts)]

            def get_metrics(self):
                return []

            def collective_rpc(self, name, **kwargs):
                rpc.append(name)
                raise AssertionError('Native fixed-depth decode does not need a gate trace RPC')

        registry = SimpleNamespace(register_model=lambda *args: None)
        fake = SimpleNamespace(ModelRegistry=registry, LLM=Engine,
                               SamplingParams=lambda **kwargs: kwargs)
        request = dict(op='generate', logprobs=None, requests=[
            dict(input_ids=[1, 2], temperature=1., top_p=1., max_new_tokens=2, seed=42+i)
            for i in range(3)])
        output = io.StringIO()
        argv=['server','--model','unused','--loops','2','--policy','fixed','--attention-backend','FLASHINFER']
        argv+=['--tensor-parallel-size',str(tensor_parallel_size)]
        if maximum is not None:argv+=['--max-num-seqs',str(maximum)]
        if rollout_ngram:argv+=['--rollout-ngram']
        with patch.dict('sys.modules', {'vllm': fake}), \
             patch('importlib.metadata.version', return_value='0.26.0'), \
             patch('sys.argv', argv), \
             patch('sys.stdin', io.StringIO(json.dumps(request)+'\n')), \
             patch('sys.stdout', output):
            module = runpy.run_path(str(ROOT/'ep_loop/fastpath/vllm_server.py'),
                                    run_name='_vllm_runtime_test')
            module['main']()
        ready, response = map(json.loads, output.getvalue().splitlines())
        self.assertTrue(ready['native_fixed_student'])
        kwargs = captures[0]
        self.assertEqual(kwargs['attention_config'], {'backend': 'FLASHINFER'})
        self.assertEqual(kwargs['hf_overrides']['total_ut_steps'], 2)
        self.assertEqual(kwargs['hf_overrides']['architectures'], ['OuroForCausalLM'])
        self.assertEqual(kwargs['dtype'],'bfloat16')
        self.assertNotIn('kv_cache_dtype',kwargs)
        self.assertEqual(kwargs['tensor_parallel_size'],tensor_parallel_size)
        self.assertEqual(ready['tensor_parallel_size'],tensor_parallel_size)
        if maximum is None:self.assertNotIn('max_num_seqs', kwargs)
        else:self.assertEqual(kwargs['max_num_seqs'],maximum)
        self.assertEqual(ready['max_num_seqs'],maximum)
        self.assertEqual(ready['rollout_ngram'],rollout_ngram)
        if rollout_ngram:
            from ep_loop.fastpath.speculation import ngram_speculative_config
            self.assertEqual(kwargs['speculative_config'],ngram_speculative_config())
            self.assertEqual(ready['speculative_config'],ngram_speculative_config())
        else:
            self.assertNotIn('speculative_config',kwargs)
            self.assertIsNone(ready['speculative_config'])
        self.assertEqual(len(captures[1][0]), 3)
        self.assertEqual(len(captures), 2)
        self.assertEqual(rpc, [])
        self.assertEqual([r['exit_depths'] for r in response['result']['responses']], [[2, 2]]*3)


if __name__ == '__main__':
    unittest.main()
