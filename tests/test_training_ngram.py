"""Training-only speculation: boundaries, persistent updates, and resume identity."""
from contextlib import nullcontext
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ep_loop.checkpointing import validate_resume_config
from ep_loop.config import load_config,validate_config
from ep_loop.fastpath.actor import VllmActor
from ep_loop.fastpath.rollout import RolloutBatcher
from ep_loop.fastpath.speculation import (ngram_speculative_config,rollout_ngram_enabled,
                                         validate_ngram_actor)

ROOT=Path(__file__).resolve().parents[1]


class TrainingNgramTests(unittest.TestCase):
    def test_current_fixed_training_profiles_enable_ngram_and_hf_does_not(self):
        for name in ('vllm_fixed_top_n','vllm_fixed_uniform','vllm_fixed_endpoint',
                     'vllm_fixed_top_n_current','vllm_fixed_d_loopopd'):
            config=validate_config(load_config(ROOT/'configs'/f'{name}.json'))
            self.assertTrue(rollout_ngram_enabled(config))
        for name in ('default','hf_shared_ep_d_loopopd','e_dynamic_js_vllm_top20'):
            self.assertFalse(rollout_ngram_enabled(validate_config(load_config(ROOT/'configs'/f'{name}.json'))))
        settings=ngram_speculative_config()
        settings['num_speculative_tokens']=1
        self.assertEqual(ngram_speculative_config()['num_speculative_tokens'],5)

    def test_invalid_flags_and_unsupported_actor_roles_fail_before_launch(self):
        for value in ('true',1,None):
            config=load_config(ROOT/'configs/vllm_fixed_top_n.json')
            config['implementation']['vllm_rollout_ngram']=value
            with self.assertRaisesRegex(ValueError,'boolean'):validate_config(config)
        config=load_config(ROOT/'configs/default.json')
        config['implementation']['vllm_rollout_ngram']=True
        with self.assertRaisesRegex(ValueError,'vllm_fixed'):validate_config(config)
        with patch('ep_loop.fastpath.actor.subprocess.Popen') as spawn:
            for policy,teacher in [('learned',False),('native',False),('fixed',True)]:
                with self.assertRaisesRegex(ValueError,'student rollout'):
                    VllmActor('unused','4','unused.log',policy=policy,teacher_trace=teacher,rollout_ngram=True)
            spawn.assert_not_called()
        with self.assertRaisesRegex(ValueError,'0.26'):
            validate_ngram_actor(True,'fixed',False,modern=False)

    def test_actor_command_is_opt_in_and_never_uses_global_rollout_environment(self):
        for enabled in (False,True):
            process=SimpleNamespace(stdout=io.StringIO('{"ready":true}\n'),poll=lambda:0)
            with tempfile.TemporaryDirectory() as directory, \
                 patch.dict('os.environ',{'VLLM_ROLLOUT_NGRAM':'1'}), \
                 patch('ep_loop.fastpath.actor.subprocess.Popen',return_value=process) as spawn:
                with VllmActor('unused','4',Path(directory)/'engine.log',policy='fixed',rollout_ngram=enabled):pass
                self.assertEqual('--rollout-ngram' in spawn.call_args.args[0],enabled)

    def test_two_rollout_updates_refresh_weights_keep_ids_and_sleep(self):
        class Observer:
            def stage(self,*args,**kwargs):return nullcontext()
            def event(self,*args,**kwargs):pass
        actors=[];versions=[];sleeps=[];requests=[]
        class Actor:
            def __init__(self,*args,**kwargs):
                actors.append(kwargs)
                self.runtime=dict(rollout_ngram=True,speculative_config=ngram_speculative_config())
            def request(self,op,**kwargs):
                if op=='weights':
                    assert Path(kwargs['path']).is_file()
                    versions.append(kwargs['version']);return dict(weight_version=kwargs['version'])
                sleeps.append(op)
            def generate(self,rows,logprobs):
                assert logprobs is None
                requests.extend(rows)
                return dict(executed_loops=2,seconds=.01,responses=[dict(token_ids=[10,2],
                    exit_depths=[2,2],executed_token_loops=2*(len(row['input_ids'])+1)) for row in rows])
            def close(self):pass
        config=load_config(ROOT/'configs/vllm_fixed_top_n_current.json')
        config['rollout'].update(responses_per_prompt=2,max_new_tokens=16384)
        model=torch.nn.Linear(2,2)
        student=SimpleNamespace(model=model,device=torch.device('cpu'),stop_token_ids={2})
        records=[dict(id='a',input_ids=[1,20]),dict(id='b',input_ids=[1,30])]
        generator=torch.Generator().manual_seed(42)
        with tempfile.TemporaryDirectory() as directory, \
             patch('ep_loop.fastpath.rollout.VllmActor',Actor),patch('torch.cuda.empty_cache'):
            batcher=RolloutBatcher(student,config,directory,0,Observer())
            for update in (0,1):
                batcher.prepare(records,[0,1],0,1,update,generator)
                for record in records:
                    for sample in (0,1):
                        response=batcher.pop(dict(id=record['id'],sample=sample))
                        self.assertEqual(response.exit_depths,[2,2])
                self.assertFalse(batcher.pending)
            batcher.close()
            self.assertFalse(list(Path(directory).glob('*.safetensors')))
        self.assertEqual(versions,[0,1]);self.assertEqual(sleeps,['sleep','sleep'])
        self.assertEqual(len(actors),1);self.assertTrue(actors[0]['rollout_ngram'])
        self.assertEqual(actors[0]['max_logprobs'],0)
        self.assertTrue(all(row['max_new_tokens']==16384 for row in requests))
        expected=torch.randint(0,2**63-1,(8,),generator=torch.Generator().manual_seed(42)).tolist()
        self.assertEqual([row['seed'] for row in requests],expected)

    def test_resume_can_keep_legacy_normal_decode_but_cannot_toggle_ngram(self):
        saved=load_config(ROOT/'configs/vllm_fixed_top_n.json')
        saved['implementation'].pop('vllm_rollout_ngram')
        current=copy.deepcopy(saved);current['implementation']['vllm_rollout_ngram']=False
        validate_resume_config(saved,current,0)
        current['implementation']['vllm_rollout_ngram']=True
        with self.assertRaisesRegex(ValueError,'unchanged'):validate_resume_config(saved,current,0)

    def test_launcher_switch_does_not_change_runtime_or_evaluation(self):
        for enabled in ('0','1'):
            env=os.environ.copy()
            for key in ('VLLM_PYTHON_BIN','FASTPATH_ROLLOUT_BACKEND','OUTPUT_ROOT','RUN_TAG'):
                env.pop(key,None)
            env.update(EXECUTION_MODE='vllm_fixed',VLLM_ROLLOUT_NGRAM=enabled,
                       MODEL_FAMILY='ouro',STAGE_SELECTION='top_n',METHOD='ours',MAX_NEW_TOKENS='16384')
            result=subprocess.run(['bash',str(ROOT/'scripts/run_fastpath_ablation.sh'),'--dry-run'],
                cwd=ROOT,env=env,text=True,capture_output=True,check=True,timeout=20)
            plan,_=json.JSONDecoder().raw_decode(result.stdout)
            self.assertEqual(plan['vllm_rollout_ngram'],enabled=='1')
            self.assertIsNone(plan['evaluation_speculative_config'])
            self.assertEqual(Path(plan['vllm_python_bin']).resolve(),(ROOT/'ep_loop_envs/vllm026/bin/python').resolve())
            self.assertEqual(plan['max_new_tokens'],16384)


if __name__=='__main__':unittest.main()
