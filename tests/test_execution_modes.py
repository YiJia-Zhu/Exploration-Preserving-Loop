"""The default must keep real early stopping and the full teacher distribution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from ep_loop.config import DEFAULT_CONFIG, load_config, validate_config
from ep_loop.execution import execution_mode

ROOT=Path(__file__).resolve().parents[1]


class ExecutionModeTests(unittest.TestCase):
    def test_default_and_optional_profiles_are_explicit(self):
        for path,mode,backend,k in [
            (DEFAULT_CONFIG,'hf_rad_opd','hf',None),
            (ROOT/'configs/hf_dynamic_js.json','hf_dynamic','hf',None),
            (ROOT/'configs/e_dynamic_js_vllm_top20.json','vllm_top20','vllm',20),
            (ROOT/'configs/e_dynamic_js_hf_top20.json','hf_top20','hf',20),
        ]:
            config=validate_config(load_config(path))
            self.assertEqual(execution_mode(config),mode)
            self.assertEqual(config['implementation']['rollout_backend'],backend)
            self.assertEqual(config['algorithm']['teacher_top_k'],k)
        default=load_config(DEFAULT_CONFIG)
        self.assertEqual(default['algorithm']['teacher_distribution'],'full_vocab')
        self.assertEqual(default['algorithm']['student_policy'],'ragged')
        self.assertIsNone(default['algorithm']['student_loops'])

    def test_approximation_cannot_be_labeled_as_the_full_default(self):
        config=load_config(DEFAULT_CONFIG)
        config['algorithm'].update(teacher_top_k=20,teacher_distribution='top20_original_mass_uniform_tail')
        with self.assertRaisesRegex(ValueError,'full-vocabulary'):
            validate_config(config)
        config=load_config(ROOT/'configs/e_dynamic_js_vllm_top20.json')
        config['implementation']['rollout_backend']='hf'
        with self.assertRaisesRegex(ValueError,'rollout_backend=vllm'):
            validate_config(config)

    def test_historical_explicit_approximation_keeps_its_identity(self):
        config=load_config(ROOT/'configs/e_dynamic_js_vllm_top20.json')
        del config['implementation']['execution_mode']
        self.assertEqual(execution_mode(validate_config(config)),'vllm_top20')
        config['implementation']['rollout_backend']='hf'
        self.assertEqual(execution_mode(validate_config(config)),'hf_top20')

    def test_both_shell_entries_default_to_hf_full_and_allow_explicit_modes(self):
        for entry in ['run_ablation.sh','run_fastpath_ablation.sh']:
            for selected in [None,'vllm_top20','hf_top20']:
                env=os.environ.copy()
                for name in ['EXECUTION_MODE','FASTPATH_ROLLOUT_BACKEND','OUTPUT_ROOT','RUN_TAG']:
                    env.pop(name,None)
                if selected:env['EXECUTION_MODE']=selected
                result=subprocess.run(['bash',str(ROOT/'scripts'/entry),'--dry-run'],
                    cwd=ROOT,env=env,text=True,capture_output=True,check=True,timeout=20)
                plan,_=json.JSONDecoder().raw_decode(result.stdout)
                mode=selected or 'hf_dynamic'
                self.assertEqual(plan['execution_mode'],mode)
                self.assertEqual(plan['rollout_backend'],'vllm' if mode=='vllm_top20' else 'hf')
                self.assertEqual(plan['evaluation_backend'],plan['rollout_backend'])
                self.assertEqual(plan['teacher_top_k'],None if mode=='hf_dynamic' else 20)
                self.assertEqual(plan['global_batch'],256)
                self.assertEqual(plan['max_new_tokens'],16384)
                self.assertEqual(plan['eval_examples'],30)
                self.assertEqual(plan['eval_suite'],'aime')
                self.assertEqual(plan['eval_samples_per_problem'],16)
                trainer='train_stage_distillation.py' if mode=='hf_dynamic' else 'train_fastpath_distillation.py'
                self.assertIn(trainer,result.stdout)

    def test_fastpath_training_api_routes_default_to_reference(self):
        from ep_loop.fastpath.training import train
        with patch('ep_loop.training.train',return_value='reference') as reference:
            self.assertEqual(train(DEFAULT_CONFIG,'unused','cpu',True),'reference')
            reference.assert_called_once_with(DEFAULT_CONFIG,'unused','cpu',True,None,None)

    def test_generic_training_api_preserves_explicit_optional_selection(self):
        from ep_loop.training import train
        path=ROOT/'configs/e_dynamic_js_hf_top20.json'
        with patch('ep_loop.fastpath.training.train',return_value='optional') as optional:
            self.assertEqual(train(path,'unused','cpu',True),'optional')
            optional.assert_called_once_with(path,'unused','cpu',True,None,None)

    def test_fast_evaluator_default_delegates_before_loading_vllm(self):
        from scripts.evaluate_fastpath import main
        arguments=['evaluate_fastpath.py','--data','unused.jsonl','--output','unused_output.jsonl',
                   '--gpus','3,2','--device','cpu','--smoke']
        with patch.object(sys,'argv',arguments),patch('os.execve') as replace:
            main()
        executable,command,environment=replace.call_args.args
        self.assertEqual(executable,sys.executable)
        self.assertTrue(command[2].endswith('evaluate_stage_distillation.py'))
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'],'3')
        self.assertIn('--smoke',command)


if __name__=='__main__':
    unittest.main()
