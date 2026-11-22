"""Every-five evaluation, model-only persistence, and distributed RNG isolation."""
from collections import deque
from contextlib import nullcontext,redirect_stdout,redirect_stderr
from datetime import timedelta
import json
import io
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ep_loop.config import load_config,validate_config
from ep_loop.fastpath.checkpoints import checkpoint_due,save_split_checkpoint
from ep_loop.fastpath.evaluation import (evaluation_due,evaluation_datasets,evaluate_checkpoint,
                                         run_periodic_evaluation)
from ep_loop.reproducibility import capture_rng_state

ROOT=Path(__file__).resolve().parents[1]


class TinyModel(torch.nn.Linear):
    def save_pretrained(self,folder,safe_serialization=True):
        from safetensors.torch import save_file
        save_file(self.state_dict(),str(Path(folder)/'model.safetensors'))
        (Path(folder)/'config.json').write_text('{}')


def config():return load_config(ROOT/'configs/vllm_fixed_top_n_current.json')


class ClosingResource:
    def __init__(self):self.pending=[];self.closed=False
    def close(self):self.closed=True


def distributed_evaluation(rank,directory,fail):
    torch.set_num_threads(1);torch.manual_seed(42+rank)
    dist.init_process_group('gloo',init_method='file://'+str(Path(directory)/'rendezvous'),
        rank=rank,world_size=2,timeout=timedelta(seconds=20))
    model=TinyModel(2,2);optimizer=torch.optim.AdamW(model.parameters())
    student=SimpleNamespace(model=model,device=torch.device('cpu'))
    batcher=ClosingResource();teacher=ClosingResource();teacher.model=model
    data=random.Random(42+rank);generator=torch.Generator().manual_seed(43+rank)
    before=capture_rng_state('cpu');saved_data=data.getstate();saved_rollout=generator.get_state()
    def evaluate(*args,**kwargs):
        assert rank==0 and batcher.closed and teacher.closed
        random.random();np.random.rand();torch.rand(());data.random();torch.rand((),generator=generator)
        if fail:raise ValueError('intentional evaluator failure')
        return dict(status='completed',results={})
    try:
        with patch.dict('os.environ',CUDA_VISIBLE_DEVICES='4,5'), \
             patch('ep_loop.fastpath.evaluation.evaluate_checkpoint',side_effect=evaluate):
            try:
                result=run_periodic_evaluation(config(),directory,5,student,optimizer,batcher,teacher,
                    'cpu',data,generator,rank,2,dist.group.WORLD,ROOT)
                assert not fail and result['status']=='completed'
            except RuntimeError as error:
                assert fail and 'intentional evaluator failure' in str(error)
        after=capture_rng_state('cpu')
        assert before['python_rng']==after['python_rng'] and before['numpy_rng']==after['numpy_rng']
        assert torch.equal(before['torch_rng'],after['torch_rng'])
        assert data.getstate()==saved_data and torch.equal(generator.get_state(),saved_rollout)
        assert batcher.closed and teacher.closed
        (Path(directory)/f'rank{rank}.ok').write_text('ok')
    finally:dist.destroy_process_group()


class PeriodicEvaluationTests(unittest.TestCase):
    def test_default_fixed_profiles_evaluate_every_five_and_resume_only_at_final(self):
        for name in ('vllm_fixed_top_n','vllm_fixed_top_n_current','vllm_fixed_d_loopopd','vllm_fixed_endpoint'):
            c=validate_config(load_config(ROOT/'configs'/f'{name}.json'))
            self.assertEqual(c['implementation']['checkpoint_layout'],'model_only')
            self.assertFalse(c['implementation']['save_optimizer'])
            self.assertTrue(c['implementation']['save_final_resume'])
            self.assertEqual(c['implementation']['save_every_updates'],5)
            self.assertEqual(c['evaluation']['every_updates'],5)
            self.assertEqual([i for i in range(1,13) if evaluation_due(c,i,12)],[5,10,12])
            self.assertEqual([i for i in range(1,13) if checkpoint_due(c['implementation'],i,12)[0]],
                             [5,10,12])
            self.assertEqual([i for i in range(1,13) if checkpoint_due(c['implementation'],i,12)[1]],
                             [12])
        c=load_config(ROOT/'configs/hf_shared_ep_d_loopopd.json')
        self.assertEqual(validate_config(c)['evaluation']['every_updates'],0)

    def test_model_only_does_not_call_torch_save_or_create_resume_files(self):
        c=config();model=TinyModel(2,2)
        student=SimpleNamespace(model=model,source_filenames=(),device=torch.device('cpu'))
        with tempfile.TemporaryDirectory() as directory,patch('ep_loop.fastpath.checkpoints.torch.save') as save:
            for update in (5,10):
                save_split_checkpoint(directory,student,None,None,update,c,None,None,0,1,ROOT,
                                      save_model=True,save_resume=False)
            save.assert_not_called()
            root=Path(directory)
            self.assertFalse((root/'resume').exists())
            self.assertEqual(len(list((root/'checkpoints').glob('update_*'))),2)
            for marker in root.glob('checkpoints/update_*/checkpoint.complete.json'):
                self.assertFalse(json.loads(marker.read_text())['resumable'])
            self.assertFalse(list(root.rglob('*.pt')))

    def test_evaluation_uses_saved_weights_and_normal_inference_for_each_dataset(self):
        c=config();commands=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'checkpoint';checkpoint.mkdir()
            (checkpoint/'checkpoint.complete.json').write_text('{"completed_updates":5}')
            tasks=[]
            for name in ('suite_a','suite_b'):
                path=root/(name+'.jsonl')
                path.write_text(json.dumps(dict(id=name,problem='1+1?',answer='2'))+'\n')
                tasks.append(dict(name=name,path=str(path),expected_examples=1))
            c['evaluation']['datasets']=tasks
            def run(command,**kwargs):
                commands.append(command)
                task=json.loads(Path(command[command.index('--config')+1]).read_text())
                self.assertFalse(task['implementation']['vllm_rollout_ngram'])
                self.assertEqual(task['evaluation']['every_updates'],0)
                self.assertNotIn('RANK',kwargs['env'])
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'4,5')
                output=Path(command[command.index('--output')+1]);output.write_text('{"text":"answer","response_tokens":3}\n')
                output.with_suffix('.summary.json').write_text(json.dumps(dict(checkpoint=str(checkpoint.resolve()),
                    speculative_config=None,results={'fixed_2':dict(accuracy=1.,response_tokens=3,mean_response_tokens=3.)})))
            with patch.dict('os.environ',RANK='0',WORLD_SIZE='2'), \
                 patch('ep_loop.fastpath.evaluation.subprocess.run',side_effect=run):
                report=evaluate_checkpoint(c,checkpoint,root/'evaluation',['4','5'],ROOT)
            self.assertEqual(report['status'],'completed')
            self.assertEqual(set(report['results']),{'suite_a','suite_b'})
            self.assertEqual(len(commands),2)
            self.assertEqual(json.loads((root/'evaluation/status.json').read_text())['completed_updates'],5)

    def test_dataset_names_cannot_escape_the_evaluation_directory(self):
        c=config()
        c['evaluation']['datasets']=[dict(name='../outside',path='unused')]
        with self.assertRaisesRegex(ValueError,'safe'):evaluation_datasets(c,ROOT)
        c=config();c['evaluation']['every_updates']=3
        with self.assertRaisesRegex(ValueError,'multiple'):validate_config(c)
        c=config();c['implementation']['save_optimizer']=True
        with self.assertRaisesRegex(ValueError,'save_optimizer=false'):validate_config(c)
        c=config();c['implementation']['save_final_resume']='true'
        with self.assertRaisesRegex(ValueError,'save_final_resume must be boolean'):validate_config(c)

    def test_existing_resume_can_change_output_and_evaluation_cadence(self):
        import copy
        from ep_loop.checkpointing import validate_resume_config
        saved=config()
        saved['evaluation'].pop('every_updates')
        saved['implementation'].update(checkpoint_layout='split',save_optimizer=True,
            write_rollout_text=True,write_rollout_records=True)
        saved['implementation'].pop('save_final_resume',None)
        current=copy.deepcopy(saved)
        current['evaluation']['every_updates']=5
        current['implementation'].update(checkpoint_layout='model_only',save_optimizer=False,
            save_final_resume=True,write_rollout_text=False,write_rollout_records=False)
        validate_resume_config(saved,current,10)
        current['optimization']['learning_rate']*=2
        with self.assertRaisesRegex(ValueError,'unchanged'):validate_resume_config(saved,current,10)

    def test_two_ranks_evaluate_once_and_restore_all_random_streams(self):
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                mp.spawn(distributed_evaluation,args=(directory,fail),nprocs=2,join=True)
                self.assertTrue(all((Path(directory)/f'rank{r}.ok').exists() for r in (0,1)))

    def test_actual_trainer_publishes_final_resume_before_evaluation(self):
        from ep_loop.data import ByteTokenizer
        from ep_loop.inference import GeneratedResponse
        from ep_loop.ouro import tiny_ouro
        from ep_loop.fastpath.training import train
        torch.set_num_threads(2)
        c=config();c['optimization'].update(updates=6,global_batch_size=2,mixed_precision='none',
            gradient_checkpointing=False,warmup_steps=0)
        c['rollout'].update(prompts_per_batch=2,responses_per_prompt=1,max_new_tokens=4)
        c['implementation'].update(teacher_backend='hf',readout_chunk_size=2)
        backend=tiny_ouro(ROOT/'models/Ouro-1.4B-Thinking')
        records=[dict(id=str(i),problem_sha256=str(i),input_ids=[1,20,30]) for i in range(2)]
        calls=[]
        class Batcher:
            def __init__(self,*args,**kwargs):self.pending=deque()
            def prepare(self,records,indices,rank,world,update,generator):
                for index in indices:
                    self.pending.append((records[index]['id'],GeneratedResponse([20,2],[2,2],True,8,.01,[2,2])))
            def pop(self,context):
                ident,response=self.pending.popleft()
                assert ident==context['id'];return response
            def close(self):pass
        def evaluate(c,output,completed,*args,**kwargs):
            from ep_loop.checkpointing import inspect_resume_checkpoint,validate_resume_state
            checkpoint=Path(output)/'checkpoints'/f'update_{completed:06d}'
            marker=json.loads((checkpoint/'checkpoint.complete.json').read_text())
            self.assertEqual(marker['completed_updates'],completed)
            self.assertEqual(marker['resumable'],completed==6)
            if completed==5:
                self.assertFalse(list(Path(output).rglob('*.pt')))
            else:
                info=inspect_resume_checkpoint(checkpoint,c,1)
                validate_resume_state(torch.load(checkpoint/'training_state.pt',weights_only=True),info)
                self.assertTrue((checkpoint/'rng_rank0.pt').is_file())
                self.assertFalse(list(checkpoint.glob('.*pending*')))
            calls.append(completed)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config_path=root/'config.json';config_path.write_text(json.dumps(c))
            with patch.dict('os.environ',CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',RANK='0',LOCAL_RANK='0',TENSORBOARD='0'), \
                 patch('ep_loop.fastpath.training.assets_status',return_value={'ready':True}), \
                 patch('ep_loop.fastpath.training.load_backend',return_value=(backend,{})), \
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=ByteTokenizer()), \
                 patch('ep_loop.fastpath.training.prepare_training_records',return_value=(records,dict(usable_prompt_hash='test'))), \
                 patch('ep_loop.fastpath.training.RolloutBatcher',Batcher), \
                 patch('ep_loop.fastpath.evaluation.run_periodic_evaluation',side_effect=evaluate), \
                 redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                train(config_path,root/'output','cpu')
            self.assertEqual(calls,[5,6])
            self.assertEqual(sorted(p.name for p in (root/'output/checkpoints').glob('update_*')),
                             ['update_000005','update_000006'])
            self.assertFalse((root/'output/resume').exists())
            self.assertFalse(list((root/'output').glob('rollouts_rank*')))


if __name__=='__main__':unittest.main()
