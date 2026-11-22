"""Retained weights, atomic rolling recovery, and matching optimizer/RNG state."""
from datetime import timedelta
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file,save_file

from ep_loop.checkpointing import inspect_resume_checkpoint,validate_resume_state
from ep_loop.config import load_config,validate_config
from ep_loop.fastpath.checkpoints import checkpoint_due,save_split_checkpoint
from ep_loop.reproducibility import restore_rng_state

ROOT=Path(__file__).resolve().parents[1]


class Model(torch.nn.Linear):
    def save_pretrained(self,folder,safe_serialization=True):
        save_file(self.state_dict(),str(Path(folder)/'model.safetensors'))
        (Path(folder)/'config.json').write_text('{}')


def config():
    value=load_config(ROOT/'configs/vllm_fixed_top_n_current.json')
    value['optimization']['updates']=100
    value['implementation'].update(checkpoint_layout='split',save_every_updates=5,
        resume_save_every_updates=5,keep_latest_checkpoints=0,save_optimizer=True)
    return value


def step(model,opt,data_rng,generator):
    opt.zero_grad()
    noise=torch.rand((),generator=generator)+torch.rand(())+data_rng.random()
    loss=(model(torch.ones(1,2))-noise).square().sum()
    loss.backward();opt.step()


def distributed_save(rank,directory,final_resume=False):
    torch.set_num_threads(1);torch.manual_seed(42)
    dist.init_process_group('gloo',init_method='file://'+str(Path(directory)/'rendezvous'),
                            rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        model=Model(2,2);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        for _ in range(5):
            opt.zero_grad();sum(p.square().sum() for p in model.parameters()).backward();opt.step()
        generator=torch.Generator().manual_seed(42+rank)
        torch.rand(rank+1,generator=generator)
        student=SimpleNamespace(model=model,source_filenames=(),device=torch.device('cpu'))
        c=config()
        if final_resume:
            c['implementation'].update(checkpoint_layout='model_only',save_optimizer=False,save_final_resume=True)
        save_split_checkpoint(directory,student,None,opt,5,c,random.Random(42+rank),generator,
                              rank,2,ROOT,data_identity='data',is_final=final_resume)
        latest=(Path(directory)/'checkpoints/update_000005' if final_resume else Path(directory)/'resume/latest')
        info=inspect_resume_checkpoint(latest,c,2)
        assert info['completed_updates']==5
        saved=torch.load(latest/f'rng_rank{rank}.pt',weights_only=True)
        assert torch.equal(saved['rollout_rng'],generator.get_state())
        assert (Path(directory)/'checkpoints/update_000005/training_state.pt').exists()==final_resume
        if final_resume:
            assert info['storage_role']=='final_resume'
            assert all((latest/f'rng_rank{r}.pt').is_file() for r in range(2))
            validate_resume_state(torch.load(latest/'training_state.pt',weights_only=True),info)
            assert not (Path(directory)/'resume').exists()
        (Path(directory)/f'rank{rank}.ok').write_text('ok')
    finally:dist.destroy_process_group()


class SplitCheckpointTests(unittest.TestCase):
    def test_intervals_do_not_save_each_update_and_final_is_recoverable(self):
        settings=dict(save_every_updates=5,resume_save_every_updates=10)
        self.assertEqual([s for s in range(1,13) if checkpoint_due(settings,s,12)[0]],[5,10,12])
        self.assertEqual([s for s in range(1,13) if checkpoint_due(settings,s,12)[1]],[10,12])
        invalid=config();invalid['implementation']['resume_save_every_updates']=6
        with self.assertRaisesRegex(ValueError,'multiple'):validate_config(invalid)
        validate_config(config())

    def test_publication_failure_keeps_old_resume_and_all_model_snapshots(self):
        torch.manual_seed(42)
        model=Model(2,2);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        rng=random.Random(42);generator=torch.Generator().manual_seed(42)
        student=SimpleNamespace(model=model,source_filenames=(),device=torch.device('cpu'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for _ in range(5):step(model,opt,rng,generator)
            save_split_checkpoint(root,student,None,opt,5,config(),rng,generator,0,1,ROOT,data_identity='data')
            old=(root/'resume/latest').resolve()
            self.assertEqual((root/'checkpoints/update_000005/model.safetensors').stat().st_ino,
                             (old/'model.safetensors').stat().st_ino)
            with self.assertRaisesRegex(ValueError,'optimizer'):
                inspect_resume_checkpoint(root/'checkpoints/update_000005',config(),1)
            for _ in range(5):step(model,opt,rng,generator)
            with patch('ep_loop.fastpath.checkpoints.torch.save',side_effect=OSError('disk write failure')):
                with self.assertRaisesRegex(OSError,'disk write'):
                    save_split_checkpoint(root,student,None,opt,10,config(),rng,generator,0,1,ROOT,data_identity='data')
            self.assertEqual((root/'resume/latest').resolve(),old)
            self.assertEqual(inspect_resume_checkpoint(old,config(),1)['completed_updates'],5)
            save_split_checkpoint(root,student,None,opt,10,config(),rng,generator,0,1,ROOT,data_identity='data')
            latest=root/'resume/latest'
            self.assertFalse(old.exists())
            self.assertEqual(sorted(p.name for p in (root/'checkpoints').iterdir()),['update_000005','update_000010'])
            self.assertFalse((root/'checkpoints/update_000010/training_state.pt').exists())
            self.assertEqual((root/'checkpoints/update_000010/model.safetensors').stat().st_ino,
                             (latest/'model.safetensors').stat().st_ino)
            info=inspect_resume_checkpoint(latest,config(),1)
            state=torch.load(latest/'training_state.pt',weights_only=True)
            validate_resume_state(state,info)
            self.assertEqual(info['completed_updates'],10)
            step(model,opt,rng,generator)
            expected={k:v.clone() for k,v in model.state_dict().items()}
            clone=Model(2,2);clone.load_state_dict(load_file(latest/'model.safetensors'))
            clone_opt=torch.optim.AdamW(clone.parameters(),lr=.01);clone_opt.load_state_dict(state['optimizer'])
            saved_rng=torch.load(latest/'rng_rank0.pt',weights_only=True)
            restored_rng=random.Random();restored_rng.setstate(saved_rng['data_rng'])
            restored_generator=torch.Generator();restored_generator.set_state(saved_rng['rollout_rng'])
            restore_rng_state(saved_rng,'cpu')
            step(clone,clone_opt,restored_rng,restored_generator)
            for name,value in clone.state_dict().items():torch.testing.assert_close(value,expected[name],rtol=0,atol=0)

    def test_two_real_ranks_publish_all_rng_files_before_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_save,args=(directory,),nprocs=2,join=True)
            self.assertTrue(all((Path(directory)/f'rank{rank}.ok').exists() for rank in range(2)))

    def test_final_resume_restores_next_step_and_failed_write_preserves_weights(self):
        torch.manual_seed(42)
        c=config();c['implementation'].update(checkpoint_layout='model_only',save_optimizer=False,
                                              save_final_resume=True)
        model=Model(2,2);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        rng=random.Random(42);generator=torch.Generator().manual_seed(42)
        student=SimpleNamespace(model=model,source_filenames=(),device=torch.device('cpu'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for _ in range(5):step(model,opt,rng,generator)
            save_split_checkpoint(root,student,None,opt,5,c,rng,generator,0,1,ROOT,
                                  save_resume=False,data_identity='data')
            self.assertFalse(list(root.rglob('*.pt')))
            for _ in range(5):step(model,opt,rng,generator)
            final=root/'checkpoints/update_000010'
            original_save=torch.save
            for failed_name in ('training_state.pt','rng_rank0.pt'):
                def fail_save(value,path,**kwargs):
                    if Path(path).name==failed_name:raise OSError('disk write failure')
                    return original_save(value,path,**kwargs)
                with patch('ep_loop.fastpath.checkpoints.torch.save',side_effect=fail_save), \
                     self.assertRaisesRegex(OSError,'disk write'):
                    save_split_checkpoint(root,student,None,opt,10,c,rng,generator,0,1,ROOT,
                                          data_identity='data',is_final=True)
                self.assertFalse(json.loads((final/'checkpoint.complete.json').read_text())['resumable'])
                with self.assertRaisesRegex(ValueError,'optimizer'):
                    inspect_resume_checkpoint(final,c,1)
                for key,value in load_file(final/'model.safetensors').items():
                    torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
            save_split_checkpoint(root,student,None,opt,10,c,rng,generator,0,1,ROOT,
                                  data_identity='data',is_final=True)
            info=inspect_resume_checkpoint(final,c,1)
            state=torch.load(final/'training_state.pt',weights_only=True)
            validate_resume_state(state,info)
            self.assertEqual(info['storage_role'],'final_resume')
            self.assertFalse((root/'resume').exists())
            self.assertFalse((final/'.resume.pending').exists())
            self.assertFalse(list((root/'checkpoints/update_000005').glob('*.pt')))
            step(model,opt,rng,generator)
            expected={key:value.clone() for key,value in model.state_dict().items()}
            clone=Model(2,2);clone.load_state_dict(load_file(final/'model.safetensors'))
            clone_opt=torch.optim.AdamW(clone.parameters(),lr=.01);clone_opt.load_state_dict(state['optimizer'])
            saved_rng=torch.load(final/'rng_rank0.pt',weights_only=True)
            restored_rng=random.Random();restored_rng.setstate(saved_rng['data_rng'])
            restored_generator=torch.Generator();restored_generator.set_state(saved_rng['rollout_rng'])
            restore_rng_state(saved_rng,'cpu')
            step(clone,clone_opt,restored_rng,restored_generator)
            for name,value in clone.state_dict().items():
                torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
            with self.assertRaisesRegex(FileExistsError,'already complete'):
                save_split_checkpoint(root,student,None,opt,10,c,rng,generator,0,1,ROOT,
                                      data_identity='data',is_final=True)
            self.assertEqual(inspect_resume_checkpoint(final,c,1)['storage_role'],'final_resume')

    def test_two_real_ranks_publish_final_optimizer_and_all_rng_files(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_save,args=(directory,True),nprocs=2,join=True)
            self.assertTrue(all((Path(directory)/f'rank{rank}.ok').exists() for rank in range(2)))


if __name__=='__main__':unittest.main()
