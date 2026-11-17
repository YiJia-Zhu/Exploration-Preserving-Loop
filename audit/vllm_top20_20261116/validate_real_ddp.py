"""Real two-GPU trainer diagnostic; avoid writing redundant 16-GiB snapshots.

Only this test substitutes the already-tested checkpoint writer with a tiny
explicit non-resumable marker. Production checkpoint code is unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch.distributed as dist
import ep_loop.fastpath.training as training


def marker(folder,student,tokenizer,optimizer,completed,config,data_rng,generator,rank,world):
    if rank==0:
        folder.mkdir(parents=True,exist_ok=False)
        training.write_json(folder/'NOT_A_RESUMABLE_CHECKPOINT.json',dict(
            diagnostic_only=True,completed_updates=completed,reason='Disk-saving test marker; production save_checkpoint is unchanged'))
    if world>1:dist.barrier()


if __name__=='__main__':
    training.save_checkpoint=marker
    training.train('audit/vllm_top20_20261116/ddp_validation_config.json',
        'audit/vllm_top20_20261116/two_gpu_real_updates','cuda')
