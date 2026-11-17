"""Two real fixed-N DDP updates; short diagnostic, no resumable checkpoint."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch.distributed as dist
import ep_loop.fastpath.training as training

def diagnostic_save(folder,student,tokenizer,optimizer,completed,config,data_rng,generator,rank,world_size):
    if rank==0:
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'DIAGNOSTIC_NOT_RESUMABLE.json').write_text(json.dumps(dict(
            diagnostic_only=True,resumable=False,completed_updates=completed,
            reason='Avoid a duplicate 16-GiB checkpoint during a finite preflight; production save function is unchanged.'))+'\n')
    if world_size>1:dist.barrier()

with patch.object(training,'save_checkpoint',new=diagnostic_save):
    training.train(sys.argv[1],sys.argv[2],'cuda')
