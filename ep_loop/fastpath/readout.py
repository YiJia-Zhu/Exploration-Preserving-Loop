"""Fixed computation budget, unchanged learned/native readout decisions."""
import torch
from ep_loop.selection import native_ouro_exits


def select_readout(states, gates, policy, stop_threshold=.5, native_threshold=.9):
    stacked=torch.stack(states)
    if policy=='learned':
        reached=torch.stack(gates)[...,0].sigmoid()>=stop_threshold
        depth=reached.to(torch.int64).argmax(0)+1
        depth=torch.where(reached.any(0),depth,len(states))
    elif policy=='native':
        depth=native_ouro_exits(torch.stack(gates),native_threshold)
    elif policy=='fixed':
        depth=torch.full(states[0].shape[:-1],len(states),dtype=torch.long,device=states[0].device)
    else:raise ValueError(policy)
    # Flatten only token/batch axes, never mix recurrent depths or requests.
    flat=stacked.reshape(len(states),-1,stacked.shape[-1])
    selected=flat[depth.reshape(-1)-1,torch.arange(depth.numel(),device=depth.device)]
    return selected.reshape_as(states[0]),depth
