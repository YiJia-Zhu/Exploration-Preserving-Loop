"""Move inactive training state to CPU without replacing optimizer parameters."""
import torch


def move_training_state(model,optimizer,device):
    """Call only between complete updates, with no live autograd/DDP wrapper."""
    device=torch.device(device)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                raise RuntimeError('Clear accumulated gradients before moving inactive training state')
            parameter.data=parameter.data.to(device)
        for buffer in model.buffers():
            buffer.data=buffer.data.to(device)
        for state in optimizer.state.values():
            for name,value in state.items():
                # AdamW's non-capturable step counter stays on CPU.
                if isinstance(value,torch.Tensor) and name!='step':state[name]=value.to(device)
    if torch.cuda.is_available():torch.cuda.empty_cache()
