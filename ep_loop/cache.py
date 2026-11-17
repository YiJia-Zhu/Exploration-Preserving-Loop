"""Request-local exact KV and loop-state caches for learned early exits.

Depths may lag behind the current prefix. Before a deeper loop is read, compute
its missing suffix from saved states at the preceding depth. Never substitute
shallow KV for deep KV. Each (position, depth, layer) is computed at most once.
These caches are inference-only and must not survive a model weight update.
"""
import weakref

import torch


class TensorBuffer:
    """Append with bounded spare capacity, including at long context lengths."""

    def __init__(self, dim):
        self.dim, self.size, self.storage = dim, 0, None

    def append(self, value):
        end = self.size + value.shape[self.dim]
        capacity = 0 if self.storage is None else self.storage.shape[self.dim]
        if end > capacity:
            shape = list(value.shape)
            # Doubling every deep KV buffer can reserve almost another complete
            # context after Adam/DDP states have been allocated. Bound the slack
            # to 1,024 positions while retaining geometric growth for short runs.
            shape[self.dim] = max(end, 16 if not capacity else capacity + min(capacity, 1024))
            storage = value.new_empty(shape)
            if self.size:
                storage.narrow(self.dim, 0, self.size).copy_(self.value())
            self.storage = storage
        self.storage.narrow(self.dim, self.size, end - self.size).copy_(value)
        self.size = end
        return self.value()

    def value(self):
        return self.storage.narrow(self.dim, 0, self.size)


class ExactKVCache:
    """Native attention update interface, with independent contiguous layers."""

    def __init__(self):
        self.keys, self.values = {}, {}

    def update(self, keys, values, layer_idx, *args, **kwargs):
        layer_idx = int(layer_idx)
        if layer_idx not in self.keys:
            self.keys[layer_idx], self.values[layer_idx] = TensorBuffer(2), TensorBuffer(2)
        return self.keys[layer_idx].append(keys), self.values[layer_idx].append(values)

    def get_seq_length(self, layer_idx=0):
        return self.keys[int(layer_idx)].size if int(layer_idx) in self.keys else 0


class ExactLoopCache:
    policy = "exact_lazy_depth"

    def __init__(self, model, loops):
        self.owner = weakref.ref(model)
        self.loops = loops
        self.backfill_chunk_size = 256
        self.kv = ExactKVCache()
        self.levels = [TensorBuffer(1) for _ in range(loops + 1)]
        # Huginn keeps its prelude injection and a separate coda cache for
        # every depth. A single shared coda cache would mix different depths.
        self.embedded = TensorBuffer(1)
        self.coda = [ExactKVCache() for _ in range(loops)]

    def validate(self, model, loops):
        if torch.is_grad_enabled() or model.training:
            raise ValueError("Exact loop caches require evaluation mode and torch.no_grad()")
        if self.owner() is not model or self.loops != loops:
            raise ValueError("A loop cache belongs to one model, request and maximum depth")


def causal_suffix_mask(hidden, start, window=None):
    """Bottom-right causal mask for an exact cached suffix, including sliding KV.

    Ordinary full-attention prefill and one-token decode use SDPA's fast paths.
    A multi-token backfill needs an explicit offset mask, or it can see future
    tokens / attend to the wrong prefix positions.
    """
    length = hidden.shape[1]
    if window is None and (start == 0 or length == 1):
        return None
    queries = torch.arange(start, start + length, device=hidden.device)[:, None]
    keys = torch.arange(start + length, device=hidden.device)[None, :]
    allowed = keys <= queries
    if window is not None:
        allowed &= keys > queries - window
    return hidden.new_zeros((1, 1, length, start + length)).masked_fill_(
        ~allowed[None, None], torch.finfo(hidden.dtype).min)
