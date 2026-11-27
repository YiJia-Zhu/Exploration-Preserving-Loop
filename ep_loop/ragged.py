"""Ouro RAD-OPD rollout and differentiable replay."""
import random
import weakref

import torch
from torch.utils.checkpoint import checkpoint

from .cache import TensorBuffer
from .ouro import LoopTrace
from .reproducibility import sample_seed


def sample_training_threshold(settings, seed, problem_hash, update):
    """Use the configured threshold for an initial update, then sample per question."""
    initial = settings.get("initial_thresholds", [])
    if update < len(initial):
        return initial[update]
    if settings.get("fixed") is not None:
        return settings["fixed"]
    rng = random.Random(sample_seed(seed, problem_hash, update))
    if rng.random() < settings["low_probability"]:
        return rng.uniform(*settings["low_range"])
    return rng.choice(settings["anchors"])


def reached_native_threshold(gates, threshold):
    """Compare the cumulative probability before the forced final exit."""
    if threshold == 1.0:
        return False
    remaining = torch.ones_like(gates[0][:, -1, 0])
    probabilities = []
    for gate in gates:
        hazard = gate[:, -1, 0].sigmoid()
        probabilities.append(hazard * remaining)
        remaining = remaining * (1 - hazard)
    return bool((torch.stack(probabilities, -1).cumsum(-1)[..., -1] >= threshold).all())


class RaggedKVCache:
    """Store one physical slot per visited position/depth, per backbone layer.

    Logical depth maps are integer aliases. Attention gathers the current
    depth's visible prefix into temporary tensors for SDPA.
    """
    policy = "ragged_kv"

    def __init__(self, model, loops):
        self.owner = weakref.ref(model)
        self.loops = loops
        self.layers = model.config.num_hidden_layers
        self.keys = [TensorBuffer(2) for _ in range(self.layers)]
        self.values = [TensorBuffer(2) for _ in range(self.layers)]
        self.indices = [TensorBuffer(0) for _ in range(loops)]
        self.positions = 0
        self.physical_slots = 0
        self._new_slots = []
        self._read_index = None

    def validate(self, model, loops):
        if torch.is_grad_enabled() or model.training:
            raise ValueError("Ragged rollout requires evaluation mode and no_grad")
        if self.owner() is not model or self.loops != loops:
            raise ValueError("A ragged cache belongs to one model, request and maximum depth")

    def begin_depth(self, positions):
        depth = len(self._new_slots)
        start = self.physical_slots
        self.physical_slots += positions.numel()
        slots = torch.arange(start, self.physical_slots, device=positions.device)
        self._new_slots.append(slots)
        self._read_index = torch.cat((self.indices[depth].value(), slots)) if self.positions else None

    def update(self, keys, values, layer_idx, *args, **kwargs):
        depth, layer = divmod(layer_idx, self.layers)
        if depth != len(self._new_slots) - 1 or self.keys[layer].size + keys.shape[2] != self.physical_slots:
            raise RuntimeError("Out-of-order ragged KV append")
        stored_k = self.keys[layer].append(keys)
        stored_v = self.values[layer].append(values)
        if self._read_index is None:
            return keys, values
        return stored_k.index_select(2, self._read_index), stored_v.index_select(2, self._read_index)

    def finish(self):
        if any(buffer.size != self.physical_slots for buffer in self.keys):
            raise RuntimeError("Ragged KV layers did not append their full slot range")
        for depth, buffer in enumerate(self.indices):
            buffer.append(self._new_slots[min(depth, len(self._new_slots) - 1)])
        self.positions += self._new_slots[0].numel()
        self._new_slots.clear()
        self._read_index = None


class ReplayRoundKV:
    """Functional KV selection for one replay round, including checkpoint recomputation."""

    def __init__(self, previous, live, detach_reused):
        self.previous = previous
        self.live = live
        self.detach_reused = detach_reused
        self.effective = []

    def update(self, keys, values, layer_idx, *args, **kwargs):
        pair = keys, values
        if self.previous:
            start = len(self.effective)
            prior = self.previous[start:start + 2]
            if self.detach_reused:
                prior = tuple(tensor.detach() for tensor in prior)
            pair = tuple(torch.where(self.live, current, old) for current, old in zip(pair, prior, strict=True))
        self.effective.extend(pair)
        return pair


def _backbone(backend, ids, loops):
    backbone = backend.model.model
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError("Ragged Ouro expects one nonempty, unpadded sequence")
    if not 1 <= loops <= backbone.total_ut_steps:
        raise ValueError("Invalid ragged Ouro maximum depth")
    if backbone.has_sliding_layers or backbone.config.attention_dropout != 0:
        raise ValueError("Ragged Ouro currently requires full attention with zero dropout")
    return backbone


def _run_round(backbone, hidden, positions, rotary, depth, cache):
    # The outer checkpoint owns the functional KV tensors. Bypass HF's
    # per-layer wrapper, which clears past_key_value during checkpointing.
    for layer in backbone.layers:
        hidden = layer.forward(hidden, attention_mask=None, position_ids=positions[None],
                               past_key_value=cache, cache_position=positions,
                               position_embeddings=rotary, current_ut=depth)
    return backbone.norm(hidden)


def rollout_trace(backend, ids, loops, threshold, cache=None):
    backbone = _backbone(backend, ids, loops)
    if not 0 < threshold <= 1:
        raise ValueError("Ragged Ouro threshold must be in (0, 1]")
    prefill = cache is None
    if prefill:
        cache = RaggedKVCache(backend.model, loops)
    if not isinstance(cache, RaggedKVCache):
        raise ValueError("Ragged rollout requires its own physical KV cache")
    cache.validate(backend.model, loops)
    length = ids.shape[1]
    if (not prefill and length != 1) or cache.positions + length > backend.max_context_length:
        raise ValueError("Invalid ragged decode length or context limit")
    with backend.autocast():
        hidden = backbone.embed_tokens(ids)
        positions = torch.arange(cache.positions, cache.positions + length, device=ids.device)
        rotary = backbone.rotary_emb(hidden, positions[None])
        states, gates = [], []
        for depth in range(loops):
            cache.begin_depth(positions)
            hidden = _run_round(backbone, hidden, positions, rotary, depth, cache)
            states.append(hidden)
            if not prefill and depth + 1 < loops and threshold < 1.0:
                gates.append(backbone.early_exit_gate(hidden))
                if reached_native_threshold(gates, threshold):
                    break
    cache.finish()
    return LoopTrace(tuple(states), tuple(gates), cache=cache,
                     exits=torch.full_like(ids, len(states)), executed_token_loops=length * len(states))


def replay_trace(backend, inputs, loops, first, rollout_depths, *, detach_reused=False):
    """Replay the predictor positions under their fixed rollout route."""
    backbone = _backbone(backend, inputs, loops)
    depths = torch.cat((inputs.new_full((first,), loops), rollout_depths)).reshape(1, -1)
    if depths.shape != inputs.shape:
        raise ValueError("Replay depths must align with the consumed tokens")
    positions = torch.arange(inputs.shape[1], device=inputs.device)
    use_checkpoint = torch.is_grad_enabled() and backend.model.is_gradient_checkpointing
    with backend.autocast():
        hidden = backbone.embed_tokens(inputs)
        rotary = backbone.rotary_emb(hidden, positions[None])
        previous = ()
        selected = None
        for depth in range(loops):
            live = (depths >= depth + 1)[:, None, :, None]

            def round_forward(hidden, *prior, depth=depth, live=live):
                cache = ReplayRoundKV(prior, live, detach_reused)
                hidden = _run_round(backbone, hidden, positions, rotary, depth, cache)
                return (hidden, *cache.effective) if depth + 1 < loops else (hidden,)

            hidden, *previous = (checkpoint(round_forward, hidden, *previous, use_reentrant=False)
                                 if use_checkpoint else round_forward(hidden, *previous))
            state = hidden[0, first:]
            selected = state if selected is None else torch.where(
                (rollout_depths == depth + 1)[:, None], state, selected)
    return LoopTrace((selected,), (), executed_token_loops=inputs.numel() * loops)
