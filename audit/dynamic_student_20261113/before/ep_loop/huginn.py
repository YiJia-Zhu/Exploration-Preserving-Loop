"""Huginn's native recurrent core, complete coda, KL evaluator and own KV.

Native teacher forcing follows generate_with_adaptive_compute at batch size 1:
the prompt is prefilled together; subsequent student tokens are consumed one at
a time. Each prefix uses latest-m4 KV lookup and actually stops at its exit.
No Huginn source or shared base weights are modified.
"""
import math

import torch
from torch.utils.checkpoint import checkpoint

from .ouro import LoopTrace, OuroBackend, author_module
from .selection import adjacent_js_from_log_probs


def huginn_module(directory):
    return author_module(directory, ("raven_config_minimal", "raven_modeling_minimal"), "_ep_huginn_")


class HuginnBackend(OuroBackend):
    source_filenames = ("raven_config_minimal.py", "raven_modeling_minimal.py")
    native_exit_reduces_executed_loops = True
    has_coda = True

    @property
    def module(self):
        import sys
        return sys.modules[type(self.model).__module__]

    @property
    def max_context_length(self):
        return self.model.config.block_size

    @property
    def stop_token_ids(self):
        # The official generator also stops on begin_text and end_turn;
        # config.eos_token_id alone would miss those native stopping tokens.
        return set(self.model._get_stops(self.model.generation_config, None, {}).tolist())

    def native_exit_depths(self, trace, threshold):
        if trace.exits is None:
            raise ValueError("Native Huginn exits must be obtained while forming the teacher's own KV")
        return trace.exits

    def _coda(self, latents, freqs, cache):
        # Exact predict_from_latents path, postponing ONLY the linear LM head.
        x = self.model.transformer.ln_f(latents)
        index = torch.tensor(0, dtype=torch.long)
        for block in self.model.transformer.coda:
            index = index - 1
            x = block(x, freqs, index, None, cache)
        return self.model.transformer.ln_f(x)

    def trace(self, ids, loops, cache=None, use_cache=False, endpoint="fixed", exit_threshold=5e-4):
        if loops < 1 or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Huginn reference backend requires positive depth and one unpadded sequence")
        if endpoint not in {"fixed", "native"}:
            raise ValueError("Unknown Huginn endpoint")
        if endpoint == "native" and (not math.isfinite(exit_threshold) or exit_threshold <= 0):
            raise ValueError("Huginn KL exit threshold must be finite and positive")
        if endpoint == "native" and torch.is_grad_enabled():
            raise ValueError("Native Huginn teacher selection must run under torch.no_grad()")
        if endpoint == "native" and not use_cache:
            raise ValueError("Native Huginn requires its independent adaptive KV stream")
        if cache is not None and not use_cache:
            raise ValueError("Cannot pass a KV cache when use_cache is false")
        if cache is None and use_cache:
            cache = self.module.HuginnDynamicCache(lookup_strategy="latest-m4")
        offset = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(offset, offset + ids.shape[1], device=ids.device)
        if offset + ids.shape[1] > self.max_context_length:
            raise ValueError("Huginn context limit exceeded")
        states, changes = [], []
        previous = None
        with self.autocast():
            embedded, index = self.model.embed_inputs(ids, cache_position=positions, past_key_values=cache)
            latents = self.model.initialize_state(embedded)
            freqs = self.model.freqs_cis[:, positions]
            evaluator = self.module.KLExitEvaluator(self.model, exit_threshold) if endpoint == "native" else None
            if evaluator is not None:
                # Official evaluator compares the first readout to a UNIFORM
                # distribution. Later checks use KL(q_previous || q_current).
                evaluator.init(latents)
            for depth in range(loops):
                args = (latents, embedded, freqs, None, cache, index, depth)
                checked = self.model.gradient_checkpointing and torch.is_grad_enabled() and cache is None
                if checked:
                    latents, index = checkpoint(self.model.core_block_forward, *args, use_reentrant=False)
                else:
                    latents, index = self.model.core_block_forward(*args)
                if evaluator is not None:
                    reached, outputs, _ = evaluator.check(self.model, latents, {
                        "cache_position": positions, "past_key_values": cache,
                    })
                    head_state = outputs.latent_states
                    current = outputs.logits[:, -1].log_softmax(-1)
                    if previous is not None:
                        changes.append(adjacent_js_from_log_probs(torch.stack((previous, current)))[0])
                    previous = current
                else:
                    head_state = (checkpoint(self._coda, latents, freqs, cache, use_reentrant=False)
                                  if checked else self._coda(latents, freqs, cache))
                states.append(head_state)
                if evaluator is not None and bool(reached.all()):
                    break
        executed = len(states)
        exits = torch.full(ids.shape, executed, dtype=torch.long, device=ids.device) if evaluator is not None else None
        js = None
        if evaluator is not None:
            js = torch.zeros((loops - 1, 1), dtype=torch.float32, device=ids.device)
            if changes:
                js[:len(changes)] = torch.stack(changes)
        # Padded depths are never eligible targets or boundaries. Keeping a
        # rectangular trace allows the same S/K stage planner for both families.
        states.extend(torch.zeros_like(states[0]) for _ in range(loops - executed))
        return LoopTrace(tuple(states), (), cache, exits, js, executed * ids.shape[1])

    def response_trace(self, ids, loops, start, endpoint="fixed", exit_threshold=5e-4):
        if endpoint == "fixed":
            trace = self.trace(ids, loops)
            return LoopTrace(tuple(h[0, start:] for h in trace.states), (),
                             executed_token_loops=trace.executed_token_loops)
        if not 0 <= start < ids.shape[1]:
            raise ValueError("Invalid first response prediction position")
        cache, states, exits, changes, cost = None, [], [], [], 0
        # The first prediction consumes the whole prompt, exactly as native
        # adaptive generation. Future prefixes consume known STUDENT tokens.
        for pos in range(start, ids.shape[1]):
            current = ids[:, :pos + 1] if pos == start else ids[:, pos:pos + 1]
            trace = self.trace(current, loops, cache, True, "native", exit_threshold)
            cache = trace.cache
            states.append(torch.stack([h[0, -1] for h in trace.states]))
            exits.append(trace.exits[0, -1])
            changes.append(trace.adjacent_js[:, 0])
            cost += trace.executed_token_loops
        stacked = torch.stack(states, dim=1)
        return LoopTrace(tuple(stacked.unbind(0)), (), exits=torch.stack(exits),
                         adjacent_js=torch.stack(changes, dim=1), executed_token_loops=cost)


def load_huginn(directory, device, mixed_precision="bf16", gradient_checkpointing=True):
    module = huginn_module(directory)
    config = module.RavenConfig.from_pretrained(directory, local_files_only=True)
    model, loading = module.RavenForCausalLM.from_pretrained(
        directory, config=config, local_files_only=True, torch_dtype=torch.float32,
        output_loading_info=True, low_cpu_mem_usage=True,
    )
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Incomplete Huginn checkpoint loading: {loading}")
    model.to(device).requires_grad_(True)
    model.gradient_checkpointing = bool(gradient_checkpointing)
    return HuginnBackend(model, mixed_precision), loading


def tiny_huginn(directory, device="cpu"):
    """Random official architecture; retain the 2+4+2 native KV layout."""
    module = huginn_module(directory)
    config = module.RavenConfig(
        n_embd=32, n_heads=4, n_layers=8, block_size=1024, vocab_size=260,
        padding_multiple=1, intermediate_size=64, n_layers_in_prelude=2,
        n_layers_in_recurrent_block=4, n_layers_in_coda=2, mean_recurrence=4,
        mean_backprop_depth=2, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = module.RavenForCausalLM(config).to(device=device, dtype=torch.float32)
    return HuginnBackend(model, "none")
