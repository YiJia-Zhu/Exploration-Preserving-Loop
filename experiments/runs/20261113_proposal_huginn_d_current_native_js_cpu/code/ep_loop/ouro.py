"""Load unchanged official Ouro source and expose its normalized loop states.

This module does not import the legacy cql-handoff diagnostic package. Depth
selection changes the native backbone loop count, not just its output head.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.util
import sys
import types

import torch


def author_module(directory, stems=("configuration_ouro", "modeling_ouro"), namespace="_ep_ouro_"):
    directory = Path(directory).resolve()
    name = namespace + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    if name + "." + stems[-1] in sys.modules:
        return sys.modules[name + "." + stems[-1]]
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    for stem in stems:
        path = directory / (stem + ".py")
        if not path.is_file():
            raise FileNotFoundError(f"Missing official model source: {path}")
        spec = importlib.util.spec_from_file_location(name + "." + stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


@contextmanager
def fixed_depth(model, loops):
    if loops < 1:
        raise ValueError("Depth must be positive")
    old = model.model.total_ut_steps, model.config.total_ut_steps
    model.model.total_ut_steps = model.config.total_ut_steps = loops
    try:
        yield
    finally:
        model.model.total_ut_steps, model.config.total_ut_steps = old


@contextmanager
def evaluation_mode(model):
    old = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(old)


@dataclass
class LoopTrace:
    states: tuple
    gates: tuple
    cache: object = None
    exits: object = None
    adjacent_js: object = None
    executed_token_loops: object = None


class OuroBackend:
    source_filenames = ("configuration_ouro.py", "modeling_ouro.py")
    native_exit_reduces_executed_loops = False
    has_coda = False

    def __init__(self, model, mixed_precision="bf16"):
        self.model = model
        self.mixed_precision = mixed_precision

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def max_context_length(self):
        return self.model.config.max_position_embeddings

    def native_exit_depths(self, trace, threshold):
        from .selection import native_ouro_exits
        return native_ouro_exits(torch.stack(trace.gates), threshold)

    def autocast(self):
        if self.device.type == "cuda" and self.mixed_precision == "bf16":
            # Separate contexts prevent no_grad teacher casts from masking
            # gradients through the same shared weights in D-LoopOPD.
            return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
        return nullcontext()

    def trace(self, ids, loops, cache=None, use_cache=False, endpoint="fixed", exit_threshold=0.9):
        with fixed_depth(self.model, loops), self.autocast():
            output, states, gates = self.model.model(
                input_ids=ids, past_key_values=cache, use_cache=use_cache,
            )
        if len(states) != loops or len(gates) != loops:
            raise RuntimeError("Native backbone did not execute the requested depth")
        return LoopTrace(tuple(states), tuple(gates), output.past_key_values,
                         executed_token_loops=ids.shape[1] * loops)

    def head(self, states):
        # Native states already passed through final norm, including the state
        # fed to the next loop. Do not normalize them a second time.
        with self.autocast():
            return self.model.lm_head(states).float()

    def response_trace(self, ids, loops, start, endpoint="fixed", exit_threshold=0.9):
        trace = self.trace(ids, loops, use_cache=False)
        return LoopTrace(tuple(h[0, start:] for h in trace.states),
                         tuple(g[0, start:] for g in trace.gates),
                         executed_token_loops=trace.executed_token_loops)


def load_ouro(directory, device, mixed_precision="bf16", gradient_checkpointing=True):
    module = author_module(directory)
    config = module.OuroConfig.from_pretrained(directory, local_files_only=True)
    config.use_cache = False
    model, loading = module.OuroForCausalLM.from_pretrained(
        directory, config=config, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="sdpa", output_loading_info=True, low_cpu_mem_usage=True,
    )
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Incomplete checkpoint loading: {loading}")
    model.to(device).requires_grad_(True)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return OuroBackend(model, mixed_precision), loading


def tiny_ouro(directory, device="cpu"):
    """Random small official architecture for semantic tests, never a benchmark."""
    module = author_module(directory)
    config = module.OuroConfig(
        vocab_size=260, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=8,
        max_position_embeddings=1024, total_ut_steps=4,
        layer_types=["full_attention"] * 2, max_window_layers=2,
        attention_dropout=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    model = module.OuroForCausalLM(config).to(device=device, dtype=torch.float32)
    return OuroBackend(model, "none")
