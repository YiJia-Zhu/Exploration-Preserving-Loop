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
    backfill_token_loops: int = 0
    coda_token_positions: int = 0
    reused_token_loops: int = 0


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

    @property
    def stop_token_ids(self):
        eos = self.model.config.eos_token_id
        return set(eos if isinstance(eos, (tuple, list)) else [eos])

    def native_exit_depths(self, trace, threshold):
        from .selection import native_ouro_exits
        return native_ouro_exits(torch.stack(trace.gates), threshold)

    def enable_student_exit(self):
        """Ouro already has a gate; keep its pretrained parameters."""
        self.model.config.ep_loop_student_policy = "learned"
        return self.model.model.early_exit_gate

    def learned_trace(self, ids, loops, threshold):
        """Actually stop the student; recompute full prefix to avoid missing KV.

This intentionally uses no approximate KV imputation. Teacher/native readout
and the original cached fixed-depth baseline remain on the author's path.
"""
        if ids.ndim != 2 or ids.shape[0] != 1 or loops < 1 or not 0 < threshold <= 1:
            raise ValueError("Invalid learned-exit prefix/depth/threshold")
        backbone = self.model.model
        module = sys.modules[type(self.model).__module__]
        with self.autocast():
            hidden = backbone.embed_tokens(ids)
            positions = torch.arange(ids.shape[1], device=ids.device)
            position_ids = positions[None]
            kwargs = dict(config=backbone.config, input_embeds=hidden,
                          attention_mask=None, cache_position=positions,
                          past_key_values=None, position_ids=position_ids)
            masks = {"full_attention": module.create_causal_mask(**kwargs)}
            if backbone.has_sliding_layers:
                masks["sliding_attention"] = module.create_sliding_window_causal_mask(**kwargs)
            rotary = backbone.rotary_emb(hidden, position_ids)
            states, gates = [], []
            for depth in range(loops):
                for layer in backbone.layers:
                    hidden = layer(hidden, attention_mask=masks[layer.attention_type],
                        position_ids=position_ids, past_key_value=None, use_cache=False,
                        cache_position=positions, position_embeddings=rotary, current_ut=depth)
                hidden = backbone.norm(hidden)
                gate = backbone.early_exit_gate(hidden)
                states.append(hidden)
                gates.append(gate)
                if bool((gate[:, -1, 0].sigmoid() >= threshold).all()):
                    break
        actual = len(states)
        exits = torch.full(ids.shape, actual, dtype=torch.long, device=ids.device)
        return LoopTrace(tuple(states), tuple(gates), exits=exits,
                         executed_token_loops=ids.shape[1]*actual)

    def learned_cached_trace(self, ids, loops, threshold, cache=None, stop_rule='local'):
        from .cache import ExactLoopCache, causal_suffix_mask
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or loops < 1 or not 0 < threshold <= 1:
            raise ValueError("Invalid learned-exit suffix/depth/threshold")
        cache = cache if cache is not None else ExactLoopCache(self.model, loops)
        if not isinstance(cache, ExactLoopCache):
            raise ValueError("Learned stopping requires its own exact loop cache")
        cache.validate(self.model, loops)
        if cache.levels[0].size + ids.shape[1] > self.max_context_length:
            raise ValueError("Ouro context limit exceeded")
        backbone = self.model.model
        states, gates, cost = [], [], 0
        with self.autocast():
            cache.levels[0].append(backbone.embed_tokens(ids))
            end = cache.levels[0].size
            for depth in range(loops):
                start = cache.levels[depth + 1].size
                # A late deep exit can require the whole historical prefix.
                # Chunk its exact causal suffix so masks/activations stay bounded.
                for begin in range(start, end, cache.backfill_chunk_size):
                    stop = min(end, begin + cache.backfill_chunk_size)
                    hidden = cache.levels[depth].value()[:, begin:stop]
                    positions = torch.arange(begin, stop, device=ids.device)
                    position_ids = positions[None]
                    rotary = backbone.rotary_emb(hidden, position_ids)
                    masks = {"full_attention": causal_suffix_mask(hidden, begin)}
                    if backbone.has_sliding_layers:
                        masks["sliding_attention"] = causal_suffix_mask(hidden, begin, backbone.config.sliding_window)
                    for layer in backbone.layers:
                        hidden = layer(hidden, attention_mask=masks[layer.attention_type],
                            position_ids=position_ids, past_key_value=cache.kv, use_cache=True,
                            cache_position=positions, position_embeddings=rotary, current_ut=depth)
                    hidden = backbone.norm(hidden)
                    cache.levels[depth + 1].append(hidden)
                cost += end - start
                # Public trace rows correspond only to the newly consumed IDs.
                states.append(cache.levels[depth + 1].value()[:, -ids.shape[1]:])
                gate = backbone.early_exit_gate(states[-1])
                gates.append(gate)
                if stop_rule == 'local':
                    stop = gate[:, -1, 0].sigmoid() >= threshold
                elif stop_rule == 'native_cdf':
                    remaining=torch.ones_like(gates[0][...,-1,0])
                    pdf=[]
                    for index,visited in enumerate(gates):
                        probability=visited[...,-1,0].sigmoid()
                        pdf.append(remaining if index==loops-1 else probability*remaining)
                        remaining=remaining*(1-probability)
                    stop=torch.stack(pdf,-1).cumsum(-1)[...,-1]>=threshold
                else:
                    raise ValueError('Unknown cached stop rule')
                if bool(stop.all()):
                    break
        actual = len(states)
        return LoopTrace(tuple(states), tuple(gates), cache=cache,
            exits=torch.full(ids.shape, actual, dtype=torch.long, device=ids.device),
            executed_token_loops=cost, backfill_token_loops=cost - actual * ids.shape[1])

    @contextmanager
    def generation_context(self):
        """Reuse BF16 weight casts for one no-grad response, then discard them.

        Never retain detached inference casts into the shared-model gradient
        replay. Inner loop forwards inherit this request-scoped setting.
        """
        if torch.is_grad_enabled():
            raise ValueError("Generation weight-cast caching requires torch.no_grad()")
        previous = getattr(self, "_generation_cast_cache", False)
        self._generation_cast_cache = True
        try:
            with self.autocast():
                yield
        finally:
            self._generation_cast_cache = previous
            torch.clear_autocast_cache()

    def autocast(self):
        if self.device.type == "cuda" and self.mixed_precision == "bf16":
            # Separate contexts prevent no_grad teacher casts from masking
            # gradients through the same shared weights in D-LoopOPD.
            return torch.autocast("cuda", dtype=torch.bfloat16,
                                  cache_enabled=getattr(self, "_generation_cast_cache", False))
        return nullcontext()

    def trace(self, ids, loops, cache=None, use_cache=False, endpoint="fixed", exit_threshold=0.9,
              attention_mask=None, position_ids=None, cache_position=None):
        if endpoint == "ragged":
            if not use_cache or any(x is not None for x in (attention_mask, position_ids, cache_position)):
                raise ValueError("RAD rollout requires internal caching and unpadded positions")
            from .ragged import rollout_trace
            return rollout_trace(self, ids, loops, exit_threshold, cache)
        if endpoint == "learned":
            if use_cache:
                return self.learned_cached_trace(ids, loops, exit_threshold, cache)
            if cache is not None:
                raise ValueError("Cannot pass a KV cache when use_cache is false")
            return self.learned_trace(ids, loops, exit_threshold)
        if endpoint == 'native_adaptive':
            if not use_cache:
                raise ValueError('Actual native adaptive execution requires exact depth KV caching')
            return self.learned_cached_trace(ids,loops,exit_threshold,cache,stop_rule='native_cdf')
        with fixed_depth(self.model, loops), self.autocast():
            output, states, gates = self.model.model(
                input_ids=ids, past_key_values=cache, use_cache=use_cache,
                attention_mask=attention_mask,position_ids=position_ids,cache_position=cache_position,
            )
        if len(states) != loops or len(gates) != loops:
            raise RuntimeError("Native backbone did not execute the requested depth")
        return LoopTrace(tuple(states), tuple(gates), output.past_key_values,
                         executed_token_loops=ids.shape[1] * loops)

    def shared_response_trace(self, ids, student_loops, teacher_loops, start):
        """One forward: differentiable shallow loops, detached deep continuation.

        The first S normalized states are simultaneously student outputs and
        detached teacher readouts. Only the K-S additional loops are charged
        as teacher backbone work. No inference KV or stale activations enter.
        """
        if ids.ndim!=2 or ids.shape[0]!=1 or not 1<=student_loops<=teacher_loops or not 0<=start<ids.shape[1]:
            raise ValueError('Invalid shared student/teacher replay shape or depths')
        if self.model.config.attention_dropout!=0:
            raise ValueError('Shared replay currently requires zero attention dropout')
        backbone=self.model.model
        module=sys.modules[type(self.model).__module__]
        shallow_states,shallow_gates=[],[]
        with self.autocast():
            hidden=backbone.embed_tokens(ids)
            positions=torch.arange(ids.shape[1],device=ids.device)
            position_ids=positions[None]
            kwargs=dict(config=backbone.config,input_embeds=hidden,attention_mask=None,
                        cache_position=positions,past_key_values=None,position_ids=position_ids)
            masks={'full_attention':module.create_causal_mask(**kwargs)}
            if backbone.has_sliding_layers:
                masks['sliding_attention']=module.create_sliding_window_causal_mask(**kwargs)
            rotary=backbone.rotary_emb(hidden,position_ids)
            def loop(state,depth):
                for layer in backbone.layers[:backbone.config.num_hidden_layers]:
                    state=layer(state,attention_mask=masks[layer.attention_type],
                        position_ids=position_ids,past_key_value=None,use_cache=False,
                        cache_position=positions,position_embeddings=rotary,current_ut=depth)
                return backbone.norm(state)
            for depth in range(student_loops):
                hidden=loop(hidden,depth)
                shallow_states.append(hidden[0,start:])
                shallow_gates.append(backbone.early_exit_gate(hidden)[0,start:])
            teacher_states=[state.detach() for state in shallow_states]
            teacher_gates=[gate.detach() for gate in shallow_gates]
            with torch.no_grad(),evaluation_mode(self.model):
                hidden=hidden.detach()
                for depth in range(student_loops,teacher_loops):
                    hidden=loop(hidden,depth)
                    teacher_states.append(hidden[0,start:])
                    teacher_gates.append(backbone.early_exit_gate(hidden)[0,start:])
        shallow_cost=ids.shape[1]*student_loops
        student=LoopTrace(tuple(shallow_states),tuple(shallow_gates),executed_token_loops=shallow_cost)
        teacher=LoopTrace(tuple(teacher_states),tuple(teacher_gates),
            executed_token_loops=ids.shape[1]*(teacher_loops-student_loops),reused_token_loops=shallow_cost)
        return student,teacher

    def head(self, states):
        # Native states already passed through final norm, including the state
        # fed to the next loop. Do not normalize them a second time.
        with self.autocast():
            return self.model.lm_head(states).float()

    def response_trace(self, ids, loops, start, endpoint="fixed", exit_threshold=0.9):
        if endpoint == 'native_adaptive':
            return self.native_adaptive_response_trace(ids,loops,start,exit_threshold)
        trace = self.trace(ids, loops, use_cache=False)
        return LoopTrace(tuple(h[0, start:] for h in trace.states),
                         tuple(g[0, start:] for g in trace.gates),
                         executed_token_loops=trace.executed_token_loops)

    def native_adaptive_response_trace(self, ids, loops, start, threshold):
        """Real CDF exits on a known student prefix, with exact historical KV.

        Unvisited states are padded solely to form rectangular label tensors;
        per-position exits mask them out of JS and KL, without extra backbone work.
        """
        if torch.is_grad_enabled() or self.model.training:
            raise ValueError('Adaptive teacher labeling requires no_grad and evaluation mode')
        if ids.shape[0]!=1 or not 0<=start<ids.shape[1]:
            raise ValueError('Invalid adaptive teacher prefix')
        positions=ids.shape[1]-start
        states=gates=exits=None
        cache=None
        cost=backfill=0
        with self.generation_context():
            for offset in range(positions):
                current=ids[:,:start+1] if offset==0 else ids[:,start+offset:start+offset+1]
                trace=self.trace(current,loops,cache,True,'native_adaptive',threshold)
                cache=trace.cache
                if states is None:
                    states=trace.states[0].new_empty((loops,positions,trace.states[0].shape[-1]))
                    gates=trace.gates[0].new_empty((loops,positions,1))
                    exits=ids.new_empty(positions)
                depth=len(trace.states)
                exits[offset]=depth
                for slot in range(loops):
                    selected=min(slot,depth-1)
                    states[slot,offset].copy_(trace.states[selected][0,-1])
                    gates[slot,offset].copy_(trace.gates[selected][0,-1])
                cost+=trace.executed_token_loops
                backfill+=trace.backfill_token_loops
        return LoopTrace(tuple(states.unbind(0)),tuple(gates.unbind(0)),exits=exits,
                         executed_token_loops=cost,backfill_token_loops=backfill)


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
