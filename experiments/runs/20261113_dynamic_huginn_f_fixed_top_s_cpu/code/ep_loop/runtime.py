"""Legacy 2026-11-08 adapters around official Ouro/cql-handoff code.

The downloaded model source stays unchanged. Training uses a frozen backbone
with FP32 LoRA parameters; disabling LoRA gives the immutable initial teacher.
The current native-exit/JS full-parameter engine is in ouro.py/objective.py.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/cql-handoff"))
from lib.ouro_readout import author_module, read_all_loops, author_stopped_forward
from lib.ouro_advisor_stagecd import Patch
from lib.synthetic import PEOPLE, BOXES, COLORS, prompts

REVISION = "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(directory):
    directory = Path(directory).resolve()
    module = author_module(directory)
    config = module.OuroConfig.from_pretrained(directory, local_files_only=True)
    config.use_cache = False
    config.early_exit_threshold = 1.0
    model, info = module.OuroForCausalLM.from_pretrained(
        directory, config=config, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        output_loading_info=True, low_cpu_mem_usage=True,
    )
    assert not any(info.get(k) for k in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), info
    model.eval().requires_grad_(False).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    return model, tokenizer, info


@contextmanager
def depth(model, loops):
    old = model.model.total_ut_steps, model.config.total_ut_steps
    model.model.total_ut_steps = model.config.total_ut_steps = loops
    try:
        yield
    finally:
        model.model.total_ut_steps, model.config.total_ut_steps = old


def logits_at_loops(model, ids, loops, positions):
    """Read already normalized loop states, preserving gradients when enabled."""
    with depth(model, loops), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        _, states, _ = model.model(
            input_ids=ids, attention_mask={"full_attention": None}, use_cache=False,
        )
        return [model.lm_head(h[:, positions, :]).float() for h in states]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_json(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def make_scenes(tokenizer, n, seed, exclude=()):
    """New independent inputs; the released paper observations are never used."""
    rng = np.random.default_rng(seed)
    seen = set(exclude)
    scenes = []
    while len(scenes) < n:
        mapping = dict(zip(PEOPLE, rng.permutation(BOXES).tolist()))
        colors = dict(zip(BOXES, rng.permutation(COLORS).tolist()))
        person = str(rng.choice(PEOPLE))
        pp = prompts(mapping, colors, rng.permutation(PEOPLE), rng.permutation(BOXES), person)
        fingerprint = hashlib.sha256(pp["two_hop"].encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        cases = {}
        for kind, prompt in pp.items():
            candidates = BOXES if kind == "person_box" else COLORS
            answer = mapping[person] if kind == "person_box" else colors[mapping[person]]
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            candidate_ids = [tokenizer.encode(" " + c, add_special_tokens=False) for c in candidates]
            assert all(len(x) == 1 for x in candidate_ids)
            offsets = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            unused_box = next(b for b in BOXES if b != mapping[person])
            line = f"Box {unused_box} contains {colors[unused_box]}."
            end = prompt.index(line) + len(line)
            control_index = max(i for i, (a, b) in enumerate(offsets) if a < end and b <= end)
            cases[kind] = dict(
                prompt=prompt, ids=ids, candidates=list(candidates),
                candidate_ids=[x[0] for x in candidate_ids],
                answer=answer, answer_index=list(candidates).index(answer),
                control_index=control_index,
            )
        scenes.append(dict(id=len(scenes), fingerprint=fingerprint, mapping=mapping,
                           colors=colors, person=person, cases=cases))
    return scenes


def case_masks(case):
    n = len(case["ids"])
    masks = {k: torch.zeros(n, dtype=torch.bool, device="cuda") for k in ("q", "cq", "F", "cF")}
    masks["q"][-1] = True
    masks["cq"][case["control_index"]] = True
    return masks


def metrics(vec, case):
    v = torch.as_tensor(np.asarray(vec).copy(), dtype=torch.float64)
    logp = v.log_softmax(-1)
    p = logp.exp()
    cand = v[case["candidate_ids"]]
    target = case["answer_index"]
    return dict(correct=bool(cand.argmax().item() == target),
                candidate_logits=cand.tolist(),
                candidate_probabilities=p[case["candidate_ids"]].tolist(),
                candidate_mass=float(p[case["candidate_ids"]].sum()),
                candidate_entropy=float(-(cand.softmax(-1) * cand.log_softmax(-1)).sum()),
                full_entropy=float(-(p * logp).sum()),
                gold_probability=float(p[case["candidate_ids"][target]]),
                full_vocab_top1_id=int(v.argmax()),
                gold_rank=int((v > v[case["candidate_ids"][target]]).sum()) + 1)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.enabled = True
        self.scaling = alpha / rank
        self.a = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device, dtype=torch.float32) * 0.01)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))

    def forward(self, x):
        output = self.base(x)
        if not self.enabled:
            return output
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
        return output + (delta * self.scaling).to(output.dtype)


def install_lora(model, rank=8):
    for layer in model.model.layers:
        for name in ("q_proj", "v_proj"):
            setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name), rank))
    return [p for p in model.parameters() if p.requires_grad]


@contextmanager
def teacher_mode(model):
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    old = [m.enabled for m in modules]
    for m in modules:
        m.enabled = False
    try:
        with torch.no_grad():
            yield
    finally:
        for m, flag in zip(modules, old):
            m.enabled = flag


def reverse_kl(student, teacher):
    logp, logq = student.log_softmax(-1), teacher.detach().log_softmax(-1)
    return (logp.exp() * (logp - logq)).sum(-1).mean()


def aggregate_log_probs(logits, candidate_ids):
    """Keep original candidate mass and a numerically stable OTHER event."""
    logp = logits.log_softmax(-1)
    selected = logp[..., candidate_ids]
    rest = logp.clone()
    rest[..., candidate_ids] = -torch.inf
    return torch.cat([selected, rest.logsumexp(-1, keepdim=True)], -1)


def preserve_kl(student, teacher, candidate_ids):
    logp = aggregate_log_probs(student, candidate_ids)
    logq = aggregate_log_probs(teacher.detach(), candidate_ids)
    return (logq.exp() * (logq - logp)).sum(-1).mean()


@torch.no_grad()
def sample_rollout(model, input_ids, max_new_tokens, generator):
    """Fresh current-S2 rollout; full-prefix recompute avoids cache ambiguity."""
    ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = logits_at_loops(model, ids, 2, slice(-1, None))[-1][0, -1]
        token = torch.multinomial(logits.softmax(-1), 1, generator=generator).view(1, 1)
        ids = torch.cat([ids, token], dim=1)
        if int(token) == model.config.eos_token_id:
            break
    return ids
