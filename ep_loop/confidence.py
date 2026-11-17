"""Teacher-only confidence instruments; never supply gold answers to probes.

Direct signals reuse original-prefix distributions. Prompt probes explicitly
change the input and cost extra forwards; they are opt-in experimental labels.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import math

import torch

from .ouro import evaluation_mode


SIGNALS = {"js", "max_probability", "normalized_entropy", "probe_know"}
KNOW_PROMPT = '\n\n[Confidence probe] Can you now give a correct final answer to the original question? Reply only Yes or No.\nAnswer:'


def confidence_changes(log_probs, signal):
    """Absolute scalar changes preserve both increases and decreases."""
    if log_probs.ndim != 3 or signal not in {"max_probability", "normalized_entropy"}:
        raise ValueError("Expected [depth, position, vocabulary] and a direct confidence signal")
    if signal == "max_probability":
        confidence = log_probs.max(-1).values.exp()
    else:
        if log_probs.shape[-1] < 2:
            raise ValueError("Normalized entropy requires at least two vocabulary entries")
        confidence = (1 + (log_probs.exp()*log_probs).sum(-1)/math.log(log_probs.shape[-1])).clamp(0, 1)
    return (confidence[1:]-confidence[:-1]).abs(), confidence


@contextmanager
def isolated_probe_rng(backend, ids, seed):
    """Probe instrumentation must not advance training/rollout random streams."""
    device = backend.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    digest = hashlib.sha256(ids.detach().cpu().numpy().tobytes()).digest()
    derived = (int.from_bytes(digest[:8], 'big')+int(seed)) % (2**63-1)
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(derived)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(derived)
        yield


class PromptConfidence:
    def __init__(self, backend, tokenizer, seed=42, observer=None):
        self.backend, self.tokenizer, self.seed = backend, tokenizer, seed
        self.observer = observer
        self.yes = tokenizer.encode(' Yes', add_special_tokens=False)
        self.no = tokenizer.encode(' No', add_special_tokens=False)
        if not self.yes or not self.no or self.yes == self.no:
            raise ValueError("Probe requires distinct nonempty Yes and No tokenizations")

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _trace(self, ids, depth, cost, initial_state=None):
        if ids.shape[1] > self.backend.max_context_length:
            raise ValueError("Confidence probe exceeds model context; shorten prompt/response or probe")
        trace = self.backend.trace(ids, depth, use_cache=False,
                                   **(dict(initial_state=initial_state) if initial_state is not None else {}))
        cost['token_loops'] += trace.executed_token_loops
        cost['coda_token_positions'] += trace.executed_token_loops if self.backend.has_coda else 0
        cost['forwards'] += 1
        return trace

    def _label_scores(self, prefix, tokens, depth, cost, initial_state=None):
        ids = torch.cat((prefix, prefix.new_tensor([tokens[:-1]])), dim=1) if len(tokens)>1 else prefix
        trace = self._trace(ids, depth, cost,initial_state[:,:ids.shape[1]] if initial_state is not None else None)
        logp = self.backend.head(torch.stack([state[:,prefix.shape[1]-1:] for state in trace.states])).log_softmax(-1)
        targets = ids.new_tensor(tokens)
        return logp[:,0, torch.arange(len(tokens), device=ids.device), targets].sum(-1)

    @torch.no_grad()
    def measure(self, inputs, first, exits, signal, positions=None, record_details=False):
        if signal != 'probe_know':
            raise ValueError("Unknown prompt confidence signal")
        selected = list(range(exits.numel())) if positions is None else list(positions)
        if any(i<0 or i>=exits.numel() for i in selected) or len(set(selected))!=len(selected):
            raise ValueError("Probe positions must be distinct valid response positions")
        k = int(exits.max())
        confidence = torch.zeros((k, len(selected)), device=inputs.device)
        cost = dict(token_loops=0, coda_token_positions=0, forwards=0, positions=len(selected))
        self.last_records = []
        stage = self.observer.stage('confidence_probes', signal=signal, positions=len(selected)) if self.observer else nullcontext({})
        iterator = self.observer.progress(selected, desc='Teacher confidence probes', unit='position', position=2, leave=False) if self.observer else selected
        with stage as stats, evaluation_mode(self.backend.model), isolated_probe_rng(self.backend, inputs, self.seed):
            for column, pos in enumerate(iterator):
                prefix = inputs[:, :first+pos+1]
                endpoint = int(exits[pos])
                branch = torch.cat((prefix, prefix.new_tensor([self._encode(KNOW_PROMPT)])), dim=1)
                # One trajectory supplies ALL depth readouts. This avoids
                # O(K^2) repeated loops and random-initialization confounding.
                initial = None
                if self.backend.has_coda:
                    length = branch.shape[1]+max(len(self.yes),len(self.no))-1
                    # Native Huginn keeps the prelude residual in embedding
                    # dtype even under BF16 autocast. Match that initialization.
                    parameter = self.backend.model.transformer.wte.weight
                    with isolated_probe_rng(self.backend,branch,self.seed):
                        initial = self.backend.model.initialize_state(parameter.new_empty((1,length,self.backend.model.config.n_embd)))
                if len(self.yes)==len(self.no)==1:
                    trace = self._trace(branch,endpoint,cost,initial)
                    readouts = torch.stack([state[:,-1:] for state in trace.states])
                    logp = self.backend.head(readouts)[:,0,0].log_softmax(-1)
                    yes,no = logp[:,self.yes[0]],logp[:,self.no[0]]
                else:
                    with isolated_probe_rng(self.backend, branch, self.seed):
                        yes = self._label_scores(branch,self.yes,endpoint,cost,initial)
                    with isolated_probe_rng(self.backend, branch, self.seed):
                        no = self._label_scores(branch,self.no,endpoint,cost,initial)
                label_log_probs = torch.stack((yes,no),dim=-1)
                confidence[:endpoint,column] = label_log_probs.softmax(-1)[:,0]
                confidence[endpoint:,column] = confidence[endpoint-1,column]
                if record_details:
                    self.last_records.append(dict(response_position=pos,
                        label_log_probs=label_log_probs.tolist()))
            stats.update(cost)
        return (confidence[1:]-confidence[:-1]).abs(), confidence, cost
