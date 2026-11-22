"""Compare fixed-K4 nested and explicit unrolled vLLM forwards."""
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from ep_loop.fastpath.actor import VllmActor
from ep_loop.gpu_resources import require_idle_gpus

WORKLOAD = OUT.parent / 'rlt_v020_runtime_probe_20261120_151153/workload.json'
VLLM_PYTHON = ROOT / 'ep_loop_envs/vllm026/bin/python'
GPU = '3'


class AuditActor(VllmActor):
    def __init__(self, model, unrolled):
        label = 'unrolled' if unrolled else 'native'
        command = [
            str(VLLM_PYTHON), '-u', str(OUT/'vllm_server.py'),
            '--model', model, '--loops', '4', '--policy', 'fixed',
            '--max-model-len', '9216', '--gpu-memory-utilization', '.75',
            '--max-logprobs', '-1', '--attention-backend', 'FLASHINFER',
        ]
        if unrolled:
            command.append('--unrolled')
        self._start(command, GPU, OUT/f'{label}.engine.log', python_paths=(OUT,))


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')


def measure(actor, jobs, cap, logprobs=None, temperature=1.0):
    requests = [dict(input_ids=job['input_ids'], seed=job['seed'], temperature=temperature,
                     top_p=.7 if temperature else 1., max_new_tokens=cap, ignore_eos=True)
                for job in jobs]
    result = actor.generate(requests, logprobs=logprobs)
    responses = result['responses']
    assert len(responses) == len(jobs)
    assert all(len(row['token_ids']) == cap and row['exit_depths'] == [4] * cap
               for row in responses)
    assert result['engine_metrics']['vllm:num_preemptions'] == 0
    return dict(seconds=result['seconds'], output_tokens=cap*len(jobs),
                token_ids=[row['token_ids'] for row in responses],
                logprobs=[row['logprobs'] for row in responses])


def run(model, workloads, unrolled):
    label = 'unrolled' if unrolled else 'native'
    startup = time.perf_counter()
    with AuditActor(model, unrolled) as actor:
        elapsed = time.perf_counter() - startup
        print(label, 'started in', round(elapsed, 2), 'seconds', flush=True)
        data = {'startup_seconds': elapsed, 'workloads': {}}
        for phase,jobs in workloads.items():
            cap = 256 if phase == 'short' else 128
            measure(actor, jobs, cap)
            runs = [measure(actor, jobs, cap) for _ in range(3)]
            test = measure(actor, jobs[:1], 4, logprobs=-1, temperature=0.)
            median = statistics.median(r['seconds'] for r in runs)
            data['workloads'][phase] = dict(times=[r['seconds'] for r in runs],
                                           median_seconds=median,
                                           output_tokens=runs[0]['output_tokens'],
                                           stochastic_token_ids=runs[0]['token_ids'],
                                           greedy_logprobs=test['logprobs'][0],
                                           greedy_token_ids=test['token_ids'][0])
            print(label, phase, 'seconds', [round(r['seconds'],3) for r in runs], flush=True)
    return data


def main():
    spec = json.loads(WORKLOAD.read_text())
    short = spec['phases']['short']['jobs']
    long = [dict(job,input_ids=job['input_ids'][:4096]) for job in spec['phases']['8192']['jobs']]
    workloads = {'short': short, '4096': long}
    with open(f'/tmp/ep-loop-gpu{GPU}.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_idle_gpus([GPU])
        native = run(spec['model'], workloads, False)
        explicit = run(spec['model'], workloads, True)
    comparison = {}
    for phase in workloads:
        a,b = (d['workloads'][phase] for d in (native,explicit))
        assert a['output_tokens']==b['output_tokens']
        same_greedy_ids = (a['greedy_token_ids']==b['greedy_token_ids'])
        max_diff = 0.
        positions = 0
        for p,q in zip(a['greedy_logprobs'], b['greedy_logprobs'], strict=True):
            p = dict(zip(p['token_ids'], p['log_probs'], strict=True))
            q = dict(zip(q['token_ids'], q['log_probs'], strict=True))
            assert len(p)==len(q)==49152 and p.keys()==q.keys()
            max_diff = max(max_diff, max(abs(p[k]-q[k]) for k in p if p[k]!=q[k])
                           if p!=q else 0.)
            positions += 1
        comparison[phase] = dict(
            native_seconds=a['times'], unrolled_seconds=b['times'],
            native_median_seconds=a['median_seconds'], unrolled_median_seconds=b['median_seconds'],
            output_tokens=a['output_tokens'],
            stochastic_sequences_equal=sum(x==y for x,y in zip(
                a['stochastic_token_ids'],b['stochastic_token_ids'],strict=True)),
            greedy_four_tokens_equal=same_greedy_ids, vocab_logprobs_positions=positions,
            vocab_logprobs_max_abs_diff=max_diff if same_greedy_ids else None,
        )
    write(OUT/'comparison.json', comparison)
    print(json.dumps(comparison,indent=2),flush=True)


if __name__ == '__main__':
    main()
