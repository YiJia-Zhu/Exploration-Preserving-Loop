"""Check numerical equivalence of explicit unrolling with and without compilation."""
import fcntl
import json
import math
from pathlib import Path

import run as bench

OUT = Path(__file__).resolve().parent


class VerifyActor(bench.VllmActor):
    def __init__(self, model, label, *, unrolled, eager):
        command = [
            str(bench.VLLM_PYTHON), '-u', str(OUT/'vllm_server.py'),
            '--model', model, '--loops', '4', '--policy', 'fixed',
            '--max-model-len', '9216', '--gpu-memory-utilization', '.75',
            '--max-logprobs', '-1', '--attention-backend', 'FLASHINFER',
        ]
        if unrolled:
            command.append('--unrolled')
        if eager:
            command.append('--eager')
        self._start(command, bench.GPU, OUT/f'verify_{label}.engine.log', python_paths=(OUT,))


def compare(a, b):
    result = dict(greedy_equal=a['token_ids'] == b['token_ids'], exact_logprobs=True,
                  max_abs_logprob_diff=0., max_total_variation=0.)
    for left,right in zip(a['logprobs'][0],b['logprobs'][0],strict=True):
        p = dict(zip(left['token_ids'],left['log_probs'],strict=True))
        q = dict(zip(right['token_ids'],right['log_probs'],strict=True))
        assert len(p) == len(q) == 49152 and p.keys() == q.keys()
        result['exact_logprobs'] &= p == q
        maximum = max((abs(p[k]-q[k]) for k in p if p[k]!=q[k]),default=0.)
        tv = .5 * math.fsum(abs(math.exp(p[k])-math.exp(q[k])) for k in p)
        result['max_abs_logprob_diff'] = max(result['max_abs_logprob_diff'],maximum)
        result['max_total_variation'] = max(result['max_total_variation'],tv)
    return result


def main():
    model_spec = json.loads(bench.WORKLOAD.read_text())
    prompts = {
        'short': model_spec['phases']['short']['jobs'][:1],
        '4096': [dict(job, input_ids=job['input_ids'][:4096])
                 for job in model_spec['phases']['8192']['jobs'][:1]],
    }
    summary = {}
    with open(f'/tmp/ep-loop-gpu{bench.GPU}.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bench.require_idle_gpus([bench.GPU])
        for eager in (True,False):
            label = 'eager' if eager else 'compiled'
            data = {}
            for unrolled in (False, True):
                kind = 'unrolled' if unrolled else 'native'
                with VerifyActor(model_spec['model'], f'{label}_{kind}', unrolled=unrolled, eager=eager) as actor:
                    data[kind] = {
                        phase: [bench.measure(actor,jobs,4,logprobs=-1,temperature=0.) for _ in range(2)]
                        for phase,jobs in prompts.items()}
            summary[label] = {
                phase: dict(
                    cross_forward=[compare(x,y) for x,y in zip(
                        data['native'][phase], data['unrolled'][phase], strict=True)],
                    repeat_native=compare(*data['native'][phase]),
                    repeat_unrolled=compare(*data['unrolled'][phase]),
                )
                for phase in prompts
            }
            bench.write(OUT/'equivalence.json',summary)
            print(label, summary[label], flush=True)


if __name__ == '__main__':
    main()
