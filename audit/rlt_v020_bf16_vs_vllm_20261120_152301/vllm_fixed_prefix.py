"""Run the official engine on the tuning workloads at equal output length."""
import fcntl
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from ep_loop.fastpath.actor import VllmActor
from ep_loop.gpu_resources import require_idle_gpus


def measure(actor, phase, cap):
    requests = [
        dict(input_ids=job['input_ids'], seed=job['seed'], temperature=1., top_p=.7,
             max_new_tokens=cap, ignore_eos=True)
        for job in phase['jobs']
    ]
    result = actor.generate(requests, logprobs=None)
    outputs = result['responses']
    if len(outputs) != len(requests) or any(len(row['token_ids']) != cap
                                            or row['exit_depths'] != [2] * cap for row in outputs):
        raise RuntimeError('Unexpected official vLLM batch length or depth')
    total = sum(len(row['token_ids']) for row in outputs)
    return dict(seconds=result['seconds'], output_tokens=total,
                output_tok_s=total / result['seconds'],
                all_first_tokens_s=max(row['request_timing']['time_to_first_token_seconds']
                                       for row in outputs),
                engine_metrics=result['engine_metrics'])


def main():
    spec = json.loads((OUT / 'native_screen/workload.json').read_text())
    gpu = '4'
    results = {}
    with open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_idle_gpus([gpu])
        with VllmActor(spec['model'], gpu, OUT / 'vllm_fixed_prefix.engine.log',
                        loops=2, policy='fixed', max_model_len=17408, memory_fraction=.75,
                        attention_backend='FLASHINFER',
                        python_bin=ROOT / 'ep_loop_envs/vllm026/bin/python') as actor:
            for name, phase in spec['phases'].items():
                measure(actor, phase, 16)
                results[name] = [measure(actor, phase, phase['generated_tokens']) for _ in range(2)]
                print(name, results[name], flush=True)
    (OUT / 'vllm_fixed_prefix.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
