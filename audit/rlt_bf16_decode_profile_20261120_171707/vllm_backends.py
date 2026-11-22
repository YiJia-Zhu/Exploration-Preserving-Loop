"""Compare official vLLM attention backends while preserving workload and sampler."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
VLLM_PYTHON = ROOT / 'ep_loop_envs/vllm026/bin/python'
PRIOR = OUT.parent / 'rlt_v020_bf16_vs_vllm_20261120_152301'
spec = importlib.util.spec_from_file_location('fixed_prefix', PRIOR / 'vllm_fixed_prefix.py')
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)


def main():
    workload = json.loads((PRIOR / 'native_screen/workload.json').read_text())
    gpu = '4'
    results = {}
    with open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixed.require_idle_gpus([gpu])
        for backend in ('FLASHINFER', 'TRITON_ATTN'):
            class Actor(fixed.VllmActor):
                def __init__(self):
                    command = [
                        str(VLLM_PYTHON), '-u', str(OUT/'vllm_server.py'),
                        '--model', workload['model'], '--policy', 'fixed', '--loops', '2',
                        '--max-model-len', '17408', '--gpu-memory-utilization', '.75',
                        '--attention-backend', backend,
                    ]
                    self._start(command, gpu, OUT/f'vllm_{backend}.engine.log')

            with Actor() as actor:
                assert not actor.runtime['rollout_ngram']
                phases = {}
                for name, phase in workload['phases'].items():
                    fixed.measure(actor, phase, 16)
                    phases[name] = [fixed.measure(actor, phase, phase['generated_tokens'])
                                    for _ in range(2)]
                    print(backend, name, phases[name], flush=True)
                results[backend] = phases
                (OUT/'vllm_backends.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
