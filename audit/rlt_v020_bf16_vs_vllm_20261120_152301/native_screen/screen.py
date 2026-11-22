"""Time the optional fused SiLU operator with the same CUDA 13 stack."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parents[1] / 'rlt_v020_runtime_probe_20261120_151153'
sys.path.insert(0, str(PREVIOUS))
import run as probe


def main():
    python = OUT.parent / 'native-venv/bin/python'
    if len(sys.argv) == 2:
        mode = sys.argv[1]
        if mode == 'fallback':
            sys.modules['vllm_rlt._C'] = None
        from vllm_rlt.kernels._extension import try_load_extension
        assert try_load_extension() == (mode == 'native')
        probe.OUT = OUT
        probe.CASES[mode] = dict(probe.CASES['packed_prefill'], block=64)
        probe.worker(mode, repetitions=2)
        return

    gpu = '7'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
           'CUDA_HOME': '/usr/local/cuda'}
    env.pop('PYTHONPATH', None)
    with open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        probe.require_idle_gpus([gpu])
        for mode in ('fallback', 'native'):
            with (OUT / f'{mode}.log').open('w') as log:
                subprocess.run([str(python), '-u', __file__, mode],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
