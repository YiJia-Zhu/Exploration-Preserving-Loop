"""Screen two larger pages without changing the BF16 or sampling policy."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
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
    options = {128: '4', 256: '6'}
    if len(sys.argv) == 2:
        block = int(sys.argv[1])
        name = f'block{block}'
        probe.CASES[name] = dict(probe.CASES['packed_prefill'], block=block)
        probe.OUT = OUT
        probe.worker(name, repetitions=2)
        return
    with ExitStack() as stack:
        for gpu in options.values():
            lock = stack.enter_context(open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        probe.require_idle_gpus(list(options.values()))

        def launch(case):
            block, gpu = case
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
            env.pop('PYTHONPATH', None)
            with (OUT / f'block{block}.log').open('w') as log:
                subprocess.run([str(probe.RELEASE_PYTHON), '-u', __file__, str(block)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        with ThreadPoolExecutor(max_workers=len(options)) as pool:
            list(pool.map(launch, options.items()))


if __name__ == '__main__':
    main()
