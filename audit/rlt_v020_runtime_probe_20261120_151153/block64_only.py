"""Change only KV page size in the previous default RLT settings."""
import fcntl
import os
import subprocess
import sys

import run


if __name__ == '__main__':
    name = 'block64_only'
    if len(sys.argv) > 1:
        run.CASES[name] = dict(run.CASES['baseline'], block=64)
        run.worker(name, phases=('short', '8192'), repetitions=1)
    else:
        gpu = '4'
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
        env.pop('PYTHONPATH', None)
        with open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run.require_idle_gpus([gpu])
            subprocess.run([str(run.RELEASE_PYTHON), '-u', __file__, '--worker'], env=env, check=True)
