"""Run one synchronous packed-prefill variant on GPU 4."""
import argparse
import fcntl
import os
import subprocess
import sys

import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-size', type=int, choices=(16, 64), default=64)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    name = f'packed_sync_block{args.block_size}'
    if args.worker:
        run.CASES[name] = dict(run.CASES['packed_prefill'], block=args.block_size)
        run.worker(name)
    else:
        gpu = '4'
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
        env.pop('PYTHONPATH', None)
        with open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run.require_idle_gpus([gpu])
            subprocess.run([str(run.RELEASE_PYTHON), '-u', __file__, '--worker',
                            '--block-size', str(args.block_size)], env=env, check=True)
