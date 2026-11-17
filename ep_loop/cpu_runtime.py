"""CPU intra-op threads per process; model/GPU process counts stay separate."""
import os


def cpu_thread_environment():
    raw = os.environ.get('CPU_THREADS', '16')
    if not raw.isdecimal() or int(raw) < 1:
        raise ValueError('CPU_THREADS must be a positive integer')
    value = str(int(raw))
    return {name: value for name in (
        'CPU_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def configure_cpu_threads():
    environment = cpu_thread_environment()
    os.environ.update(environment)
    import torch
    threads = int(environment['CPU_THREADS'])
    torch.set_num_threads(threads)
    return threads
