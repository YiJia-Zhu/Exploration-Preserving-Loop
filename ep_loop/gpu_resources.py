"""Read-only GPU admission check; small CUDA reservations also count as busy."""
import csv
import io
import subprocess


def gpu_status():
    def query(fields, kind):
        raw=subprocess.check_output(['nvidia-smi',f'--query-{kind}={fields}',
                                     '--format=csv,noheader,nounits'],text=True)
        return list(csv.reader(io.StringIO(raw),skipinitialspace=True))
    devices={row[1].strip():dict(index=row[0].strip(),uuid=row[1].strip(),
                               memory_used_MiB=int(row[2]),processes=[])
             for row in query('index,uuid,memory.used','gpu') if row}
    for row in query('gpu_uuid,pid,process_name','compute-apps'):
        if row and row[0].strip() in devices:
            devices[row[0].strip()]['processes'].append(dict(pid=int(row[1]),name=row[2].strip()))
    return sorted(devices.values(),key=lambda row:int(row['index']))


def require_idle_gpus(indices):
    wanted={str(x) for x in indices}
    devices={row['index']:row for row in gpu_status()}
    if wanted-set(devices):raise ValueError(f'Unknown GPU IDs: {sorted(wanted-set(devices))}')
    busy=[devices[i] for i in sorted(wanted)
          if devices[i]['processes'] or devices[i]['memory_used_MiB']>64]
    if busy:raise RuntimeError(f'GPUs are occupied or reserved; no inference/training started: {busy}')
    return [devices[str(i)] for i in indices]
