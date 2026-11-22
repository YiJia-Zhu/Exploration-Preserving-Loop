"""Repeat the matched 8-question speed test with the installed RLT release."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess

AUDIT = Path(__file__).resolve().parent
PRIOR_AUDIT = AUDIT.parent / 'ouro_base_8q_backend_speed_20261120_143308'
RELEASE_AUDIT = AUDIT.parent / 'vllm_rlt_v020_release_20261120_144351'
RELEASE = RELEASE_AUDIT / 'vllm-rlt'
PYTHON = RELEASE_AUDIT / 'benchmark-venv/bin/python'
TAG = 'c08b9d79834bb8b4ff283b15bae2c14c559f6b52'

spec = importlib.util.spec_from_file_location('original_benchmark', PRIOR_AUDIT / 'run.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class InstalledRltActor(benchmark.RltActor):
    def __init__(self, model, gpu, log_path, *, loops, max_model_len,
                 attention_backend, prefix_caching):
        command = [
            str(PYTHON), '-u', str(AUDIT / 'rlt_server.py'), '--kv-block-size', '64',
            '--model', str(model), '--loops', str(loops),
            '--max-model-len', str(max_model_len),
            '--attention-backend', attention_backend,
        ]
        if prefix_caching:
            command.append('--enable-prefix-caching')
        self._start(command, gpu, log_path)


def main():
    os.environ.pop('PYTHONPATH', None)
    installed = json.loads(subprocess.check_output([
        str(PYTHON), '-c',
        'import importlib.metadata as m, json, vllm_rlt, torch; '
        'print(json.dumps(dict(rlt=m.version("vllm-rlt"), path=vllm_rlt.__file__,'
        ' torch=torch.__version__, cuda=torch.version.cuda, triton=m.version("triton"))))',
    ], text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}))
    head = subprocess.check_output(['git', '-C', RELEASE, 'rev-parse', 'HEAD'], text=True).strip()
    if installed['rlt'] != '0.2.0' or not Path(installed['path']).resolve().is_relative_to(RELEASE):
        raise RuntimeError(f'Unexpected installed module: {installed}')
    if head != TAG:
        raise RuntimeError(f'Unexpected release commit: {head}')
    benchmark.write_json(AUDIT / 'runtime.json', dict(commit=head, **installed))

    workload = json.loads((PRIOR_AUDIT / 'workload.json').read_text())
    workload['rlt_revision'] = head
    workload['gpus'] = {'vllm': '0', 'rlt': '3'}
    workload['rlt_kv_block_size'] = 64
    benchmark.write_json(AUDIT / 'workload.json', workload)
    benchmark.OUT = AUDIT
    benchmark.RltActor = InstalledRltActor
    benchmark.configure_cpu_threads()
    tokenizer = benchmark.AutoTokenizer.from_pretrained(benchmark.MODEL, local_files_only=True)
    with ExitStack() as stack:
        for gpu in workload['gpus'].values():
            lock = stack.enter_context(open(f'/tmp/ep-loop-gpu{gpu}.lock', 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        benchmark.require_idle_gpus(list(workload['gpus'].values()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(benchmark.run_backend, name, gpu, workload['jobs'],
                                         workload['warm_jobs'], tokenizer)
                       for name, gpu in workload['gpus'].items()}
            results = {name: future.result() for name, future in futures.items()}

    summary = {
        'runtime': installed, 'rlt_commit': head,
        'engines': {name: report['runtime'] for name, report in results.items()},
        'previous_comparison': str(RELEASE_AUDIT / 'comparison.json'),
    }
    for phase in ('short', 'full'):
        vllm, rlt = (results[name][phase] for name in ('vllm', 'rlt'))
        fields = ('seconds', 'output_tokens', 'tokens_per_second', 'stopped_at_limit', 'correct')
        pairs = list(zip(vllm['answers'], rlt['answers'], workload['jobs'], strict=True))
        for vllm_answer, rlt_answer, job in pairs:
            for answer in (vllm_answer, rlt_answer):
                if (answer['id'], answer['sampling_seed'], answer['prompt_tokens']) != (
                    job['id'], job['seed'], len(job['input_ids'])
                ):
                    raise RuntimeError(f'Unmatched question or seed: {job["id"]}')
        summary[phase] = dict(
            vllm={field: vllm[field] for field in fields},
            rlt={field: rlt[field] for field in fields},
            throughput_ratio_vllm_over_rlt=vllm['tokens_per_second'] / rlt['tokens_per_second'],
            batch_speedup_vllm_over_rlt=rlt['seconds'] / vllm['seconds'],
            questions=[
                dict(id=job['id'], vllm_tokens=vllm_answer['response_tokens'],
                     rlt_tokens=rlt_answer['response_tokens'],
                     vllm_finish=vllm_answer['finish_reason'], rlt_finish=rlt_answer['finish_reason'])
                for vllm_answer, rlt_answer, job in pairs
            ],
        )
    benchmark.write_json(AUDIT / 'comparison.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
