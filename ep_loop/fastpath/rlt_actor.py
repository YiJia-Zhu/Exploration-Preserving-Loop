"""Evaluation adapter for the standalone vLLM-RLT engine."""
import json
from pathlib import Path
import subprocess
import sys

from .actor import VllmActor


RELEASE = Path(__file__).resolve().parents[2] / 'audit/vllm_rlt_v020_release_20261120_144351'
RLT_COMMIT = 'c08b9d79834bb8b4ff283b15bae2c14c559f6b52'


def verified_release():
    python_bin = RELEASE / 'benchmark-venv/bin/python'
    checkout = RELEASE / 'vllm-rlt'
    version, source = json.loads(subprocess.check_output([str(python_bin), '-I', '-c',
        'import importlib.metadata as m, importlib.util as u, json; '
        'print(json.dumps([m.version("vllm-rlt"), u.find_spec("vllm_rlt").origin]))'], text=True))
    commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if version != '0.2.0' or commit != RLT_COMMIT or dirty or not Path(source).resolve().is_relative_to(checkout):
        raise RuntimeError('Expected the installed, unmodified RLT v0.2.0 release')
    return dict(version=version, commit=commit, source=source, python_bin=str(python_bin))


class RltActor(VllmActor):
    @classmethod
    def from_release(cls, model, gpu, log_path, **options):
        release = verified_release()
        actor = cls(model, gpu, log_path, python_bin=release['python_bin'], **options)
        source = actor.runtime.get('module_path')
        if not source or Path(source).resolve() != Path(release['source']).resolve():
            actor.close()
            raise RuntimeError('The running RLT worker imported a different module')
        actor.runtime.update(release)
        return actor

    def __init__(self, model, gpu, log_path, *, loops, max_model_len, eager=False,
                 max_num_seqs=None, max_num_batched_tokens=None, prefill_chunk_size=None,
                 attention_backend='triton', async_scheduling=False, prefix_caching=False,
                 python_bin=None, kv_block_size=None):
        checkout = Path(__file__).resolve().parents[2] / 'external/vllm-rlt'
        if python_bin is None and not (checkout / 'vllm_rlt').is_dir():
            raise FileNotFoundError(f'Clone InferMatrixLab/vllm-rlt into {checkout}')
        command = [str(python_bin or sys.executable), '-u', '-m', 'ep_loop.fastpath.rlt_server',
                   '--model', str(model), '--loops', str(loops),
                   '--max-model-len', str(max_model_len),
                   '--attention-backend', attention_backend]
        for flag, value in (
            ('--max-num-seqs', max_num_seqs),
            ('--max-num-batched-tokens', max_num_batched_tokens),
            ('--prefill-chunk-size', prefill_chunk_size),
            ('--kv-block-size', kv_block_size),
        ):
            if value is not None:
                command.extend([flag, str(value)])
        if eager:
            command.append('--eager')
        if async_scheduling:
            command.append('--async-scheduling')
        if prefix_caching:
            command.append('--enable-prefix-caching')
        self._start(command, gpu, log_path, python_paths=(checkout,) if python_bin is None else ())

    def iter_responses(self, requests):
        self.process.stdin.write(json.dumps(dict(op='generate', requests=requests)) + '\n')
        self.process.stdin.flush()
        seen = set()
        while True:
            message = self._read()
            if not message.get('ok'):
                raise RuntimeError(message)
            if message.get('done'):
                if len(seen) != len(requests):
                    raise RuntimeError('vLLM-RLT returned an incomplete batch')
                return
            index = message['index']
            if index in seen or not 0 <= index < len(requests):
                raise RuntimeError(f'Unexpected vLLM-RLT response index {index}')
            seen.add(index)
            yield index, message['response'], message['seconds']
