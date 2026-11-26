"""One local vLLM actor with explicitly scoped GPU/process ownership."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from ep_loop.cpu_runtime import cpu_thread_environment


def align_teacher_traces(traces, request_ids):
    """Map v0.26 internal UUID-suffixed IDs back to public output IDs exactly."""
    result=[];used=set()
    for request_id in request_ids:
        matches=[key for key in traces if key==request_id or
                 re.fullmatch(re.escape(request_id)+r'-[0-9a-f]{8}',key)]
        if len(matches)!=1 or matches[0] in used:
            raise RuntimeError(f'Teacher request/trace mapping is not one-to-one: {request_id}, {matches}')
        used.add(matches[0]);result.append(traces[matches[0]])
    if used!=set(traces):raise RuntimeError('Teacher returned unmatched traces')
    return result


def validate_weight_workers(workers, version, tensor_parallel_size):
    if (len(workers) != tensor_parallel_size or
            any(row.get('weight_version') != version for row in workers)):
        raise RuntimeError(f'Incomplete or stale TP weight refresh: {workers}')
    return dict(workers[0], workers=workers, tensor_parallel_size=tensor_parallel_size)


def leader_teacher_traces(workers, tensor_parallel_size):
    if len(workers) != tensor_parallel_size or any(workers[1:]):
        raise RuntimeError('Only TP rank zero may write teacher trace files')
    return workers[0]


class VllmActor:
    def __init__(self, model, gpu, log_path, loops=4, policy='learned',
                 stop_threshold=.5, native_threshold=.9, max_model_len=17408,
                 memory_fraction=.7, eager=False, sleep=False, teacher_trace=False,
                 python_bin=None, attention_backend=None, max_logprobs=None,
                 max_num_seqs=None, tensor_parallel_size=1, rollout_ngram=False,
                 prefix_caching=False):
        from .speculation import validate_ngram_actor
        validate_ngram_actor(rollout_ngram, policy, teacher_trace)
        if max_num_seqs is not None and (type(max_num_seqs) is not int or max_num_seqs<1):
            raise ValueError('max_num_seqs must be a positive integer or None for the engine default')
        from .parallel import gpu_groups
        groups = gpu_groups(str(gpu).split(','), tensor_parallel_size)
        if len(groups) != 1:
            raise ValueError('One actor must receive exactly tensor_parallel_size visible GPUs')
        python_bin=python_bin or os.environ.get('VLLM_PYTHON_BIN') or sys.executable
        attention_backend=attention_backend or os.environ.get('VLLM_ATTENTION_BACKEND')
        command=[str(python_bin),'-u','-m','ep_loop.fastpath.vllm_server',
            '--model',str(Path(model).resolve()),'--loops',str(loops),'--policy',policy,
            '--stop-threshold',str(stop_threshold),'--native-threshold',str(native_threshold),
            '--max-model-len',str(max_model_len),
            '--gpu-memory-utilization',str(memory_fraction),
            '--tensor-parallel-size',str(tensor_parallel_size)]
        if eager:command+=['--eager']
        if sleep:command+=['--sleep']
        if teacher_trace:command+=['--teacher-trace']
        if attention_backend:command+=['--attention-backend',attention_backend]
        if max_logprobs is not None:command+=['--max-logprobs',str(max_logprobs)]
        if max_num_seqs is not None:command+=['--max-num-seqs',str(max_num_seqs)]
        if rollout_ngram:command+=['--rollout-ngram']
        if prefix_caching:command+=['--enable-prefix-caching']
        self._start(command,gpu,log_path)

    def _start(self, command, gpu, log_path, python_paths=()):
        root=Path(__file__).resolve().parents[2]
        self.log_path=Path(log_path);self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.log=self.log_path.open('a',buffering=1)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
             'VLLM_WORKER_MULTIPROC_METHOD':'spawn','PYTHONHASHSEED':'42',
             **cpu_thread_environment(),'TOKENIZERS_PARALLELISM':'false',
             'MAX_JOBS':os.environ.get('MAX_JOBS','16'),
             'PYTHONPATH':os.pathsep.join([str(root),*map(str,python_paths),os.environ.get('PYTHONPATH','')])}
        # The selected Python environment can supply JIT tools such as ninja.
        bin_dir = os.path.dirname(command[0])
        if bin_dir:
            env['PATH'] = os.pathsep.join((str((root / bin_dir).resolve()), env.get('PATH') or os.defpath))
        # The rollout actor owns a separate TP communicator. Never let a
        # torchrun trainer's rank/store leak into its subprocess hierarchy.
        for key in list(env):
            if key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK',
                       'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
                env.pop(key,None)
        # HF benefits from expandable segments, but vLLM sleep/wake owns a
        # CuMem memory pool that rejects that allocator option. Isolate only
        # the actor environment; never change the training process allocator.
        for key in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF'):
            if 'expandable_segments:True' in env.get(key,''):
                env.pop(key,None)
        self.process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,start_new_session=True)
        try:
            ready=self._read()
            if ready.get('ready') is not True:raise RuntimeError(ready)
            self.runtime=ready
        except BaseException:
            self.close();raise

    def _read(self):
        for line in self.process.stdout:
            try:return json.loads(line)
            except json.JSONDecodeError:self.log.write(line)
        raise RuntimeError(f'vLLM actor exited ({self.process.poll()}); see {self.log_path}')

    def request(self, op, **fields):
        self.process.stdin.write(json.dumps(dict(op=op,**fields))+'\n');self.process.stdin.flush()
        response=self._read()
        if response.get('ok') is not True:raise RuntimeError(response)
        return response['result']

    def generate(self, requests, logprobs=20):
        return self.request('generate',requests=requests,logprobs=logprobs)

    def iter_responses(self, requests):
        result = self.generate(requests, logprobs=None)
        responses = result['responses']
        if len(responses) != len(requests):
            raise RuntimeError('vLLM returned an incomplete batch')
        for index, response in enumerate(responses):
            timing = response.get('request_timing')
            seconds = sum(timing.values()) if timing else result['seconds']
            yield index, response, seconds

    def teacher_trace(self, requests, output):
        return self.request('teacher_trace',requests=requests,output=str(output))

    def close(self):
        process=getattr(self,'process',None)
        if process is not None:
            try:
                if process.poll() is None:
                    process.stdin.write('{"op":"close"}\n');process.stdin.flush()
                    process.wait(timeout=20)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
        if getattr(self,'log',None) is not None:self.log.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
