"""One local vLLM actor with explicitly scoped GPU/process ownership."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from ep_loop.cpu_runtime import cpu_thread_environment


class VllmActor:
    def __init__(self, model, gpu, log_path, loops=4, policy='learned',
                 stop_threshold=.5, native_threshold=.9, max_model_len=17408,
                 memory_fraction=.7, eager=False, sleep=False, teacher_trace=False):
        root=Path(__file__).resolve().parents[2]
        self.log_path=Path(log_path);self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.log=self.log_path.open('a',buffering=1)
        command=[sys.executable,'-u','-m','ep_loop.fastpath.vllm_server',
            '--model',str(Path(model).resolve()),'--loops',str(loops),'--policy',policy,
            '--stop-threshold',str(stop_threshold),'--native-threshold',str(native_threshold),
            '--max-model-len',str(max_model_len),
            '--gpu-memory-utilization',str(memory_fraction)]
        if eager:command+=['--eager']
        if sleep:command+=['--sleep']
        if teacher_trace:command+=['--teacher-trace']
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
             'VLLM_WORKER_MULTIPROC_METHOD':'spawn','PYTHONHASHSEED':'42',
             **cpu_thread_environment(),'TOKENIZERS_PARALLELISM':'false',
             'PYTHONPATH':str(root)+os.pathsep+os.environ.get('PYTHONPATH','')}
        # The rollout actor owns a separate one-GPU communicator. Never let a
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
