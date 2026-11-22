"""Small local JSON-lines actor. All library extensions are repo-local copies.

Run in a separate process with its own physical CUDA_VISIBLE_DEVICES group.
Stdout is the protocol; engine logs go to stderr. No weight/environment installs.
"""
import argparse
from contextlib import redirect_stdout
import json
from importlib.metadata import version
import os
from pathlib import Path
import sys
import sysconfig
import time
from ep_loop.fastpath.actor import align_teacher_traces, validate_weight_workers, leader_teacher_traces
from ep_loop.fastpath.speculation import ngram_speculative_config, validate_ngram_actor

vllm_version = version('vllm')
modern = tuple(int(x) for x in vllm_version.split('.')[:2]) >= (0, 26)
if modern:
    # FlashInfer's SM120 JIT needs the CUDA13 toolkit matching this wheel stack,
    # rather than inheriting the HF trainer's CUDA12.8 compiler.
    toolkit=Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'
    if (toolkit/'bin/nvcc').is_file():
        os.environ['CUDA_HOME']=os.environ['CUDA_PATH']=str(toolkit)
        os.environ['PATH']=str(toolkit/'bin')+os.pathsep+os.environ.get('PATH','')

# Register at module import so multiprocessing spawn inherits the same entry.
with redirect_stdout(sys.stderr):
    from ep_loop.cpu_runtime import configure_cpu_threads
    cpu_threads = configure_cpu_threads()
    from vllm import ModelRegistry
    model_module = 'vllm_ouro_v026' if modern else 'vllm_ouro'
    ModelRegistry.register_model('EPLoopOuroForCausalLM',f'ep_loop.fastpath.{model_module}:OuroForCausalLM')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--loops',type=int,default=4)
    parser.add_argument('--policy',choices=['learned','native','fixed'],default='learned')
    parser.add_argument('--stop-threshold',type=float,default=.5)
    parser.add_argument('--native-threshold',type=float,default=.9)
    parser.add_argument('--max-model-len',type=int,default=17408)
    parser.add_argument('--gpu-memory-utilization',type=float,default=.70)
    parser.add_argument('--eager',action='store_true')
    parser.add_argument('--sleep',action='store_true')
    parser.add_argument('--teacher-trace',action='store_true')
    parser.add_argument('--attention-backend',choices=['FLASHINFER','FLASH_ATTN','TRITON_ATTN'])
    parser.add_argument('--max-logprobs',type=int,default=20)
    parser.add_argument('--max-num-seqs',type=int)
    parser.add_argument('--tensor-parallel-size',type=int,default=1)
    parser.add_argument('--rollout-ngram',action='store_true',help='N-gram5 for the training student rollout only')
    args=parser.parse_args()
    protocol=sys.stdout
    def emit(row):
        protocol.write(json.dumps(row,ensure_ascii=False)+'\n');protocol.flush()
    try:
        if args.tensor_parallel_size<1:raise ValueError('tensor_parallel_size must be positive')
        if modern and args.policy != 'fixed':
            raise ValueError('vLLM 0.26 adapter currently supports fixed-depth Ouro only')
        native_fixed = modern and not args.teacher_trace
        validate_ngram_actor(args.rollout_ngram,args.policy,args.teacher_trace,modern)
        if modern and args.teacher_trace:
            # The teacher's all-token trace hook belongs to the V1 runner.
            os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
        with redirect_stdout(sys.stderr):
            from vllm import LLM,SamplingParams
            extra={}
            if args.rollout_ngram:
                extra['speculative_config']=ngram_speculative_config()
            if args.max_num_seqs is not None:
                if args.max_num_seqs<1:raise ValueError('max_num_seqs must be positive')
                extra['max_num_seqs']=args.max_num_seqs
            if modern:
                if args.attention_backend:extra['attention_config']={'backend':args.attention_backend}
                extra['disable_log_stats']=False
                if args.teacher_trace:extra['async_scheduling']=False
            elif args.attention_backend:
                os.environ['VLLM_ATTENTION_BACKEND']=args.attention_backend
            engine=LLM(model=args.model,trust_remote_code=True,dtype='bfloat16',
                generation_config='vllm',
                tensor_parallel_size=args.tensor_parallel_size,seed=42,enforce_eager=args.eager,
                enable_prefix_caching=False,enable_chunked_prefill=modern,
                enable_sleep_mode=args.sleep,max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_logprobs=args.max_logprobs,
                worker_extension_cls='ep_loop.fastpath.vllm_worker.OuroWorkerExtension',
                hf_overrides=dict(architectures=['OuroForCausalLM' if native_fixed else 'EPLoopOuroForCausalLM'],
                    total_ut_steps=args.loops,ep_loop_readout_policy=args.policy,
                    ep_loop_teacher_trace=args.teacher_trace,
                    ep_loop_stop_threshold=args.stop_threshold,ep_loop_native_threshold=args.native_threshold),**extra)
        runtime=dict(vllm_version=vllm_version,attention_backend=args.attention_backend,
                     native_fixed_student=native_fixed,teacher_trace=args.teacher_trace,
                     max_num_seqs=args.max_num_seqs,tensor_parallel_size=args.tensor_parallel_size,
                     rollout_ngram=args.rollout_ngram,
                     speculative_config=ngram_speculative_config() if args.rollout_ngram else None,
                     token_loop_accounting=('Scheduled teacher prefill positions, including recomputation'
                        if args.teacher_trace else
                        'Committed prefix/output only; excludes rejected drafts and KV recomputation'))
        print(json.dumps(dict(event='ep_loop_engine_ready',**runtime)),file=sys.stderr,flush=True)
        emit(dict(ready=True,policy=args.policy,executed_loops=args.loops,cpu_threads=cpu_threads,**runtime))
        def counters():
            if not modern:return {}
            keys={'vllm:num_preemptions','vllm:generation_tokens','vllm:prompt_tokens'}
            result={}
            for metric in engine.get_metrics():
                if ((metric.name in keys or metric.name.startswith('vllm:spec_decode_'))
                        and isinstance(getattr(metric,'value',None),(int,float))):
                    result[metric.name]=result.get(metric.name,0)+metric.value
            return result
        sleeping=False
        for line in sys.stdin:
            request=json.loads(line)
            try:
                with redirect_stdout(sys.stderr):
                    op=request['op']
                    if op=='close':break
                    if op=='sleep':
                        engine.sleep(level=1);sleeping=True;result=dict(sleeping=True)
                    elif op=='wake':
                        engine.wake_up();sleeping=False;result=dict(sleeping=False)
                    elif op=='weights':
                        if sleeping:engine.wake_up();sleeping=False
                        engine.reset_prefix_cache()
                        workers=engine.collective_rpc('ep_load_weights',args=(request['path'],request['version']))
                        result=validate_weight_workers(workers,request['version'],args.tensor_parallel_size)
                    elif op=='health':
                        workers=engine.collective_rpc('ep_health')
                        if len(workers)!=args.tensor_parallel_size:raise RuntimeError('Missing TP health workers')
                        result=dict(workers[0],workers=workers,tensor_parallel_size=args.tensor_parallel_size)
                    elif op=='teacher_trace':
                        if not args.teacher_trace:raise ValueError('Use a teacher-trace engine')
                        if sleeping:engine.wake_up();sleeping=False
                        engine.reset_prefix_cache()
                        engine.collective_rpc('ep_begin_teacher_trace',args=(request['output'],))
                        start=time.perf_counter()
                        try:
                            sampling=[SamplingParams(max_tokens=1,temperature=1.,seed=42+i,
                                      logprobs=None,detokenize=False) for i,_ in enumerate(request['requests'])]
                            outputs=engine.generate([dict(prompt_token_ids=r['input_ids']) for r in request['requests']],
                                                    sampling,use_tqdm=False)
                        finally:
                            traces=leader_teacher_traces(engine.collective_rpc('ep_end_teacher_trace'),args.tensor_parallel_size)
                        responses=[]
                        ordered_traces=align_teacher_traces(traces,[row.request_id for row in outputs])
                        for row,trace in zip(outputs,ordered_traces):
                            if trace['input_positions']!=len(row.prompt_token_ids):
                                raise RuntimeError('Teacher prefill trace differs from supplied student text')
                            responses.append(trace)
                        result=dict(responses=responses,seconds=time.perf_counter()-start,executed_loops=args.loops)
                    elif op=='generate':
                        if sleeping:engine.wake_up();sleeping=False
                        sampling=[SamplingParams(temperature=r['temperature'],top_p=r['top_p'],
                            max_tokens=r['max_new_tokens'],seed=r['seed'],logprobs=request.get('logprobs',20),
                            min_tokens=r.get('min_new_tokens',0),ignore_eos=r.get('ignore_eos',False),
                            detokenize=False) for r in request['requests']]
                        if not native_fixed:engine.collective_rpc('ep_start_trace')
                        before=counters()
                        start=time.perf_counter()
                        outputs=engine.generate([dict(prompt_token_ids=r['input_ids']) for r in request['requests']],sampling,use_tqdm=False)
                        elapsed=time.perf_counter()-start
                        depths={} if native_fixed else engine.collective_rpc('ep_finish_trace')[0]
                        after=counters()
                        responses=[]
                        for row in outputs:
                            completion=row.outputs[0]
                            selected=([args.loops]*len(completion.token_ids) if native_fixed else depths[row.request_id]['depths'])
                            if len(selected)!=len(completion.token_ids):
                                raise RuntimeError(f'Depth count differs from generated length: {len(selected)} vs {len(completion.token_ids)}')
                            sparse=[]
                            for probabilities in completion.logprobs or []:
                                top=sorted(probabilities.items(),key=lambda pair:pair[1].logprob,reverse=True)
                                if request.get('logprobs',20)!=-1:top=top[:20]
                                sparse.append(dict(token_ids=[int(token) for token,_ in top],
                                                   log_probs=[value.logprob for _,value in top]))
                            metrics=getattr(row,'metrics',None)
                            timing={}
                            if metrics is not None:
                                timing=dict(time_to_first_token_seconds=metrics.first_token_latency,
                                    first_to_last_token_seconds=metrics.last_token_ts-metrics.first_token_ts)
                            responses.append(dict(request_id=row.request_id,token_ids=list(completion.token_ids),
                                exit_depths=selected,finish_reason=completion.finish_reason,
                                gate_logits=[] if native_fixed else depths[row.request_id]['gate_logits'],
                                executed_token_loops=args.loops*(len(row.prompt_token_ids)+len(completion.token_ids)-1),
                                logprobs=sparse,request_timing=timing))
                        result=dict(responses=responses,seconds=elapsed,executed_loops=args.loops,policy=args.policy,
                                    engine_metrics={key:after[key]-before.get(key,0) for key in after},runtime=runtime)
                    else:raise ValueError(f'Unknown operation {op}')
                emit(dict(ok=True,result=result))
            except Exception as exc:
                import traceback
                traceback.print_exc(file=sys.stderr)
                emit(dict(ok=False,error_type=type(exc).__name__,error=str(exc)))
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit(dict(ready=False,error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':main()
