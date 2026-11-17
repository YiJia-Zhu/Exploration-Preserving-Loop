"""Small local JSON-lines actor. All library extensions are repo-local copies.

Run in a separate process with a single physical CUDA_VISIBLE_DEVICES entry.
Stdout is the protocol; engine logs go to stderr. No weight/environment installs.
"""
import argparse
from contextlib import redirect_stdout
import json
import sys
import time

# Register at module import so multiprocessing spawn inherits the same entry.
with redirect_stdout(sys.stderr):
    from ep_loop.cpu_runtime import configure_cpu_threads
    cpu_threads = configure_cpu_threads()
    from vllm import ModelRegistry
    ModelRegistry.register_model('EPLoopOuroForCausalLM','ep_loop.fastpath.vllm_ouro:OuroForCausalLM')


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
    args=parser.parse_args()
    protocol=sys.stdout
    def emit(row):
        protocol.write(json.dumps(row,ensure_ascii=False)+'\n');protocol.flush()
    try:
        with redirect_stdout(sys.stderr):
            from vllm import LLM,SamplingParams
            engine=LLM(model=args.model,trust_remote_code=True,dtype='bfloat16',
                generation_config='vllm',
                tensor_parallel_size=1,seed=42,enforce_eager=args.eager,
                enable_prefix_caching=False,enable_chunked_prefill=False,
                enable_sleep_mode=args.sleep,max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                worker_extension_cls='ep_loop.fastpath.vllm_worker.OuroWorkerExtension',
                hf_overrides=dict(architectures=['EPLoopOuroForCausalLM'],
                    total_ut_steps=args.loops,ep_loop_readout_policy=args.policy,
                    ep_loop_teacher_trace=args.teacher_trace,
                    ep_loop_stop_threshold=args.stop_threshold,ep_loop_native_threshold=args.native_threshold))
        emit(dict(ready=True,policy=args.policy,executed_loops=args.loops,cpu_threads=cpu_threads))
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
                        result=engine.collective_rpc('ep_load_weights',args=(request['path'],request['version']))[0]
                    elif op=='health':
                        result=engine.collective_rpc('ep_health')[0]
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
                            traces=engine.collective_rpc('ep_end_teacher_trace')[0]
                        responses=[]
                        for row in outputs:
                            trace=traces[row.request_id]
                            if trace['input_positions']!=len(row.prompt_token_ids):
                                raise RuntimeError('Teacher prefill trace differs from supplied student text')
                            responses.append(trace)
                        result=dict(responses=responses,seconds=time.perf_counter()-start,executed_loops=args.loops)
                    elif op=='generate':
                        if sleeping:engine.wake_up();sleeping=False
                        sampling=[SamplingParams(temperature=r['temperature'],top_p=r['top_p'],
                            max_tokens=r['max_new_tokens'],seed=r['seed'],logprobs=request.get('logprobs',20),
                            detokenize=False) for r in request['requests']]
                        engine.collective_rpc('ep_start_trace')
                        start=time.perf_counter()
                        outputs=engine.generate([dict(prompt_token_ids=r['input_ids']) for r in request['requests']],sampling,use_tqdm=False)
                        elapsed=time.perf_counter()-start
                        depths=engine.collective_rpc('ep_finish_trace')[0]
                        responses=[]
                        for row in outputs:
                            completion=row.outputs[0]
                            selected=depths[row.request_id]['depths']
                            if len(selected)!=len(completion.token_ids):
                                raise RuntimeError(f'Depth count differs from generated length: {len(selected)} vs {len(completion.token_ids)}')
                            sparse=[]
                            for probabilities in completion.logprobs or []:
                                top=sorted(probabilities.items(),key=lambda pair:pair[1].logprob,reverse=True)[:20]
                                sparse.append(dict(token_ids=[int(token) for token,_ in top],
                                                   log_probs=[value.logprob for _,value in top]))
                            responses.append(dict(request_id=row.request_id,token_ids=list(completion.token_ids),
                                exit_depths=selected,finish_reason=completion.finish_reason,
                                gate_logits=depths[row.request_id]['gate_logits'],
                                executed_token_loops=args.loops*(len(row.prompt_token_ids)+len(completion.token_ids)-1),
                                logprobs=sparse))
                        result=dict(responses=responses,seconds=elapsed,executed_loops=args.loops,policy=args.policy)
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
