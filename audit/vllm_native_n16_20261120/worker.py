"""Measure official vLLM native n=16 versus separate child requests."""
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import sys
import time

# Sets the CUDA toolkit and CPU threads the same way as the evaluation actor.
from ep_loop.fastpath import vllm_server  # noqa: F401
from vllm import LLM, SamplingParams


def main():
    folder=Path(__file__).resolve().parent
    spec=json.loads((folder/'workload.json').read_text())
    use_cache=bool(int(sys.argv[1]))
    n=spec['samples_per_problem']
    common=dict(temperature=1.0,top_p=.7,max_tokens=spec['max_new_tokens'],
                ignore_eos=True,detokenize=False)
    jobs=spec['prompts']
    prompts=[dict(prompt_token_ids=r['input_ids']) for r in jobs]
    children=[prompt for prompt in prompts for _ in range(n)]
    native=[SamplingParams(n=n,seed=r['seed'],**common) for r in jobs]
    manual=[SamplingParams(n=1,seed=r['seed']+i,**common) for r in jobs for i in range(n)]
    llm=LLM(model=spec['checkpoint'],trust_remote_code=True,dtype='bfloat16',
            tensor_parallel_size=1,seed=42,generation_config='vllm',
            max_model_len=17408,max_num_batched_tokens=17408,
            gpu_memory_utilization=.75,enable_chunked_prefill=True,
            attention_config={'backend':'FLASHINFER'},enable_prefix_caching=use_cache,
            disable_log_stats=False,hf_overrides=dict(architectures=['OuroForCausalLM'],total_ut_steps=2))
    def counters():
        vals=Counter()
        for m in llm.get_metrics():
            if m.name in {'vllm:prefix_cache_queries','vllm:prefix_cache_hits','vllm:prompt_tokens'}:
                vals[m.name]+=m.value
        return vals
    def generate(grouped):
        before=counters()
        start=time.perf_counter()
        outputs=llm.generate(prompts if grouped else children,native if grouped else manual,use_tqdm=False)
        wall=time.perf_counter()-start
        tokens=[]
        for output in outputs:
            assert output.finished
            assert len(output.outputs)==(n if grouped else 1)
            tokens.extend(list(o.token_ids) for o in sorted(output.outputs,key=lambda o:o.index))
        assert len(tokens)==len(jobs)*n and all(len(t)==spec['max_new_tokens'] for t in tokens)
        return tokens,dict(method='native_n16' if grouped else 'manual',seconds=wall,
             output_tokens=sum(map(len,tokens)),cache_counters=dict(counters()-before))
    # Warm both API paths and compiled kernels, clear prefix blocks between timed calls.
    for grouped in (True,False):
        if use_cache:
            assert llm.reset_prefix_cache()
        generate(grouped)
    recorded=dict(native_n16=[],manual=[])
    timings=[]
    for grouped in (True,False,False,True,True,False):
        if use_cache:
            assert llm.reset_prefix_cache()
        tokens,run=generate(grouped)
        recorded[run['method']].append(tokens)
        timings.append(run)
        print(json.dumps(run),flush=True)
    native_rows,manual_rows=recorded['native_n16'][0],recorded['manual'][0]
    report=dict(cache=use_cache,vllm=version('vllm'),runs=timings,
        native_api_requests=len(jobs),manual_api_requests=len(jobs)*n,
        identical_output_sequences=sum(a==b for a,b in zip(native_rows,manual_rows)),
        pair_size=len(native_rows),
        repeats_identical={name:all(rows[0]==other for other in rows) for name,rows in recorded.items()},
        median_seconds={name:statistics.median(x['seconds'] for x in timings if x['method']==name)
                        for name in recorded})
    (folder/f'cache_{int(use_cache)}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
