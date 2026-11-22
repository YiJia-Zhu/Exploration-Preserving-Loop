"""Compare TP/DP arrangements on the same GPUs, prefixes and forced lengths.

This is an inference microbenchmark, not a full-rollout or training speed claim.
All requests per engine are submitted in one call; max_num_seqs is omitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',required=True,help='Use a PCIe pair, for example 2,3')
    parser.add_argument('--python-bin',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int)
    parser.add_argument('--new-tokens',type=int)
    parser.add_argument('--model',default='models/Ouro-1.4B-Thinking')
    parser.add_argument('--manifest',type=Path,help='Use the fixed workload from benchmark_runtime.py')
    args=parser.parse_args()
    from ep_loop.fastpath.actor import VllmActor
    from ep_loop.fastpath.parallel import gpu_groups
    from ep_loop.gpu_resources import require_idle_gpus
    devices=args.gpus.split(',')
    if len(devices)!=2:parser.error('This matched comparison uses exactly two GPUs')
    work = json.loads(args.manifest.read_text()) if args.manifest else None
    if work is not None and (not work['ignore_eos'] or work['smoke'] or work['loops'] != 2
            or len(work['jobs']) != 8 or work['profile'] not in ('short','8192','14336')):
        parser.error('Use a full fixed-work N2 eight-question manifest')
    if work and args.new_tokens is not None:
        parser.error('The manifest already defines the number of output tokens per request')
    args.repeats = args.repeats if args.repeats is not None else (work['repeats'] if work else 2)
    args.new_tokens = args.new_tokens if args.new_tokens is not None else (work['max_new_tokens'] if work else 128)
    if args.repeats<1 or args.new_tokens<2:parser.error('Positive repeats and at least two output tokens required')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    require_idle_gpus(devices)
    report=dict(status='initializing',gpus=devices,started_at=datetime.now().astimezone().isoformat(),
        model=work['model'] if work else str((ROOT/args.model).resolve()),seed=42,max_num_seqs=None,cases=[],
        manifest=str(args.manifest.resolve()) if work else None,
        scope='Matched inference batches, fixed N2 BF16/FlashInfer with prefill and forced output length.')
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    workloads=[(8,work['profile'])] if work else [(32,1024),(32,8192),(16,16384)]
    try:
        for tp in [1,2]:
            groups=gpu_groups(devices,tp)
            with ExitStack() as stack,ThreadPoolExecutor(max_workers=len(groups)) as pool:
                def start(gpu):
                    return VllmActor(report['model'],gpu,args.output.with_suffix(f'.tp{tp}.gpu{gpu}.engine.log'),
                        loops=2,policy='fixed',tensor_parallel_size=tp,max_model_len=17408,
                        memory_fraction=.75,sleep=True,python_bin=args.python_bin,attention_backend='FLASHINFER')
                actors=[stack.enter_context(future.result()) for future in [pool.submit(start,gpu) for gpu in groups]]
                for total,prefix in workloads:
                    if work:
                        requests=[dict(input_ids=job['input_ids'],temperature=work['temperature'],top_p=work['top_p'],
                            seed=job['seed'],max_new_tokens=args.new_tokens,ignore_eos=True) for job in work['jobs']]
                    else:
                        requests=[dict(input_ids=[40]*prefix,temperature=1.,top_p=1.,seed=42+i,
                            max_new_tokens=args.new_tokens,min_new_tokens=args.new_tokens,ignore_eos=True)
                            for i in range(total)]
                    def run(rows):
                        begin=time.perf_counter()
                        futures=[pool.submit(actor.generate,rows[index::len(actors)],logprobs=None)
                                 for index,actor in enumerate(actors)]
                        results=[future.result() for future in futures]
                        wall=time.perf_counter()-begin
                        responses=[r for result in results for r in result['responses']]
                        assert len(responses)==total
                        assert all(len(r['token_ids'])==rows[0]['max_new_tokens'] for r in responses)
                        preemptions=sum(result.get('engine_metrics',{}).get('vllm:num_preemptions',0) for result in results)
                        if work and preemptions:
                            raise RuntimeError('Matched benchmark preempted sequences')
                        latency=[r['request_timing']['first_to_last_token_seconds'] for r in responses]
                        return dict(wall_seconds=wall,output_tokens=sum(len(r['token_ids']) for r in responses),
                            output_tokens_per_second=sum(len(r['token_ids']) for r in responses)/wall,
                            mean_first_to_last_token_seconds=statistics.mean(latency),
                            mean_tpot_ms=statistics.mean(latency)/(rows[0]['max_new_tokens']-1)*1000,
                            preemptions=preemptions,engine_seconds=[result['seconds'] for result in results])
                    warm=run([dict(row,max_new_tokens=16,min_new_tokens=0 if work else 16) for row in requests])
                    measurements=[run(requests) for _ in range(args.repeats)]
                    row=dict(tp=tp,dp=len(actors),total_requests=total,
                        prefix_tokens=[len(req['input_ids']) for req in requests] if work else prefix,
                        forced_new_tokens=args.new_tokens,warmup=warm,measurements=measurements,
                        mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in measurements),
                        mean_output_tokens_per_second=statistics.mean(r['output_tokens_per_second'] for r in measurements),
                        median_wall_seconds=statistics.median(r['wall_seconds'] for r in measurements),
                        median_output_tokens_per_second=statistics.median(r['output_tokens_per_second'] for r in measurements),
                        total_preemptions=sum(r['preemptions'] for r in measurements))
                    report['cases'].append(row);report['status']='measuring';save()
                    print(json.dumps(row),flush=True)
                for actor in actors:actor.request('sleep')
        comparisons=[]
        for total,prefix in workloads:
            rows=[r for r in report['cases'] if r['total_requests']==total and (work or r['prefix_tokens']==prefix)]
            first,second=rows
            metric = 'median_wall_seconds' if work else 'mean_wall_seconds'
            comparisons.append(dict(total_requests=total,prefix_tokens=first['prefix_tokens'],
                tp1_dp2_seconds=first[metric],tp2_dp1_seconds=second[metric],
                speedup=first[metric]/second[metric]))
        report.update(status='completed',comparisons=comparisons,finished_at=datetime.now().astimezone().isoformat())
        save();print(json.dumps(comparisons,indent=2),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
