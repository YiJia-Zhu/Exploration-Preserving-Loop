"""Isolated native Ouro N2 rollout tuning; never trains or saves checkpoints.

Screen BF16 cache/scheduler settings on matched forced-token workloads, then
confirm on the same 32 x 8 natural-EOS requests with alternative GPU sharding.
Parent runs in the HF environment; each worker uses the isolated vLLM runtime.
"""
import argparse
from array import array
from contextlib import ExitStack
from datetime import datetime
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import sysconfig
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

BASE=dict(memory_fraction=.75,max_num_seqs=None,max_num_batched_tokens=17408)
CANDIDATES=[('mem90',dict(memory_fraction=.90)),
    ('seq16',dict(max_num_seqs=16)),('seq32',dict(max_num_seqs=32)),
    ('mem90_seq8',dict(memory_fraction=.90,max_num_seqs=8)),
    ('mem90_seq16',dict(memory_fraction=.90,max_num_seqs=16)),
    ('mem90_seq32',dict(memory_fraction=.90,max_num_seqs=32)),
    ('mem90_seq16_batch4096',dict(memory_fraction=.90,max_num_seqs=16,max_num_batched_tokens=4096)),
    ('mem90_seq32_batch4096',dict(memory_fraction=.90,max_num_seqs=32,max_num_batched_tokens=4096))]
WORKLOADS=[('short',64,1024,128),('pressure8k',64,8192,128),
           ('pressure16k',32,16384,128),('decode4k',64,4096,512)]

def now():return datetime.now().astimezone().isoformat()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
def partition(requests,count,strategy):
    if count<1 or strategy not in {'question','sample'}:raise ValueError('Invalid inference partition')
    shards=[[] for _ in range(count)]
    for index,request in enumerate(requests):
        owner=request['question_index']%count if strategy=='question' else index%count
        shards[owner].append(request)
    return shards

def confirmation_variants(best):
    """Spend long-rollout time only on a meaningful, non-regressing candidate."""
    ratios=best['speedup_by_workload']
    eligible=(best['long_context_geomean_speedup']>=1.05 and
              min(ratios[key] for key in ('pressure8k','pressure16k','decode4k'))>=.90)
    options=best['options'] if eligible else BASE
    variants=[('real_baseline','question',BASE)]
    if eligible:variants.append(('real_tuned','question',options))
    variants.append(('real_spread','sample',options))
    return variants

def worker(case_path):
    case=json.loads(case_path.read_text());output=Path(case['result'])
    report=dict(case=case,status='initializing',started_at=now(),measurements=[])
    write(output,report)
    toolkit=Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'
    if (toolkit/'bin/nvcc').exists():
        os.environ['CUDA_HOME']=os.environ['CUDA_PATH']=str(toolkit)
        os.environ['PATH']=str(toolkit/'bin')+os.pathsep+os.environ.get('PATH','')
    from ep_loop.cpu_runtime import configure_cpu_threads
    configure_cpu_threads()
    from vllm import LLM,SamplingParams
    import torch
    options=case['options'];extra={}
    if options['max_num_seqs'] is not None:extra['max_num_seqs']=options['max_num_seqs']
    if options.get('kv_cache_dtype'):extra['kv_cache_dtype']=options['kv_cache_dtype']
    started=time.perf_counter()
    engine=LLM(model=case['model'],trust_remote_code=True,dtype='bfloat16',generation_config='vllm',
        tensor_parallel_size=1,seed=42,enforce_eager=False,enable_sleep_mode=True,
        enable_prefix_caching=False,enable_chunked_prefill=True,max_model_len=17408,
        max_num_batched_tokens=options['max_num_batched_tokens'],gpu_memory_utilization=options['memory_fraction'],
        attention_config={'backend':options.get('attention_backend','FLASHINFER')},disable_log_stats=False,max_logprobs=0,
        hf_overrides=dict(architectures=['OuroForCausalLM'],total_ut_steps=2),**extra)
    assert engine.llm_engine.model_config.hf_config.total_ut_steps==2
    report.update(initialization_seconds=time.perf_counter()-started,status='ready',ready_at=now())
    write(output,report)
    def counters():
        values={}
        for item in engine.get_metrics():
            if item.name in {'vllm:num_preemptions','vllm:generation_tokens','vllm:prompt_tokens'}:
                values[item.name]=values.get(item.name,0)+item.value
        return values
    def run(requests,label):
        params=[SamplingParams(temperature=1.,top_p=1.,max_tokens=r['max_new_tokens'],
            min_tokens=r.get('min_new_tokens',0),ignore_eos=r.get('ignore_eos',False),seed=r['seed'],
            logprobs=None,detokenize=False) for r in requests]
        before=counters();start=time.perf_counter()
        rows=engine.generate([dict(prompt_token_ids=r['input_ids']) for r in requests],params,use_tqdm=False)
        seconds=time.perf_counter()-start;after=counters()
        assert len(rows)==len(requests)
        responses=[]
        for response_index,(request,row) in enumerate(zip(requests,rows)):
            tokens=list(row.outputs[0].token_ids);metrics=row.metrics
            if case.get('capture_fidelity') and label=='natural_eos' and response_index<2:
                report.setdefault('fidelity_sequences',[]).append(dict(id=request.get('id'),
                    sample=request.get('sample'),prompt_tokens=len(request['input_ids']),
                    input_ids=request['input_ids']+tokens[:8192]))
            if request.get('ignore_eos'):assert len(tokens)==request['max_new_tokens']
            responses.append(dict(id=request.get('id'),question_index=request.get('question_index'),
                sample=request.get('sample'),tokens=len(tokens),finish_reason=row.outputs[0].finish_reason,
                token_sha256=hashlib.sha256(array('I',tokens).tobytes()).hexdigest(),
                ttft_seconds=metrics.first_token_latency if metrics is not None else None,
                first_to_last_seconds=(metrics.last_token_ts-metrics.first_token_ts) if metrics is not None else None))
        total=sum(r['tokens'] for r in responses)
        return dict(label=label,seconds=seconds,output_tokens=total,tokens_per_second=total/seconds,
            preemptions=after.get('vllm:num_preemptions',0)-before.get('vllm:num_preemptions',0),
            prompt_tokens_computed=after.get('vllm:prompt_tokens',0)-before.get('vllm:prompt_tokens',0),
            responses=responses)
    if case['phase']=='screen':
        for label,count,prefix,length in WORKLOADS:
            requests=[dict(input_ids=[40]*prefix,max_new_tokens=length,min_new_tokens=length,
                ignore_eos=True,seed=42+i) for i in range(count)]
            report.update(status='warming',active_workload=label);write(output,report)
            run([dict(r,max_new_tokens=16,min_new_tokens=16) for r in requests],label+'_warmup')
            for repeat in range(case['repeats']):
                report.update(status='measuring',active_workload=label,repeat=repeat);write(output,report)
                report['measurements'].append(run(requests,label));write(output,report)
    else:
        requests=case['requests']
        run([dict(requests[0],max_new_tokens=16)],'warmup')
        report.update(status='waiting_at_barrier');write(output,report)
        deadline=time.monotonic()+900
        while not Path(case['barrier']).exists():
            if time.monotonic()>deadline:raise TimeoutError('Inference start barrier timed out')
            time.sleep(.1)
        report.update(status='measuring',active_workload='natural_eos',generation_started_at=now());write(output,report)
        report['measurements'].append(run(requests,case.get('workload_label','natural_eos')))
    report.update(status='completed',finished_at=now(),peak_torch_cuda_allocated=torch.cuda.max_memory_allocated())
    write(output,report)
    del engine;gc.collect()

def build_requests(model):
    from transformers import AutoTokenizer
    from ep_loop.config import load_config
    from ep_loop.data import prepare_training_records,sample_rollout_indices
    from ep_loop.reproducibility import sample_seed
    config=load_config(ROOT/'configs/hf_shared_ep_d_loopopd.json')
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    records,identity=prepare_training_records(config,tokenizer,ROOT)
    indices=sample_rollout_indices(random.Random(42),len(records),32)
    requests=[]
    for question,index in enumerate(indices):
        record=records[index]
        for sample in range(8):
            requests.append(dict(id=record['id'],question_index=question,sample=sample,
                input_ids=record['input_ids'],seed=sample_seed(42,record['problem_sha256'],sample),
                max_new_tokens=16384))
    return requests,identity

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--gpus',default='0,4,5,7')
    parser.add_argument('--python-bin',default=str(ROOT/'ep_loop_envs/vllm026/bin/python'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--phase',choices=['screen','confirm'],default='screen')
    args=parser.parse_args()
    if args.worker:
        try:worker(args.worker)
        except BaseException as error:
            case=json.loads(args.worker.read_text());path=Path(case['result'])
            report=json.loads(path.read_text()) if path.exists() else dict(case=case)
            report.update(status='failed',error=repr(error),finished_at=now());write(path,report)
            raise
        return
    if args.output is None:parser.error('--output required')
    devices=args.gpus.split(',')
    if not 1<=len(devices)<=4 or len(set(devices))!=len(devices):parser.error('Use one to four distinct authorized GPU IDs')
    if args.repeats<1:parser.error('Positive repeats required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    from ep_loop.gpu_resources import require_idle_gpus,gpu_status
    from ep_loop.process_control import process_record,terminate_experiment
    model=str((ROOT/'models/Ouro-1.4B-Thinking').resolve())
    children=[]
    def stop(*unused):raise KeyboardInterrupt('Benchmark interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    report=dict(status='admission',phase=args.phase,gpus=devices,model=model,started_at=now(),pid=os.getpid(),
        start_ticks=process_record(os.getpid())['start_ticks'],training=False,checkpoints=False,cases=[])
    summary_path=output/(args.phase+'.json')
    if summary_path.exists():raise FileExistsError(summary_path)
    write(summary_path,report)
    def run_wave(specs):
        children.clear();cases=[];logs=[]
        require_idle_gpus([gpu for _,gpu,_,_ in specs])
        try:
            for name,gpu,options,requests in specs:
                path=output/(name+'.case.json');result=output/(name+'.result.json')
                if result.exists():raise FileExistsError(result)
                case=dict(name=name,gpu=gpu,options=options,model=model,phase=args.phase,
                    repeats=args.repeats,result=str(result))
                if requests is not None:case.update(requests=requests,barrier=str(output/(name.rsplit('_gpu',1)[0]+'.go')),
                    capture_fidelity=name.startswith('real_baseline'))
                write(path,case);cases.append(case)
                env=os.environ|dict(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CPU_THREADS='16',OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',
                    NUMEXPR_NUM_THREADS='16',VLLM_WORKER_MULTIPROC_METHOD='spawn',TOKENIZERS_PARALLELISM='false',
                    PYTHONHASHSEED='42',PYTHONUNBUFFERED='1')
                for key in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF'):env.pop(key,None)
                log=(output/(name+'.engine.log')).open('wb');logs.append(log)
                child=subprocess.Popen([args.python_bin,'-u',str(Path(__file__).resolve()),'--worker',str(path)],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child)
                case['pid']=child.pid
            report['active_cases']=[dict(name=c['name'],gpu=c['gpu'],pid=c['pid']) for c in cases]
            write(summary_path,report)
            started=time.monotonic();last=0
            while any(child.poll() is None for child in children):
                values=[json.loads(Path(c['result']).read_text()) if Path(c['result']).exists() else dict(status='starting') for c in cases]
                for child,value,case in zip(children,values,cases):
                    if child.poll() not in (None,0):raise RuntimeError(f"{case['name']} failed: {value.get('error')}; see engine log")
                if args.phase=='confirm' and all(v['status'] in {'waiting_at_barrier','measuring','completed'} for v in values):
                    barrier=Path(cases[0]['barrier'])
                    if not barrier.exists():barrier.write_text(now()+'\n')
                if time.monotonic()-last>=30:
                    progress=[dict(name=c['name'],gpu=c['gpu'],status=v['status'],
                        workload=v.get('active_workload'),finished_measurements=len(v.get('measurements',[]))) for c,v in zip(cases,values)]
                    report['progress']=progress;write(summary_path,report)
                    print(json.dumps(dict(time=now(),phase=args.phase,elapsed_seconds=time.monotonic()-started,progress=progress)),flush=True)
                    last=time.monotonic()
                if time.monotonic()-started>(2400 if args.phase=='screen' else 7200):raise TimeoutError('Benchmark wave timeout')
                time.sleep(2)
            for child,case in zip(children,cases):
                if child.returncode:raise RuntimeError(f"{case['name']} exited {child.returncode}")
                value=json.loads(Path(case['result']).read_text());assert value['status']=='completed'
                report['cases'].append(dict(name=case['name'],gpu=case['gpu'],result=case['result']))
            write(summary_path,report)
            for attempt in range(30):
                try:require_idle_gpus([c['gpu'] for c in cases]);break
                except RuntimeError:
                    if attempt==29:raise
                    time.sleep(1)
        finally:
            for child in children:
                if child.poll() is None:terminate_experiment(child.pid)
            for log in logs:log.close()
    try:
        with ExitStack() as stack:
            registries=[ROOT.parent/'steer_opd/.runtime',*[ROOT.parent/'steer_opd/exploration'/name/'.runtime' for name in ('readout_sv','residual_sv')]]
            reserved=set();admission_handles=[]
            for registry in registries:
                path=registry/'gpu_admission.lock'
                if path.exists():
                    handle=stack.enter_context(path.open('r'))
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);admission_handles.append(handle)
                path=registry/'gpu_leases.json'
                for lease in json.loads(path.read_text()) if path.exists() else []:
                    current=process_record(lease['pid'])
                    if current and current['state']!='Z' and str(current['start_ticks'])==str(lease['start_ticks']):reserved.update(lease['uuids'])
            admission=require_idle_gpus(devices)
            assert not reserved.intersection(row['uuid'] for row in admission),'A selected GPU is leased'
            for gpu in devices:fcntl.flock(stack.enter_context(open('/tmp/ep-loop-gpu'+gpu+'.lock','a')),fcntl.LOCK_EX|fcntl.LOCK_NB)
            report.update(status='running',admission=admission);write(summary_path,report)
            # Keep per-GPU locks, but do not block admission on unrelated GPUs.
            for handle in admission_handles:fcntl.flock(handle,fcntl.LOCK_UN)
            if args.phase=='screen':
                run_wave([(f'baseline_gpu{gpu}',gpu,BASE,None) for gpu in devices])
                for begin in range(0,len(CANDIDATES),len(devices)):
                    run_wave([(name+'_gpu'+gpu,gpu,BASE|change,None) for (name,change),gpu in zip(CANDIDATES[begin:begin+len(devices)],devices)])
                scores=[]
                baselines={c['gpu']:json.loads(Path(c['result']).read_text()) for c in report['cases'] if c['name'].startswith('baseline_')}
                def means(value):
                    return {name:statistics.mean(r['seconds'] for r in value['measurements'] if r['label']==name) for name,_,_,_ in WORKLOADS}
                for case in report['cases']:
                    if case['name'].startswith('baseline_'):continue
                    value=json.loads(Path(case['result']).read_text());before=means(baselines[case['gpu']]);after=means(value)
                    ratios={key:before[key]/after[key] for key in before}
                    score=math.prod(ratios[key] for key in ('pressure8k','pressure16k','decode4k'))**(1/3)
                    scores.append(dict(name=case['name'],gpu=case['gpu'],options=value['case']['options'],
                        speedup_by_workload=ratios,long_context_geomean_speedup=score))
                scores.sort(key=lambda row:row['long_context_geomean_speedup'],reverse=True)
                report.update(scores=scores,best=scores[0])
            else:
                screen=json.loads((output/'screen.json').read_text());best=screen['best']
                requests,identity=build_requests(model);write(output/'natural_requests.json',dict(requests=requests,data_identity=identity))
                variants=confirmation_variants(best)
                report['confirmation_variants']=[dict(name=name,partition=strategy,options=options) for name,strategy,options in variants]
                report['selection_rule']='Screen gain >=5%, with no long workload regression >10%; otherwise retain original BF16 settings and isolate sample spreading.'
                write(summary_path,report)
                for name,strategy,options in variants:
                    shards=partition(requests,len(devices),strategy)
                    assert sum(map(len,shards))==len(requests)
                    run_wave([(name+'_gpu'+gpu,gpu,options,shard) for gpu,shard in zip(devices,shards)])
            report.update(status='completed',finished_at=now());write(summary_path,report)
    except BaseException as error:
        report.update(status='failed_or_interrupted',error=repr(error),finished_at=now());write(summary_path,report)
        raise

if __name__=='__main__':main()
