"""Standalone Ouro N2 kernel/speculation screening and matched math accuracy.

No training, checkpoint saving, production configuration or installed-package edits.
Run the parent in .venv and workers in ep_loop_envs/vllm026.
"""
import argparse
from array import array
from contextlib import ExitStack
import fcntl
import functools
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_ouro_rollout import now, write

BASE = dict(memory_fraction=.75, block_size=16, kv_cache_dtype='auto')
CANDIDATES = [
    ('cuda_core_decode', dict(cuda_core_decode=True)),
    ('block64', dict(block_size=64)),
    ('ngram3', dict(speculative_config=dict(method='ngram', num_speculative_tokens=3,
        prompt_lookup_min=2, prompt_lookup_max=5, rejection_sample_method='standard'))),
    ('block32', dict(block_size=32)),
    ('ngram5', dict(speculative_config=dict(method='ngram', num_speculative_tokens=5,
        prompt_lookup_min=2, prompt_lookup_max=5, rejection_sample_method='standard'))),
    ('fp8', dict(kv_cache_dtype='fp8_e4m3')),
]


def patch_decode_class(wrapper):
    """Process-local diagnostic: compare CUDA-core decode with default TC decode."""
    if getattr(wrapper, '_ep_cuda_core_patch', False):
        return
    original_init, original_plan = wrapper.__init__, wrapper.plan
    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        kwargs['use_tensor_cores'] = False
        original_init(self, *args, **kwargs)
        assert not self.use_tensor_cores
        if not getattr(wrapper, '_ep_patch_observed', False):
            print(f'EP_LOOP_CUDA_CORE_DECODE_ACTIVE pid={os.getpid()}', file=sys.stderr, flush=True)
            wrapper._ep_patch_observed = True
    @functools.wraps(original_plan)
    def plan(self, *args, **kwargs):
        # vLLM passes -1, whereas FlashInfer CUDA-core plan requires None.
        if not self.use_tensor_cores and kwargs.get('fixed_split_size', -1) == -1:
            kwargs['fixed_split_size'] = None
        return original_plan(self, *args, **kwargs)
    wrapper.__init__, wrapper.plan = initialize, plan
    wrapper._ep_cuda_core_patch = True


# multiprocessing spawn reimports this entry point. Apply only inside the
# explicitly opted-in benchmark processes, including their EngineCore children.
if os.environ.get('EP_LOOP_BENCH_CUDA_CORE') == '1':
    toolkit = Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'
    if (toolkit/'bin/nvcc').exists():
        os.environ['CUDA_HOME'] = os.environ['CUDA_PATH'] = str(toolkit)
        os.environ['PATH'] = str(toolkit/'bin')+os.pathsep+os.environ.get('PATH', '')
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    patch_decode_class(BatchDecodeWithPagedKVCacheWrapper)


def identity(row):
    return row['benchmark'], row['id'], row['sample']


def validate_requests(expected, actual):
    a, b = {identity(r): r for r in expected}, {identity(r): r for r in actual}
    if len(a) != len(expected) or len(b) != len(actual) or a != b:
        raise ValueError('Missing, duplicated, or changed benchmark requests')


def paired_accuracy(reference, candidate, repeats=10000):
    """Cluster bootstrap by question, retaining all repeated samples together."""
    import numpy as np
    a, b = {identity(r): r for r in reference}, {identity(r): r for r in candidate}
    if len(a) != len(reference) or len(b) != len(candidate) or a.keys() != b.keys():
        raise ValueError('Accuracy comparison requires unique matching response IDs')
    if any(a[key]['seed'] != b[key]['seed'] for key in a):
        raise ValueError('Paired accuracy requires identical per-request seeds')
    groups = {}
    for key in a:
        groups.setdefault(key[:2], []).append(float(b[key]['correct'])-float(a[key]['correct']))
    differences = np.array([np.mean(v) for v in groups.values()], dtype=np.float64)
    rng = np.random.default_rng(42)
    bootstrap = differences[rng.integers(len(differences), size=(repeats, len(differences)))].mean(1)
    low, high = np.quantile(bootstrap, [.025, .975])
    return dict(delta_avg=float(differences.mean()), ci95_question_bootstrap=[float(low),float(high)],
        reference_avg=sum(r['correct'] for r in reference)/len(reference),
        candidate_avg=sum(r['correct'] for r in candidate)/len(candidate),
        lost_correct=sum(a[k]['correct'] and not b[k]['correct'] for k in a),
        gained_correct=sum(not a[k]['correct'] and b[k]['correct'] for k in a),
        questions=len(groups), responses=len(a), bootstrap_repeats=repeats,
        claim='Pilot estimate; a small or zero observed difference does not establish accuracy equivalence')


def build_accuracy_requests(model):
    from transformers import AutoTokenizer
    from ep_loop.data import prompt_ids
    from ep_loop.evaluation_data import load_evaluation_records
    from ep_loop.reproducibility import sample_seed
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    requests, sources = [], []
    datasets = [
        ('DAPO_DEV64', ROOT/'data/dapo_ablation/ouro_seed42_trainall_dev64_eval64_prompt1024_aime30_clean/dev.jsonl', 64, 1.),
        ('AIME2024', ROOT/'data/aime/aime2024.jsonl', 30, .7),
    ]
    for benchmark, path, count, top_p in datasets:
        records, metadata = load_evaluation_records(path, count, 'aime' if benchmark=='AIME2024' else None)
        sources.append(metadata | dict(benchmark=benchmark))
        for record in records:
            tokens = prompt_ids(tokenizer, record, True, True)
            assert 0 < len(tokens) <= 1024
            for sample in range(2):
                requests.append(dict(benchmark=benchmark, id=record['id'], sample=sample,
                    input_ids=tokens, seed=sample_seed(42, record['problem_sha256'], sample),
                    temperature=1., top_p=top_p, max_new_tokens=16384,
                    answer=record['answer'], messages=record['messages']))
    assert len(requests) == 188
    validate_requests(requests, requests)
    return requests, sources


def screen_workloads(previous):
    original = json.loads((previous/'natural_requests.json').read_text())['requests'][:64]
    sequences = []
    for path in sorted(previous.glob('real_baseline_gpu*.result.json')):
        for row in json.loads(path.read_text())['fidelity_sequences']:
            if len(row['input_ids']) > 8192:
                sequences.append(row)
    assert sequences
    long = [dict(id=f'prefix{i}', sample=0, benchmark='screen',
        input_ids=sequences[i % len(sequences)]['input_ids'][:8192], seed=42+i,
        temperature=1., top_p=1., max_new_tokens=128, min_new_tokens=128, ignore_eos=True)
        for i in range(16)]
    growth = [dict(r, benchmark='screen', temperature=1., top_p=1.,
        max_new_tokens=2048, min_new_tokens=2048, ignore_eos=True) for r in original]
    return [('real_prefix8k', long), ('real_growth2k', growth)]


def worker(path):
    case = json.loads(path.read_text())
    output = Path(case['result'])
    report = dict(status='initializing', case=case, started_at=now(), measurements=[])
    write(output, report)
    toolkit = Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'
    if (toolkit/'bin/nvcc').exists():
        os.environ['CUDA_HOME'] = os.environ['CUDA_PATH'] = str(toolkit)
        os.environ['PATH'] = str(toolkit/'bin')+os.pathsep+os.environ.get('PATH', '')
    from ep_loop.cpu_runtime import configure_cpu_threads
    configure_cpu_threads()
    from vllm import LLM, SamplingParams
    options = case['options']
    extra = {key: options[key] for key in ('speculative_config','calculate_kv_scales') if key in options}
    start = time.perf_counter()
    engine = LLM(model=case['model'], trust_remote_code=True, dtype='bfloat16',
        generation_config='vllm', seed=42, tensor_parallel_size=1,
        max_model_len=17408, max_num_batched_tokens=17408,
        gpu_memory_utilization=options['memory_fraction'], block_size=options['block_size'],
        kv_cache_dtype=options['kv_cache_dtype'], enable_prefix_caching=False,
        enable_chunked_prefill=True, enable_sleep_mode=True,
        max_logprobs=0, disable_log_stats=False, attention_config={'backend':'FLASHINFER'},
        hf_overrides=dict(architectures=['OuroForCausalLM'], total_ut_steps=2), **extra)
    assert engine.llm_engine.model_config.hf_config.total_ut_steps == 2
    report.update(initialization_seconds=time.perf_counter()-start, status='ready')
    write(output, report)
    def counters():
        values = {}
        for metric in engine.get_metrics():
            if isinstance(getattr(metric, 'value', None), (int, float)):
                values[metric.name] = values.get(metric.name, 0)+metric.value
        return values
    def run(requests, label, save_answers=False):
        parameters = [SamplingParams(temperature=r['temperature'], top_p=r['top_p'],
            seed=r['seed'], max_tokens=r['max_new_tokens'], min_tokens=r.get('min_new_tokens',0),
            ignore_eos=r.get('ignore_eos',False), logprobs=None, detokenize=False) for r in requests]
        before = counters()
        started = time.perf_counter()
        outputs = engine.generate([dict(prompt_token_ids=r['input_ids']) for r in requests],
                                  parameters, use_tqdm=False)
        seconds = time.perf_counter()-started
        after = counters()
        assert len(outputs) == len(requests)
        responses = []
        if save_answers:
            from transformers import AutoTokenizer
            from ep_loop.inference import verify_integer_response
            tokenizer = AutoTokenizer.from_pretrained(case['model'], local_files_only=True)
        for request, result in zip(requests, outputs):
            assert result.prompt_token_ids == request['input_ids']
            tokens = list(result.outputs[0].token_ids)
            if request.get('ignore_eos'):
                assert len(tokens) == request['max_new_tokens']
            row = dict(id=request['id'], sample=request['sample'], benchmark=request['benchmark'],
                seed=request['seed'], tokens=len(tokens), finish_reason=result.outputs[0].finish_reason,
                token_sha256=hashlib.sha256(array('I',tokens).tobytes()).hexdigest())
            if save_answers:
                text = tokenizer.decode(tokens)
                row.update(text=text, **verify_integer_response(text, request['answer'], request['messages']))
            responses.append(row)
        if save_answers:
            answers = output.with_suffix('.responses.jsonl')
            with answers.open('x') as stream:
                for row in responses:
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        total = sum(row['tokens'] for row in responses)
        metrics = {key:after[key]-before.get(key,0) for key in after
                   if key in {'vllm:num_preemptions','vllm:prompt_tokens','vllm:generation_tokens'} or 'spec_decode' in key}
        return dict(label=label, seconds=seconds, output_tokens=total,
                    tokens_per_second=total/seconds, engine_metrics=metrics,
                    responses=[{k:v for k,v in row.items() if k!='text'} for row in responses])
    if case['phase'] == 'screen':
        for label, requests in case['workloads']:
            report.update(status='warming', active_workload=label)
            write(output, report)
            run([dict(r,max_new_tokens=16,min_new_tokens=16) for r in requests], label+'_warmup')
            for repeat in range(2):
                report.update(status='measuring', active_workload=label, repeat=repeat)
                write(output, report)
                report['measurements'].append(run(requests,label))
                write(output, report)
    else:
        requests = case['requests']
        run([dict(requests[0],max_new_tokens=16)], 'warmup')
        report.update(status='waiting_at_barrier')
        write(output, report)
        deadline = time.monotonic()+900
        while not Path(case['barrier']).exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Accuracy start barrier timed out')
            time.sleep(.1)
        report.update(status='measuring',generation_started_at=now(),active_workload='accuracy')
        write(output, report)
        report['measurements'].append(run(requests,'accuracy',True))
    report.update(status='completed', finished_at=now())
    write(output, report)
    del engine
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', default='0,4,5')
    parser.add_argument('--phase', choices=['screen','accuracy'], default='screen')
    parser.add_argument('--variants', type=Path, help='Explicit named option dictionaries for accuracy')
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except BaseException as error:
            case = json.loads(args.worker.read_text())
            path = Path(case['result'])
            report = json.loads(path.read_text()) if path.exists() else dict(case=case)
            report.update(status='failed',error=repr(error),finished_at=now())
            write(path, report)
            raise
        return
    if args.output is None:
        parser.error('--output required')
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or not set(gpus) <= {'0','4','5','7'}:
        parser.error('Only the user-authorized GPUs 0,4,5,7 may be used')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    summary_path = output/(args.phase+'.json')
    if summary_path.exists():
        raise FileExistsError(summary_path)
    from ep_loop.gpu_resources import require_idle_gpus, gpu_status
    from ep_loop.process_control import process_record, terminate_experiment, experiment_processes
    model = str((ROOT/'models/Ouro-1.4B-Thinking').resolve())
    report = dict(status='admission',phase=args.phase,pid=os.getpid(),
        start_ticks=process_record(os.getpid())['start_ticks'],started_at=now(),gpus=gpus,cases=[],
        training=False,checkpoints=False,production_defaults_changed=False)
    write(summary_path,report)
    children = []
    def stop(*_):
        raise KeyboardInterrupt('Benchmark interrupted')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    def wave(specifications, allow_failure=False):
        children.clear()
        require_idle_gpus([spec[1] for spec in specifications])
        cases = []
        try:
            for name,gpu,options,data in specifications:
                path = output/(name+'.case.json')
                result = output/(name+'.result.json')
                if result.exists():
                    raise FileExistsError(result)
                case = dict(name=name,gpu=gpu,model=model,phase=args.phase,options=options,result=str(result))
                case.update(dict(workloads=data) if args.phase=='screen' else
                            dict(requests=data,barrier=str(output/(name.rsplit('_gpu',1)[0]+'.go'))))
                write(path,case)
                env = os.environ | dict(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CPU_THREADS='16',OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',
                    OPENBLAS_NUM_THREADS='16',NUMEXPR_NUM_THREADS='16',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                    TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='42',PYTHONUNBUFFERED='1',
                    EP_LOOP_BENCH_CUDA_CORE='1' if options.get('cuda_core_decode') else '0')
                for key in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF'):
                    env.pop(key,None)
                with (output/(name+'.engine.log')).open('wb') as log:
                    child = subprocess.Popen([str(ROOT/'ep_loop_envs/vllm026/bin/python'),'-u',
                        str(Path(__file__).resolve()),'--worker',str(path)],cwd=ROOT,env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child)
                cases.append(case | process_record(child.pid))
            report['active_cases']=[{key:case[key] for key in ('name','gpu','pid','start_ticks')} for case in cases]
            write(summary_path,report)
            start, last = time.monotonic(), 0
            while any(child.poll() is None for child in children):
                values=[json.loads(Path(case['result']).read_text()) if Path(case['result']).exists()
                        else dict(status='starting') for case in cases]
                failed=[case['name'] for case,child in zip(cases,children) if child.poll() not in (None,0)]
                if failed and not allow_failure:
                    raise RuntimeError(f'Worker failed: {failed}')
                if args.phase=='accuracy' and all(value['status'] in {'waiting_at_barrier','measuring','completed'} for value in values):
                    barrier=Path(cases[0]['barrier'])
                    if not barrier.exists():
                        barrier.write_text(now()+'\n')
                if time.monotonic()-last >= 30:
                    owned={row['pid'] for row in experiment_processes(os.getpid())}
                    foreign=[row for row in gpu_status() if row['index'] in gpus
                             and any(p['pid'] not in owned for p in row['processes'])]
                    if foreign:
                        raise RuntimeError(f'Unrelated GPU processes detected; stop only this experiment: {foreign}')
                    report['progress']=[dict(name=case['name'],status=value['status'],
                        workload=value.get('active_workload'),measurements=len(value.get('measurements',[])))
                        for case,value in zip(cases,values)]
                    write(summary_path,report)
                    print(json.dumps(dict(time=now(),progress=report['progress'])),flush=True)
                    last=time.monotonic()
                if time.monotonic()-start > (2400 if args.phase=='screen' else 7200):
                    raise TimeoutError('Benchmark wave timeout')
                time.sleep(2)
            for child,case in zip(children,cases):
                value=json.loads(Path(case['result']).read_text())
                if child.returncode and not allow_failure:
                    raise RuntimeError(case['name'])
                report['cases'].append(dict(name=case['name'],gpu=case['gpu'],
                    options=case['options'],status=value['status'],result=case['result']))
            write(summary_path,report)
            for attempt in range(30):
                try:
                    require_idle_gpus([case['gpu'] for case in cases])
                    break
                except RuntimeError:
                    if attempt==29:
                        raise
                    time.sleep(1)
        finally:
            for child in children:
                if child.poll() is None:
                    terminate_experiment(child.pid)
    try:
        with ExitStack() as stack:
            admission_handles=[]
            reserved=set()
            for registry in [ROOT.parent/'steer_opd/.runtime',
                *[ROOT.parent/'steer_opd/exploration'/name/'.runtime' for name in ('readout_sv','residual_sv')]]:
                lock=registry/'gpu_admission.lock'
                if lock.exists():
                    handle=stack.enter_context(lock.open('r'))
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    admission_handles.append(handle)
                leases=registry/'gpu_leases.json'
                for lease in json.loads(leases.read_text()) if leases.exists() else []:
                    current=process_record(lease['pid'])
                    if current and current['state']!='Z' and str(current['start_ticks'])==str(lease['start_ticks']):
                        reserved.update(lease['uuids'])
            admission=require_idle_gpus(gpus)
            if reserved.intersection(row['uuid'] for row in admission):
                raise RuntimeError('An authorized GPU is currently leased')
            for gpu in gpus:
                handle=stack.enter_context(open('/tmp/ep-loop-gpu'+gpu+'.lock','a'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for handle in admission_handles:
                fcntl.flock(handle,fcntl.LOCK_UN)
            report.update(status='running',admission=admission)
            write(summary_path,report)
            if args.phase=='screen':
                workloads=screen_workloads(ROOT/'audit/ouro_rollout_optimization_20261119_133009')
                wave([(f'baseline_gpu{gpu}',gpu,BASE,workloads) for gpu in gpus])
                for begin in range(0,len(CANDIDATES),len(gpus)):
                    wave([(name+'_gpu'+gpu,gpu,BASE|changes,workloads)
                          for (name,changes),gpu in zip(CANDIDATES[begin:begin+len(gpus)],gpus)], True)
                baselines={case['gpu']:json.loads(Path(case['result']).read_text())
                    for case in report['cases'] if case['name'].startswith('baseline_')}
                def means(value):
                    return {label:statistics.mean(m['seconds'] for m in value['measurements'] if m['label']==label)
                            for label,_ in workloads}
                scores=[]
                for case in report['cases']:
                    if case['name'].startswith('baseline_') or case['status']!='completed':
                        continue
                    reference=means(baselines[case['gpu']])
                    candidate=means(json.loads(Path(case['result']).read_text()))
                    ratios={name:reference[name]/candidate[name] for name in reference}
                    scores.append(case | dict(speedup_by_workload=ratios,
                                             geomean_speedup=math.prod(ratios.values())**(1/len(ratios))))
                report['scores']=sorted(scores,key=lambda row:row['geomean_speedup'],reverse=True)
            else:
                if args.variants is None:
                    raise ValueError('--variants is required for accuracy')
                variants=json.loads(args.variants.read_text())
                requests,sources=build_accuracy_requests(model)
                write(output/'accuracy_requests.json',dict(requests=requests,sources=sources))
                report['variants']=variants
                write(summary_path,report)
                for variant in variants:
                    shards=[requests[index::len(gpus)] for index in range(len(gpus))]
                    validate_requests(requests,[row for shard in shards for row in shard])
                    wave([(variant['name']+'_gpu'+gpu,gpu,BASE|variant['options'],shard)
                          for gpu,shard in zip(gpus,shards)])
            report.update(status='completed',finished_at=now())
            write(summary_path,report)
    except BaseException as error:
        report.update(status='failed_or_interrupted',error=repr(error),finished_at=now())
        write(summary_path,report)
        raise


if __name__=='__main__':
    main()
