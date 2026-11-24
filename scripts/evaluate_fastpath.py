"""Sharded Ouro evaluation with stable sample seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from ep_loop.cpu_runtime import configure_cpu_threads
    cpu_threads = configure_cpu_threads()
    from ep_loop.config import DEFAULT_CONFIG, load_config, validate_config, student_budget
    from ep_loop.execution import execution_mode, VLLM_EVAL_MODES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(DEFAULT_CONFIG))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--gpus',default='2,3')
    p.add_argument('--backend',choices=['vllm','vllm-rlt'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples-per-problem',type=int)
    p.add_argument('--loops',action='append',type=int)
    p.add_argument('--threshold',action='append',type=float)
    p.add_argument('--phase-zero',action='store_true')
    p.add_argument('--max-new-tokens',type=int)
    p.add_argument('--write-token-ids',action='store_true')
    p.add_argument('--eager',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='HF modes only')
    p.add_argument('--smoke',action='store_true',help='CPU-only HF semantic check')
    args=p.parse_args()
    config=validate_config(load_config(args.config))
    backend=args.backend or config['evaluation'].get('inference_backend','vllm')
    if backend not in {'vllm','vllm-rlt'}:p.error(f'Unknown evaluation backend: {backend}')
    devices=args.gpus.split(',')
    if len(set(devices))!=len(devices) or any(not gpu.isdecimal() for gpu in devices):raise ValueError('Distinct numeric GPU IDs required')
    if execution_mode(config) not in VLLM_EVAL_MODES:
        if backend=='vllm-rlt':p.error('vLLM-RLT requires fixed-depth Ouro evaluation')
        command=[sys.executable,'-u',str(Path(__file__).with_name('evaluate_stage_distillation.py')),
                 '--config',args.config,'--data',str(args.data),'--output',str(args.output),'--device',args.device]
        if args.checkpoint is not None:command+=['--checkpoint',str(args.checkpoint)]
        if args.samples_per_problem is not None:command+=['--samples-per-problem',str(args.samples_per_problem)]
        for depth in args.loops or []:command+=['--loops',str(depth)]
        for threshold in args.threshold or []:command+=['--threshold',str(threshold)]
        if args.phase_zero:command+=['--phase-zero']
        if args.max_new_tokens is not None:command+=['--max-new-tokens',str(args.max_new_tokens)]
        if args.write_token_ids:command+=['--write-token-ids']
        if args.smoke:command+=['--smoke']
        os.execve(sys.executable,command,{**os.environ,'CUDA_VISIBLE_DEVICES':devices[0]})
        return
    if args.phase_zero or args.threshold or args.max_new_tokens is not None or args.write_token_ids:
        p.error('RAD-OPD evaluation options require the HF evaluator')
    if args.smoke or args.device!='cuda':p.error('vLLM evaluation requires CUDA; use HF training modes for CPU smoke checks')
    from transformers import AutoTokenizer
    from ep_loop.data import prompt_ids
    from ep_loop.inference import verify_integer_response,math_metrics
    from ep_loop.observability import RunLogger
    from ep_loop.reproducibility import sample_seed
    from ep_loop.fastpath.actor import VllmActor
    from ep_loop.fastpath.parallel import gpu_groups
    tensor_parallel_size=config['implementation'].get('vllm_tensor_parallel_size',1)
    if backend=='vllm-rlt' and tensor_parallel_size!=1:
        p.error('vLLM-RLT evaluation uses one replica per GPU; set evaluation config TP to 1')
    inference_groups=gpu_groups(devices,tensor_parallel_size)
    from ep_loop.evaluation_data import load_evaluation_records, benchmark_math_metrics
    if args.output.exists():raise FileExistsError(args.output)
    settings=dict(config['evaluation'])
    if backend=='vllm-rlt':
        for key in ('rlt_max_num_seqs','rlt_max_num_batched_tokens','rlt_prefill_chunk_size'):
            if key in settings and (type(settings[key]) is not int or settings[key] < 1):
                p.error(f'evaluation.{key} must be a positive integer')
        for key in ('rlt_async_scheduling','rlt_prefix_caching'):
            if type(settings.get(key,False)) is not bool:
                p.error(f'evaluation.{key} must be a boolean')
    if config['model']['family']!='ouro':raise ValueError('Ouro-only opt-in fastpath')
    if not settings.get('use_cache',True):raise ValueError('Use the original HF evaluator for recompute-only cache diagnostics')
    policy=settings.get('student_policy',config['algorithm'].get('student_policy','fixed'))
    if policy=='learned' and args.loops:raise ValueError('Learned students must retain their own stopping head')
    if args.checkpoint is None and policy not in {'native','native_adaptive'}:raise ValueError('Trained students require --checkpoint')
    samples=args.samples_per_problem if args.samples_per_problem is not None else settings['samples_per_problem']
    if samples<1:raise ValueError('samples-per-problem must be positive')
    settings['samples_per_problem']=samples
    records,data_metadata=load_evaluation_records(args.data,settings.get('expected_examples'),settings.get('suite'))
    if not records or len({row['id'] for row in records})!=len(records):raise ValueError('Unique nonempty evaluation records required')
    directory=(args.checkpoint or ROOT/config['model']['path']).resolve()
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    encoded=[prompt_ids(tokenizer,record,config['model']['enable_thinking'],
        config['model']['repair_template_newlines']) for record in records]
    if any(len(ids)>config['data']['max_prompt_tokens'] for ids in encoded):
        raise ValueError('Evaluation prompt exceeds max_prompt_tokens; preserve the full problem and adjust the limit explicitly')
    jobs=[dict(index=index,record=record,sample=sample,input_ids=encoded[index])
        for index,record in enumerate(records) for sample in range(samples)]
    modes=[('fixed',depth) for depth in (args.loops or settings['student_loops'])] if args.loops or policy=='fixed' else [(policy,student_budget(config))]
    if backend=='vllm-rlt' and any(mode!='fixed' for mode,_ in modes):
        p.error('vLLM-RLT evaluation requires fixed depths')
    settings['inference_backend']=backend
    args.output.parent.mkdir(parents=True,exist_ok=True)
    questions_path=args.output.with_suffix('.questions.jsonl')
    questions=[dict(index=index,id=record['id'],problem_sha256=record['problem_sha256'],
        benchmark=record.get('benchmark','unspecified'),
        problem=record.get('problem','\n'.join(m['content'] for m in record['messages'] if m['role']=='user')),
        messages=record['messages'],reference_answer=record['answer'],prompt_tokens=len(encoded[index]))
        for index,record in enumerate(records)]
    with questions_path.open('x') as question_file:
        for row in questions:question_file.write(json.dumps(row,ensure_ascii=False)+'\n')
    all_results=[];summaries={};begin=time.perf_counter()
    with RunLogger(args.output.with_suffix('.events.jsonl'),args.output.with_suffix('.console.log'),'fastpath_evaluation') as observer:
        with args.output.open('x') as output,observer.stage('evaluation_run',echo=True,
                gpus=devices,data_sha256=data_metadata['sha256'],settings=settings):
            for endpoint,loops in modes:
                messages=queue.Queue();rows=[]
                def shard(gpu,subset):
                    try:
                        if not subset:return
                        log=args.output.with_suffix(f'.gpu{gpu}.engine.log')
                        max_model_len=config['data']['max_prompt_tokens']+settings['max_new_tokens']
                        requests=[dict(input_ids=j['input_ids'],temperature=settings['temperature'],
                            top_p=settings['top_p'],max_new_tokens=settings['max_new_tokens'],
                            seed=sample_seed(config['seed'],j['record']['problem_sha256'],j['sample'])) for j in subset]
                        if backend=='vllm-rlt':
                            from ep_loop.fastpath.rlt_actor import RltActor
                            actor=RltActor(directory,gpu,log,loops=loops,max_model_len=max_model_len,
                                           eager=args.eager,max_num_seqs=settings.get('rlt_max_num_seqs'),
                                           max_num_batched_tokens=settings.get('rlt_max_num_batched_tokens'),
                                           prefill_chunk_size=settings.get('rlt_prefill_chunk_size'),
                                           attention_backend=settings.get('rlt_attention_backend','triton'),
                                           async_scheduling=settings.get('rlt_async_scheduling',False),
                                           prefix_caching=settings.get('rlt_prefix_caching',False))
                            cache_policy='rlt_last_exited'
                        else:
                            actor=VllmActor(directory,gpu,log,loops=loops,
                                policy=endpoint,stop_threshold=config['algorithm'].get('student_stop_threshold',.5),
                                native_threshold=config['algorithm']['exit_threshold'],
                                max_model_len=max_model_len,memory_fraction=.75,eager=args.eager,rollout_ngram=False,
                                python_bin=config['implementation'].get('vllm_python_bin'),
                                tensor_parallel_size=tensor_parallel_size,
                                attention_backend=config['implementation'].get('vllm_attention_backend'),
                                **({'max_num_seqs':config['implementation']['vllm_max_num_seqs']}
                                   if config['implementation'].get('vllm_max_num_seqs') else {}))
                            cache_policy='vllm_fixed_depth' if endpoint=='fixed' else 'vllm_fixed_k_gate_readout'
                        with actor:
                            if backend=='vllm-rlt':
                                responses=actor.iter_responses(requests)
                            else:
                                result=actor.generate(requests,logprobs=None)
                                if len(result['responses'])!=len(subset):
                                    raise RuntimeError('vLLM returned a different number of evaluation responses')
                                responses=((i,response,result['seconds']) for i,response in enumerate(result['responses']))
                            for i,response,seconds in responses:
                                job=subset[i]
                                record=job['record'];text=tokenizer.decode(response['token_ids'])
                                row=dict(index=job['index'],id=record['id'],problem_sha256=record['problem_sha256'],
                                    benchmark=record.get('benchmark','unspecified'),
                                    problem=questions[job['index']]['problem'],messages=record['messages'],
                                    reference_answer=record['answer'],
                                    policy=endpoint,max_loops=loops,sample=job['sample'],
                                    sampling_seed=sample_seed(config['seed'],record['problem_sha256'],job['sample']),
                                    text=text,response_tokens=len(response['token_ids']),reached_eos=response['finish_reason']=='stop',
                                    actual_student_exit_depths=response['exit_depths'],actual_execution_depths=[loops]*len(response['token_ids']),
                                    executed_token_loops=response['executed_token_loops'],backfill_token_loops=0,coda_token_positions=0,
                                    cache_policy=cache_policy,gpu=gpu,
                                    **({'request_elapsed_seconds':seconds} if backend=='vllm-rlt' else {'batch_wall_seconds':seconds}),
                                    **verify_integer_response(text,record['answer'],record['messages']))
                                messages.put(('row',row))
                    except BaseException as error:messages.put(('error',error))
                    finally:messages.put(('done',gpu))
                with ThreadPoolExecutor(max_workers=len(inference_groups)) as pool:
                    futures=[pool.submit(shard,gpu,jobs[i::len(inference_groups)]) for i,gpu in enumerate(inference_groups)]
                    done=0;bar=observer.progress(total=len(jobs),desc=f'Evaluate {endpoint} K={loops}',unit='response')
                    while done<len(inference_groups):
                        kind,value=messages.get()
                        if kind=='error':raise value
                        if kind=='done':done+=1;continue
                        rows.append(value);output.write(json.dumps(value,ensure_ascii=False)+'\n');output.flush()
                        observer.event('evaluation_sample_complete',echo=True,id=value['id'],sample=value['sample'],gpu=value['gpu'],response_tokens=value['response_tokens'])
                        bar.update(1)
                    bar.close()
                    for future in futures:future.result()
                correctness={record['id']:[] for record in records}
                for row in rows:correctness[row['id']].append(row['correct'])
                depths=[depth for row in rows for depth in row['actual_student_exit_depths']]
                summaries[f'{endpoint}_{loops}']=dict(**math_metrics(correctness),
                    benchmarks=benchmark_math_metrics(records,correctness),
                    selected_exit_depth_hist=[depths.count(d) for d in range(1,loops+1)],
                    mean_selected_depth=sum(depths)/len(depths),mean_actual_depth=float(loops),
                    executed_token_loops=sum(row['executed_token_loops'] for row in rows),
                    response_tokens=sum(row['response_tokens'] for row in rows),
                    mean_response_tokens=sum(row['response_tokens'] for row in rows)/len(rows),
                    correct_responses=sum(row['correct'] for row in rows),
                    accuracy=sum(row['correct'] for row in rows)/len(rows),
                    backfill_token_loops=0,truncated_responses=sum(not row['reached_eos'] for row in rows))
                all_results+=rows
    # Stable content ordering despite parallel completion; seeds already depend
    # only on the original problem fingerprint and sample, never GPU/shard order.
    all_results.sort(key=lambda row:(row['policy'],row['max_loops'],row['index'],row['sample']))
    sorted_output=args.output.with_suffix('.sorted.tmp')
    with sorted_output.open('w') as stream:
        for row in all_results:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    sorted_output.replace(args.output)
    report=dict(results=summaries,settings=settings,gpus=devices,wall_seconds=time.perf_counter()-begin,
        checkpoint=str(directory),speculative_config=None,
        questions=str(questions_path.resolve()),
        tensor_parallel_size=tensor_parallel_size,inference_gpu_groups=inference_groups,
        cpu_threads_per_process=cpu_threads,
        data=data_metadata,
        backend=backend,
        inference_python=sys.executable if backend=='vllm-rlt' else config['implementation'].get('vllm_python_bin') or os.environ.get('VLLM_PYTHON_BIN') or sys.executable,
        attention_backend=settings.get('rlt_attention_backend','triton') if backend=='vllm-rlt' else config['implementation'].get('vllm_attention_backend') or os.environ.get('VLLM_ATTENTION_BACKEND'),
        scope='Same per-problem seeds, gate policy and sampling settings. Fixed K execution; do not compare parallel throughput with isolated single-request latency.')
    args.output.with_suffix('.summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
