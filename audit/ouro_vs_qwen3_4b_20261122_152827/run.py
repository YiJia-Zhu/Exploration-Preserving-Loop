"""Compare native Ouro and Qwen3 at fixed work on one GPU."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import sysconfig
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from ep_loop.gpu_resources import require_idle_gpus

VLLM_PYTHON = ROOT / 'ep_loop_envs/vllm026/bin/python'
WORKLOAD = ROOT / 'audit/vllm_rlt_v020_release_20261120_144351/workload.json'
REASONING = WORKLOAD.parent / 'rlt.json'
MODELS = {'qwen3_4b': ('Qwen3-4B', None), 'ouro_k2': ('Ouro-1.4B-Thinking', 2),
          'ouro_k4': ('Ouro-1.4B-Thinking', 4)}
PHASES = {'short_c1': (1, 256, 128), 'short_c8': (8, 256, 256), 'long_c8': (8, 8192, 128)}
GPU = '3'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def prepare(model):
    from safetensors import safe_open
    from transformers import AutoTokenizer

    path = ROOT/'models'/model
    config = read(path/'config.json')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    reasoning = {row['id']: row['text'] for row in read(REASONING)['full']['answers']}
    jobs = []
    for record in read(WORKLOAD)['jobs']:
        initial = tokenizer.apply_chat_template(record['messages'], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True)
        initial = initial.replace('\\n', '\n')
        ids = tokenizer.encode(initial, add_special_tokens=False, verbose=False)
        if not 0 < len(ids) <= 256:
            raise ValueError(f'Original question does not fit into 256 tokens: {record["id"]}')
        ending = tokenizer.encode(reasoning[record['id']], add_special_tokens=False, verbose=False)
        if not ending:
            raise ValueError('Empty continuation')
        tail = (ending * math.ceil((8192-len(ids))/len(ending)))[:8192-len(ids)]
        jobs.append(dict(id=record['id'], seed=record['seed'], original_prompt_tokens=len(ids),
                         input_ids=ids+tail))
    matrices = 0
    for shard in path.glob('*.safetensors'):
        with safe_open(shard, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.startswith('model.layers.'):
                    shape = f.get_slice(key).get_shape()
                    if len(shape)==2:
                        matrices += math.prod(shape)
    if matrices < 1:
        raise RuntimeError(f'Missing transformer matrices for {path}')
    return dict(model=str(path.resolve()), config=config, jobs=jobs, layer_matrix_parameters=matrices)


def worker(case):
    from ep_loop.cpu_runtime import configure_cpu_threads
    configure_cpu_threads()
    toolkit = Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'
    if (toolkit/'bin/nvcc').is_file():
        os.environ['CUDA_HOME'] = os.environ['CUDA_PATH'] = str(toolkit)
        os.environ['PATH'] = str(toolkit/'bin')+os.pathsep+os.environ.get('PATH','')
    from vllm import LLM, SamplingParams

    model, loops = MODELS[case]
    spec = read(OUT/f'workload_{model}.json')
    start = time.perf_counter()
    engine = LLM(
        model=spec['model'], trust_remote_code=True, dtype='bfloat16', seed=42,
        generation_config='vllm', tensor_parallel_size=1,
        attention_config={'backend':'FLASHINFER'}, gpu_memory_utilization=.75,
        max_model_len=9216, max_num_batched_tokens=9216,
        enable_prefix_caching=False, enable_chunked_prefill=True, disable_log_stats=False,
        **({'hf_overrides': {'total_ut_steps':loops}} if loops is not None else {}),
    )
    print(case, 'initialized', round(time.perf_counter()-start,2), flush=True)

    def count_preemptions():
        return sum(m.value for m in engine.get_metrics() if m.name=='vllm:num_preemptions')

    def measure(count, prefix, cap):
        jobs=spec['jobs'][:count]
        prompts=[dict(prompt_token_ids=job['input_ids'][:prefix]) for job in jobs]
        params=[SamplingParams(temperature=1.,top_p=.7,seed=job['seed'],max_tokens=cap,
                              ignore_eos=True,detokenize=False) for job in jobs]
        prior=count_preemptions()
        start=time.perf_counter()
        rows=engine.generate(prompts,params,use_tqdm=False)
        duration=time.perf_counter()-start
        assert len(rows)==count
        assert all(len(row.prompt_token_ids)==prefix and len(row.outputs[0].token_ids)==cap
                   for row in rows)
        assert count_preemptions()==prior
        return dict(
            seconds=duration, output_tokens=count*cap, output_tok_s=(count*cap)/duration,
            mean_tpot_ms=1000*statistics.mean(
                (r.metrics.last_token_ts-r.metrics.first_token_ts)/(cap-1) for r in rows),
            all_first_token_s=max(r.metrics.first_token_latency for r in rows),
        )

    result={}
    for name,(count,prefix,cap) in PHASES.items():
        measure(count,prefix,cap)
        runs=[measure(count,prefix,cap) for _ in range(3)]
        result[name]=dict(concurrency=count,prompt_tokens_each=prefix,
                          output_tokens_each=cap,runs=runs,median={
                              k:statistics.median(r[k] for r in runs) for k in runs[0]})
        print(case,name,result[name]['median'],flush=True)
        write(OUT/f'{case}.json',result)


def main():
    specs={}
    for model,_ in MODELS.values():
        if model not in specs:
            specs[model]=prepare(model)
            write(OUT/f'workload_{model}.json',specs[model])
    with open(f'/tmp/ep-loop-gpu{GPU}.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_idle_gpus([GPU])
        for case in MODELS:
            env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
                 'VLLM_WORKER_MULTIPROC_METHOD':'spawn','PYTHONHASHSEED':'42'}
            with (OUT/f'{case}.log').open('w') as log:
                subprocess.run([str(VLLM_PYTHON),'-u',__file__,'--worker',case],
                               env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    summary={}
    for case,(model,loops) in MODELS.items():
        spec=specs[model]
        cfg=spec['config']
        visits=cfg['num_hidden_layers']*(loops or 1)
        summary[case]=dict(
            layer_visits=visits, kv_heads=cfg['num_key_value_heads'],
            theoretical_bf16_kv_bytes_per_token=visits*2*cfg['num_key_value_heads']*cfg['head_dim']*2,
            effective_layer_matrix_parameters=spec['layer_matrix_parameters']*(loops or 1),
            results=read(OUT/f'{case}.json'),
        )
    write(OUT/'comparison.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=list(MODELS))
    args=parser.parse_args()
    worker(args.worker) if args.worker else main()
