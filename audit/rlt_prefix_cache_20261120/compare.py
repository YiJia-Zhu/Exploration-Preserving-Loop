"""Measure RLT prefix KV reuse with identical short AIME jobs, swapping GPUs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import mean

from ep_loop.config import load_config
from ep_loop.data import normalize_record, prompt_ids
from ep_loop.gpu_resources import require_idle_gpus
from transformers import AutoTokenizer

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
DEVICES = ('0', '1')
baseline = json.loads((ROOT/'audit/checkpoint_eval_20261120_004854/plan.json').read_text())
checkpoint = Path(next(item['checkpoint'] for item in baseline['checkpoints'] if item['step']==5))
config = load_config(ROOT/'configs/evaluation_vllm_rlt.json')
source = ROOT/config['evaluation']['data_path']
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
entries = [json.loads(line) for line in source.open()]
entries.sort(key=lambda row:len(prompt_ids(tokenizer, normalize_record(row),
    config['model']['enable_thinking'], config['model']['repair_template_newlines'])))
selected = [entries[i] for i in (4, 12, 20, 29)]
dataset = AUDIT/'questions.jsonl'
dataset.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected))
config['evaluation'].update(suite=None, expected_examples=len(selected), data_path=str(dataset),
                            every_updates=0, max_new_tokens=256)
for key in ('rlt_max_num_seqs','rlt_max_num_batched_tokens','rlt_prefill_chunk_size'):
    config['evaluation'].pop(key,None)
for mode in (False,True):
    (AUDIT/f'cache_{mode}.config.json').write_text(json.dumps(config|{
        'evaluation':config['evaluation']|dict(rlt_prefix_caching=mode)},indent=2)+'\n')

def run(gpu, enabled, wave):
    prefix=f'wave{wave}_gpu{gpu}_cache{int(enabled)}'
    output=AUDIT/f'{prefix}.responses.jsonl'
    command=[sys.executable,'-u','scripts/evaluate_fastpath.py','--config',
        str(AUDIT/f'cache_{enabled}.config.json'),'--checkpoint',str(checkpoint),
        '--data',str(dataset),'--gpus',gpu,'--output',str(output)]
    with (AUDIT/f'{prefix}.log').open('x') as log:
        subprocess.run(command,cwd=ROOT,env=os.environ|dict(CUDA_VISIBLE_DEVICES=gpu),
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    records=[json.loads(line) for line in output.open()]
    events=[json.loads(line) for line in output.with_suffix(f'.gpu{gpu}.engine.log').open() if line.startswith('{"event": "rlt_')]
    ready,batch=events
    assert ready['enable_prefix_caching'] is enabled
    assert len(records)==batch['requests']==64 and all(row['max_loops']==2 for row in records)
    assert batch['cached_prompt_tokens']>0 if enabled else batch['cached_prompt_tokens']==0
    stats=dict(gpu=gpu,prefix_cache=enabled,wall_seconds=batch['seconds'],
        output_tokens=sum(row['response_tokens'] for row in records),
        response_file=str(output),engine=ready,
        cached_prompt_tokens=batch['cached_prompt_tokens'],prompt_tokens=batch['prompt_tokens'])
    stats['tokens_per_second']=stats['output_tokens']/stats['wall_seconds']
    print(json.dumps(stats),flush=True)
    return stats,{(row['id'],row['sample'],row['sampling_seed']):row['text'] for row in records}

def main():
    cases=[]
    contents=[]
    with ExitStack() as locks:
        for gpu in DEVICES:
            fcntl.flock(locks.enter_context(open(f'/tmp/ep-loop-gpu{gpu}.lock','a')),fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_idle_gpus(DEVICES)
        for wave in range(2):
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(run,gpu,i==wave,wave) for i,gpu in enumerate(DEVICES)]
                for future in futures:
                    stats,texts=future.result()
                    cases.append(stats);contents.append(texts)
    keys=contents[0].keys()
    assert all(z.keys()==keys for z in contents)
    report=dict(checkpoint=str(checkpoint),questions=len(selected),samples_per_question=16,max_new_tokens=256,
        cases=cases,exact_text_all_runs=sum(all(row[key]==contents[0][key] for row in contents) for key in keys))
    for enabled in (False,True):
        subset=[r for r in cases if r['prefix_cache']==enabled]
        wall=mean(r['wall_seconds'] for r in subset)
        report['cache_on' if enabled else 'cache_off']=dict(mean_wall_seconds=wall,
            mean_tokens_per_second=mean(r['tokens_per_second'] for r in subset),
            cache_fraction=sum(r['cached_prompt_tokens'] for r in subset)/sum(r['prompt_tokens'] for r in subset))
    (AUDIT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)

if __name__=='__main__':
    main()
