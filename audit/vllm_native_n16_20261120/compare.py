"""Prepare a shared real AIME workload, then run the matched API benchmarks."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess

from transformers import AutoTokenizer
from ep_loop.data import normalize_record,prompt_ids
from ep_loop.gpu_resources import require_idle_gpus

AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[1]
DEVICES=('0','1')


def run(gpu,cache):
    with (AUDIT/f'cache_{cache}.log').open('x') as log:
        subprocess.run([str(ROOT/'ep_loop_envs/vllm026/bin/python'),'-u',str(AUDIT/'worker.py'),str(cache)],
            cwd=ROOT, env=os.environ|dict(CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,
            check=True)
    return json.loads((AUDIT/f'cache_{cache}.json').read_text())


def main():
    baseline=json.loads((ROOT/'audit/checkpoint_eval_20261120_004854/plan.json').read_text())
    model=Path(next(row['checkpoint'] for row in baseline['checkpoints'] if row['step']==5))
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    with (ROOT/'audit/rlt_prefix_cache_20261120/questions.jsonl').open() as f:
        rows=[normalize_record(json.loads(line)) for line in f]
    prompts=[dict(id=row['id'],input_ids=prompt_ids(tok,row,True,True),seed=42+1009*i)
             for i,row in enumerate(rows)]
    plan=dict(checkpoint=str(model),prompts=prompts,max_new_tokens=256,samples_per_problem=16,
              ignore_eos=True,temperature=1.0,top_p=.7,gpus=DEVICES,
              rlt_revision='bed34006ea8dc5b6ba1a3abd99a9fd81bd91a31b')
    (AUDIT/'workload.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ExitStack() as locks:
        for gpu in DEVICES:
            fcntl.flock(locks.enter_context(open(f'/tmp/ep-loop-gpu{gpu}.lock','a')),fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_idle_gpus(DEVICES)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(run,gpu,cache) for cache,gpu in enumerate(DEVICES)]
            results=[f.result() for f in futures]
    report=dict(questions=len(rows),samples_per_question=16,max_new_tokens=256,results=results)
    (AUDIT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
