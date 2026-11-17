"""Short real-engine check with framework scheduler defaults and whole batches."""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ep_loop.fastpath.actor import VllmActor
from transformers import AutoTokenizer
from ep_loop.data import prompt_ids

output=Path(__file__).with_name('vllm_scheduler_default_validation.json')
if output.exists():raise FileExistsError(output)
model=ROOT/'models/Ouro-1.4B-Thinking'
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
data=ROOT/'data/dapo_ablation/ouro_seed42_train256_dev64_eval64_prompt1024/eval.jsonl'
records=[json.loads(line) for line in data.read_text().splitlines()][:8]
prompts=[prompt_ids(tokenizer,row) for row in records]
started=time.perf_counter();cases=[]
with VllmActor(model,'2',output.with_suffix('.engine.log'),loops=2,policy='fixed',
               max_model_len=17408,memory_fraction=.45,sleep=True) as actor:
    for name,inputs in [('training_batch8',prompts),('whole_shard32_max_prompt1024',[(prompts[i%8]*1024)[:1024] for i in range(32)])]:
        requests=[dict(input_ids=ids,temperature=1.,top_p=.7,max_new_tokens=16,seed=42+i) for i,ids in enumerate(inputs)]
        result=actor.generate(requests,logprobs=None)
        assert len(result['responses'])==len(requests)
        assert result['executed_loops']==2
        for row in result['responses']:
            assert 1<=len(row['token_ids'])<=16
            assert row['exit_depths']==[2]*len(row['token_ids'])
        cases.append(dict(name=name,requests=len(requests),response_tokens=sum(len(row['token_ids']) for row in result['responses']),seconds=result['seconds']))
        actor.request('sleep')
report=dict(status='PASS',gpu='2',cases=cases,wall_seconds=time.perf_counter()-started,
    scheduler='No sequence-count argument is passed; installed vLLM defaults',
    scope='Real Ouro weights, compiled fixed-N2 engine, short generation and sleep/wake; not an accuracy or long-generation throughput result.')
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
