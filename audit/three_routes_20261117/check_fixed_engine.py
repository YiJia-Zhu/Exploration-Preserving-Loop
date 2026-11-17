"""Finite real-weight check that fixed-N vLLM instantiates N recurrent steps."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from ep_loop.data import prompt_ids
from ep_loop.fastpath.actor import VllmActor

directory=ROOT/'models/Ouro-1.4B-Thinking'
tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
data=ROOT/'data/dapo_ablation/ouro_seed42_train256_dev64_eval64_prompt1024/eval.jsonl'
records=[json.loads(line) for line in data.read_text().splitlines()[:2]]
report=[]
for depth in [1,2,4]:
    started=time.perf_counter()
    with VllmActor(directory,'2',Path(__file__).parent/f'fixed_N{depth}.engine.log',
                   loops=depth,policy='fixed',max_model_len=2048,max_num_seqs=2,
                   memory_fraction=.55,eager=False) as actor:
        health=actor.request('health')
        assert health['configured_recurrent_steps']==depth,health
        requests=[dict(input_ids=prompt_ids(tokenizer,row,True,True),max_new_tokens=16,
                       temperature=1.,top_p=.7,seed=42+i) for i,row in enumerate(records)]
        result=actor.generate(requests,logprobs=None)
        assert result['executed_loops']==depth
        for request,response in zip(requests,result['responses']):
            assert response['exit_depths']==[depth]*len(response['token_ids'])
            assert response['executed_token_loops']==depth*(len(request['input_ids'])+len(response['token_ids'])-1)
        row=dict(depth=depth,health=health,requests=len(result['responses']),
                 tokens=sum(len(item['token_ids']) for item in result['responses']),
                 generation_seconds=result['seconds'],wall_seconds=time.perf_counter()-started)
        report.append(row)
        print(json.dumps(row),flush=True)
(Path(__file__).parent/'fixed_engine_validation.json').write_text(json.dumps(dict(status='PASS',cases=report,
    scope='Real base weights and compiled actor, two prompts x 16-token cap per N; no optimizer updates or task-quality claims.'),indent=2)+'\n')
