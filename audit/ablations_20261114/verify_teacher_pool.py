"""Teacher-correct cache generation and actual train-pool loading, tiny mocked generation."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from ep_loop.config import load_config,merge
from ep_loop.data import ByteTokenizer,normalize_record,prepare_training_records
from ep_loop.inference import GeneratedResponse
from ep_loop.ouro import tiny_ouro
from scripts.build_teacher_filter import main
root=Path(__file__).resolve().parents[2];audit=root/'audit/ablations_20261114'
torch.set_num_threads(2);torch.manual_seed(42)
raw=[dict(id=str(i),problem=f'{i}+1?',answer=str(i+1)) for i in range(4)]
data=audit/'pool_input.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in raw))
out=audit/'teacher_pool.jsonl'
cfg=merge(load_config(root/'configs/e_dynamic_js_teacher_correct.json'),dict(data=dict(train_path=str(data),expected_examples=4,correctness_cache=str(out),max_prompt_tokens=128),model=dict(revision='random-tiny-cache-test')))
path=audit/'teacher_pool_config.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
backend=tiny_ouro(root/'models/Ouro-1.4B-Thinking');tokenizer=ByteTokenizer();index=[0]
def generated(*args,**kwargs):
 i=index[0];index[0]+=1
 # First two pass; third is wrong; fourth is correct but truncated.
 text='\\boxed{'+str(i+1 if i!=2 else 99)+'}'
 tokens=tokenizer.encode(text)+[2]
 return GeneratedResponse(tokens,[1]*len(tokens),i!=3,len(tokens),seconds=.001,execution_depths=[1]*len(tokens))
with patch('ep_loop.backends.load_backend',return_value=(backend,{})),patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer),patch('ep_loop.inference.generate',side_effect=generated),patch.object(sys,'argv',['build_teacher_filter.py','--config',str(path),'--output',str(out),'--device','cpu']):main()
records,report=prepare_training_records(cfg,tokenizer,root);assert [r['id'] for r in records]==['0','1']
assert report['correctness_rejected']==2
summary=json.loads(out.with_suffix('.summary.json').read_text());assert summary['accepted_problems']==2
# Count-matched uses same cache, sample count, and seed; does not use correctness in choosing IDs.
matched,_=prepare_training_records(merge(cfg,dict(distillation=dict(question_filter='random_matched'))),tokenizer,root)
assert len(matched)==len(records)
(audit/'teacher_pool_validation.json').write_text(json.dumps(dict(status='PASS',correct_teacher_pool_ids=['0','1'],truncation_rejected=True,random_matched_size=2,scope='Controlled tiny mock, not pretrained teacher answer accuracy'),indent=2)+'\n')
