"""Exercise filter CLI logging using explicitly random tiny teacher assets."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from ep_loop.config import load_config,merge
from ep_loop.data import ByteTokenizer
from ep_loop.ouro import tiny_ouro
from scripts.build_teacher_filter import main
root=Path(__file__).resolve().parents[2];audit=root/'audit/logging_20261114'
torch.set_num_threads(2);torch.manual_seed(42)
config=merge(load_config(root/'configs/e_dynamic_js.json'),dict(model=dict(revision='random-tiny-logging-test-not-pretrained'),data=dict(train_path=str(audit/'launcher_input.jsonl'),expected_examples=3,max_prompt_tokens=128),rollout=dict(max_new_tokens=4)))
cfg=audit/'filter_tiny_config.json';cfg.write_text(json.dumps(config,indent=2)+'\n')
out=audit/'filter_tiny.jsonl';backend=tiny_ouro(root/'models/Ouro-1.4B-Thinking')
with patch('ep_loop.backends.load_backend',return_value=(backend,{})),patch('transformers.AutoTokenizer.from_pretrained',return_value=ByteTokenizer()),patch.object(sys,'argv',['build_teacher_filter.py','--config',str(cfg),'--output',str(out),'--device','cpu']):
 main()
rows=[json.loads(line) for line in out.with_suffix('.events.jsonl').read_text().splitlines()]
assert rows[-1]['event']=='run_end' and rows[-1]['status']=='completed'
assert len([r for r in rows if r.get('stage')=='generation' and r['event']=='stage_end'])==3
assert len(out.read_text().splitlines())==4
(audit/'filter_validation.json').write_text(json.dumps(dict(status='PASS',scope='Random tiny mocked loader only; no pretrained filter labels',three_answers_logged=True),indent=2)+'\n')
