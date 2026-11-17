"""Real teacher readouts across chunked prefills at the production context limit."""
import json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from safetensors.torch import load_file
from ep_loop.fastpath.actor import VllmActor
output=Path(__file__).with_name('vllm_teacher_long_context.json')
torch.set_num_threads(4);started=time.perf_counter()
with tempfile.TemporaryDirectory(dir=output.parent) as directory, VllmActor(ROOT/'models/Ouro-1.4B-Thinking','6',
    output.with_suffix('.engine.log'),loops=4,policy='fixed',teacher_trace=True,sleep=True,
    max_model_len=17408,memory_fraction=.65) as actor:
 result=actor.teacher_trace([dict(input_ids=[40]*17407) for _ in range(2)],directory)
 assert len(result['responses'])==2
 for row in result['responses']:
  tensors=load_file(row['path'])
  assert row['input_positions']==17407
  assert tensors['states'].shape==(4,17407,2048)
  assert tensors['gates'].shape==(4,17407,1)
  assert tensors['states'].isfinite().all() and tensors['gates'].isfinite().all()
  assert row['executed_token_loops']==4*17407
 actor.request('sleep')
report=dict(status='PASS',requests=2,known_prefix_positions=17407,total_model_context=17408,
    physical_teacher_loops=4,teacher_batch_seconds=result['seconds'],wall_seconds=time.perf_counter()-started,
    scope='Long known-prefix teacher prefill and full readout transport with chunking; no autoregressive speed or task-score claim.')
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
