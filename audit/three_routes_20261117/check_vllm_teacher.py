"""Compare vLLM teacher-forced all-loop readouts with frozen HF Ouro."""
import gc,json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from ep_loop.ouro import load_ouro
from ep_loop.data import prompt_ids
from ep_loop.fastpath.actor import VllmActor
from ep_loop.selection import native_ouro_exits
from ep_loop.reproducibility import seed_everything
seed_everything(42);torch.set_num_threads(4)
output=Path(__file__).with_name('vllm_teacher_validation.json')
model=ROOT/'models/Ouro-1.4B-Thinking'
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
records=[json.loads(l) for l in (ROOT/'data/dapo_ablation/ouro_seed42_train256_dev64_eval64_prompt1024/train.jsonl').read_text().splitlines()][:2]
requests=[dict(input_ids=prompt_ids(tokenizer,r)) for r in records]
backend,_=load_ouro(model,'cuda','bf16',False);backend.model.eval();reference=[]
with torch.no_grad():
 for request in requests:
  trace=backend.trace(torch.tensor([request['input_ids']],device='cuda'),4,use_cache=False)
  states=torch.stack([s[0] for s in trace.states]);gates=torch.stack([s[0] for s in trace.gates])
  logq=backend.head(states[:,-16:]).log_softmax(-1).cpu()
  reference.append(dict(states=states.cpu(),gates=gates.cpu(),logq=logq))
 head=backend.model.lm_head.weight.detach().cpu().clone()
del backend,trace,states,gates,logq;gc.collect();torch.cuda.empty_cache()
checks=[];started=time.perf_counter()
with tempfile.TemporaryDirectory(dir=output.parent) as temporary, VllmActor(model,'0',output.with_suffix('.engine.log'),loops=4,policy='fixed',
       max_model_len=17408,memory_fraction=.65,sleep=True,teacher_trace=True) as actor:
 result=actor.teacher_trace(requests,Path(temporary)/'readouts')
 assert len(result['responses'])==len(requests)
 for actual,expected in zip(result['responses'],reference):
  traced=load_file(actual['path']);s=traced['states'];g=traced['gates']
  assert s.shape==expected['states'].shape
  with torch.no_grad():
   logp=torch.nn.functional.linear(s[:,-16:].to('cuda'),head.to('cuda',dtype=s.dtype)).float().log_softmax(-1).cpu()
  midpoint=torch.logaddexp(logp,expected['logq'])-__import__('math').log(2)
  js=.5*((logp.exp()*(logp-midpoint)).sum(-1)+(expected['logq'].exp()*(expected['logq']-midpoint)).sum(-1))
  exit_agreement=float((native_ouro_exits(g,.9)==native_ouro_exits(expected['gates'],.9)).float().mean())
  checks.append(dict(input_positions=s.shape[1],state_max_abs_error=float((s.float()-expected['states'].float()).abs().max()),
       max_full_vocab_js=float(js.max()),exit_agreement=exit_agreement,executed_token_loops=actual['executed_token_loops']))
 actor.request('sleep')
report=dict(status='PASS' if all(x['max_full_vocab_js']<.001 and x['exit_agreement']>.99 for x in checks) else 'CHECK_FAILED',checks=checks,
            teacher_batch_seconds=result['seconds'],wall_seconds=time.perf_counter()-started,scope='Real-weight teacher-forced all-loop backend parity; not task efficacy or long-context speed evidence.')
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if report['status']!='PASS':raise SystemExit(2)
