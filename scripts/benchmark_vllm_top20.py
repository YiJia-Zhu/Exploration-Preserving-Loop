"""Opt-in real-model consistency/throughput check; never changes active runs."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='2')
    p.add_argument('--tokens',type=int,default=128,help='Short throughput diagnostic only; training/eval cap remains 16384')
    p.add_argument('--examples',type=int,default=4)
    p.add_argument('--eager',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    from transformers import AutoTokenizer
    from ep_loop.data import prompt_ids
    from ep_loop.ouro import load_ouro,evaluation_mode
    from ep_loop.inference import generate
    from ep_loop.reproducibility import seed_everything,sample_seed
    from ep_loop.fastpath.actor import VllmActor
    from ep_loop.fastpath.sparse import TopKDistribution,adjacent_js
    torch.set_num_threads(4);seed_everything(42)
    root=Path(__file__).resolve().parents[1]
    source=root/'data/dapo_ablation/ouro_seed42_train256_dev64_eval64_prompt1024/eval.jsonl'
    records=[json.loads(line) for line in source.read_text().splitlines()][:args.examples]
    model=root/'models/Ouro-1.4B-Thinking'
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    requests=[dict(input_ids=prompt_ids(tokenizer,r),temperature=1.,top_p=.7,max_new_tokens=args.tokens,
                   seed=sample_seed(42,r['problem_sha256'],0)) for r in records]
    backend,_=load_ouro(model,'cuda','bf16',False);backend.enable_student_exit();backend.model.eval()
    reference=[]
    for request in requests:
        with torch.no_grad(),evaluation_mode(backend.model),backend.generation_context():
            trace=backend.trace(torch.tensor([request['input_ids']],device='cuda'),4,use_cache=True,endpoint='learned',exit_threshold=.5)
            depth=int(trace.exits[0,-1])
            logits=backend.head(trace.states[depth-1][:,-1:])[0]
            sparse=TopKDistribution.from_logits(logits)
            reference.append(dict(depth=depth,gate_logits=[float(g[0,-1,0]) for g in trace.gates],
                ids=sparse.token_ids.cpu(),log_probs=sparse.log_probs.cpu(),tail=sparse.log_tail_mass.cpu()))
        del trace,logits,sparse
    # Warm up outside timed diagnostic, preserving a fresh per-request RNG.
    generate(backend,requests[0]['input_ids'],4,4,1.,.7,torch.Generator(device='cuda').manual_seed(42),'learned',.5,True)
    hf=[]
    for request in requests:
        out=generate(backend,request['input_ids'],4,args.tokens,1.,.7,
            torch.Generator(device='cuda').manual_seed(request['seed']),'learned',.5,True)
        hf.append(dict(tokens=len(out.token_ids),seconds=out.seconds,executed_token_loops=out.executed_token_loops))
    del backend;gc.collect();torch.cuda.empty_cache()
    with VllmActor(model,args.gpu,args.output.with_suffix('.engine.log'),max_model_len=2048,
                   memory_fraction=.45,eager=args.eager) as actor:
        one=[{**request,'max_new_tokens':1} for request in requests]
        scores=actor.generate(one)
        checks=[]
        for actual,expected in zip(scores['responses'],reference):
            row=actual['logprobs'][0]
            ids=torch.tensor(row['token_ids'],dtype=torch.int32)[None]
            log_probs=torch.tensor(row['log_probs'])[None]
            mass=(1-log_probs.exp().sum(-1)).clamp_min(torch.finfo(torch.float32).tiny)
            paired=TopKDistribution(torch.stack((expected['ids'],ids)),
                torch.stack((expected['log_probs'],log_probs)),torch.stack((expected['tail'],mass.log())),49152)
            checks.append(dict(hf_depth=expected['depth'],vllm_depth=actual['exit_depths'][0],
                sparse_js=float(adjacent_js(paired)),hf_gate_logits=expected['gate_logits'],
                vllm_gate_logits=actual['gate_logits'][0]))
        warm=actor.generate([{**request,'max_new_tokens':4} for request in requests])
        actual=actor.generate(requests)
    hf_tokens=sum(row['tokens'] for row in hf);hf_seconds=sum(row['seconds'] for row in hf)
    vllm_tokens=sum(len(row['token_ids']) for row in actual['responses'])
    report=dict(status='PASS' if all(row['hf_depth']==row['vllm_depth'] and row['sparse_js']<.0005 for row in checks) else 'CHECK_FAILED',
        consistency=checks,examples=len(requests),diagnostic_max_new_tokens=args.tokens,
        production_max_new_tokens=16384,seed=42,eager=args.eager,hf=hf,
        hf_tokens_per_second=hf_tokens/hf_seconds,vllm_tokens=vllm_tokens,vllm_seconds=actual['seconds'],
        vllm_tokens_per_second=vllm_tokens/actual['seconds'],
        throughput_ratio=(vllm_tokens/actual['seconds'])/(hf_tokens/hf_seconds),
        scope='Real DAPO prompts, selected-depth consistency and short stochastic generation throughput. Framework RNG/rounding can change continuations; no task-score claim or 16k speed extrapolation.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='PASS':raise SystemExit(2)


if __name__=='__main__':main()
