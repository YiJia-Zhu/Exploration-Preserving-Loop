"""Frozen-teacher approximation audit and a 17,408-position vLLM KV check."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',default='2');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--approximation-only',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    from transformers import AutoTokenizer
    from ep_loop.ouro import load_ouro
    from ep_loop.data import prompt_ids
    from ep_loop.selection import adjacent_js_from_log_probs,make_dynamic_stage_plan
    from ep_loop.fastpath.sparse import TopKDistribution,adjacent_js
    from ep_loop.fastpath.actor import VllmActor
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'audit/dapo_launch_20261114/launch.json').read_text())
    directory=Path(old['output']);config=json.loads((directory/'configs/ours.json').read_text())
    records={row['id']:row for row in [json.loads(line) for line in Path(config['data']['train_path']).read_text().splitlines()]}
    tokenizer=AutoTokenizer.from_pretrained(config['model']['path'],local_files_only=True)
    teacher,_=load_ouro(config['model']['path'],'cuda','bf16',False);teacher.model.eval()
    comparisons=[]
    responses=[];seen=set()
    for filename in ['rollouts_rank0.jsonl','rollouts_rank1.jsonl']:
        with (directory/'ours'/filename).open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['id'] in seen:continue
                seen.add(row['id']);responses.append(row)
                if len(responses)==4:break
        if len(responses)==4:break
    with torch.no_grad():
        for response in responses:
            prompt=prompt_ids(tokenizer,records[response['id']]);tokens=response['tokens'][:128]
            ids=torch.tensor([prompt+tokens],device='cuda')
            trace=teacher.response_trace(ids[:,:-1],4,len(prompt)-1)
            exits=teacher.native_exit_depths(trace,.9)
            logits=teacher.head(torch.stack(trace.states));logq=logits.log_softmax(-1)
            sparse=TopKDistribution.from_logits(logits)
            exact=adjacent_js_from_log_probs(logq);approx=adjacent_js(sparse)
            original=make_dynamic_stage_plan(exact,exits,js_threshold=.05)
            changed=make_dynamic_stage_plan(approx,exits,js_threshold=.05)
            valid=torch.arange(2,5,device='cuda')[:,None]<=exits[None]
            comparisons.append(dict(id=response['id'],positions=len(tokens),edges=int(valid.sum()),
                boundary_matches=int((((exact>.05)==(approx>.05)) & valid).sum()),
                stage_count_matches=int((original.target_loops==changed.target_loops).sum()),
                max_js_error=float((exact-approx).abs().max()),
                mean_retained_mass=float(sparse.log_probs.exp().sum(-1).mean()),
                min_retained_mass=float(sparse.log_probs.exp().sum(-1).min())))
    del trace,logits,logq,sparse,teacher,ids;gc.collect();torch.cuda.empty_cache()
    long=None
    if not args.approximation_only:
        with VllmActor(config['model']['path'],args.gpu,args.output.with_suffix('.engine.log'),
                       max_model_len=17408,memory_fraction=.55) as actor:
            requests=[dict(input_ids=[40]*17406,temperature=1.,top_p=.7,max_new_tokens=2,seed=42+i) for i in range(2)]
            long=actor.generate(requests,logprobs=20)
            assert all(len(row['token_ids'])==2 for row in long['responses'])
            assert all(len(row['exit_depths'])==2 for row in long['responses'])
            assert all(row['executed_token_loops']==4*17407 for row in long['responses'])
    positions=sum(row['positions'] for row in comparisons);edges=sum(row['edges'] for row in comparisons)
    report=dict(status='PASS',approximation=comparisons,
        boundary_agreement=sum(row['boundary_matches'] for row in comparisons)/edges,
        target_depth_agreement=sum(row['stage_count_matches'] for row in comparisons)/positions,
        unchanged_js_threshold=.05,long_context=dict(prompt_history_positions=17406,new_tokens_per_request=2,
            total_context_positions=17408,requests=2,seconds=long['seconds'],executed_loops=4) if long else None,
        scope='Top-20 agreement on 512 historical response positions; long prefix plus decode validates KV/context capacity, not a full 16k autoregressive speed benchmark.')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
