"""Compare JS, scalar confidence and prompt probes on sampled teacher prefixes.

No training, no fixed student policy, and no gold answer in instrument prompts.
Optional forced-answer branches are measurements, not the deployed policy.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def forced_answer(instrument, prefix, depth, max_tokens, cost):
    """Independent answerability diagnostic, never a confidence label."""
    import torch
    suffix='\n\n[Answer probe] Stop explaining and give only your best final answer now, in \\boxed{}.\nAnswer:'
    current=torch.cat((prefix,prefix.new_tensor([instrument.tokenizer.encode(suffix,add_special_tokens=False)])),dim=1)
    tokens=[]
    for _ in range(max_tokens):
        trace=instrument._trace(current,depth,cost)
        token=int(instrument.backend.head(trace.states[depth-1][:,-1:])[0,0].argmax())
        tokens.append(token)
        if token in instrument.backend.stop_token_ids:break
        current=torch.cat((current,current.new_tensor([[token]])),dim=1)
    return instrument.tokenizer.decode(tokens)


def main():
    from ep_loop.config import DEFAULT_CONFIG
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(DEFAULT_CONFIG))
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,default=64)
    p.add_argument('--positions-per-response',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--forced-answer-max-tokens',type=int,default=16)
    p.add_argument('--signals',nargs='+',choices=['js','max_probability','normalized_entropy','probe_know'],
                   default=['js','max_probability','normalized_entropy'])
    p.add_argument('--forced-answer-check',action='store_true',help='Extra per-depth forced answers, verified separately; expensive')
    p.add_argument('--smoke',action='store_true',help='Explicit random tiny model, not a pretrained result')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.limit,args.positions_per_response,args.max_new_tokens,args.forced_answer_max_tokens)<1:p.error('Counts must be positive')
    from ep_loop.observability import RunLogger
    with RunLogger(args.output.with_suffix('.events.jsonl'),args.output.with_suffix('.console.log'),'confidence_diagnostic') as observer:
        with observer.stage('confidence_diagnostic',echo=True):
            diagnose(args,observer)


def diagnose(args,observer):
    import torch
    from ep_loop.backends import load_backend
    from ep_loop.confidence import PromptConfidence,confidence_changes,isolated_probe_rng
    from ep_loop.config import load_config,smoke_config,validate_config
    from ep_loop.data import ByteTokenizer,normalize_record,prompt_ids
    from ep_loop.inference import generate,integer_answer,verify_integer_response
    from ep_loop.observability import model_progress
    from ep_loop.reproducibility import execution_provenance,sample_seed,seed_everything
    from ep_loop.selection import adjacent_js_from_log_probs
    root=Path(__file__).resolve().parents[1]
    config=load_config(args.config)
    if args.smoke:
        if args.device!='cpu':raise ValueError('--smoke requires --device cpu')
        config=smoke_config(config)
    validate_config(config)
    records=[normalize_record(json.loads(line)) for line in args.data.read_text().splitlines() if line.strip()][:args.limit]
    if not records:raise ValueError('No diagnostic questions')
    if args.forced_answer_check and any(integer_answer(r['answer']) is None for r in records):
        raise ValueError('Forced-answer verification requires integer ground truth')
    seed_everything(config['seed']);torch.set_num_threads(2)
    with observer.stage('model_load',echo=True),model_progress(observer.show_progress):
        backend,_=load_backend(config['model'],root/config['model']['path'],args.device,
            config['optimization']['mixed_precision'],False,smoke=args.smoke)
    backend.model.eval()
    if args.smoke:tokenizer=ByteTokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(root/config['model']['path'],local_files_only=True)
    instrument=PromptConfidence(backend,tokenizer,config['seed'],observer=observer)
    totals=dict(generation_token_loops=0,teacher_token_loops=0,probe_token_loops=0,probe_coda_token_positions=0,
                probe_forwards=0,forced_answer_token_loops=0,forced_answer_forwards=0)
    with args.output.open('x') as out:
        for record in observer.progress(records,desc='Confidence questions',unit='problem',position=0):
            seed=sample_seed(config['seed'],record['problem_sha256'])
            seed_everything(seed)
            prompt=prompt_ids(tokenizer,record,config['model']['enable_thinking'],config['model']['repair_template_newlines'])
            response=generate(backend,prompt,config['algorithm']['teacher_max_loops'],args.max_new_tokens,
                config['rollout']['temperature'],config['rollout']['top_p'],torch.Generator(device=backend.device).manual_seed(seed),
                endpoint='native',exit_threshold=config['algorithm']['exit_threshold'],observer=observer,
                log_context=dict(id=record['id']),progress=True,progress_desc=f"Diagnostic {record['id']}",progress_position=1)
            inputs=torch.tensor([prompt+response.token_ids[:-1]],device=backend.device)
            first=len(prompt)-1;k=config['algorithm']['teacher_max_loops']
            with torch.no_grad(),observer.stage('teacher_prefix_trace'):
                trace=backend.response_trace(inputs,k,first,'native',config['algorithm']['exit_threshold'])
                exits=backend.native_exit_depths(trace,config['algorithm']['exit_threshold']).reshape(-1)
                n=len(response.token_ids);count=min(args.positions_per_response,n)
                positions=sorted({i*(n-1)//max(1,count-1) for i in range(count)})
                hidden=torch.stack([state[positions] for state in trace.states])
                logq=backend.head(hidden).log_softmax(-1)
                js=adjacent_js_from_log_probs(logq)
                measured={}
                probe_details={}
                for signal in args.signals:
                    if signal=='js':measured[signal]=(js,None)
                    elif signal.startswith('probe_'):
                        changes,confidence,cost=instrument.measure(inputs,first,exits,signal,positions,record_details=True)
                        measured[signal]=(changes,confidence)
                        probe_details[signal] = list(instrument.last_records)
                        totals['probe_token_loops']+=cost['token_loops']
                        totals['probe_coda_token_positions']+=cost['coda_token_positions']
                        totals['probe_forwards']+=cost['forwards']
                    else:measured[signal]=confidence_changes(logq,signal)
                for column,pos in enumerate(positions):
                    endpoint=int(exits[pos]);signals={}
                    for signal,(changes,confidence) in measured.items():
                        signals[signal]=dict(changes=changes[:endpoint-1,column].tolist(),
                            confidence=confidence[:endpoint,column].tolist() if confidence is not None else None)
                    forced=[]
                    if args.forced_answer_check:
                        prefix=inputs[:,:first+pos+1]
                        for depth in range(1,endpoint+1):
                            with isolated_probe_rng(backend,prefix,seed):
                                cost=dict(token_loops=0,coda_token_positions=0,forwards=0)
                                candidate=forced_answer(instrument,prefix,depth,args.forced_answer_max_tokens,cost)
                                forced.append(dict(depth=depth,text=candidate,**verify_integer_response(candidate,record['answer'],record['messages'])))
                                totals['forced_answer_token_loops']+=cost['token_loops']
                                totals['forced_answer_forwards']+=cost['forwards']
                    out.write(json.dumps(dict(id=record['id'],problem_sha256=record['problem_sha256'],
                        response_position=pos,native_teacher_exit=endpoint,signals=signals,forced_answer_checks=forced,
                        probe_details={signal:details[column] for signal,details in probe_details.items()},
                        dynamic_generation_exit_depths=response.exit_depths,smoke=args.smoke),ensure_ascii=False)+'\n')
                    out.flush()
                totals['generation_token_loops']+=response.executed_token_loops
                totals['teacher_token_loops']+=trace.executed_token_loops
    report=dict(problems=len(records),signals=args.signals,smoke=args.smoke,cost=totals,
        scope='Sampled prefixes from native dynamic teacher continuations; original-prefix signals versus input-changing prompt instruments. Gold is used only by the external forced-answer verifier, never in probe prompts. Not on-policy training or proof of necessary exploration.',
        reproducibility=execution_provenance(root,backend.device,config['seed'],phase='confidence_diagnostic'))
    args.output.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
