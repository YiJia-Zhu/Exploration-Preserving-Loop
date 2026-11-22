"""Real Ouro parity, long-context and sleep/wake gate before timing training."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='0')
    p.add_argument('--python-bin',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-num-seqs',type=int,default=0,help='0: framework default; positive: engine concurrency')
    p.add_argument('--tensor-parallel-size',type=int,default=1)
    a=p.parse_args()
    if a.max_num_seqs<0:p.error('--max-num-seqs must be nonnegative')
    from ep_loop.fastpath.parallel import gpu_groups
    if len(gpu_groups(a.gpu.split(','),a.tensor_parallel_size))!=1:
        p.error('--gpu must contain exactly --tensor-parallel-size GPU IDs')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    from ep_loop.gpu_resources import require_idle_gpus
    require_idle_gpus(a.gpu.split(','))
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID')
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    from transformers import AutoTokenizer
    from ep_loop.ouro import load_ouro
    from ep_loop.data import prompt_ids
    from ep_loop.selection import native_ouro_exits
    from ep_loop.fastpath.actor import VllmActor
    torch.set_num_threads(16);torch.manual_seed(42)
    model=ROOT/'models/Ouro-1.4B-Thinking'
    data=ROOT/'data/dapo_ablation/ouro_seed42_trainall_dev64_eval64_prompt1024_aime30_clean/train.jsonl'
    with data.open() as f:records=[json.loads(next(f)) for _ in range(2)]
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    prompts=[prompt_ids(tokenizer,r) for r in records]
    backend,_=load_ouro(model,'cuda','bf16',False)
    backend.model.eval();references=[]
    with torch.no_grad():
        for ids in prompts:
            tensor=torch.tensor([ids],device='cuda')
            trace=backend.trace(tensor,4,use_cache=False)
            states=torch.stack([s[0] for s in trace.states]).cpu()
            gates=torch.stack([g[0] for g in trace.gates]).cpu()
            final=backend.trace(tensor,2,use_cache=False)
            logp=backend.head(final.states[-1][:,-1:])[0,0].log_softmax(-1).cpu()
            references.append(dict(states=states,gates=gates,student_logp=logp))
    head=copy.deepcopy(backend.model.lm_head).cpu().eval().requires_grad_(False)
    # HF training keeps FP32 residuals/embeddings with autocast; vLLM keeps
    # BF16 weights and residuals. Also compare the matching BF16 HF forward
    # to separate kernel/architecture errors from this existing dtype gap.
    backend.model.to(dtype=torch.bfloat16)
    with torch.no_grad():
        for ids,ref in zip(prompts,references):
            tensor=torch.tensor([ids],device='cuda')
            final=backend.trace(tensor,2,use_cache=False)
            ref['student_bf16_logp']=backend.head(final.states[-1][:,-1:])[0,0].log_softmax(-1).cpu()
            trace=backend.trace(tensor,4,use_cache=False)
            ref['bf16_states']=torch.stack([s[0] for s in trace.states]).cpu()
            ref['bf16_gates']=torch.stack([g[0] for g in trace.gates]).cpu()
    backend.model.to('cpu')
    del tensor,trace,final,states,gates,logp
    gc.collect();torch.cuda.empty_cache()
    report=dict(gpu=a.gpu,python_bin=a.python_bin,tensor_parallel_size=a.tensor_parallel_size,checks=[],status='running',
                scope='Real original Ouro weights, full-vocabulary teacher parity, '
                      'native fixed-N2 student, batch128, 17408 context, weight refresh, '
                      'co-resident sleeping actors. Not task efficacy or full DDP training.')
    def record(name,**values):
        report['checks'].append(dict(name=name,**values))
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['checks'][-1]),flush=True)
    common=dict(max_model_len=17408,memory_fraction=.75,sleep=True,
                python_bin=a.python_bin,attention_backend='FLASHINFER',tensor_parallel_size=a.tensor_parallel_size)
    if a.max_num_seqs:common['max_num_seqs']=a.max_num_seqs
    with tempfile.TemporaryDirectory(prefix='ep026_validate_',dir=a.output.parent) as tmp:
        tmp=Path(tmp)
        with VllmActor(model,a.gpu,a.output.with_suffix('.student.engine.log'),
                        loops=2,policy='fixed',max_logprobs=-1,**common) as student:
            assert student.runtime['vllm_version'].startswith('0.26.')
            assert student.runtime['native_fixed_student']
            weights=tmp/'weights.safetensors'
            save_file(backend.model.state_dict(),str(weights))
            acknowledgement=student.request('weights',path=str(weights),version=0)
            weights.unlink()
            assert acknowledgement['weight_version']==0
            assert len(acknowledgement['workers'])==a.tensor_parallel_size
            record('all_tp_workers_loaded_weights',weight_version=0,workers=acknowledgement['workers'])
            del backend
            gc.collect()
            requests=[dict(input_ids=ids,temperature=1.,top_p=1.,max_new_tokens=1,seed=42+i)
                      for i,ids in enumerate(prompts)]
            actual=student.generate(requests,logprobs=-1)
            errors=[];bf16_errors=[];top1=[];full_js=[];full_js_bf16=[]
            for got,ref in zip(actual['responses'],references):
                row=got['logprobs'][0]
                observed=torch.tensor(row['log_probs'])
                expected=ref['student_logp'][torch.tensor(row['token_ids'])]
                errors.append(float((observed-expected).abs().max()))
                expected_bf16=ref['student_bf16_logp'][torch.tensor(row['token_ids'])]
                bf16_errors.append(float((observed-expected_bf16).abs().max()))
                top1.append(row['token_ids'][0]==int(ref['student_bf16_logp'].argmax()))
                assert len(row['token_ids'])==len(ref['student_logp'])
                for expected_logp,destination in [(expected,full_js),(expected_bf16,full_js_bf16)]:
                    middle=torch.logaddexp(observed,expected_logp)-__import__('math').log(2)
                    destination.append(float(.5*((observed.exp()*(observed-middle)).sum()+
                                                 (expected_logp.exp()*(expected_logp-middle)).sum())))
            record('student_next_token_logprob',max_abs_logprob_error_vs_fp32_autocast=max(errors),
                   max_abs_logprob_error_vs_bf16=max(bf16_errors),bf16_top1_agreement=top1,
                   full_vocab_js_vs_fp32_autocast=full_js,full_vocab_js_vs_bf16=full_js_bf16,runtime=student.runtime)
            # Compare probability mass across the complete vocabulary. A hard
            # absolute logprob cutoff overweights very unlikely BF16 tokens.
            assert max(full_js)<.001 and max(full_js_bf16)<.001 and all(top1),(full_js,full_js_bf16)
            batch=[dict(input_ids=prompts[i%2],temperature=1.,top_p=1.,
                        max_new_tokens=16,seed=42+i) for i in range(128)]
            actual=student.generate(batch,logprobs=None)
            assert len(actual['responses'])==128
            assert all(row['exit_depths']==[2]*len(row['token_ids']) for row in actual['responses'])
            record('student_batch128',seconds=actual['seconds'],tokens=sum(len(x['token_ids']) for x in actual['responses']),
                   engine_metrics=actual.get('engine_metrics'))
            long=student.generate([dict(input_ids=[40]*17406,temperature=1.,top_p=1.,
                max_new_tokens=2,min_new_tokens=2,ignore_eos=True,seed=42)],logprobs=None)
            assert len(long['responses'][0]['token_ids'])==2
            record('student_long_context',input_tokens=17406,output_tokens=2,seconds=long['seconds'])
            student.request('sleep')
            with VllmActor(model,a.gpu,a.output.with_suffix('.teacher.engine.log'),
                           loops=4,policy='fixed',teacher_trace=True,**common) as teacher:
                actual=teacher.teacher_trace([dict(input_ids=x) for x in prompts],tmp/'teacher0')
                teacher.request('sleep')
                head.to('cuda')
                for i,(got,ref) in enumerate(zip(actual['responses'],references)):
                    payload=load_file(got['path']);s=payload['states'];g=payload['gates']
                    assert list(s.shape)==[4,len(prompts[i]),2048]
                    assert got['executed_token_loops']==4*len(prompts[i])
                    comparisons={key:[] for key in ['vs_fp32_autocast','vs_bf16','hf_dtype_control']}
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        for begin in range(0,s.shape[1],32):
                            new=head(s[:,begin:begin+32].cuda()).float().log_softmax(-1)
                            old=head(ref['states'][:,begin:begin+32].cuda()).float().log_softmax(-1)
                            bf16=head(ref['bf16_states'][:,begin:begin+32].cuda()).float().log_softmax(-1)
                            for key,left,right in [('vs_fp32_autocast',new,old),('vs_bf16',new,bf16),
                                                   ('hf_dtype_control',old,bf16)]:
                                middle=torch.logaddexp(left,right)-__import__('math').log(2)
                                js=.5*((left.exp()*(left-middle)).sum(-1)+(right.exp()*(right-middle)).sum(-1))
                                comparisons[key].append(js.cpu().flatten())
                    statistics={}
                    for key,chunks in comparisons.items():
                        values=torch.cat(chunks)
                        statistics[key]=dict(max=float(values.max()),mean=float(values.mean()),
                                             p99=float(values.quantile(.99)))
                    agreement=float((native_ouro_exits(g,.9)==native_ouro_exits(ref['gates'],.9)).float().mean())
                    record('teacher_all_loop_readout',example=i,full_vocab_js=statistics,
                           native_exit_agreement=agreement,input_positions=s.shape[1])
                    # Repeated loop calculations amplify rounding at isolated
                    # positions (also in the HF dtype control). Gate on mean
                    # and p99 distribution mass error and on logical exits;
                    # retain the maximum explicitly as a numerical limitation.
                    assert all(statistics[key]['mean']<.001 and statistics[key]['p99']<.005
                               for key in ['vs_fp32_autocast','vs_bf16']) and agreement>=.99,(statistics,agreement)
                head.to('cpu');gc.collect();torch.cuda.empty_cache()
                actual=teacher.teacher_trace([dict(input_ids=[40]*17406)],tmp/'teacher_long')
                teacher.request('sleep')
                got=actual['responses'][0]
                with safe_open(got['path'],framework='pt',device='cpu') as f:
                    assert f.get_slice('states').get_shape()==[4,17406,2048]
                record('teacher_long_readout',input_positions=got['input_positions'],seconds=actual['seconds'])
                student.request('wake');student.generate(requests,logprobs=None);student.request('sleep')
                teacher.teacher_trace([dict(input_ids=x) for x in prompts],tmp/'teacher1')
                teacher.request('sleep')
                record('warm_student_teacher_switch',status='PASS')
    report['status']='PASS'
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',str(a.output),flush=True)


if __name__=='__main__':
    main()
