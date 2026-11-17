"""Two real-weight updates, weight-version/KV lifecycle and sparse gradients.

Diagnostic rollout cap is 64 to exercise the update boundary promptly. The
production launcher keeps 16384 and every original learning hyperparameter.
No full checkpoint or model-directory copy is created.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',default='2');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import torch
    from transformers import AutoTokenizer
    from ep_loop.config import load_config
    from ep_loop.data import normalize_record,prompt_ids
    from ep_loop.ouro import load_ouro
    from ep_loop.observability import RunLogger
    from ep_loop.reproducibility import seed_everything
    from ep_loop.fastpath.objective import DistillationObjective
    from ep_loop.fastpath.rollout import RolloutBatcher
    torch.set_num_threads(4);seed_everything(42)
    root=Path(__file__).resolve().parents[1]
    previous=json.loads((root/'experiments/ablations/latest_run.json').read_text())
    config=json.loads((Path(previous['output'])/'configs/ours.json').read_text())
    config['rollout']['max_new_tokens']=64
    config['implementation'].update(vllm_memory_fraction=.45,vllm_eager=False)
    tokenizer=AutoTokenizer.from_pretrained(config['model']['path'],local_files_only=True)
    records=[normalize_record(json.loads(line),index) for index,line in enumerate(Path(config['data']['train_path']).read_text().splitlines())]
    for record in records:record['input_ids']=prompt_ids(tokenizer,record)
    student,_=load_ouro(config['model']['path'],'cuda','bf16',True)
    teacher=type(student)(copy.deepcopy(student.model).eval().requires_grad_(False),'bf16')
    student.enable_student_exit();student.model.train()
    objective=DistillationObjective(student,teacher,config['algorithm'],128,True)
    opt=config['optimization']
    optimizer=torch.optim.AdamW(student.model.parameters(),lr=opt['learning_rate'],weight_decay=opt['weight_decay'],betas=tuple(opt['betas']),eps=opt['eps'])
    generator=torch.Generator(device='cuda').manual_seed(42);data_rng=random.Random(42)
    teacher_gate=teacher.model.model.early_exit_gate.weight.detach().clone()
    initial_gate=student.model.model.early_exit_gate.weight.detach().clone()
    stats=[]
    with RunLogger(args.output/'events.jsonl',args.output/'console.log','fastpath_validation') as observer:
        batcher=RolloutBatcher(student,config,args.output,0,observer)
        try:
            for update in range(2):
                start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                indices=data_rng.sample(range(len(records)),config['rollout']['prompts_per_batch'])
                batcher.prepare(records,indices,0,1,update,generator)
                loss_sum=tokens=0
                for index in indices:
                    record=records[index]
                    for sample in range(config['rollout']['responses_per_prompt']):
                        response=batcher.pop(dict(id=record['id'],sample=sample))
                        ids=torch.tensor([record['input_ids']+response.token_ids],device='cuda')
                        loss,info=objective(ids,len(record['input_ids']),response.exit_depths)
                        assert torch.isfinite(loss)
                        (loss/16).backward();loss_sum+=float(loss.detach());tokens+=len(response.token_ids)
                norm=torch.nn.utils.clip_grad_norm_(student.model.parameters(),float('inf'))
                assert torch.isfinite(norm)
                optimizer.step()
                assert all(parameter.grad is None for parameter in teacher.model.parameters())
                torch.testing.assert_close(teacher.model.model.early_exit_gate.weight,teacher_gate,atol=0,rtol=0)
                row=dict(update=update+1,responses=16,response_tokens=tokens,mean_loss=loss_sum/16,
                    gradient_norm=float(norm),seconds=time.perf_counter()-start,
                    gate_weight_max_change=float((student.model.model.early_exit_gate.weight.detach()-initial_gate).abs().max()))
                stats.append(row);print(json.dumps(row),flush=True)
                assert not list(args.output.glob('.vllm_weights*'))
        finally:batcher.close()
    report=dict(status='PASS',updates=stats,seed=42,global_batch=16,diagnostic_max_new_tokens=64,
        production_max_new_tokens=16384,teacher_frozen=True,weight_versions=[0,1],
        temporary_weight_files_removed=True,scope='Real Ouro weights, two optimizer updates; diagnostic short generations, not a training efficacy result.')
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
