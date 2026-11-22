"""CPU/DDP integration proof: direct10 versus target5 then resumed target10.

Uses random tiny architectures and synthetic prompts, never pretrained GPU work.
All temporary checkpoints are deleted; only lightweight verification reports stay.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(args):
    from ep_loop.config import load_config,smoke_config
    if args.variant=='fixed':
        from ep_loop.fastpath import training
        config_path=ROOT/'configs/vllm_fixed_top_n.json'
    else:
        from ep_loop import training
        config_path=ROOT/'configs/hf_adaptive.json'
    config=smoke_config(load_config(config_path))
    config['optimization'].update(updates=args.target,global_batch_size=2)
    config['rollout'].update(prompts_per_batch=2,responses_per_prompt=1,max_new_tokens=3)
    config['implementation'].update(save_every_updates=args.target,keep_latest_checkpoints=1)
    training.smoke_config=lambda unused:config
    training.train(config_path,args.work_dir,'cpu',True,resume=args.resume)


def equivalent(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    return a==b


def validate(args):
    import torch
    from safetensors.torch import load_file
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    args.report_dir.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','CPU_THREADS':'1','OMP_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TENSORBOARD':'1',
         'EP_LOOP_PROGRESS':'0','EP_LOOP_TOKEN_PROGRESS':'0','EP_LOOP_STAGE_TIMESTAMPS':'0',
         'EP_LOOP_HEARTBEAT_SECONDS':'0','PYTHONHASHSEED':'42'}
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    results=[]
    with tempfile.TemporaryDirectory(prefix='ep_loop_resume_') as directory:
        workspace=Path(directory)
        for variant in args.variants.split(','):
            base=workspace/variant;full=base/'full';split=base/'split'
            phases=[('direct10',full,10,None),('first5',split,5,None),
                    ('resume_to10',split,10,split/'checkpoints/update_000005')]
            with (args.report_dir/(variant+'.console.log')).open('w') as log:
                for label,output,target,resume in phases:
                    command=[sys.executable,'-m','torch.distributed.run','--standalone',
                        f'--nproc-per-node={args.world_size}',str(Path(__file__).resolve()),
                        '--worker','--variant',variant,'--work-dir',str(output),'--target',str(target)]
                    if resume is not None:command+=['--resume',str(resume)]
                    result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:raise RuntimeError(f'{variant}/{label} failed; see {log.name}')
                    print(json.dumps(dict(variant=variant,phase=label,status='passed')),flush=True)
            checkpoints=[folder/'checkpoints/update_000010' for folder in (full,split)]
            weights=[load_file(str(folder/'model.safetensors')) for folder in checkpoints]
            assert equivalent(*weights),'Model weights differ'
            delta=max(float((weights[0][k]-weights[1][k]).abs().max()) for k in weights[0])
            states=[torch.load(folder/'training_state.pt',map_location='cpu',weights_only=True) for folder in checkpoints]
            assert equivalent(*states),'Optimizer or training state differs'
            for rank in range(args.world_size):
                rng=[torch.load(folder/f'rng_rank{rank}.pt',map_location='cpu',weights_only=True) for folder in checkpoints]
                assert equivalent(*rng),f'Rank{rank} RNG differs'
                trajectories=[(folder/f'rollouts_rank{rank}.jsonl').read_text() for folder in (full,split)]
                assert trajectories[0]==trajectories[1],f'Rank{rank} on-policy trajectories differ'
            metrics=[[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()] for folder in (full,split)]
            assert [r['mean_response_loss'] for r in metrics[0]]==[r['mean_response_loss'] for r in metrics[1]],'Loss sequence differs'
            event_log=EventAccumulator(str(split/'tensorboard'),size_guidance={'scalars':0}).Reload()
            steps=[e.step for e in event_log.Scalars('train/loss')]
            assert steps==list(range(1,11)),steps
            result=dict(variant=variant,world_size=args.world_size,comparison='direct10 vs target5 then resumed target10',
                model_max_abs_difference=delta,optimizer_and_training_state_equal=True,
                all_rank_rng_equal=True,on_policy_trajectories_equal=True,loss_sequence_equal=True,
                tensorboard_steps=steps,scope='Random tiny CPU/DDP semantic proof; no GPU/vLLM numerical identity claim')
            results.append(result);print(json.dumps(result),flush=True)
    (args.report_dir/'validation.json').write_text(json.dumps(results,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'audit/resume_tensorboard_20261117/consistency')
    parser.add_argument('--world-size',type=int,default=2)
    parser.add_argument('--variants',default='fixed,dynamic')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--variant',choices=['fixed','dynamic'])
    parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--target',type=int)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args)
    else:validate(args)


if __name__=='__main__':main()
