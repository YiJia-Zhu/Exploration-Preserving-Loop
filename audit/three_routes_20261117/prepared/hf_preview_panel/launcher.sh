#!/usr/bin/env bash
# Three research routes. This entry defaults to hybrid; the historical
# run_ablation.sh default is retained for reproducibility of earlier runs.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
ROUTE="${ROUTE:-hybrid}"                    # hf | vllm | hybrid
GPUS="${GPUS:-2,3}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
MODEL_FAMILY="${MODEL_FAMILY:-ouro}"
MODEL_PATH="${MODEL_PATH:-models/Ouro-1.4B-Thinking}"
TEACHER_MAX_LOOPS="${TEACHER_MAX_LOOPS:-4}"
TEACHER_UPDATE="${TEACHER_UPDATE:-frozen}"
TEACHER_EXIT_THRESHOLD="${TEACHER_EXIT_THRESHOLD:-0.9}"
STUDENT_LOOPS="${STUDENT_LOOPS:-2}"           # vLLM: actual N in rollout, replay AND evaluation.
ABLATION_GROUPS="${ABLATION_GROUPS:-endpoint,uniform_n,top_n}"
STAGE_SELECTION="${STAGE_SELECTION:-threshold}" # hf/hybrid: all JS-threshold stages.
JS_THRESHOLD="${JS_THRESHOLD:-0.05}"
STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-1.0}"  # hf/hybrid; fixed-vLLM overrides to zero.
STUDENT_STOP_THRESHOLD="${STUDENT_STOP_THRESHOLD:-0.5}"
BOUNDARY_SIGNAL="${BOUNDARY_SIGNAL:-js}"
QUESTION_FILTER="${QUESTION_FILTER:-all}"
DATA_SOURCE="${DATA_SOURCE:-data/dapo_math17k/train.jsonl}"
SOURCE_EXPECTED_EXAMPLES="${SOURCE_EXPECTED_EXAMPLES:-17391}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-256}"
DEV_EXAMPLES="${DEV_EXAMPLES:-64}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-64}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
SEED="${SEED:-42}"
UPDATES="${UPDATES:-20}"
PROMPTS_PER_BATCH="${PROMPTS_PER_BATCH:-4}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-16384}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
CHECKPOINT_READOUT="${CHECKPOINT_READOUT:-1}"
READOUT_CHUNK_SIZE="${READOUT_CHUNK_SIZE:-128}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.0}"
TRAIN_TOP_P="${TRAIN_TOP_P:-1.0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_INITIAL="${EVAL_INITIAL:-1}"             # vLLM panel: base K and base N only before first group.
EVAL_UPDATES="${EVAL_UPDATES:-20}"           # Final-only by default to avoid duplicate long evaluation.
EVAL_SAMPLES_PER_PROBLEM="${EVAL_SAMPLES_PER_PROBLEM:-1}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.7}"
EVAL_STUDENT_LOOPS="${EVAL_STUDENT_LOOPS:-1,2,3,4}" # hybrid: same checkpoint, multiple fixed N.
SAVE_EVERY_UPDATES="${SAVE_EVERY_UPDATES:-10}"
KEEP_LATEST_CHECKPOINTS="${KEEP_LATEST_CHECKPOINTS:-2}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
VLLM_MEMORY_FRACTION="${VLLM_MEMORY_FRACTION:-0.45}"
VLLM_EAGER="${VLLM_EAGER:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/three_route_ablations}" # Put on a volume with enough checkpoint space.
RUN_TAG="${RUN_TAG:-}"
export PROJECT_ROOT
set +a
cd -- "$PROJECT_ROOT"
exec "$PYTHON_BIN" - "$@" <<'PY'
import argparse,json,os,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from ep_loop.observability import RunLogger,run_command

p=argparse.ArgumentParser(description='Three-route ablations; hybrid is the research default')
p.add_argument('--route',choices=['hf','vllm','hybrid'],default=os.environ['ROUTE'])
p.add_argument('--gpus',default=os.environ['GPUS'])
actions=p.add_mutually_exclusive_group()
actions.add_argument('--dry-run',action='store_true')
actions.add_argument('--check',action='store_true')
actions.add_argument('--prepare-only',action='store_true')
a=p.parse_args()
root=Path(os.environ['PROJECT_ROOT'])
gpus=[x.strip() for x in a.gpus.split(',')]
if not gpus or any(not x.isdecimal() for x in gpus) or len(gpus)!=len(set(gpus)):p.error('Distinct numeric GPUs required')
if os.environ['MODEL_FAMILY']!='ouro':p.error('The three-route entry currently targets Ouro; historical Huginn HF configs remain available')
mode={'hf':'hf_adaptive','vllm':'vllm_fixed','hybrid':'hybrid'}[a.route]
groups=[x.strip() for x in os.environ['ABLATION_GROUPS'].split(',')] if a.route=='vllm' else [os.environ['STAGE_SELECTION']]
if a.route=='vllm' and (not groups or len(set(groups))!=len(groups) or any(x not in {'endpoint','uniform_n','top_n'} for x in groups)):
    p.error('ABLATION_GROUPS must contain distinct endpoint,uniform_n,top_n choices')
if not 1<=int(os.environ['STUDENT_LOOPS'])<=int(os.environ['TEACHER_MAX_LOOPS']):p.error('Require 1 <= N <= K')
tag=os.environ['RUN_TAG'] or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+f"_{a.route}_K{os.environ['TEACHER_MAX_LOOPS']}_N{os.environ['STUDENT_LOOPS']}_seed{os.environ['SEED']}"
if Path(tag).name!=tag or tag in {'.','..'}:p.error('RUN_TAG must be a directory name')
output=(root/os.environ['OUTPUT_ROOT']).resolve()
panel=output/(tag+'_panel')
model=(root/os.environ['MODEL_PATH']).resolve()
model_bytes=sum(path.stat().st_size for path in model.glob('*.safetensors'))
updates,save,keep=(int(os.environ[key]) for key in ['UPDATES','SAVE_EVERY_UPDATES','KEEP_LATEST_CHECKPOINTS'])
if updates<1 or save<1 or keep<0:p.error('Invalid checkpoint/update counts')
retained=set(range(save,updates+1,save))|{updates}|set(range(max(1,updates-keep+1),updates+1)) if keep else set(range(save,updates+1,save))|{updates}
per_run_peak=model_bytes*(6*(min(keep,2)+2)+4)
required=per_run_peak+max(0,len(groups)-1)*len(retained)*model_bytes*6
volume=output
while not volume.exists():volume=volume.parent
free=shutil.disk_usage(volume).free
commands=[]
for index,group in enumerate(groups):
    initial=os.environ['EVAL_INITIAL'] if index==0 else '0'
    overrides=dict(EXECUTION_MODE=mode,GPUS=','.join(gpus),METHOD='ours',STAGE_SELECTION=group,
        RUN_TAG=tag+'_'+group,OUTPUT_ROOT=str(output),EVAL_INITIAL=initial,
        EVAL_INITIAL_FIXED='1' if a.route=='vllm' and initial in {'1','true'} else '0')
    if a.route=='vllm':overrides.update(STOP_LOSS_WEIGHT='0',BASELINE_STUDENT_LOOPS=os.environ['STUDENT_LOOPS'])
    entry='run_fastpath_ablation.sh' if a.route=='vllm' else 'run_ablation.sh'
    cmd=['bash',str(root/'scripts'/entry)]
    if a.dry_run:cmd+=['--dry-run']
    elif a.check:cmd+=['--check']
    elif a.prepare_only:cmd+=['--prepare-only']
    commands.append(dict(group=group,command=cmd,environment=overrides))
plan=dict(stage='ablation',research_route=a.route,execution_mode=mode,gpus=gpus,
    teacher_loops=int(os.environ['TEACHER_MAX_LOOPS']),student_loops=int(os.environ['STUDENT_LOOPS']) if a.route=='vllm' else 'dynamic_training',
    deployment_loops=os.environ['EVAL_STUDENT_LOOPS'] if a.route=='hybrid' else os.environ['STUDENT_LOOPS'] if a.route=='vllm' else 'dynamic',
    groups=groups,updates_per_group=updates,teacher_distribution='full_vocab',
    global_batch=int(os.environ['PROMPTS_PER_BATCH'])*int(os.environ['RESPONSES_PER_PROMPT']),
    train_examples=int(os.environ['TRAIN_EXAMPLES']),eval_examples=int(os.environ['EVAL_EXAMPLES']),
    max_new_tokens=int(os.environ['MAX_NEW_TOKENS']),eval_max_new_tokens=int(os.environ['EVAL_MAX_NEW_TOKENS']),
    output=str(output),panel=str(panel),checkpoint_free_gib=free/2**30,
    checkpoint_required_gib=required/2**30,checkpoint_space_ready=free>=required,commands=commands)
print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
if not a.dry_run and not a.prepare_only and free<required:
    p.error(f'Checkpoint volume has {free/2**30:.1f} GiB free; this panel needs about {required/2**30:.1f} GiB. Set OUTPUT_ROOT to a suitable volume. No inference/training started.')
if a.dry_run or a.check:
    for item in commands:
        subprocess.run(item['command'],cwd=root,env=os.environ|item['environment'],check=True)
    raise SystemExit(0)
if panel.exists():p.error('Panel already exists; choose a new RUN_TAG')
with RunLogger(panel/'panel.events.jsonl',panel/'panel.console.log','three_routes') as observer:
    (panel/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(root/'scripts/run_three_route_ablation.sh',panel/'launcher.sh')
    with observer.stage('three_route_panel',echo=True,route=a.route,groups=groups):
        for item in commands:
            run_command(item['command'],env=os.environ|item['environment'],cwd=root,logger=observer,
                        label=item['group'],log_path=panel/(item['group']+'.log'))
PY
