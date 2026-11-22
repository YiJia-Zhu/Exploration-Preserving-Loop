#!/usr/bin/env bash
# Three research routes. This entry defaults to hybrid; the historical
# run_ablation.sh default is retained for reproducibility of earlier runs.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
ROUTE="${ROUTE:-hybrid}"                    # hf | vllm | hybrid
GPUS="${GPUS:-2,3}"
CPU_THREADS="${CPU_THREADS:-16}"             # CPU threads per process, independent of GPU/rank count.
TENSORBOARD="${TENSORBOARD:-1}"             # Rank0 scalars under each run's ours/tensorboard/.
TENSORBOARD_FLUSH_SECONDS="${TENSORBOARD_FLUSH_SECONDS:-10}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-}"       # Empty: trainer environment; otherwise isolated inference environment.
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}" # Empty: auto; FLASHINFER or FLASH_ATTN.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}" # Per-engine concurrency; 0 leaves the vLLM default unchanged.
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}" # 2: two GPUs per inference engine; HF DDP still uses every GPU.
VLLM_ROLLOUT_NGRAM="${VLLM_ROLLOUT_NGRAM:-1}" # vLLM training rollouts only; never passed to evaluation actors.
MODEL_FAMILY="${MODEL_FAMILY:-ouro}"
MODEL_PATH="${MODEL_PATH:-models/Ouro-1.4B-Thinking}"
TEACHER_MAX_LOOPS="${TEACHER_MAX_LOOPS:-4}"
TEACHER_UPDATE="${TEACHER_UPDATE:-frozen}"
OPD_MODE="${OPD_MODE:-}"                    # loopopd | d_loopopd | both; empty follows TEACHER_UPDATE.
COMPARISON_ROLE="${COMPARISON_ROLE:-method_ablation}"
TEACHER_EXIT_THRESHOLD="${TEACHER_EXIT_THRESHOLD:-0.9}"
STUDENT_LOOPS="${STUDENT_LOOPS:-2}"           # vLLM: actual N in rollout, replay AND evaluation.
ABLATION_GROUPS="${ABLATION_GROUPS:-endpoint,top_n}"
TEACHER_BACKEND="${TEACHER_BACKEND:-vllm}" # vLLM route: teacher-forced all-loop prefills.
TEACHER_ENDPOINT="${TEACHER_ENDPOINT:-native}" # top_n logical endpoint; endpoint control stays K.
PHASE_OFFLOAD="${PHASE_OFFLOAD:-1}"       # Inactive HF model/optimizer on CPU; vLLM sleep between phases.
TEACHER_TRACE_DIR="${TEACHER_TRACE_DIR:-/dev/shm}" # Temporary loop states on RAM-backed tmpfs; deleted after loading.
SHALLOW_TARGET_PLACEMENT="${SHALLOW_TARGET_PLACEMENT:-prefix}" # n_t<N: supervise only first n_t student loops.
STAGE_SELECTION="${STAGE_SELECTION:-threshold}" # hf/hybrid: all JS-threshold stages.
JS_THRESHOLD="${JS_THRESHOLD:-0.05}"
STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-1.0}"  # hf/hybrid; fixed-vLLM overrides to zero.
STUDENT_STOP_THRESHOLD="${STUDENT_STOP_THRESHOLD:-0.5}"
BOUNDARY_SIGNAL="${BOUNDARY_SIGNAL:-js}"
QUESTION_FILTER="${QUESTION_FILTER:-all}"
DATA_SOURCE="${DATA_SOURCE:-data/dapo_math17k/train.jsonl}"
SOURCE_EXPECTED_EXAMPLES="${SOURCE_EXPECTED_EXAMPLES:-17391}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-all}"     # All usable source questions except the held-out dev split.
DEV_EXAMPLES="${DEV_EXAMPLES:-64}"
EVAL_SUITE="${EVAL_SUITE:-aime}"            # aime | dapo_dev (historical development evaluation).
if [[ "$EVAL_SUITE" == aime ]]; then
  EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"        # Default follows the resolved route.
  EVAL_EXAMPLES="${EVAL_EXAMPLES:-}"
  EVAL_SAMPLES_PER_PROBLEM="${EVAL_SAMPLES_PER_PROBLEM:-16}"
  TRAIN_DECONTAMINATION_PATH="${TRAIN_DECONTAMINATION_PATH:-data/aime/eval.jsonl}" # Keep the same training pool for future years.
elif [[ "$EVAL_SUITE" == dapo_dev ]]; then
  EVAL_DATA_PATH=""
  EVAL_EXAMPLES="${EVAL_EXAMPLES:-64}"
  EVAL_SAMPLES_PER_PROBLEM="${EVAL_SAMPLES_PER_PROBLEM:-1}"
else
  echo "EVAL_SUITE must be aime or dapo_dev" >&2; exit 2
fi
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
SEED="${SEED:-42}"
UPDATES="${UPDATES:-100}"
PROMPTS_PER_BATCH="${PROMPTS_PER_BATCH:-32}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}"
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
EVAL_UPDATES="${EVAL_UPDATES:-${UPDATES}}"   # Final-only by default to avoid duplicate long evaluation.
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.7}"
EVAL_STUDENT_LOOPS="${EVAL_STUDENT_LOOPS:-1,2,3,4}" # hybrid: same checkpoint, multiple fixed N.
if [[ "$ROUTE" == vllm ]]; then
  SAVE_EVERY_UPDATES="${SAVE_EVERY_UPDATES:-5}"
  KEEP_LATEST_CHECKPOINTS="${KEEP_LATEST_CHECKPOINTS:-0}"
  CHECKPOINT_LAYOUT="${CHECKPOINT_LAYOUT:-model_only}"
  SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-0}"
  SAVE_FINAL_RESUME="${SAVE_FINAL_RESUME:-1}"
  EVAL_EVERY_UPDATES="${EVAL_EVERY_UPDATES:-5}"
else
  SAVE_EVERY_UPDATES="${SAVE_EVERY_UPDATES:-${UPDATES}}"
  KEEP_LATEST_CHECKPOINTS="${KEEP_LATEST_CHECKPOINTS:-1}"
  CHECKPOINT_LAYOUT="${CHECKPOINT_LAYOUT:-full}"
  SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-1}"
  SAVE_FINAL_RESUME="${SAVE_FINAL_RESUME:-0}"
  EVAL_EVERY_UPDATES="${EVAL_EVERY_UPDATES:-0}"
fi
RESUME_SAVE_EVERY_UPDATES="${RESUME_SAVE_EVERY_UPDATES:-${SAVE_EVERY_UPDATES}}"
VLLM_MEMORY_FRACTION="${VLLM_MEMORY_FRACTION:-0.75}"
VLLM_EAGER="${VLLM_EAGER:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/three_route_ablations}" # Put on a volume with enough checkpoint space.
RUN_TAG="${RUN_TAG:-}"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-10}" # Includes setup, generation, training and evaluation; 0 disables.
export PROJECT_ROOT
set +a
cd -- "$PROJECT_ROOT"
exec "$PYTHON_BIN" - "$@" <<'PY'
import argparse,json,os,shutil,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from ep_loop.cpu_runtime import cpu_thread_environment
os.environ.update(cpu_thread_environment())
from ep_loop.observability import RunLogger,run_command
started_monotonic=time.monotonic()

p=argparse.ArgumentParser(description='Three-route ablations; hybrid is the research default')
p.add_argument('--route',choices=['hf','vllm','hybrid'],default=os.environ['ROUTE'])
p.add_argument('--gpus',default=os.environ['GPUS'])
actions=p.add_mutually_exclusive_group()
actions.add_argument('--dry-run',action='store_true')
actions.add_argument('--check',action='store_true')
actions.add_argument('--prepare-only',action='store_true')
a=p.parse_args()
if os.environ['EVAL_SUITE']=='aime':
    if not os.environ['EVAL_DATA_PATH']:
        os.environ['EVAL_DATA_PATH']='data/aime/eval.jsonl' if a.route=='vllm' else 'data/aime/aime2024.jsonl'
    if not os.environ['EVAL_EXAMPLES']:
        is_combined_aime=Path(os.environ['EVAL_DATA_PATH']).parts[-3:]==('data','aime','eval.jsonl')
        os.environ['EVAL_EXAMPLES']='90' if is_combined_aime else '30'
max_hours=float(os.environ['MAX_RUNTIME_HOURS'])
if not 0<=max_hours<float('inf'):p.error('MAX_RUNTIME_HOURS must be finite and nonnegative')
root=Path(os.environ['PROJECT_ROOT'])
gpus=[x.strip() for x in a.gpus.split(',')]
if not gpus or any(not x.isdecimal() for x in gpus) or len(gpus)!=len(set(gpus)):p.error('Distinct numeric GPUs required')
inference_tp=int(os.environ['VLLM_TENSOR_PARALLEL_SIZE'])
if inference_tp<1 or len(gpus)%inference_tp:p.error('GPU count must be divisible by positive VLLM_TENSOR_PARALLEL_SIZE')
if a.route!='vllm' and inference_tp!=1:p.error('Paired TP training currently supports the fixed-vLLM route')
if os.environ['MODEL_FAMILY']!='ouro':p.error('The three-route entry currently targets Ouro; historical Huginn HF configs remain available')
if os.environ['TEACHER_UPDATE'] not in {'frozen','current'}:p.error('TEACHER_UPDATE must be frozen or current')
opd_mode=os.environ['OPD_MODE'] or ('d_loopopd' if os.environ['TEACHER_UPDATE']=='current' else 'loopopd')
if opd_mode not in {'loopopd','d_loopopd','both'}:p.error('OPD_MODE must be loopopd, d_loopopd or both')
opd_modes=['loopopd','d_loopopd'] if opd_mode=='both' else [opd_mode]
teacher_updates={'loopopd':'frozen','d_loopopd':'current'}
mode={'hf':'hf_adaptive','vllm':'vllm_fixed','hybrid':'hybrid'}[a.route]
groups=[x.strip() for x in os.environ['ABLATION_GROUPS'].split(',')] if a.route=='vllm' else [os.environ['STAGE_SELECTION']]
if a.route=='vllm' and (not groups or len(set(groups))!=len(groups) or any(x not in {'endpoint','uniform_n','top_n'} for x in groups)):
    p.error('ABLATION_GROUPS must contain distinct endpoint,uniform_n,top_n choices')
if not 1<=int(os.environ['STUDENT_LOOPS'])<=int(os.environ['TEACHER_MAX_LOOPS']):p.error('Require 1 <= N <= K')
run_groups=[(opd,group) for opd in opd_modes for group in groups]
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
required=per_run_peak+max(0,len(run_groups)-1)*len(retained)*model_bytes*6
if os.environ['CHECKPOINT_LAYOUT']=='split':
    if a.route!='vllm':p.error('Split checkpoints require the vLLM training route')
    snapshots=len(set(range(save,updates+1,save))|{updates})
    required=model_bytes*(2*snapshots*len(run_groups)+8+2*len(gpus))
elif os.environ['CHECKPOINT_LAYOUT']=='model_only':
    snapshots=len(set(range(save,updates+1,save))|{updates})
    final_resume_bytes=4 if os.environ['SAVE_FINAL_RESUME'].lower() in {'1','true'} else 0
    required=model_bytes*((2*snapshots+final_resume_bytes)*len(run_groups)+2*len(gpus))
volume=output
while not volume.exists():volume=volume.parent
free=shutil.disk_usage(volume).free
commands=[]
for index,(opd,group) in enumerate(run_groups):
    initial=os.environ['EVAL_INITIAL'] if index==0 else '0'
    label=f'{opd}_{group}' if len(opd_modes)>1 else group
    overrides=dict(EXECUTION_MODE=mode,GPUS=','.join(gpus),
        METHOD='baseline' if a.route=='vllm' and group=='endpoint' else 'ours',STAGE_SELECTION=group,
        TEACHER_UPDATE=teacher_updates[opd],
        RUN_TAG=tag+'_'+label,OUTPUT_ROOT=str(output),EVAL_INITIAL=initial,
        EVAL_INITIAL_FIXED='1' if a.route=='vllm' and initial in {'1','true'} else '0')
    if a.route=='vllm':overrides.update(STOP_LOSS_WEIGHT='0',BASELINE_STUDENT_LOOPS=os.environ['STUDENT_LOOPS'],
        VLLM_ROLLOUT_NGRAM=os.environ['VLLM_ROLLOUT_NGRAM'])
    entry='run_fastpath_ablation.sh' if a.route=='vllm' else 'run_ablation.sh'
    cmd=['bash',str(root/'scripts'/entry)]
    if a.dry_run:cmd+=['--dry-run']
    elif a.check:cmd+=['--check']
    elif a.prepare_only:cmd+=['--prepare-only']
    commands.append(dict(group=label,stage_selection=group,opd_mode=opd,
                         teacher_update=teacher_updates[opd],command=cmd,environment=overrides))
plan=dict(stage='ablation',research_route=a.route,execution_mode=mode,gpus=gpus,
    comparison_role=os.environ['COMPARISON_ROLE'],opd_modes=opd_modes,
    teacher_updates={opd:teacher_updates[opd] for opd in opd_modes},
    warmup_steps={opd:(10 if opd=='d_loopopd' else 0) if os.environ.get('WARMUP_STEPS','auto')=='auto'
                  else int(os.environ['WARMUP_STEPS']) for opd in opd_modes},
    vllm_max_num_seqs=int(os.environ['VLLM_MAX_NUM_SEQS']) or None,
    vllm_rollout_ngram=a.route=='vllm' and os.environ['VLLM_ROLLOUT_NGRAM'].lower() in {'1','true'},
    evaluation_speculative_config=None,
    checkpoint_layout=os.environ['CHECKPOINT_LAYOUT'],save_every_updates=save,
    resume_save_every_updates=int(os.environ['RESUME_SAVE_EVERY_UPDATES']),
    save_optimizer=os.environ['SAVE_OPTIMIZER'].lower() in {'1','true'},
    save_final_resume=os.environ['SAVE_FINAL_RESUME'].lower() in {'1','true'},
    eval_every_updates=int(os.environ['EVAL_EVERY_UPDATES']) if a.route=='vllm' and os.environ['RUN_EVAL'].lower() in {'1','true'} else 0,
    vllm_tensor_parallel_size=inference_tp,inference_replicas=len(gpus)//inference_tp,
    inference_gpu_groups=[gpus[start:start+inference_tp] for start in range(0,len(gpus),inference_tp)],
    hf_training_world_size=len(gpus),
    cpu_threads_per_process=int(os.environ['CPU_THREADS']),
    max_runtime_hours=max_hours,
    teacher_backend=os.environ['TEACHER_BACKEND'] if a.route=='vllm' else 'hf',
    phase_offload=os.environ['PHASE_OFFLOAD'].lower() in {'1','true'} if a.route=='vllm' else False,
    teacher_logical_endpoints={item['group']:'fixed' if item['stage_selection'] in {'endpoint','uniform_n'} else os.environ['TEACHER_ENDPOINT']
                              for item in commands} if a.route=='vllm' else 'route_native',
    teacher_loops=int(os.environ['TEACHER_MAX_LOOPS']),student_loops=int(os.environ['STUDENT_LOOPS']) if a.route=='vllm' else 'dynamic_training',
    deployment_loops=os.environ['EVAL_STUDENT_LOOPS'] if a.route=='hybrid' else os.environ['STUDENT_LOOPS'] if a.route=='vllm' else 'dynamic',
    groups=[item['group'] for item in commands],stage_selection_groups=groups,
    updates_per_group=updates,teacher_distribution='full_vocab',
    global_batch=int(os.environ['PROMPTS_PER_BATCH'])*int(os.environ['RESPONSES_PER_PROMPT']),
    train_examples='all' if os.environ['TRAIN_EXAMPLES']=='all' else int(os.environ['TRAIN_EXAMPLES']),
    prompts_per_batch=int(os.environ['PROMPTS_PER_BATCH']),responses_per_prompt=int(os.environ['RESPONSES_PER_PROMPT']),
    target_rollout_responses=updates*int(os.environ['PROMPTS_PER_BATCH'])*int(os.environ['RESPONSES_PER_PROMPT']),
    target_question_draws=updates*int(os.environ['PROMPTS_PER_BATCH']),
    teacher_trace_dir=os.environ['TEACHER_TRACE_DIR'] if a.route=='vllm' else None,
    eval_suite=os.environ['EVAL_SUITE'],eval_data_path=os.environ['EVAL_DATA_PATH'],
    train_decontamination_path=os.environ.get('TRAIN_DECONTAMINATION_PATH') if os.environ['EVAL_SUITE']=='aime' else None,
    eval_examples=int(os.environ['EVAL_EXAMPLES']),eval_samples_per_problem=int(os.environ['EVAL_SAMPLES_PER_PROBLEM']),
    eval_responses_per_checkpoint=int(os.environ['EVAL_EXAMPLES'])*int(os.environ['EVAL_SAMPLES_PER_PROBLEM']),
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
    if max_hours and not a.prepare_only:
        with (panel/'time_budget.console.log').open('ab') as budget_log:
            watchdog=subprocess.Popen([sys.executable,'-u',str(root/'scripts/watch_run_budget.py'),
                '--pid',str(os.getpid()),'--hours',str(max_hours),'--started-monotonic',str(started_monotonic),
                '--panel',str(panel),'--output-root',str(output),'--run-prefix',tag],
                cwd=root,stdin=subprocess.DEVNULL,stdout=budget_log,stderr=subprocess.STDOUT,start_new_session=True)
        (panel/'time_budget.pid').write_text(str(watchdog.pid)+'\n')
    with observer.stage('three_route_panel',echo=True,route=a.route,groups=groups):
        for item in commands:
            status=dict(group=item['group'],status='running',
                        output=str(output/item['environment']['RUN_TAG']),
                        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
            temporary=panel/'current_group.tmp'
            temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(panel/'current_group.json')
            try:
                run_command(item['command'],env=os.environ|item['environment'],cwd=root,logger=observer,
                            label=item['group'],log_path=panel/(item['group']+'.log'))
            except subprocess.CalledProcessError as error:
                status.update(status='failed',returncode=error.returncode)
                temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(panel/'current_group.json')
                raise
            status.update(status='prepared' if a.prepare_only else 'completed',
                          finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
            temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(panel/'current_group.json')
PY
