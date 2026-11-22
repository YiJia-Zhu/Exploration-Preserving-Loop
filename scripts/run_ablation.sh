#!/usr/bin/env bash
# Ouro/Huginn method ablations. Edit parameters here, or override them via env.
# Current stage: choose the method on development data; parameter sweeps and
# formal baseline comparisons follow after the method is fixed.
# Examples:
#   bash scripts/run_ablation.sh --dry-run
#   GPUS=2,3 bash scripts/run_ablation.sh
#   GPUS=7 STAGE_SELECTION=threshold_uniform bash scripts/run_ablation.sh
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER_PATH="${PROJECT_ROOT}/scripts/$(basename -- "${BASH_SOURCE[0]}")"

# ==================== Editable experiment parameters ====================
set -a
EXECUTION_MODE="${EXECUTION_MODE:-hf_dynamic}" # Default: full HF dynamic; optional: vllm_fixed | vllm_top20 | hf_top20.
GPUS="${GPUS:-2,3}"                       # Physical GPU IDs; one process per GPU.
CPU_THREADS="${CPU_THREADS:-16}"          # CPU threads per process, independent of GPU/rank count.
TENSORBOARD="${TENSORBOARD:-1}"             # Rank0 completed-update scalars.
TENSORBOARD_FLUSH_SECONDS="${TENSORBOARD_FLUSH_SECONDS:-10}"
METHOD="${METHOD:-ours}"                  # ours (default) | both | baseline | recompute
# baseline/both are optional LoopOPD controls within this ablation stage.
MODEL_FAMILY="${MODEL_FAMILY:-ouro}"      # ouro | huginn
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" # Reduce fragmentation between long responses.

# Model defaults depend on family. MODEL_PATH/TEACHER_MAX_LOOPS may be overridden.
case "$MODEL_FAMILY" in
  ouro)
    MODEL_PATH="${MODEL_PATH:-models/Ouro-1.4B-Thinking}"
    TEACHER_MAX_LOOPS="${TEACHER_MAX_LOOPS:-4}"
    BASELINE_STUDENT_LOOPS="${BASELINE_STUDENT_LOOPS:-2}"
    TEACHER_EXIT_THRESHOLD="${TEACHER_EXIT_THRESHOLD:-0.9}"
    READOUT_CHUNK_SIZE="${READOUT_CHUNK_SIZE:-128}"
    ;;
  huginn)
    MODEL_PATH="${MODEL_PATH:-models/huginn-0125}"
    TEACHER_MAX_LOOPS="${TEACHER_MAX_LOOPS:-32}"
    BASELINE_STUDENT_LOOPS="${BASELINE_STUDENT_LOOPS:-8}"
    TEACHER_EXIT_THRESHOLD="${TEACHER_EXIT_THRESHOLD:-0.0005}"
    READOUT_CHUNK_SIZE="${READOUT_CHUNK_SIZE:-16}"
    ;;
  *) echo "MODEL_FAMILY must be ouro or huginn" >&2; exit 2 ;;
esac
MODEL_REVISION="${MODEL_REVISION:-}"       # Empty: use the family's pinned revision.
TEACHER_UPDATE="${TEACHER_UPDATE:-frozen}" # frozen | current
JS_THRESHOLD="${JS_THRESHOLD:-0.05}"       # Natural-log JS; development value.
if [[ "$EXECUTION_MODE" == vllm_fixed ]]; then
  STAGE_SELECTION="${STAGE_SELECTION:-top_n}"
else
  STAGE_SELECTION="${STAGE_SELECTION:-threshold}" # E/D/C dynamic phase targets.
fi
BOUNDARY_SIGNAL="${BOUNDARY_SIGNAL:-js}"   # js|max_probability|normalized_entropy|probe_know
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.05}" # Absolute confidence change; tune separately.
# 教师正确性筛选消融：all=默认不过滤；teacher_correct=仅保留初始教师答对的训练题目。
# random_matched=同数量随机题目对照。学生生成的回答不按正确性过滤。
QUESTION_FILTER="${QUESTION_FILTER:-all}"
FILTER_REQUIRE_EOS="${FILTER_REQUIRE_EOS:-1}" # 仅开启教师筛选时生效：要求教师完整回答。
STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-1.0}"
STUDENT_STOP_THRESHOLD="${STUDENT_STOP_THRESHOLD:-0.5}"
BASELINE_USE_CACHE="${BASELINE_USE_CACHE:-1}"
OURS_USE_CACHE="${OURS_USE_CACHE:-1}"         # Exact lazy depth KV; 0: old recompute diagnostic.
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"         # Rank 0 training/evaluation progress.
TOKEN_PROGRESS="${TOKEN_PROGRESS:-1}"       # One token bar per generated response.
PROGRESS_MIN_INTERVAL="${PROGRESS_MIN_INTERVAL:-1.0}" # Seconds between refreshes.
LOG_TIMEZONE="${LOG_TIMEZONE:-Asia/Shanghai}"
LOG_STAGE_TIMESTAMPS="${LOG_STAGE_TIMESTAMPS:-1}" # Absolute start/end time for each stage.
LOG_HEARTBEAT_SECONDS="${LOG_HEARTBEAT_SECONDS:-30}" # Periodic time/progress records; 0 disables heartbeat.

# Input must be an already prepared local JSONL; this script never downloads data.
DATA_SOURCE="${DATA_SOURCE:-data/dapo_math17k/train.jsonl}"
SOURCE_EXPECTED_EXAMPLES="${SOURCE_EXPECTED_EXAMPLES:-17391}" # Public release: 17,398 unique prompts minus 7 conflicting labels.
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-all}"     # All usable source questions except the held-out dev split.
DEV_EXAMPLES="${DEV_EXAMPLES:-64}"
EVAL_SUITE="${EVAL_SUITE:-aime}"            # aime | dapo_dev.
if [[ "$EVAL_SUITE" == aime ]]; then
  EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/aime/aime2024.jsonl}"
  EVAL_EXAMPLES="${EVAL_EXAMPLES:-30}"
  EVAL_SAMPLES_PER_PROBLEM="${EVAL_SAMPLES_PER_PROBLEM:-16}"
  TRAIN_DECONTAMINATION_PATH="${TRAIN_DECONTAMINATION_PATH:-data/aime/eval.jsonl}"
  SPLIT_EVAL_EXAMPLES="$DEV_EXAMPLES"
  EVAL_SPLIT_TAG="_aime${EVAL_EXAMPLES}_clean"
elif [[ "$EVAL_SUITE" == dapo_dev ]]; then
  EVAL_DATA_PATH=""
  EVAL_EXAMPLES="${EVAL_EXAMPLES:-64}"
  EVAL_SAMPLES_PER_PROBLEM="${EVAL_SAMPLES_PER_PROBLEM:-1}"
  SPLIT_EVAL_EXAMPLES="$EVAL_EXAMPLES"
  EVAL_SPLIT_TAG=""
else
  echo "EVAL_SUITE must be aime or dapo_dev" >&2; exit 2
fi
SEED="${SEED:-42}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
SPLIT_DIR="${SPLIT_DIR:-data/dapo_ablation/${MODEL_FAMILY}_seed${SEED}_train${TRAIN_EXAMPLES}_dev${DEV_EXAMPLES}_eval${SPLIT_EVAL_EXAMPLES}_prompt${MAX_PROMPT_TOKENS}${EVAL_SPLIT_TAG}}"

UPDATES="${UPDATES:-100}"
PROMPTS_PER_BATCH="${PROMPTS_PER_BATCH:-32}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}" # Global batch is their product.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
WARMUP_STEPS="${WARMUP_STEPS:-auto}"        # auto: frozen=0, current=10.
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-none}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"   # bf16 | none; master params are FP32.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
CHECKPOINT_READOUT="${CHECKPOINT_READOUT:-1}"
SAVE_EVERY_UPDATES="${SAVE_EVERY_UPDATES:-${UPDATES}}"
KEEP_LATEST_CHECKPOINTS="${KEEP_LATEST_CHECKPOINTS:-1}" # Rotate one full recovery point; permanently retain the final snapshot.
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-21600}" # Allow long, unequal on-policy rollouts across ranks.
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.0}"
TRAIN_TOP_P="${TRAIN_TOP_P:-1.0}"

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_INITIAL="${EVAL_INITIAL:-1}"           # Initial native dynamic reference.
EVAL_INITIAL_FIXED="${EVAL_INITIAL_FIXED:-0}" # Optional fixed-depth reference curve.
EVAL_UPDATES="${EVAL_UPDATES:-${UPDATES}}"  # Final-only by default.
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-16384}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.7}"
EVAL_MATCHED_CACHE="${EVAL_MATCHED_CACHE:-0}" # Optional old baseline recompute diagnostic.
EVAL_STUDENT_LOOPS="${EVAL_STUDENT_LOOPS:-1,2,3,4}" # Hybrid: fixed deployment budgets, one trained checkpoint.

if [[ "$EXECUTION_MODE" == hf_dynamic ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/ablations}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/ablations_fastpath}"
fi
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_${EXECUTION_MODE}_${MODEL_FAMILY}_${STAGE_SELECTION}_${BOUNDARY_SIGNAL}_${QUESTION_FILTER}}"
set +a
# ========================================================================

case "$EXECUTION_MODE" in
  hf_dynamic|hf_adaptive|hybrid) ;;
  vllm_fixed|vllm_top20|hf_top20) exec bash "$PROJECT_ROOT/scripts/run_fastpath_ablation.sh" "$@" ;;
  *) echo "EXECUTION_MODE must be hf_dynamic (default), vllm_fixed, vllm_top20 or hf_top20" >&2; exit 2 ;;
esac

if [[ ! "$SEED" =~ ^[0-9]{1,10}$ ]] || (( 10#$SEED >= 4294967296 )); then
  echo "SEED must be an integer in [0, 2**32); default is 42." >&2
  exit 2
fi
# Must be set before Python starts; do not enable strict deterministic GPU modes.
export PYTHONHASHSEED="$SEED"
export EP_LOOP_PROGRESS="$SHOW_PROGRESS" EP_LOOP_TOKEN_PROGRESS="$TOKEN_PROGRESS"
export EP_LOOP_PROGRESS_MIN_INTERVAL="$PROGRESS_MIN_INTERVAL" EP_LOOP_LOG_TIMEZONE="$LOG_TIMEZONE"
export EP_LOOP_STAGE_TIMESTAMPS="$LOG_STAGE_TIMESTAMPS" EP_LOOP_HEARTBEAT_SECONDS="$LOG_HEARTBEAT_SECONDS"
export PROJECT_ROOT LAUNCHER_PATH
cd -- "$PROJECT_ROOT"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN. Set PYTHON_BIN to your project interpreter." >&2
  exit 2
fi

# Replace Bash so it cannot reread a changed script offset after the long run.
exec "$PYTHON_BIN" - "$@" <<'PY'
import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time

from ep_loop.cpu_runtime import cpu_thread_environment
os.environ.update(cpu_thread_environment())
from ep_loop.config import assets_status, load_config, merge, validate_config
from ep_loop.data import normalize_record, prompt_ids, split_training_records, PROMPT_SAMPLING
from ep_loop.evaluation_data import load_evaluation_records, exclude_evaluation_overlaps
from ep_loop.reproducibility import SAMPLE_SEED_SCHEME, seed_everything
from ep_loop.observability import RunLogger, run_command, utc_now

launcher_started_at = utc_now()
observer_stack = ExitStack()

parser = argparse.ArgumentParser(description="Prepare and run method ablations on development data; no downloads or automatic parameter sweep.")
action = parser.add_mutually_exclusive_group()
action.add_argument('--dry-run', action='store_true', help='Print parameters/commands; no writes, model loads or training')
action.add_argument('--check', action='store_true', help='Read-only model/data preflight; no model weights or GPU allocation')
action.add_argument('--prepare-only', action='store_true', help='Create CPU-tokenized data splits and run configs; no training')
parser.add_argument('--method', choices=['both', 'baseline', 'ours', 'recompute'], default=os.environ['METHOD'],
                    help='Default: ours. baseline/both explicitly add a LoopOPD ablation control')
parser.add_argument('--gpus', default=os.environ['GPUS'])
args = parser.parse_args()
root = Path(os.environ['PROJECT_ROOT'])

def fail(message):
    raise ValueError(message)

def integer(name, minimum=1):
    value = int(os.environ[name])
    if value < minimum: fail(f'{name} must be >= {minimum}')
    return value

def number(name):
    value = float(os.environ[name])
    if not math.isfinite(value): fail(f'{name} must be finite')
    return value

def boolean(name):
    value = os.environ[name].lower()
    if value not in {'0', '1', 'false', 'true'}: fail(f'{name} must be 0/1 or false/true')
    return value in {'1', 'true'}

def absolute(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root/path).resolve()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

try:
    gpus = [value.strip() for value in args.gpus.split(',')]
    if not gpus or any(not value.isdecimal() for value in gpus) or len(set(map(int,gpus))) != len(gpus):
        fail('GPUS must contain distinct numeric GPU IDs, for example 2,3')
    gpus = [str(int(value)) for value in gpus]
    methods = ['baseline', 'ours'] if args.method == 'both' else [args.method]
    if any(method not in {'baseline','ours','recompute'} for method in methods): fail('Invalid METHOD')
    family = os.environ['MODEL_FAMILY']
    execution_mode = os.environ['EXECUTION_MODE']
    if execution_mode=='hf_adaptive' and family!='ouro': fail('The streamed native-CDF teacher currently supports Ouro only')
    stage_selection = os.environ['STAGE_SELECTION']
    if stage_selection not in {'threshold','threshold_uniform','threshold_endpoint'}:
        fail('STAGE_SELECTION must be threshold, threshold_uniform or threshold_endpoint; all keep learned dynamic stopping')
    teacher_update = os.environ['TEACHER_UPDATE']
    if teacher_update not in {'frozen','current'}: fail('TEACHER_UPDATE must be frozen or current')
    prefix = '' if family == 'ouro' else 'huginn_'
    source = absolute(os.environ['DATA_SOURCE'])
    split = absolute(os.environ['SPLIT_DIR'])
    eval_suite = os.environ['EVAL_SUITE']
    eval_path = absolute(os.environ['EVAL_DATA_PATH']) if eval_suite=='aime' else split/'eval.jsonl'
    tag = os.environ['RUN_TAG']
    if not tag or Path(tag).name != tag or tag in {'.','..'}: fail('RUN_TAG must be a directory name')
    run = absolute(os.environ['OUTPUT_ROOT'])/tag
    train_count = None if os.environ['TRAIN_EXAMPLES']=='all' else integer('TRAIN_EXAMPLES')
    dev_count = integer('DEV_EXAMPLES')
    eval_count = integer('EVAL_EXAMPLES')
    seed,updates = integer('SEED',0),integer('UPDATES')
    if number('LOG_HEARTBEAT_SECONDS') < 0: fail('LOG_HEARTBEAT_SECONDS must be nonnegative')
    prompts,responses = integer('PROMPTS_PER_BATCH'),integer('RESPONSES_PER_PROMPT')
    if prompts % len(gpus): fail('PROMPTS_PER_BATCH must be divisible by the number of GPUs')
    run_eval = boolean('RUN_EVAL')
    split_eval_count = integer('SPLIT_EVAL_EXAMPLES')
    if split_eval_count > dev_count: fail('DAPO split evaluation size exceeds DEV_EXAMPLES')
    warmup = (10 if teacher_update == 'current' else 0) if os.environ['WARMUP_STEPS'] == 'auto' else integer('WARMUP_STEPS',0)
    clip = None if os.environ['GRADIENT_CLIP_NORM'].lower() == 'none' else number('GRADIENT_CLIP_NORM')
    if clip is not None and clip <= 0: fail('GRADIENT_CLIP_NORM must be positive or none')
    if os.environ['MIXED_PRECISION'] not in {'bf16','none'}: fail('MIXED_PRECISION must be bf16 or none')
    betas = [number('ADAM_BETA1'),number('ADAM_BETA2')]
    if any(not 0 <= beta < 1 for beta in betas) or number('ADAM_EPS') <= 0: fail('Invalid AdamW betas/eps')
    if number('STOP_LOSS_WEIGHT') <= 0 or not 0 < number('STUDENT_STOP_THRESHOLD') <= 1: fail('Invalid student stop loss/threshold')
    save_every = integer('SAVE_EVERY_UPDATES')
    if os.environ['EVAL_UPDATES'] == 'auto':
        saved = list(range(save_every,updates,save_every))
        midpoint = min(saved,key=lambda step:abs(step-updates/2)) if saved else None
        eval_steps = sorted({updates} | ({midpoint} if midpoint is not None else set()))
    else:
        eval_steps = sorted({int(value.strip()) for value in os.environ['EVAL_UPDATES'].split(',')})
    if run_eval and any(step < 1 or step > updates or (step != updates and step % save_every) for step in eval_steps):
        fail('EVAL_UPDATES must refer to saved checkpoints within UPDATES')
    selected_model = dict(path=str(absolute(os.environ['MODEL_PATH'])))
    if os.environ['MODEL_REVISION']: selected_model['revision'] = os.environ['MODEL_REVISION']
    common = dict(
        experiment=dict(stage='ablation',purpose='method_selection_on_development_data',
            comparison_role=os.environ.get('COMPARISON_ROLE','method_ablation'),
            opd_mode='d_loopopd' if teacher_update=='current' else 'loopopd'),
        model=selected_model,
        data=dict(train_path=str(split/'train.jsonl'),expected_examples=train_count,
                  max_prompt_tokens=integer('MAX_PROMPT_TOKENS'),correctness_cache=None),
        algorithm=dict(teacher_update=teacher_update,teacher_max_loops=integer('TEACHER_MAX_LOOPS'),
            teacher_distribution='full_vocab',teacher_top_k=None),
        optimization=dict(updates=updates,global_batch_size=prompts*responses,
            learning_rate=number('LEARNING_RATE'),weight_decay=number('WEIGHT_DECAY'),betas=betas,
            eps=number('ADAM_EPS'),warmup_steps=warmup,gradient_clip_norm=clip,
            mixed_precision=os.environ['MIXED_PRECISION'],gradient_checkpointing=boolean('GRADIENT_CHECKPOINTING')),
        rollout=dict(prompts_per_batch=prompts,responses_per_prompt=responses,
            max_new_tokens=integer('MAX_NEW_TOKENS'),temperature=number('TRAIN_TEMPERATURE'),top_p=number('TRAIN_TOP_P')),
        evaluation=dict(max_new_tokens=integer('EVAL_MAX_NEW_TOKENS'),
            suite=eval_suite,data_path=str(eval_path),expected_examples=eval_count,
            scope='AIME2024/2025/2026 integer-answer evaluation' if eval_suite=='aime' else 'Held-out DAPO development evaluation',
            samples_per_problem=integer('EVAL_SAMPLES_PER_PROBLEM'),temperature=number('EVAL_TEMPERATURE'),top_p=number('EVAL_TOP_P')),
        implementation=dict(readout_chunk_size=integer('READOUT_CHUNK_SIZE'),
            execution_mode=execution_mode,rollout_backend='hf',
            checkpoint_readout=boolean('CHECKPOINT_READOUT'),save_every_updates=save_every,
            keep_latest_checkpoints=integer('KEEP_LATEST_CHECKPOINTS',0),
            distributed_timeout_seconds=integer('DISTRIBUTED_TIMEOUT_SECONDS')),seed=seed)
    if os.environ['QUESTION_FILTER'] not in {'all','teacher_correct','random_matched'}:fail('Invalid QUESTION_FILTER')
    filter_path=run/'teacher_correctness.jsonl'
    common['distillation']=dict(question_filter=os.environ['QUESTION_FILTER'],
        require_eos=boolean('FILTER_REQUIRE_EOS'))
    if os.environ['QUESTION_FILTER']!='all':
        common['data']['correctness_cache']=str(filter_path)
    configs = {}
    for method in sorted(set(methods) | {'baseline','reference'}):
        filename = prefix+('e_dynamic_js.json' if method == 'ours' else 'native_base.json' if method == 'reference' else 'a_loopopd.json')
        config = merge(load_config(root/'configs'/filename),common)
        config['run_name'] = f'{tag}_{method}'
        config['paper']['status'] = 'Method ablation on development data; not a parameter sweep or formal baseline benchmark'
        if method == 'ours':
            if config['algorithm'].get('student_policy') != 'learned' or config['algorithm'].get('student_loops') is not None:
                fail('Our main method requires learned per-token stopping, with no fixed student_loops')
            if config['evaluation'].get('student_policy') != 'learned':
                fail('Our main evaluation must use the learned stopping policy')
            config['algorithm'].update(student_max_loops=integer('TEACHER_MAX_LOOPS'),
                stage_selection=stage_selection,
                exit_threshold=number('TEACHER_EXIT_THRESHOLD'),js_threshold=number('JS_THRESHOLD'),
                boundary_signal=os.environ['BOUNDARY_SIGNAL'],confidence_threshold=number('CONFIDENCE_THRESHOLD'),
                stop_loss_weight=number('STOP_LOSS_WEIGHT'),student_stop_threshold=number('STUDENT_STOP_THRESHOLD'))
            config['evaluation']['student_loops'] = [integer('TEACHER_MAX_LOOPS')]
            if execution_mode=='hf_adaptive':
                config['algorithm']['teacher_endpoint']='native_adaptive'
            elif execution_mode=='hybrid':
                budgets=sorted({int(value.strip()) for value in os.environ['EVAL_STUDENT_LOOPS'].split(',')})
                if not budgets or any(not 1<=depth<=integer('TEACHER_MAX_LOOPS') for depth in budgets):
                    fail('Hybrid evaluation budgets must lie in [1,K]')
                config['evaluation'].update(student_policy='fixed',student_loops=budgets)
            config['rollout']['use_cache'] = boolean('OURS_USE_CACHE')
            config['evaluation']['use_cache'] = boolean('OURS_USE_CACHE')
        elif method == 'reference':
            config['algorithm'].update(student_max_loops=integer('TEACHER_MAX_LOOPS'),exit_threshold=number('TEACHER_EXIT_THRESHOLD'))
            config['evaluation'].update(student_loops=[integer('TEACHER_MAX_LOOPS')],use_cache=True)
            if execution_mode=='hf_adaptive':
                config['algorithm']['student_policy']='native_adaptive'
                config['evaluation']['student_policy']='native_adaptive'
        else:
            config['algorithm']['student_loops'] = integer('BASELINE_STUDENT_LOOPS')
            config['rollout']['use_cache'] = method != 'recompute' and boolean('BASELINE_USE_CACHE')
            config['evaluation'].update(student_loops=[integer('BASELINE_STUDENT_LOOPS')],use_cache=config['rollout']['use_cache'])
        validate_config(config)
        configs[method] = config
    # Shared pool is always selected by the same INITIAL native teacher,
    # regardless of whether B uses a fixed teacher endpoint during OPD.
    if os.environ['QUESTION_FILTER']!='all':
        from ep_loop.data import filter_policy
        shared_policy=filter_policy(configs['reference'])
        for method in methods:
            if filter_policy(configs[method])!=shared_policy:
                fail('Teacher-correct pool requires identical native teacher sampling/exit settings across groups')
    configs['baseline_matched_cache'] = copy.deepcopy(configs['baseline'])
    configs['baseline_matched_cache']['evaluation']['use_cache'] = False
    plan = dict(experiment_stage='ablation',purpose='method_selection_on_development_data',
        cpu_threads_per_process=int(os.environ['CPU_THREADS']),
        execution_mode=execution_mode,rollout_backend='hf',evaluation_backend='vllm' if execution_mode=='hybrid' else 'hf',
        teacher_distribution='full_vocab',teacher_top_k=None,
        learned_student_execution='actual_early_stop_with_exact_lazy_depth_cache',
        teacher_execution='actual_cdf_early_stop_with_exact_backfill' if execution_mode=='hf_adaptive' else 'full_K_then_native_readout',
        deployment_budgets=configs['ours']['evaluation']['student_loops'] if 'ours' in configs else None,
        methods=methods,gpus=gpus,model=family,teacher_update=teacher_update,stage_selection=stage_selection,
        opd_mode='d_loopopd' if teacher_update=='current' else 'loopopd',
        loopopd_role='primary_comparison' if os.environ.get('COMPARISON_ROLE')=='primary_opd' else 'optional_ablation_control',
        seed=seed,python_hash_seed=os.environ['PYTHONHASHSEED'],
        evaluation_seed_scheme=SAMPLE_SEED_SCHEME,strict_deterministic_kernels=False,
        teacher_max_loops=integer('TEACHER_MAX_LOOPS'),baseline_student_loops=integer('BASELINE_STUDENT_LOOPS'),
        dynamic_student='1..K, learned stopping',js_threshold=number('JS_THRESHOLD'),stop_loss_weight=number('STOP_LOSS_WEIGHT'),
        boundary_signal=os.environ['BOUNDARY_SIGNAL'],confidence_threshold=number('CONFIDENCE_THRESHOLD'),
        question_filter=os.environ['QUESTION_FILTER'],
        filter_require_eos=boolean('FILTER_REQUIRE_EOS'),
        ours_use_cache=boolean('OURS_USE_CACHE'),baseline_use_cache=boolean('BASELINE_USE_CACHE'),
        progress=dict(show=boolean('SHOW_PROGRESS'),tokens=boolean('TOKEN_PROGRESS'),
                      min_interval=number('PROGRESS_MIN_INTERVAL'),timezone=os.environ['LOG_TIMEZONE'],
                      stage_timestamps=boolean('LOG_STAGE_TIMESTAMPS'),heartbeat_seconds=number('LOG_HEARTBEAT_SECONDS')),
        student_stop_threshold=number('STUDENT_STOP_THRESHOLD'),train_examples='all' if train_count is None else train_count,dev_examples=dev_count,
        prompt_sampling=PROMPT_SAMPLING,target_rollout_responses=updates*prompts*responses,target_question_draws=updates*prompts,
        updates=updates,global_batch=prompts*responses,prompts_per_batch=prompts,responses_per_prompt=responses,
        learning_rate=number('LEARNING_RATE'),max_new_tokens=integer('MAX_NEW_TOKENS'),
        eval_suite=eval_suite,eval_data_path=str(eval_path),eval_samples_per_problem=integer('EVAL_SAMPLES_PER_PROBLEM'),
        eval_responses_per_checkpoint=eval_count*integer('EVAL_SAMPLES_PER_PROBLEM'),
        eval_examples=eval_count,eval_max_new_tokens=integer('EVAL_MAX_NEW_TOKENS'),
        distributed_timeout_seconds=integer('DISTRIBUTED_TIMEOUT_SECONDS'),
        evaluation_updates=eval_steps if run_eval else [],initial_references=run_eval and boolean('EVAL_INITIAL'),
        initial_fixed_references=run_eval and boolean('EVAL_INITIAL') and boolean('EVAL_INITIAL_FIXED'),
        output=str(run),launcher_sha256=digest(Path(os.environ['LAUNCHER_PATH'])))
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)

    def command(cmd, gpu_list, label='command'):
        print('+ CUDA_VISIBLE_DEVICES='+','.join(gpu_list)+' '+shlex.join(cmd),flush=True)
        if not args.dry_run:
            run_command(cmd, cwd=root, env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(gpu_list)},
                        logger=observer, label=label, log_path=run/'logs'/(label+'.log'))

    def train_command(method):
        cmd = [sys.executable,'-u']
        if len(gpus) > 1:
            cmd += ['-m','torch.distributed.run','--standalone',f'--nproc-per-node={len(gpus)}']
        return cmd+['scripts/train_stage_distillation.py',
            '--config',str(run/'configs'/(method+'.json')),'--device','cuda','--output',str(run/method)]

    if args.dry_run:
        print('Dry run only: no files, data preparation, model loading or training.',flush=True)
        for method in methods: command(train_command(method),gpus)
        sys.exit(0)

    # Preflight checks the full source count before any GPU/weight allocation.
    preflight_at, preflight_clock = utc_now(), time.perf_counter()
    probe = merge(configs[methods[0]],{'data':{'train_path':str(source),
        'expected_examples':integer('SOURCE_EXPECTED_EXAMPLES')}})
    status = assets_status(probe,root)
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
    if not status['ready']:
        print('Prepare the verified source JSONL or edit DATA_SOURCE/SOURCE_EXPECTED_EXAMPLES; no training started.',file=sys.stderr)
        sys.exit(2)
    model_config = json.loads((Path(probe['model']['path'])/'config.json').read_text())
    context = model_config.get('max_position_embeddings',model_config.get('block_size'))
    if context and common['data']['max_prompt_tokens']+max(common['rollout']['max_new_tokens'],common['evaluation']['max_new_tokens']) > context:
        fail('Prompt plus response limit exceeds model context length')
    evaluation_records = []
    if eval_suite=='aime':
        if not eval_path.is_file(): fail('AIME JSONL missing; run scripts/prepare_aime_data.py --download')
        evaluation_records,evaluation_metadata = load_evaluation_records(eval_path,eval_count,eval_suite)
        decontamination_records,decontamination_metadata = load_evaluation_records(
            absolute(os.environ['TRAIN_DECONTAMINATION_PATH']),suite='aime')
        plan['evaluation_dataset'] = evaluation_metadata
        plan['training_decontamination_dataset'] = decontamination_metadata
    if args.check: sys.exit(0)
    if run.exists(): fail(f'Output already exists: {run}. Choose a new RUN_TAG; nothing was overwritten.')
    observer = observer_stack.enter_context(RunLogger(run/'launcher.events.jsonl', run/'launcher.console.log', 'ablation'))
    observer.event('launcher_parameters', parameters=plan, launcher_started_at=launcher_started_at)
    observer.event('preflight_complete', started_at=preflight_at, finished_at=utc_now(),
                   seconds=time.perf_counter()-preflight_clock, ready=status['ready'])
    plan['observability'] = observer.metadata()
    ablation_span = observer.start('ablation', echo=True, methods=methods, stage_selection=stage_selection)

    # CPU-only split: deduplicate by message content, filter long prompts, then sample.
    data_span = observer.start('data_split', echo=True)
    seed_everything(seed)
    from transformers import AutoTokenizer
    with observer.stage('tokenizer_load', echo=True):
        tokenizer = AutoTokenizer.from_pretrained(probe['model']['path'],local_files_only=True)
    records,seen,ids = [],{},set()
    too_long=duplicates=0
    with source.open() as stream:
        for index,line in observer.progress(enumerate(stream), total=integer('SOURCE_EXPECTED_EXAMPLES'),
                                           desc='Prepare ablation split', unit='row', leave=False):
            if not line.strip(): continue
            record = normalize_record(json.loads(line),index)
            key = record['problem_sha256']
            if key in seen:
                if seen[key] != record['answer']: fail('Source has conflicting answers for identical messages')
                duplicates += 1
                continue
            seen[key] = record['answer']
            encoded = prompt_ids(tokenizer,record,probe['model']['enable_thinking'],probe['model']['repair_template_newlines'])
            if len(encoded)>integer('MAX_PROMPT_TOKENS'):
                too_long += 1
                continue
            if record['id'] in ids: fail('Source contains conflicting duplicate IDs')
            ids.add(record['id']);records.append(record)
    if eval_suite=='aime':
        records,overlap_report = exclude_evaluation_overlaps(records,decontamination_records)
        plan['evaluation_overlap_exclusion'] = overlap_report
    partitions = split_training_records(records, train_count, dev_count, split_eval_count, seed)
    train_count = len(partitions['train.jsonl'])
    if train_count < prompts: fail('Training pool must contain at least PROMPTS_PER_BATCH distinct questions')
    plan['train_examples'] = train_count
    for config in configs.values(): config['data']['expected_examples'] = train_count
    if run_eval and eval_suite=='dapo_dev':
        from ep_loop.inference import integer_answer
        if any(integer_answer(row['answer']) is None for row in partitions['eval.jsonl']):
            fail('Bundled evaluation requires integer ground truth; use suitable data or set RUN_EVAL=0')
    texts = {name:''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows) for name,rows in partitions.items()}
    manifest = dict(source=str(source),source_sha256=digest(source),seed=seed,model_revision=probe['model']['revision'],
        model_family=family,max_prompt_tokens=integer('MAX_PROMPT_TOKENS'),
        source_expected_examples=integer('SOURCE_EXPECTED_EXAMPLES'),unique_usable_examples=len(records),
        duplicates_removed=duplicates,overlong_removed=too_long,paper_dataset_identity_verified=False,
        partition_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in texts.items()},
        partition_sizes={name:len(rows) for name,rows in partitions.items()})
    if eval_suite=='aime':
        manifest.update(evaluation_dataset=evaluation_metadata,evaluation_overlap_exclusion=overlap_report,
                        training_decontamination_dataset=decontamination_metadata)
    paths = [split/name for name in [*partitions,'manifest.json']]
    if any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths) or json.loads((split/'manifest.json').read_text()) != manifest:
            fail('Existing split differs or is incomplete; choose a new SPLIT_DIR')
        if any((split/name).read_text()!=text for name,text in texts.items()): fail('Existing split content changed')
        print('Reusing matching split: '+str(split),flush=True)
    else:
        split.mkdir(parents=True,exist_ok=True)
        for name,text in texts.items():
            with (split/name).open('x') as stream: stream.write(text)
        dump(split/'manifest.json',manifest)
    observer.finish(data_span, train_examples=train_count, dev_examples=dev_count, usable_examples=len(records))
    config_span = observer.start('save_configs', echo=True)
    (run/'configs').mkdir(parents=True,exist_ok=False)
    for method,config in configs.items(): dump(run/'configs'/(method+'.json'),config)
    dump(run/'launcher_parameters.json',plan)
    dump(run/'data_manifest.json',manifest)
    shutil.copy2(os.environ['LAUNCHER_PATH'],run/'launcher.sh')
    observer.finish(config_span)
    print('Prepared data and configs: '+str(run),flush=True)
    if args.prepare_only:
        observer.finish(ablation_span, status='prepared_only')
        sys.exit(0)

    if os.environ['QUESTION_FILTER']!='all':
        command([sys.executable,'-u','scripts/build_teacher_filter.py','--config',str(run/'configs/reference.json'),
                 '--output',str(filter_path),'--device','cuda'],gpus[:1],label='teacher_correctness')

    def evaluate(method,checkpoint,label,fixed_loops=None):
        if method == 'ours' and fixed_loops:
            fail('Our main evaluation must not force --loops; use its learned stopping head')
        hybrid=execution_mode=='hybrid'
        cmd=[sys.executable,'-u','scripts/evaluate_fastpath.py' if hybrid else 'scripts/evaluate_stage_distillation.py',
             '--config',str(run/'configs'/(method+'.json')),
             '--data',str(eval_path),'--output',str(run/'evaluation'/(label+'.jsonl'))]
        cmd += ['--gpus',','.join(gpus)] if hybrid else ['--device','cuda']
        if checkpoint is not None: cmd += ['--checkpoint',str(checkpoint)]
        for depth in fixed_loops or []: cmd += ['--loops',str(depth)]
        command(cmd,gpus if hybrid else gpus[:1],label='eval_'+label)

    if run_eval and boolean('EVAL_INITIAL'):
        evaluate('reference',None,'initial_native')
        if boolean('EVAL_INITIAL_FIXED'):
            evaluate('reference',None,'initial_fixed',sorted({integer('BASELINE_STUDENT_LOOPS'),integer('TEACHER_MAX_LOOPS')}))
    for method in methods:
        command(train_command(method),gpus,label='train_'+method)
        if run_eval:
            # Evaluating saved snapshots after training is equivalent to inspecting
            # their weights at those updates, and does not interrupt optimizer RNG.
            for step in eval_steps:
                checkpoint=run/method/'checkpoints'/f'update_{step:06d}'
                evaluate(method,checkpoint,f'{method}_update_{step:06d}')
                if method=='baseline' and boolean('EVAL_MATCHED_CACHE') and configs[method]['rollout']['use_cache']:
                    evaluate('baseline_matched_cache',checkpoint,f'baseline_recompute_update_{step:06d}')
    observer.finish(ablation_span)
    print('Completed: '+str(run),flush=True)
except (ValueError,KeyError,FileNotFoundError) as exc:
    print('ERROR: '+str(exc),file=sys.stderr)
    sys.exit(2)
except subprocess.CalledProcessError as exc:
    print(f'Command failed (exit {exc.returncode}); stopping subsequent work.',file=sys.stderr)
    sys.exit(exc.returncode)
finally:
    observer_stack.__exit__(*sys.exc_info())
PY
