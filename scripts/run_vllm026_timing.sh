#!/usr/bin/env bash
# Same fixed-depth gate+JS method; three full updates for stage timing.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
GPUS="${GPUS:-0,1,2,3}" # TP1 replicas by default; TP2 pairs are selectable. HF uses four DDP ranks.
CPU_THREADS="${CPU_THREADS:-16}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}" # Existing HF training runtime.
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-${PROJECT_ROOT}/ep_loop_envs/vllm026/bin/python}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
VLLM_MEMORY_FRACTION="${VLLM_MEMORY_FRACTION:-0.75}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}" # Omit the engine concurrency limit by default; submit all requests once.
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}" # 2: paired TP2 x DP2; slower in the matched long-prefix short test, so opt in.
VLLM_ROLLOUT_NGRAM="${VLLM_ROLLOUT_NGRAM:-1}" # Fixed N2 training generation; independent inference/evaluation stays normal.
VLLM_EAGER="${VLLM_EAGER:-0}"
ROUTE=vllm
ABLATION_GROUPS=top_n
MODEL_PATH="${MODEL_PATH:-models/Ouro-1.4B-Thinking}"
TEACHER_MAX_LOOPS="${TEACHER_MAX_LOOPS:-4}"
STUDENT_LOOPS="${STUDENT_LOOPS:-2}"
TEACHER_UPDATE=frozen
TEACHER_BACKEND=vllm
TEACHER_ENDPOINT=native
TEACHER_EXIT_THRESHOLD="${TEACHER_EXIT_THRESHOLD:-0.9}"
SHALLOW_TARGET_PLACEMENT=prefix
PHASE_OFFLOAD=1
UPDATES="${UPDATES:-3}"
PROMPTS_PER_BATCH="${PROMPTS_PER_BATCH:-32}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
SEED="${SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.0}"
TRAIN_TOP_P="${TRAIN_TOP_P:-1.0}"
MIXED_PRECISION=bf16
GRADIENT_CHECKPOINTING=1
CHECKPOINT_READOUT=1
READOUT_CHUNK_SIZE="${READOUT_CHUNK_SIZE:-128}"
QUESTION_FILTER=all
TRAIN_EXAMPLES=all
DEV_EXAMPLES=64
RUN_EVAL=0                         # This run measures training, no AIME generation.
EVAL_INITIAL=0
KEEP_LATEST_CHECKPOINTS=1
SAVE_EVERY_UPDATES="${SAVE_EVERY_UPDATES:-${UPDATES}}"
TENSORBOARD=1
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/ep_loop_runs/Exploration-Preserving-Loop/vllm026_timing}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_vllm026_flashinfer_K4_N2_B256_tp${VLLM_TENSOR_PARALLEL_SIZE}_u${UPDATES}_seed${SEED}}"
TEACHER_TRACE_DIR="${TEACHER_TRACE_DIR:-/dev/shm/ep_loop_${RUN_TAG}}"
VALIDATE_ENGINE="${VALIDATE_ENGINE:-1}"
export PROJECT_ROOT
set +a
cd -- "$PROJECT_ROOT"
if [[ "${1:-}" != --dry-run && "${1:-}" != --prepare-only ]]; then
  "$VLLM_PYTHON_BIN" -c 'import importlib.metadata as m; assert m.version("vllm")=="0.26.0"; print("vLLM",m.version("vllm"),"FlashInfer",m.version("flashinfer-python"),flush=True)'
  "$PYTHON_BIN" - <<'PY'
import os
from ep_loop.gpu_resources import require_idle_gpus
require_idle_gpus(os.environ['GPUS'].split(','))
PY
  if [[ "$VALIDATE_ENGINE" == 1 ]]; then
    validation_gpus="$("$PYTHON_BIN" -c 'import os; print(",".join(os.environ["GPUS"].split(",")[:int(os.environ["VLLM_TENSOR_PARALLEL_SIZE"])]))')"
    "$PYTHON_BIN" -u scripts/validate_vllm026.py --gpu "$validation_gpus" \
      --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
      --python-bin "$VLLM_PYTHON_BIN" \
      --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
      --output "audit/vllm026_flashinfer_20261118/${RUN_TAG}.validation.json"
  fi
fi
exec bash scripts/run_three_route_ablation.sh "$@"
