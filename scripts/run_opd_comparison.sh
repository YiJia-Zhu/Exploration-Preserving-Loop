#!/usr/bin/env bash
# Primary fixed-depth comparisons: LoopOPD and D-LoopOPD, each with endpoint
# distillation and gate+JS supervision. Preparation/dry-run flags pass through.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ROUTE=vllm COMPARISON_ROLE=primary_opd
export OPD_MODE="${OPD_MODE:-both}"              # loopopd | d_loopopd | both
export ABLATION_GROUPS="${ABLATION_GROUPS:-endpoint,top_n}"
export VLLM_ROLLOUT_NGRAM="${VLLM_ROLLOUT_NGRAM:-1}"
exec bash "$PROJECT_ROOT/scripts/run_three_route_ablation.sh" "$@"
