#!/usr/bin/env bash
# Run the matched single-GPU and two-GPU inference comparisons in one job.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ $# -lt 2 || $# -gt 3 || ( $1 != slurm && ! $1 =~ ^[0-9]+,[0-9]+$ ) || ( -n ${3:-} && $3 != --dry-run ) ]]; then
    printf 'Usage: bash %s GPU0,GPU1|slurm OUTPUT_DIRECTORY [--dry-run]\n' "$0" >&2
    exit 2
fi
gpus=$1
if [[ $gpus == slurm ]]; then
    gpus=${SLURM_JOB_GPUS:-}
    if [[ ! $gpus =~ ^[0-9]+,[0-9]+$ ]]; then
        printf 'Expected two GPU IDs from SLURM_JOB_GPUS, got %s\n' "$gpus" >&2
        exit 2
    fi
fi
primary_gpu=${gpus%,*}
if [[ $primary_gpu == "${gpus#*,}" ]]; then
    printf 'Provide two distinct GPU indices.\n' >&2
    exit 2
fi
out=$2
python_bin=.venv/bin/python
dry_run=()
if [[ ${3:-} == --dry-run ]]; then
    dry_run=(--dry-run)
else
    "$python_bin" - "$gpus" <<'PY'
import subprocess
import sys
from ep_loop.gpu_resources import require_idle_gpus

try:
    devices = require_idle_gpus(sys.argv[1].split(','))
except subprocess.CalledProcessError as error:
    raise SystemExit(f"GPU preflight failed: {error.output.strip() or error}") from None
for device in devices:
    print(f"GPU {device['index']}: {device['uuid']}")
PY
fi
mkdir -p -- "$(dirname -- "$out")"
mkdir -- "$out"

failed=0
benchmark() {
    local name=$1
    shift
    printf '\n%s\n' "$name"
    if ! "$python_bin" scripts/benchmark_runtime.py --gpu "$primary_gpu" \
        --output "$out/$name" "${dry_run[@]}" "$@"; then
        printf 'Failed: %s\n' "$name" | tee -a "$out/failures.log" >&2
        failed=1
    fi
}

for depth in 2 4; do
    serial=(vllm rlt hf)
    if [[ $depth == 4 ]]; then
        serial+=(backfill-dense ragged-dense)
    fi
    for profile in short 8192 14336; do
        benchmark "${profile}_n${depth}_c8" --profile "$profile" --loops "$depth" --concurrency 8 \
            --case vllm vllm-triton rlt rlt-block16 rlt-block128
        benchmark "${profile}_n${depth}_c1" --profile "$profile" --loops "$depth" --concurrency 1 \
            --case "${serial[@]}"
    done
    benchmark "natural_n${depth}_c8" --profile natural --loops "$depth" --concurrency 8 --case vllm rlt
done
benchmark natural_n2_c1 --profile natural --loops 2 --concurrency 1 --case vllm rlt hf
benchmark natural_n4_c1 --profile natural --loops 4 --concurrency 1 --tau 0.20 --case vllm rlt hf backfill ragged
benchmark repeated_n2_c64 --profile repeated --loops 2 --concurrency 64 \
    --case vllm vllm-prefix-cache rlt rlt-prefix-cache

for profile in short 8192 14336; do
    manifest="$out/${profile}_n2_c8/manifest.json"
    if [[ ${3:-} == --dry-run && -s $manifest ]]; then
        printf 'Planned: tp_%s using %s\n' "$profile" "$manifest"
    elif [[ ! -s $manifest ]] || ! "$python_bin" scripts/benchmark_vllm_parallel.py \
        --gpus "$gpus" --python-bin ep_loop_envs/vllm026/bin/python \
        --manifest "$manifest" --output "$out/tp_$profile.json"; then
        printf 'Failed: tp_%s\n' "$profile" | tee -a "$out/failures.log" >&2
        failed=1
    fi
done
exit "$failed"
