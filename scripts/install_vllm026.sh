#!/usr/bin/env bash
# Isolated inference runtime; the HF training environment is left untouched.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ENV="${VLLM_ENV:-${PROJECT_ROOT}/ep_loop_envs/vllm026}"
UV_BIN="${UV_BIN:-${VLLM_ENV}/bin/uv}"
"$UV_BIN" pip install --python "${VLLM_ENV}/bin/python" -r "${PROJECT_ROOT}/requirements-vllm026.txt"
"${VLLM_ENV}/bin/python" - <<'PY'
from pathlib import Path
import sysconfig

# NVIDIA's split CUDA wheels use lib/, whereas FlashInfer JIT expects lib64/
# and the unversioned linker name. These links stay inside the private venv.
toolkit = Path(sysconfig.get_paths()['purelib']) / 'nvidia/cu13'
for link, target in [(toolkit / 'lib64', 'lib'),
                     (toolkit / 'lib/libcudart.so', 'libcudart.so.13')]:
    if not link.exists():
        link.symlink_to(target)
    if not link.resolve().exists():
        raise RuntimeError(f'Broken private CUDA link: {link}')
PY
"$UV_BIN" pip check --python "${VLLM_ENV}/bin/python"
