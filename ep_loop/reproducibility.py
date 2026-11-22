"""Control cheap random sources without forcing slower deterministic kernels.

Training uses persistent per-rank streams. Evaluation/filtering use stable
per-problem/sample streams, independent of example order or earlier lengths.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess


RANK_SEED_STRIDE = 100003
SAMPLE_SEED_SCHEME = "sha256_base_seed_problem_fingerprint_sample_v1"


def rank_seed(seed, rank):
    return int(seed) + int(rank) * RANK_SEED_STRIDE


def sample_seed(seed, problem_fingerprint, sample=0):
    """Use a stable digest, never Python's process-randomized hash()."""
    key = json.dumps([int(seed), str(problem_fingerprint), int(sample)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def seed_everything(seed):
    """Seed Python, global NumPy and Torch CPU/CUDA; leave math flags alone.

    torch.manual_seed also seeds CUDA generators lazily; it does not allocate
    a context on every visible GPU. Never call this inside a token/loop step.
    """
    import numpy as np
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed) % (1 << 32))
    torch.manual_seed(int(seed))


def capture_rng_state(device):
    """Primitive NumPy encoding stays compatible with torch.load(weights_only=True)."""
    import numpy as np
    import torch
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return dict(python_rng=random.getstate(),
        numpy_rng=dict(algorithm=str(algorithm), keys=keys.tolist(), position=int(position),
                       has_gauss=int(has_gauss), cached_gaussian=float(cached_gaussian)),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None)


def restore_rng_state(state, device):
    """Restore new states, while accepting older checkpoints missing Python/NumPy."""
    import numpy as np
    import torch
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if "numpy_rng" in state:
        value = state["numpy_rng"]
        np.random.set_state((value["algorithm"], np.asarray(value["keys"], dtype=np.uint32),
                            value["position"], value["has_gauss"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch_rng"])
    if torch.device(device).type == "cuda" and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], device)


def execution_provenance(root, device, seed, rank=0, world_size=1, phase="training"):
    """Collect small startup metadata only; never hash large weights/datasets."""
    import torch
    root, device = Path(root), torch.device(device)
    packages = {}
    for package in ("torch", "numpy", "transformers", "tokenizers", "accelerate", "safetensors", "tqdm", "tensorboard"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    git = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]),
                      ("tracked_changes", ["status", "--porcelain", "--untracked-files=no"])):
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5)
            git[key] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            git[key] = None
    sources = [*root.glob("ep_loop/*.py"), *root.glob("scripts/*.py"), *root.glob("scripts/*.sh")]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(sources)}
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = dict(name=properties.name, capability=[properties.major, properties.minor],
                   total_memory=properties.total_memory, uuid=str(getattr(properties, "uuid", "unknown")))
    return dict(phase=phase, base_seed=int(seed), initialization_seed=int(seed), data_order_seed=int(seed),
        rank_seed_stride=RANK_SEED_STRIDE,
        rank_stochastic_seed=rank_seed(seed, rank) if phase == "training" else None,
        rollout_seed=rank_seed(seed, rank) if phase == "training" else None,
        random_stream_policy="persistent_rank_streams" if phase == "training" else "per_problem_sample_streams",
        rank=int(rank), world_size=int(world_size), python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        evaluation_seed_scheme=SAMPLE_SEED_SCHEME, python=platform.python_version(),
        platform=platform.platform(), packages=packages, cuda_version=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), device=str(device), gpu=gpu,
        num_threads=torch.get_num_threads(),
        cpu_thread_environment={name:os.environ.get(name) for name in (
            'CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        pytorch_cuda_alloc_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        git=git, source_sha256=source_hashes,
        scope="Best-effort seeded execution. No deterministic-kernel enforcement; exact CUDA/DDP results across hardware/software/world sizes are not promised.")
