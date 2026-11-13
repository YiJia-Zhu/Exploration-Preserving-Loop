"""JSON experiment configurations with explicit paper/implementation provenance."""
from pathlib import Path
import copy
import json
import math


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Circular configuration inheritance")
    config = json.loads(path.read_text())
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, (*seen, path)), config)
    return config


def validate_config(config):
    a, o, r, d = (config[k] for k in ("algorithm", "optimization", "rollout", "data"))
    if config["model"]["family"] != "ouro":
        raise ValueError("The implemented native backend is Ouro; Huginn requires its coda/adaptive backend")
    if not 1 <= a["student_loops"] <= a["teacher_max_loops"]:
        raise ValueError("Require 1 <= S <= K")
    if a["teacher_update"] not in {"frozen", "current"}:
        raise ValueError("Teacher update must be frozen or current")
    if a["teacher_endpoint"] not in {"fixed", "native"}:
        raise ValueError("Teacher endpoint must be fixed or native")
    if a["stage_selection"] not in {"endpoint", "uniform", "js"}:
        raise ValueError("Unknown stage selection")
    if not 0 < a["exit_threshold"] <= 1 or not math.isfinite(a["js_threshold"]) or a["js_threshold"] < 0:
        raise ValueError("Invalid exit/JS threshold")
    if r["prompts_per_batch"] * r["responses_per_prompt"] != o["global_batch_size"]:
        raise ValueError("Global batch must equal prompts per batch times responses per prompt")
    if min(o["updates"], o["global_batch_size"], r["max_new_tokens"], d["max_prompt_tokens"]) < 1:
        raise ValueError("Counts must be positive")
    if o["learning_rate"] <= 0 or o["weight_decay"] < 0 or o["warmup_steps"] < 0:
        raise ValueError("Invalid optimizer parameters")
    if o["parameter_dtype"] != "float32":
        raise ValueError("Full training keeps FP32 parameters/master optimizer state")
    if config["implementation"]["microbatch_responses"] != 1:
        raise ValueError("The reference engine currently uses one unpadded response per microbatch")
    for settings in (r, config["evaluation"]):
        if settings["temperature"] <= 0 or not 0 < settings["top_p"] <= 1:
            raise ValueError("Invalid sampling parameters")
    return config


def smoke_config(config):
    """Small official random architecture, expressly outside paper-scale claims."""
    return merge(config, {
        "run_name": config["run_name"] + "_cpu_smoke",
        "data": {"train_path": None, "expected_examples": None, "max_prompt_tokens": 128,
                 "correctness_cache": None},
        "rollout": {"prompts_per_batch": 2, "responses_per_prompt": 2, "max_new_tokens": 4},
        "optimization": {"updates": 2, "global_batch_size": 4, "learning_rate": 1e-3,
                         "warmup_steps": 0, "mixed_precision": "none", "gradient_checkpointing": False},
        "implementation": {"readout_chunk_size": 2, "save_every_updates": 1,
                           "save_optimizer": True},
        "algorithm": {"js_threshold": 0.0},
    })
