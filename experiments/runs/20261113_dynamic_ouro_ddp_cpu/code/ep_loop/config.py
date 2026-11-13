"""JSON experiment configurations with explicit paper/implementation provenance."""
from pathlib import Path
import copy
import json
import math

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/default.json"


def student_budget(config):
    a = config["algorithm"]
    return a["student_max_loops"] if a.get("student_policy", "fixed") != "fixed" else a["student_loops"]


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
    family = config["model"]["family"]
    if family not in {"ouro", "huginn"}:
        raise ValueError("Supported native backends are ouro and huginn")
    policy = a.get("student_policy", "fixed")
    if policy not in {"fixed", "learned", "native"}:
        raise ValueError("Student policy must be fixed, learned or native")
    if policy == "fixed":
        if not isinstance(a["student_loops"], int) or not 1 <= a["student_loops"] <= a["teacher_max_loops"]:
            raise ValueError("Require integer 1 <= S <= K")
    elif a.get("student_max_loops") != a["teacher_max_loops"] or a.get("student_loops") is not None:
        raise ValueError("Dynamic student uses maximum K and has no fixed student_loops")
    if a["teacher_update"] not in {"frozen", "current"}:
        raise ValueError("Teacher update must be frozen or current")
    if a["teacher_endpoint"] not in {"fixed", "native"}:
        raise ValueError("Teacher endpoint must be fixed or native")
    if a["stage_selection"] not in {"endpoint", "uniform", "js", "threshold", "threshold_uniform", "threshold_endpoint"}:
        raise ValueError("Unknown stage selection")
    if policy == "learned":
        if a["stage_selection"] not in {"threshold", "threshold_uniform", "threshold_endpoint"}:
            raise ValueError("Learned student must preserve all JS-threshold stage counts")
        if not 0 < a["student_stop_threshold"] <= 1 or not math.isfinite(a["stop_loss_weight"]) or a["stop_loss_weight"] <= 0:
            raise ValueError("Invalid student stop threshold/BCE coefficient")
        if r["use_cache"]:
            raise ValueError("First dynamic student implementation explicitly recomputes full prefixes; set use_cache=false")
    elif a["stage_selection"].startswith("threshold"):
        raise ValueError("Threshold-only phase counts require a learned dynamic student")
    if (not math.isfinite(a["exit_threshold"]) or a["exit_threshold"] <= 0
            or (family == "ouro" and a["exit_threshold"] > 1)
            or not math.isfinite(a["js_threshold"]) or a["js_threshold"] < 0):
        raise ValueError("Invalid exit/JS threshold")
    if family == "huginn" and policy == "native" and not r["use_cache"]:
        raise ValueError("Huginn reference generation requires independent native KV caches")
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
    result = merge(config, {
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
    if config["algorithm"].get("student_policy") == "learned":
        result["algorithm"]["js_threshold"] = config["algorithm"]["js_threshold"]
    if config["model"]["family"] == "huginn":
        # Real K=32/S=8 or16 remain in the original config; CPU smoke uses 4/2.
        dynamic = config["algorithm"].get("student_policy", "fixed") != "fixed"
        result = merge(result, {"algorithm": {"teacher_max_loops": 4, "student_loops": None if dynamic else 2}})
        if dynamic:
            result["algorithm"]["student_max_loops"] = 4
    return result


def assets_status(config, root):
    """Cheap preflight: no model load, download, CUDA allocation, or file writes."""
    root = Path(root)
    model = (root/config["model"]["path"]).resolve()
    data = (root/config["data"]["train_path"]).resolve()
    missing = []
    source_names = (("configuration_ouro.py", "modeling_ouro.py") if config["model"]["family"] == "ouro"
                    else ("raven_config_minimal.py", "raven_modeling_minimal.py"))
    for name in ("config.json", *source_names, "tokenizer_config.json"):
        if not (model/name).is_file():
            missing.append(str(model/name))
    index = model/"model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
    missing.extend(str(model/name) for name in shards if not (model/name).is_file())
    metadata = model/".cache/huggingface/download/config.json.metadata"
    observed_revision = metadata.read_text().splitlines()[0] if metadata.is_file() else None
    revision_matches = observed_revision == config["model"]["revision"]
    if not data.is_file():
        missing.append(str(data))
        rows = None
    else:
        with data.open() as f:
            rows = sum(bool(line.strip()) for line in f)
    count_matches = config["data"]["expected_examples"] is None or rows == config["data"]["expected_examples"]
    return dict(ready=not missing and revision_matches and count_matches,
                missing=missing, observed_model_revision=observed_revision,
                requested_model_revision=config["model"]["revision"],
                model_revision_metadata_matches=revision_matches,
                observed_training_rows=rows, expected_training_rows=config["data"]["expected_examples"],
                training_count_matches=count_matches, dataset_identity_verified=False,
                note="Metadata/count checks do not independently establish the original paper dataset identity")
