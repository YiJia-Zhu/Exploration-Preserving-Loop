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


def validate_rad_opd(config):
    a = config["algorithm"]
    if (config["model"]["family"] != "ouro" or a["teacher_update"] != "frozen"
            or a["teacher_endpoint"] != "fixed" or a["teacher_max_loops"] != 4
            or a["stage_selection"] != "endpoint" or a.get("stop_loss_weight") != 0):
        raise ValueError("RAD-OPD requires Ouro, a frozen dense four-loop endpoint teacher and zero gate loss")
    if not config["rollout"]["use_cache"] or not config["evaluation"].get("use_cache", True):
        raise ValueError("RAD-OPD requires a KV cache for rollout and evaluation")
    if config["evaluation"].get("student_policy", "ragged") not in {"ragged", "fixed"}:
        raise ValueError("RAD-OPD evaluation supports ragged and fixed policies")
    if config.get("distillation", {}).get("question_filter", "all") != "all":
        raise ValueError("The RAD-OPD MVP retains all questions and rollout responses")
    if type(a.get("rad_detach_reused_kv", False)) is not bool:
        raise ValueError("rad_detach_reused_kv must be a boolean")
    tau = a.get("rad_tau")
    if not isinstance(tau, dict):
        raise ValueError("RAD-OPD requires a threshold mixture object")
    valid_threshold = lambda value: type(value) in (int, float) and 0 < value <= 1
    fixed = tau.get("fixed")
    low, anchors, p = tau.get("low_range", []), tau.get("anchors", []), tau.get("low_probability")
    if (type(p) not in (int, float) or not 0 <= p <= 1
            or not isinstance(low, list) or len(low) != 2 or not all(map(valid_threshold, low)) or low[0] > low[1]
            or not isinstance(anchors, list) or not anchors or not all(map(valid_threshold, anchors))
            or (fixed is not None and not valid_threshold(fixed))):
        raise ValueError("Invalid RAD-OPD threshold mixture")
    thresholds = config["evaluation"].get("ragged_thresholds", [])
    if (not isinstance(thresholds, list) or not thresholds or not all(map(valid_threshold, thresholds))
            or len(set(thresholds)) != len(thresholds)):
        raise ValueError("Provide distinct RAD-OPD evaluation thresholds in (0, 1]")
    budgets = config["evaluation"].get("budget_slices", [])
    if (not isinstance(budgets, list) or not budgets or any(type(cap) is not int or cap < 1 for cap in budgets)
            or budgets != sorted(set(budgets)) or budgets[-1] != config["evaluation"]["max_new_tokens"]):
        raise ValueError("RAD-OPD budget slices must increase to the evaluation generation cap")


def validate_config(config):
    from .execution import validate_execution
    validate_execution(config)
    from .fastpath.speculation import rollout_ngram_enabled
    rollout_ngram_enabled(config)
    a, o, r, d = (config[k] for k in ("algorithm", "optimization", "rollout", "data"))
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32), default 42")
    family = config["model"]["family"]
    if family not in {"ouro", "huginn"}:
        raise ValueError("Supported native backends are ouro and huginn")
    policy = a.get("student_policy", "fixed")
    signal = a.get('boundary_signal', 'js')
    if signal not in {'js','max_probability','normalized_entropy','probe_know'}:
        raise ValueError('Unknown boundary signal')
    if signal != 'js' and policy != 'learned':
        raise ValueError('Alternative boundary signals require a dynamic learned student')
    if signal != 'js' and (not math.isfinite(a.get('confidence_threshold', .05)) or not 0<=a.get('confidence_threshold', .05)<=1):
        raise ValueError('Confidence change threshold must lie in [0,1]')
    if config.get('distillation',{}).get('response_filter','all') != 'all':
        raise ValueError('Student response filtering has been removed; every student response is distilled')
    if type(config.get('distillation',{}).get('require_eos',True)) is not bool:
        raise ValueError('distillation.require_eos must be a boolean')
    question_filter = config.get('distillation',{}).get('question_filter','teacher_correct' if d.get('correctness_cache') else 'all')
    if question_filter not in {'all','teacher_correct','random_matched'}:
        raise ValueError('Question filter must be all, teacher_correct or random_matched')
    if question_filter!='all' and not d.get('correctness_cache'):
        raise ValueError('Question filtering requires data.correctness_cache from the frozen initial teacher')
    if question_filter=='all' and d.get('correctness_cache'):
        raise ValueError('Remove correctness_cache for unfiltered questions or select teacher_correct')
    if policy not in {"fixed", "learned", "native", "native_adaptive", "ragged"}:
        raise ValueError("Unknown student policy")
    if policy == "fixed":
        if not isinstance(a["student_loops"], int) or not 1 <= a["student_loops"] <= a["teacher_max_loops"]:
            raise ValueError("Require integer 1 <= S <= K")
    elif a.get("student_max_loops") != a["teacher_max_loops"] or a.get("student_loops") is not None:
        raise ValueError("Dynamic student uses maximum K and has no fixed student_loops")
    if a["teacher_update"] not in {"frozen", "current"}:
        raise ValueError("Teacher update must be frozen or current")
    if a["teacher_endpoint"] not in {"fixed", "native", "native_adaptive"}:
        raise ValueError("Teacher endpoint must be fixed or native")
    if (policy=='native_adaptive' or a['teacher_endpoint']=='native_adaptive') and family!='ouro':
        raise ValueError('The exact streamed native-CDF teacher currently supports Ouro only')
    if a["stage_selection"] not in {"endpoint", "uniform", "js", "top_n", "uniform_n", "threshold", "threshold_uniform", "threshold_endpoint"}:
        raise ValueError("Unknown stage selection")
    if a['stage_selection'] in {'top_n','uniform_n'} and policy!='fixed':
        raise ValueError('top_n/uniform_n requires a fixed student budget')
    if a['stage_selection']=='uniform_n' and a['teacher_endpoint']!='fixed':
        raise ValueError('uniform_n requires a fixed teacher endpoint')
    if a.get('shallow_target_placement','last') not in {'prefix','last'}:
        raise ValueError('shallow_target_placement must be prefix or last')
    teacher_backend=config['implementation'].get('teacher_backend','hf')
    if teacher_backend not in {'hf','vllm'}:raise ValueError('Unknown teacher backend')
    if teacher_backend=='vllm' and (family!='ouro' or
            config['implementation'].get('execution_mode')!='vllm_fixed'):
        raise ValueError('The vLLM teacher supports Ouro in vllm_fixed only')
    if type(config['implementation'].get('phase_offload',False)) is not bool:
        raise ValueError('phase_offload must be boolean')
    hf_batch=config['implementation'].get('hf_rollout_batch_size',1)
    if type(hf_batch) is not int or hf_batch<1:
        raise ValueError('hf_rollout_batch_size must be a positive integer')
    capacity=config['implementation'].get('hf_static_cache_capacity')
    if capacity is not None and (type(capacity) is not int or capacity<1):
        raise ValueError('hf_static_cache_capacity must be a positive integer or null')
    if type(config['implementation'].get('optimizer_offload_for_rollout',False)) is not bool:
        raise ValueError('optimizer_offload_for_rollout must be boolean')
    if type(a.get('teacher_reuse_student',False)) is not bool:
        raise ValueError('teacher_reuse_student must be boolean')
    max_seqs=config['implementation'].get('vllm_max_num_seqs',0)
    if type(max_seqs) is not int or max_seqs<0:
        raise ValueError('vllm_max_num_seqs must be a nonnegative integer; 0 uses the engine default')
    tp=config['implementation'].get('vllm_tensor_parallel_size',1)
    if type(tp) is not int or tp<1:
        raise ValueError('vllm_tensor_parallel_size must be a positive integer')
    if tp>1 and (family!='ouro' or config['implementation'].get('execution_mode')!='vllm_fixed'
                 or not config['implementation'].get('phase_offload',False)
                 or config['implementation'].get('rollout_backend','vllm')!='vllm'):
        raise ValueError('Paired TP inference requires Ouro vllm_fixed, vLLM rollout and phase_offload')
    if policy == "learned":
        if a["stage_selection"] not in {"threshold", "threshold_uniform", "threshold_endpoint"}:
            raise ValueError("Learned student must preserve all JS-threshold stage counts")
        if not 0 < a["student_stop_threshold"] <= 1 or not math.isfinite(a["stop_loss_weight"]) or a["stop_loss_weight"] <= 0:
            raise ValueError("Invalid student stop threshold/BCE coefficient")
    elif a["stage_selection"].startswith("threshold"):
        raise ValueError("Threshold-only phase counts require a learned dynamic student")
    if policy == "ragged":
        validate_rad_opd(config)
    offline = d.get("rad_offline_responses")
    if offline is not None and (policy != "ragged" or not isinstance(offline, str) or not offline.strip()):
        raise ValueError("data.rad_offline_responses requires a RAD-OPD JSONL path")
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
    keep_latest = config['implementation'].get('keep_latest_checkpoints', 0)
    if type(keep_latest) is not int or keep_latest < 0:
        raise ValueError('keep_latest_checkpoints must be a nonnegative integer')
    layout=config['implementation'].get('checkpoint_layout','full')
    if layout not in {'full','split','model_only'}:
        raise ValueError('checkpoint_layout must be full, split or model_only')
    if layout=='model_only':
        if config['implementation']['save_optimizer'] or keep_latest:
            raise ValueError('model_only requires save_optimizer=false and keep_latest_checkpoints=0')
    if type(config['implementation'].get('save_final_resume',False)) is not bool:
        raise ValueError('save_final_resume must be boolean')
    if layout=='split':
        from .execution import execution_mode,VLLM_FIXED
        if execution_mode(config)!=VLLM_FIXED:
            raise ValueError('Split checkpoints currently require the fixed-vLLM training route')
        interval=config['implementation'].get('resume_save_every_updates',config['implementation']['save_every_updates'])
        if (type(interval) is not int or interval<1 or interval%config['implementation']['save_every_updates']):
            raise ValueError('resume_save_every_updates must be a positive multiple of save_every_updates')
        if not config['implementation']['save_optimizer']:
            raise ValueError('Split recovery requires save_optimizer=true')
    evaluation_every=config['evaluation'].get('every_updates',0)
    if type(evaluation_every) is not int or evaluation_every<0:
        raise ValueError('evaluation.every_updates must be a nonnegative integer')
    if evaluation_every:
        from .execution import execution_mode,VLLM_FIXED
        if execution_mode(config)!=VLLM_FIXED or policy!='fixed':
            raise ValueError('Periodic evaluation currently requires fixed-depth vLLM training')
        if not config['implementation'].get('phase_offload',False):
            raise ValueError('Periodic evaluation requires phase_offload=true to release training state')
        save_every=config['implementation']['save_every_updates']
        if type(save_every) is not int or save_every<1 or evaluation_every%save_every:
            raise ValueError('evaluation.every_updates must be a multiple of save_every_updates')
        from .fastpath.evaluation import evaluation_datasets
        evaluation_datasets(config,Path(__file__).resolve().parents[1])
    for settings in (r, config["evaluation"]):
        if type(settings.get("use_cache", True)) is not bool:
            raise ValueError("use_cache must be a boolean")
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
    if config["algorithm"].get("student_policy") == "ragged":
        result["evaluation"].update(expected_examples=None, suite=None)
    if result['implementation'].get('checkpoint_layout')=='model_only':
        result['implementation']['save_optimizer']=False
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
    offline = config["data"].get("rad_offline_responses")
    if offline is not None and not (root / offline).is_file():
        missing.append(str(root / offline))
    count_matches = config["data"]["expected_examples"] is None or rows == config["data"]["expected_examples"]
    return dict(ready=not missing and revision_matches and count_matches,
                missing=missing, observed_model_revision=observed_revision,
                requested_model_revision=config["model"]["revision"],
                model_revision_metadata_matches=revision_matches,
                observed_training_rows=rows, expected_training_rows=config["data"]["expected_examples"],
                training_count_matches=count_matches, dataset_identity_verified=False,
                note="Metadata/count checks do not independently establish the original paper dataset identity")
