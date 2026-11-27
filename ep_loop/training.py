"""Full-parameter RAD-OPD, LoopOPD/D-LoopOPD and stage distillation.

Every optimizer update uses B*G fresh student responses. Microbatches
only accumulate gradients: weights/teacher stay unchanged until the batch ends.
Distributed ranks have equal response counts, so DDP means equal response means.
"""
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
import copy
import fcntl
import json
import os
import random
import shutil
import sys
from tempfile import TemporaryDirectory
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import assets_status, load_config, smoke_config, validate_config, student_budget
from .execution import execution_mode, HF_SHARED
from .backends import load_backend
from .data import ByteTokenizer, canonical_hash, prepare_training_records, sample_rollout_indices
from .cpu_runtime import configure_cpu_threads
from .checkpointing import inspect_resume_checkpoint, validate_resume_state, validate_resume_data, copy_resume_metrics
from .tensorboard_logging import ScalarLogger, tensorboard_enabled
from .inference import generate
from .objective import DistillationObjective
from .observability import RunLogger, model_progress
from .reproducibility import (capture_rng_state, execution_provenance, rank_seed,
                              restore_rng_state, seed_everything)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def append_json(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False)+"\n")


def read_checkpoint_weights(model, folder):
    folder = Path(folder)
    if (folder / "model.safetensors.index.json").is_file():
        from transformers.modeling_utils import load_sharded_checkpoint
        loaded_keys = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        missing, unexpected = load_sharded_checkpoint(model, folder, strict=False, prefer_safe=True)
    else:
        from safetensors.torch import load_file
        weights = load_file(folder / "model.safetensors")
        missing, unexpected = model.load_state_dict(weights, strict=False)
        loaded_keys = weights
    # HF removes duplicate tied embedding tensors when saving Huginn.
    # Permit only missing aliases of an actually loaded shared Parameter,
    # for both single-file and sharded full-size checkpoints.
    names = dict(model.named_parameters(remove_duplicate=False))
    aliases = {key for key in missing if key in names and any(
        other in loaded_keys and parameter is names[key] for other, parameter in names.items())}
    if set(missing) - aliases or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def save_checkpoint(folder, student, tokenizer, optimizer, completed, config,
                    data_rng, generator, rank, world_size, data_identity=None):
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=False)
        student.model.save_pretrained(folder, safe_serialization=True)
        source = (ROOT / config["model"]["path"]).resolve()
        for filename in student.source_filenames:
            shutil.copy2(source / filename, folder / filename)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(folder)
        write_json(folder / "run_config.json", config)
        if config["implementation"]["save_optimizer"]:
            torch.save(dict(optimizer=optimizer.state_dict(), completed_updates=completed,
                            world_size=world_size, config_hash=canonical_hash(config),data_identity=data_identity),
                       folder / "training_state.pt")
    if world_size > 1:
        dist.barrier()
    torch.save(dict(data_rng=data_rng.getstate(), rollout_rng=generator.get_state(),
                    **capture_rng_state(student.device)),
               folder / f"rng_rank{rank}.pt")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        write_json(folder/'checkpoint.complete.tmp',dict(completed_updates=completed,world_size=world_size,
                    format_version=2,data_identity=data_identity,
                    config_hash=canonical_hash(config),resumable=config['implementation']['save_optimizer']))
        (folder/'checkpoint.complete.tmp').replace(folder/'checkpoint.complete.json')


def train(config_path, output, device_name="auto", smoke=False, resume=None,
          stop_after_updates=None):
    from .execution import execution_mode, HF_TRAIN_MODES
    if execution_mode(load_config(config_path)) not in HF_TRAIN_MODES:
        from .fastpath.training import train as optional_train
        return optional_train(config_path, output, device_name, smoke, resume, stop_after_updates)
    rank = int(os.environ.get("RANK", "0"))
    output = Path(output).resolve()
    with RunLogger(output/f"events_rank{rank}.jsonl", output/f"console_rank{rank}.log", "train", rank) as observer:
        with observer.stage("training_run", echo=True, config_path=str(config_path), resume=str(resume) if resume else None):
            return _train(config_path, output, device_name, smoke, resume, stop_after_updates, observer)


def _train(config_path, output, device_name, smoke, resume, stop_after_updates, observer):
    with observer.stage("config_validation", echo=True):
        config = load_config(config_path)
        if smoke:
            config = smoke_config(config)
        validate_config(config)
    if config["algorithm"].get("student_policy") in {"native", "native_adaptive"}:
        raise ValueError("Native base is an inference baseline; use evaluate_stage_distillation.py")
    policy = config["algorithm"].get("student_policy", "fixed")
    ragged = policy == "ragged"
    implementation = config['implementation']
    accelerated = implementation.get('rollout_backend') == 'vllm-rlt' and not smoke
    rollout_backend = 'vllm-rlt' if accelerated else 'hf'
    teacher_backend = 'vllm' if accelerated else 'hf'
    if ragged:
        from .ragged import RaggedKVCache, sample_training_threshold
    if not smoke:
        with observer.stage("asset_preflight", echo=True):
            status = assets_status(config, ROOT)
            if not status["ready"]:
                raise ValueError("Full-training asset preflight failed: " + json.dumps(status,ensure_ascii=False))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if config["rollout"]["prompts_per_batch"] % world:
        raise ValueError("Prompts per batch must be divisible by the number of ranks")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{local_rank}" if device_name == "cuda" else device_name)
    if accelerated and (device.type != 'cuda' or world != 1 or implementation.get('vllm_tensor_parallel_size', 1) != 1):
        raise ValueError('RAD-OPD vLLM-RLT and vLLM phases require one CUDA rank')
    if device.type == "cuda":
        torch.cuda.set_device(device)
    observer.event('cpu_runtime', echo=True, num_threads=configure_cpu_threads())
    seed_everything(config["seed"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    if world > 1:
        timeout_seconds = int(config['implementation'].get('distributed_timeout_seconds', 21600))
        if timeout_seconds <= 0:
            raise ValueError('distributed_timeout_seconds must be positive')
        with observer.stage("distributed_init", echo=True, world_size=world, timeout_seconds=timeout_seconds):
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                    timeout=timedelta(seconds=timeout_seconds))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    tensorboard = None
    resume_info = None
    teacher = None
    try:
        if resume:resume_info = inspect_resume_checkpoint(resume,config,world)
        if rank == 0:
            lock = (output / "run.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (output / "metrics.jsonl").exists() and not resume:
                raise FileExistsError("Use a new run output or --resume; existing observations are protected")
            if resume and (output / "metrics.jsonl").exists():
                logged = [json.loads(line)["update"] for line in (output/"metrics.jsonl").read_text().splitlines()]
                if logged and max(logged) > resume_info["completed_updates"]:
                    raise ValueError("Run log has updates newer than the chosen checkpoint; resume into a new output")
        if world > 1:
            dist.barrier()
        started = time.perf_counter()
        model_dir = (ROOT / config["model"]["path"]).resolve()
        with observer.stage("model_load", echo=True, model_path=str(model_dir), device=str(device)), model_progress(observer.show_progress):
            student, loading = load_backend(config["model"], model_dir, device,
                config["optimization"]["mixed_precision"], config["optimization"]["gradient_checkpointing"], smoke)
        with observer.stage("tokenizer_load", echo=True):
            if smoke:
                tokenizer = ByteTokenizer()
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        # Clone before loading a resumed student: frozen means the INITIAL model.
        with observer.stage("teacher_student_setup", echo=True):
            if accelerated:
                from .fastpath.teacher import VllmTeacher
                teacher = VllmTeacher(student, config, output, local_rank, observer)
            elif config["algorithm"]["teacher_update"] == "frozen":
                teacher = type(student)(copy.deepcopy(student.model).eval().requires_grad_(False), student.mixed_precision)
            else:
                teacher = student
            if config["algorithm"].get("student_policy") == "learned":
                student.enable_student_exit()
        if resume:
            with observer.stage("resume_weights", echo=True, checkpoint=str(resume)):
                read_checkpoint_weights(student.model, resume)
        with observer.stage("data_prepare", echo=True):
            records, data_report = prepare_training_records(config, tokenizer, ROOT, smoke, observer=observer)
            offline = None
            offline_path = config["data"].get("rad_offline_responses")
            if offline_path is not None:
                from .data import OfflineResponses
                offline = OfflineResponses(ROOT / offline_path, records, config["rollout"]["responses_per_prompt"])
                data_report["offline_response_sha256"] = offline.sha256
                data_report["usable_prompt_hash"] = canonical_hash(
                    [data_report["usable_prompt_hash"], offline.sha256])
            if accelerated:
                teacher.max_model_len = (max(len(record['input_ids']) for record in records)
                                         + config['rollout']['max_new_tokens'])
        validate_resume_data(resume_info,data_report)
        if rank == 0:
            write_json(output / "resolved_config.json", config)
            write_json(output / "data_report.json", data_report)
            write_json(output / "provenance.json", dict(
                started_at=datetime.now(timezone.utc).isoformat(), torch=torch.__version__,
                world_size=world, device=str(device), loading=loading,
                original_training_code_used=False, original_paper_scores_reproduced=False,
                source_proposal="docs/rad_opd_mvp.md" if ragged else "docs/loop_distillation_research_proposal_zh.md",
                model_source_directory=str(model_dir), smoke=smoke,
                teacher_parameter_refresh=config["algorithm"]["teacher_update"],
                teacher_reuses_student_forward=config['algorithm'].get('teacher_reuse_student',False),
                model_instances_per_rank=1 if config['algorithm']['teacher_update']=='current' else 2,
                rollout_backend=rollout_backend, teacher_backend=teacher_backend, phase_offload=accelerated,
                hf_rollout_batch_size=None if accelerated else implementation.get('hf_rollout_batch_size',1),
                native_exit_reduces_executed_loops=ragged or student.native_exit_reduces_executed_loops,
                teacher_execution_policy=config['algorithm']['teacher_endpoint'],
                teacher_native_exit_reduces_executed_loops=config['algorithm']['teacher_endpoint']=='native_adaptive',
                model_family=config["model"]["family"],
                experiment=config.get('experiment',{}),
                student_policy=config["algorithm"].get("student_policy", "fixed"),
                learned_student_uses_full_prefix_recomputation=config["algorithm"].get("student_policy") == "learned" and not config["rollout"]["use_cache"],
                rollout_cache_policy='rlt_last_exited' if accelerated else RaggedKVCache.policy if ragged else 'hf_batched_static_prefix' if execution_mode(config)==HF_SHARED else ("exact_lazy_depth" if policy == "learned"
                                      else "native") if config["rollout"]["use_cache"] else "none",
                token_loop_accounting="Forward backbone only; excludes checkpoint replay, backward, LM head; teacher coda token-positions logged separately",
                reproducibility=execution_provenance(ROOT, device, config["seed"], rank, world),
                observability=observer.metadata(),
            ))
            sources = [*sorted((ROOT/"ep_loop").glob("*.py")), ROOT/"scripts/train_stage_distillation.py"]
            if accelerated:
                sources.extend(sorted((ROOT/'ep_loop/fastpath').glob('*.py')))
            for source in sources:
                destination = output/"code"/source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source, destination)
        student.model.train()
        setup_span = observer.start("optimizer_ddp_setup", echo=True)
        from .confidence import PromptConfidence
        confidence_probe = PromptConfidence(teacher,tokenizer,config['seed'],observer=observer) if config['algorithm'].get('boundary_signal','js')=='probe_know' else None
        objective = DistillationObjective(student, teacher, config["algorithm"],
            config["implementation"]["readout_chunk_size"], config["implementation"]["checkpoint_readout"], observer=observer,
            confidence_probe=confidence_probe)
        wrapped = DistributedDataParallel(objective, device_ids=[local_rank] if device.type=="cuda" else None,
                                          find_unused_parameters=True, gradient_as_bucket_view=True) if world>1 else objective
        opt = config["optimization"]
        optimizer = torch.optim.AdamW(student.model.parameters(), lr=opt["learning_rate"],
                                     weight_decay=opt["weight_decay"], betas=tuple(opt["betas"]), eps=opt["eps"])
        observer.finish(setup_span)
        # Keep initialization identical across ranks, then give stochastic model
        # states independent rank streams. Data order remains the shared seed 42.
        seed_everything(rank_seed(config["seed"], rank))
        data_rng = random.Random(config["seed"])
        generator = torch.Generator(device=device).manual_seed(rank_seed(config["seed"], rank))
        completed = 0
        if resume:
            resume_span = observer.start("resume_optimizer_rng", echo=True)
            state = torch.load(Path(resume)/"training_state.pt", map_location="cpu", weights_only=True)
            validate_resume_state(state,resume_info)
            optimizer.load_state_dict(state["optimizer"])
            completed = state["completed_updates"]
            rng = torch.load(Path(resume)/f"rng_rank{rank}.pt", map_location="cpu", weights_only=True)
            data_rng.setstate(rng["data_rng"])
            generator.set_state(rng["rollout_rng"])
            restore_rng_state(rng, device)
            observer.finish(resume_span, completed_updates=completed,target_extended=resume_info['target_extended'])
        if rank==0 and resume:copy_resume_metrics(resume,output,completed)
        if rank==0 and tensorboard_enabled():
            tensorboard = ScalarLogger(output,resumed_at=completed if resume else None,device=device)
            tensorboard.backfill(output/'metrics.jsonl',through_step=completed)
            observer.event('tensorboard_ready',echo=True,path=str(tensorboard.directory),resume_step=completed)
        k, s = config["algorithm"]["teacher_max_loops"], student_budget(config)
        dynamic = policy != "fixed"
        rollout = config["rollout"]
        rollout_seeds = rollout.get("sample_seeds")
        n_local = opt["global_batch_size"] // world
        final_update = min(opt["updates"], stop_after_updates) if stop_after_updates else opt["updates"]
        updates_bar = observer.progress(total=opt["updates"], initial=completed,
            desc="Train updates", unit="update", position=0)
        batcher=None
        if execution_mode(config)==HF_SHARED:
            from .hf_rollout import HFRolloutBatcher
            batcher=HFRolloutBatcher(student,config,observer)
        for update in range(completed, final_update):
            update_span = observer.start("update", echo=True, update=update+1, total_updates=opt["updates"])
            start = time.perf_counter()
            warmup = min(1.0, (update+1)/opt["warmup_steps"]) if opt["warmup_steps"] else 1.0
            for group in optimizer.param_groups:
                group["lr"] = opt["learning_rate"] * warmup
            optimizer.zero_grad(set_to_none=True)
            count = rollout["prompts_per_batch"]
            indices = sample_rollout_indices(data_rng, len(records), count)
            if accelerated:
                from .fastpath.residency import move_training_state
                from .fastpath.rlt_rollout import generate_rlt_rollouts
                with observer.stage('training_state_offload', echo=True, update=update+1):
                    if not teacher.current:
                        teacher.model.to('cpu')
                    move_training_state(student.model, optimizer, 'cpu')
                selected_records = [records[index] for index in indices]
                with TemporaryDirectory(prefix='.inference_weights_', dir=output) as snapshot:
                    with observer.stage('inference_weights_save', echo=True, update=update+1, weight_version=update):
                        weights = student.model.state_dict()
                        student.model.save_pretrained(snapshot, state_dict=weights, safe_serialization=True,
                                                      max_shard_size=sum(tensor.nbytes for tensor in weights.values()))
                        del weights
                    responses = generate_rlt_rollouts(student, snapshot, config, selected_records, update,
                                                      generator, output, teacher.gpu, observer)
                    teacher.prepare([(ident, sample, response) for ident, sample, response, _ in responses],
                                    selected_records, update,
                                    weights_path=Path(snapshot)/'model.safetensors' if teacher.current else None)
                rlt_responses = {(ident, sample): (response, seed) for ident, sample, response, seed in responses}
                with observer.stage('training_state_restore', echo=True, update=update+1):
                    move_training_state(student.model, optimizer, device)
                    if not teacher.current:
                        teacher.model.to(device)
            elif batcher is not None:
                from .hf_rollout import move_optimizer_state
                offload=config['implementation'].get('optimizer_offload_for_rollout',False) and device.type=='cuda'
                if offload:
                    with observer.stage('optimizer_state_offload',echo=True,update=update+1):
                        move_optimizer_state(optimizer,'cpu')
                batcher.prepare(records,indices,rank,world,update,generator)
                if offload:
                    with observer.stage('optimizer_state_restore',echo=True,update=update+1):
                        move_optimizer_state(optimizer,device)
            # [loss, responses, tokens, truncations, endpoint-only, shallow exits,
            #  rollout token-loops, teacher loss token-loops, JS sum, JS edge count,
            #  student loss token-loops, teacher coda token-positions]
            totals = torch.zeros(29, dtype=torch.float64, device=device)
            exit_hist = torch.zeros(k, device=device, dtype=torch.long)
            stage_hist = torch.zeros(s+1, device=device, dtype=torch.long)
            mapping = torch.zeros(s,k,device=device,dtype=torch.long)
            target_hist = torch.zeros(s, device=device, dtype=torch.long)
            actual_hist = torch.zeros(s, device=device, dtype=torch.long)
            confusion = torch.zeros(s,s,device=device,dtype=torch.long)
            # Per-layer physical slots, processed decode inputs, decode loops and depth-one exits.
            ragged_cost = torch.zeros(4, device=device, dtype=torch.long) if ragged else None
            micro = 0
            response_bar = observer.progress(total=n_local, desc=f"Update {update+1}: rank {rank} responses",
                unit="response", position=1, leave=False)
            for index in indices[rank::world]:
                record = records[index]
                threshold = (sample_training_threshold(config["algorithm"]["rad_tau"], config["seed"],
                                                       record["problem_sha256"], update)
                             if ragged else config["algorithm"].get("student_stop_threshold", .5))
                for sample in range(rollout["responses_per_prompt"]):
                    log_context = dict(update=update+1, id=record["id"], sample=sample)
                    if accelerated:
                        response, sampling_seed = rlt_responses.pop((record['id'], sample))
                        log_context.update(sampling_seed=sampling_seed, rollout_weight_version=update,
                                           teacher_weight_version=teacher.weight_version)
                        teacher.use_response(record['id'], sample)
                    elif rollout_seeds is not None:
                        sampling_seed = rollout_seeds[sample]
                        generator.manual_seed(sampling_seed)
                        log_context["sampling_seed"] = sampling_seed
                    response_span = observer.start("response", **log_context)
                    if batcher is not None:
                        response = batcher.pop(record['id'], sample)
                    elif not accelerated:
                        response = generate(student, record["input_ids"], s, rollout["max_new_tokens"],
                                        rollout["temperature"], rollout["top_p"], generator,
                                        use_cache=rollout["use_cache"], endpoint=policy,
                                        exit_threshold=threshold,
                                        observer=observer, log_context=log_context,
                                        progress=True, progress_desc=f"Generate {record['id']} #{sample}",
                                        forced_token_ids=offline.tokens(record["problem_sha256"], sample) if offline else None)
                    decoded = tokenizer.decode(response.token_ids) if config['implementation']['write_rollout_text'] else None
                    ids = torch.tensor([record["input_ids"]+response.token_ids], device=device, dtype=torch.long)
                    sync = wrapped.no_sync() if world>1 and micro<n_local-1 else nullcontext()
                    with sync:
                        with observer.stage("objective_forward", update=update+1):
                            loss, info = wrapped(ids, len(record["input_ids"]), response.exit_depths)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite response loss")
                        with observer.stage("backward", update=update+1):
                            (loss/n_local).backward()
                    loss_value = float(loss.detach())
                    totals += torch.tensor([loss_value, 1, len(response.token_ids), int(not response.reached_eos),
                        info["endpoint_only_tokens"], info["exit_not_deeper_than_student"],
                        response.executed_token_loops, info["teacher_executed_token_loops"],
                        float(info["js_sum"]), int(info["js_count"]),
                        info["student_executed_token_loops"], info["teacher_coda_token_positions"],
                        float(info["stage_loss"]), float(info["stop_loss"]),
                        info["required_response_stage_loops"], info["replay_additional_required_loops"],
                        info["stop_depth_absolute_error"], info["uncompressed_tokens"],
                        info["target_at_max_tokens"], response.seconds,
                        response.backfill_token_loops,response.coda_token_positions,
                        info['confidence_probe_token_loops'],info['confidence_probe_coda_token_positions'],
                        info['confidence_probe_forwards'],float(info['boundary_signal_sum']),
                        info.get('teacher_backfill_token_loops',0),
                        info.get('teacher_reused_token_loops',0),info.get('replay_unique_token_loops',0)],
                        device=device, dtype=torch.float64)
                    exit_hist += info["exit_hist"]
                    stage_hist += info["stage_hist"]
                    mapping += info["mapping_hist"]
                    target_hist += info["target_depth_hist"]
                    actual_hist += torch.bincount(torch.tensor(response.exit_depths,device=device),minlength=s+1)[1:]
                    confusion += info["target_actual_confusion"]
                    ragged_record = {}
                    if ragged:
                        decode_depths = response.exit_depths[1:]
                        if (response.backfill_token_loops or (not accelerated and
                                response.physical_kv_token_slots != response.executed_token_loops)):
                            raise RuntimeError("RAD-OPD rollout did not preserve Ragged KV accounting")
                        ragged_cost += torch.tensor([response.physical_kv_token_slots or 0, len(decode_depths),
                                                     sum(decode_depths), decode_depths.count(1)], device=device)
                        ragged_record.update(exit_threshold=threshold,
                                             physical_kv_token_slots=response.physical_kv_token_slots,
                                             response_source="offline_teacher" if offline else "online_student")
                    append_json(output/f"rollouts_rank{rank}.jsonl", dict(log_context,
                        problem_sha256=record["problem_sha256"], tokens=response.token_ids,
                        text=decoded if config["implementation"]["write_rollout_text"] else None,
                        reached_eos=response.reached_eos, student_executed_loops=None if dynamic else s,
                        student_exit_depths=response.exit_depths,
                        cache_policy=response.cache_policy,backfill_token_loops=response.backfill_token_loops,
                        coda_token_positions=response.coda_token_positions,
                        executed_token_loops=response.executed_token_loops, **ragged_record))
                    micro += 1
                    observer.finish(response_span, response_tokens=len(response.token_ids), loss=loss_value)
                    response_bar.update(1)
            response_bar.refresh()
            response_bar.close()
            if micro != n_local:
                raise RuntimeError("Unexpected local response count")
            with observer.stage("optimizer_step", update=update+1):
                grad_norm = torch.nn.utils.clip_grad_norm_(student.model.parameters(),
                    opt["gradient_clip_norm"] if opt["gradient_clip_norm"] is not None else float("inf"))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
            completed = update+1
            if world>1:
                with observer.stage("metrics_reduce", update=completed):
                    for value in (totals, exit_hist, stage_hist, mapping, target_hist, actual_hist, confusion):
                        dist.all_reduce(value)
                    if ragged:
                        dist.all_reduce(ragged_cost)
            row = dict(update=completed, mean_response_loss=float(totals[0]/totals[1]),
                responses=int(totals[1]), response_tokens=int(totals[2]), truncated_responses=int(totals[3]),
                endpoint_only_tokens=int(totals[4]), exit_not_deeper_than_student=int(totals[5]),
                rollout_executed_token_loops=int(totals[6]), teacher_loss_executed_token_loops=int(totals[7]),
                mean_adjacent_js_nats=float(totals[8]/totals[9]) if totals[9]>0 else None,
                exit_hist=exit_hist.tolist(), stage_count_hist=stage_hist.tolist(),
                teacher_depth_by_student_slot=mapping.tolist(), learning_rate=optimizer.param_groups[0]["lr"],
                gradient_norm=float(grad_norm),
                teacher_weight_version=update if config['algorithm']['teacher_update']=='current' else 'initial',
                student_loss_executed_token_loops=int(totals[10]),
                teacher_coda_token_positions=int(totals[11]),
                teacher_executed_loops=info["teacher_executed_loops"], student_executed_loops=None if dynamic else s,
                student_policy=policy, student_max_loops=s,
                mean_stage_loss=float(totals[12]/totals[1]), mean_stop_loss=float(totals[13]/totals[1]),
                target_student_depth_hist=target_hist.tolist(), actual_student_depth_hist=actual_hist.tolist(),
                target_actual_depth_confusion=confusion.tolist(),
                mean_target_student_depth=float(totals[14]/totals[2]),
                mean_actual_student_depth=float(sum((i+1)*v for i,v in enumerate(actual_hist.tolist()))/totals[2]),
                replay_additional_required_loops=int(totals[15]),
                stop_depth_mean_absolute_error=float(totals[16]/totals[2]),
                uncompressed_tokens=int(totals[17]), target_at_max_tokens=int(totals[18]),
                generation_seconds_sum=float(totals[19]),
                rollout_backfill_token_loops=int(totals[20]),rollout_coda_token_positions=int(totals[21]),
                confidence_probe_token_loops=int(totals[22]),confidence_probe_coda_token_positions=int(totals[23]),
                confidence_probe_forwards=int(totals[24]),boundary_signal=config['algorithm'].get('boundary_signal','js'),
                mean_boundary_change=float(totals[25]/totals[9]) if totals[9]>0 else None,
                teacher_backfill_token_loops=int(totals[26]),
                teacher_reused_token_loops=int(totals[27]),replay_unique_token_loops=int(totals[28]),
                teacher_reuses_student_forward=config['algorithm'].get('teacher_reuse_student',False),
                rollout_backend=rollout_backend, teacher_backend=teacher_backend,
                hf_rollout_batch_size=None if accelerated else implementation.get('hf_rollout_batch_size',1),
                teacher_execution_policy=config['algorithm']['teacher_endpoint'],
                teacher_max_loops=k,
                native_exit_reduces_executed_loops=ragged or student.native_exit_reduces_executed_loops,
                seconds=time.perf_counter()-start)
            if ragged:
                slots, positions, loops, depth_one = ragged_cost.tolist()
                row.update(physical_kv_token_slots=None if accelerated else slots, decode_token_positions=positions,
                           decode_token_loops=loops, mean_decode_depth=loops/positions if positions else None,
                           decode_depth_one_fraction=depth_one/positions if positions else None,
                           response_source="offline_teacher" if offline else "online_student")
            if rank==0:
                append_json(output/"metrics.jsonl",row)
                if tensorboard is not None:tensorboard.record(row)
                from tqdm.auto import tqdm
                tqdm.write(json.dumps(row), file=sys.stdout)
                observer.event("update_metrics", update=completed, metrics=row)
            updates_bar.set_postfix(loss=row["mean_response_loss"], depth=row["mean_actual_student_depth"], refresh=False)
            updates_bar.update(1)
            keep_latest = implementation.get('keep_latest_checkpoints', 0)
            if implementation.get('checkpoint_layout') == 'split':
                from .fastpath.checkpoints import checkpoint_due, save_split_checkpoint
                model_due, resume_due = checkpoint_due(implementation, completed, final_update)
                if model_due or resume_due:
                    path = output / 'checkpoints' / f'update_{completed:06d}' if model_due else output / 'resume/latest'
                    with observer.stage('checkpoint_save', echo=True, update=completed, checkpoint=str(path),
                                        save_model=model_due, save_resume=resume_due):
                        save_split_checkpoint(output, student, tokenizer, optimizer, completed, config,
                                              data_rng, generator, rank, world, ROOT,
                                              save_model=model_due, save_resume=resume_due,
                                              data_identity=data_report['usable_prompt_hash'])
            elif keep_latest or completed % implementation['save_every_updates'] == 0 or completed == final_update:
                folder = output/"checkpoints"/f"update_{completed:06d}"
                with observer.stage("checkpoint_save", echo=True, update=completed, checkpoint=str(folder)):
                    save_checkpoint(folder, student, tokenizer, optimizer, completed, config, data_rng, generator, rank, world,
                                    data_identity=data_report['usable_prompt_hash'])
                if rank == 0 and keep_latest:
                    for previous in sorted(folder.parent.glob('update_*')):
                        step = int(previous.name.removeprefix('update_'))
                        if step <= completed-keep_latest and step % config['implementation']['save_every_updates']:
                            shutil.rmtree(previous)
                            observer.event('checkpoint_pruned', update=step, checkpoint=str(previous))
            if accelerated:
                teacher.clear_batch()
            if accelerated or batcher is not None:
                # Release the final training graph before allocating rollout caches.
                del loss,info,ids
                if device.type=='cuda':torch.cuda.empty_cache()
            observer.finish(update_span, update_metrics_seconds=row["seconds"])
        updates_bar.close()
        if rank==0:
            write_json(output/"summary.json", dict(
                status="completed" if completed==opt["updates"] else "stopped_at_requested_update",
                completed_updates=completed, configured_updates=opt["updates"], smoke=smoke,
                experiment=config.get('experiment',{}),
                world_size=world, teacher_update=config["algorithm"]["teacher_update"],
                teacher_reuses_student_forward=config['algorithm'].get('teacher_reuse_student',False),
                model_instances_per_rank=1 if teacher is student else 2,
                rollout_backend=rollout_backend, teacher_backend=teacher_backend,
                hf_rollout_batch_size=None if accelerated else implementation.get('hf_rollout_batch_size',1),
                student_policy=policy,
                teacher_endpoint=config["algorithm"]["teacher_endpoint"], stage_selection=config["algorithm"]["stage_selection"],
                seconds=time.perf_counter()-started, data_report=data_report,
                started_at=observer.started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                observability=observer.metadata(),
                peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
                result_scope="Execution/semantics evidence only; benchmark scores require separate evaluation"))
    except Exception as exc:
        if rank==0:
            write_json(output/"failure.json",dict(error=type(exc).__name__,message=str(exc)))
        raise
    finally:
        try:
            if accelerated and teacher is not None:
                teacher.close()
        finally:
            if tensorboard is not None:tensorboard.close()
            if lock is not None:
                lock.close()
        # An OOM rank must exit promptly so torchrun can terminate its peers.
        # Collective cleanup on an exceptional path can wait the full multi-hour
        # rollout timeout while another rank is still inside backward/all-reduce.
        if world>1 and dist.is_initialized() and sys.exc_info()[0] is None:
            dist.destroy_process_group()
