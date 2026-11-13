"""Full-parameter LoopOPD/D-LoopOPD and native-exit stage distillation.

Every optimizer update uses B*G fresh fixed-S student responses. Microbatches
only accumulate gradients: weights/teacher stay unchanged until the batch ends.
Distributed ranks have equal response counts, so DDP means equal response means.
"""
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import copy
import fcntl
import json
import os
import random
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import assets_status, load_config, smoke_config, validate_config
from .backends import load_backend
from .data import ByteTokenizer, canonical_hash, prepare_training_records
from .inference import generate
from .objective import DistillationObjective

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
        load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True)
    else:
        from safetensors.torch import load_file
        weights = load_file(folder / "model.safetensors")
        missing, unexpected = model.load_state_dict(weights, strict=False)
        # HF removes duplicate tied embedding tensors when saving Huginn.
        # Permit only missing aliases of an actually loaded shared Parameter.
        names = dict(model.named_parameters(remove_duplicate=False))
        aliases = {key for key in missing if key in names and any(
            other in weights and parameter is names[key] for other, parameter in names.items())}
        if set(missing) - aliases or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def save_checkpoint(folder, student, tokenizer, optimizer, completed, config,
                    data_rng, generator, rank, world_size):
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
                            world_size=world_size, config_hash=canonical_hash(config)),
                       folder / "training_state.pt")
    if world_size > 1:
        dist.barrier()
    torch.save(dict(data_rng=data_rng.getstate(), rollout_rng=generator.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state() if student.device.type == "cuda" else None),
               folder / f"rng_rank{rank}.pt")
    if world_size > 1:
        dist.barrier()


def train(config_path, output, device_name="auto", smoke=False, resume=None,
          stop_after_updates=None):
    config = load_config(config_path)
    if smoke:
        config = smoke_config(config)
    validate_config(config)
    if not smoke:
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
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    try:
        if rank == 0:
            lock = (output / "run.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (output / "metrics.jsonl").exists() and not resume:
                raise FileExistsError("Use a new run output or --resume; existing observations are protected")
            if resume and (output / "metrics.jsonl").exists():
                saved_state = torch.load(Path(resume)/"training_state.pt", map_location="cpu", weights_only=True)
                logged = [json.loads(line)["update"] for line in (output/"metrics.jsonl").read_text().splitlines()]
                if logged and max(logged) > saved_state["completed_updates"]:
                    raise ValueError("Run log has updates newer than the chosen checkpoint; resume into a new output")
        if world > 1:
            dist.barrier()
        started = time.perf_counter()
        model_dir = (ROOT / config["model"]["path"]).resolve()
        student, loading = load_backend(config["model"], model_dir, device,
            config["optimization"]["mixed_precision"], config["optimization"]["gradient_checkpointing"], smoke)
        if smoke:
            tokenizer = ByteTokenizer()
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        # Clone before loading a resumed student: frozen means the INITIAL model.
        if config["algorithm"]["teacher_update"] == "frozen":
            teacher = type(student)(copy.deepcopy(student.model).eval().requires_grad_(False), student.mixed_precision)
        else:
            teacher = student
        if resume:
            read_checkpoint_weights(student.model, resume)
        records, data_report = prepare_training_records(config, tokenizer, ROOT, smoke)
        if rank == 0:
            write_json(output / "resolved_config.json", config)
            write_json(output / "data_report.json", data_report)
            write_json(output / "provenance.json", dict(
                started_at=datetime.now(timezone.utc).isoformat(), torch=torch.__version__,
                world_size=world, device=str(device), loading=loading,
                original_training_code_used=False, original_paper_scores_reproduced=False,
                source_proposal="docs/loop_distillation_research_proposal_zh.md",
                model_source_directory=str(model_dir), smoke=smoke,
                teacher_parameter_refresh=config["algorithm"]["teacher_update"],
                native_exit_reduces_executed_loops=student.native_exit_reduces_executed_loops,
                model_family=config["model"]["family"],
            ))
            for source in [*sorted((ROOT/"ep_loop").glob("*.py")), ROOT/"scripts/train_stage_distillation.py"]:
                destination = output/"code"/source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source, destination)
        student.model.train()
        objective = DistillationObjective(student, teacher, config["algorithm"],
            config["implementation"]["readout_chunk_size"], config["implementation"]["checkpoint_readout"])
        wrapped = DistributedDataParallel(objective, device_ids=[local_rank] if device.type=="cuda" else None,
                                          find_unused_parameters=True) if world>1 else objective
        opt = config["optimization"]
        optimizer = torch.optim.AdamW(student.model.parameters(), lr=opt["learning_rate"],
                                     weight_decay=opt["weight_decay"], betas=tuple(opt["betas"]), eps=opt["eps"])
        data_rng = random.Random(config["seed"])
        generator = torch.Generator(device=device).manual_seed(config["seed"]+rank*100003)
        completed = 0
        if resume:
            state = torch.load(Path(resume)/"training_state.pt", map_location="cpu", weights_only=True)
            if state["config_hash"] != canonical_hash(config) or state["world_size"] != world:
                raise ValueError("Resume requires the same configuration and world size")
            optimizer.load_state_dict(state["optimizer"])
            completed = state["completed_updates"]
            rng = torch.load(Path(resume)/f"rng_rank{rank}.pt", map_location="cpu", weights_only=True)
            data_rng.setstate(rng["data_rng"])
            generator.set_state(rng["rollout_rng"])
            torch.set_rng_state(rng["torch_rng"])
            if device.type == "cuda" and rng["cuda_rng"] is not None:
                torch.cuda.set_rng_state(rng["cuda_rng"], device)
        k, s = config["algorithm"]["teacher_max_loops"], config["algorithm"]["student_loops"]
        rollout = config["rollout"]
        n_local = opt["global_batch_size"] // world
        final_update = min(opt["updates"], stop_after_updates) if stop_after_updates else opt["updates"]
        for update in range(completed, final_update):
            start = time.perf_counter()
            warmup = min(1.0, (update+1)/opt["warmup_steps"]) if opt["warmup_steps"] else 1.0
            for group in optimizer.param_groups:
                group["lr"] = opt["learning_rate"] * warmup
            optimizer.zero_grad(set_to_none=True)
            count = rollout["prompts_per_batch"]
            indices = data_rng.sample(range(len(records)), count) if len(records)>=count else data_rng.choices(range(len(records)), k=count)
            # [loss, responses, tokens, truncations, endpoint-only, shallow exits,
            #  rollout token-loops, teacher loss token-loops, JS sum, JS edge count,
            #  student loss token-loops, teacher coda token-positions]
            totals = torch.zeros(12, dtype=torch.float64, device=device)
            exit_hist = torch.zeros(k, device=device, dtype=torch.long)
            stage_hist = torch.zeros(s+1, device=device, dtype=torch.long)
            mapping = torch.zeros(s,k,device=device,dtype=torch.long)
            micro = 0
            for index in indices[rank::world]:
                record = records[index]
                for sample in range(rollout["responses_per_prompt"]):
                    response = generate(student, record["input_ids"], s, rollout["max_new_tokens"],
                                        rollout["temperature"], rollout["top_p"], generator,
                                        use_cache=rollout["use_cache"])
                    ids = torch.tensor([record["input_ids"]+response.token_ids], device=device, dtype=torch.long)
                    sync = wrapped.no_sync() if world>1 and micro<n_local-1 else nullcontext()
                    with sync:
                        loss, info = wrapped(ids, len(record["input_ids"]))
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite response loss")
                        (loss/n_local).backward()
                    totals += torch.tensor([float(loss.detach()), 1, len(response.token_ids), int(not response.reached_eos),
                        info["endpoint_only_tokens"], info["exit_not_deeper_than_student"],
                        response.executed_token_loops, info["teacher_executed_token_loops"],
                        float(info["js_sum"]), int(info["js_count"]),
                        info["student_executed_token_loops"], info["teacher_coda_token_positions"]],
                        device=device, dtype=torch.float64)
                    exit_hist += info["exit_hist"]
                    stage_hist += info["stage_hist"]
                    mapping += info["mapping_hist"]
                    append_json(output/f"rollouts_rank{rank}.jsonl", dict(
                        update=update+1, id=record["id"], sample=sample,
                        problem_sha256=record["problem_sha256"], tokens=response.token_ids,
                        text=tokenizer.decode(response.token_ids) if config["implementation"]["write_rollout_text"] else None,
                        reached_eos=response.reached_eos, student_executed_loops=s))
                    micro += 1
            if micro != n_local:
                raise RuntimeError("Unexpected local response count")
            grad_norm = torch.nn.utils.clip_grad_norm_(student.model.parameters(),
                opt["gradient_clip_norm"] if opt["gradient_clip_norm"] is not None else float("inf"))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            completed = update+1
            if world>1:
                for value in (totals, exit_hist, stage_hist, mapping):
                    dist.all_reduce(value)
            row = dict(update=completed, mean_response_loss=float(totals[0]/totals[1]),
                responses=int(totals[1]), response_tokens=int(totals[2]), truncated_responses=int(totals[3]),
                endpoint_only_tokens=int(totals[4]), exit_not_deeper_than_student=int(totals[5]),
                rollout_executed_token_loops=int(totals[6]), teacher_loss_executed_token_loops=int(totals[7]),
                mean_adjacent_js_nats=float(totals[8]/totals[9]) if totals[9]>0 else None,
                exit_hist=exit_hist.tolist(), stage_count_hist=stage_hist.tolist(),
                teacher_depth_by_student_slot=mapping.tolist(), learning_rate=optimizer.param_groups[0]["lr"],
                gradient_norm=float(grad_norm), teacher_weight_version=update if teacher is student else "initial",
                student_loss_executed_token_loops=int(totals[10]),
                teacher_coda_token_positions=int(totals[11]),
                teacher_executed_loops=info["teacher_executed_loops"], student_executed_loops=s,
                teacher_max_loops=k,
                native_exit_reduces_executed_loops=student.native_exit_reduces_executed_loops,
                seconds=time.perf_counter()-start)
            if rank==0:
                append_json(output/"metrics.jsonl",row)
                print(json.dumps(row),flush=True)
            if completed % config["implementation"]["save_every_updates"] == 0 or completed == final_update:
                folder = output/"checkpoints"/f"update_{completed:06d}"
                save_checkpoint(folder, student, tokenizer, optimizer, completed, config, data_rng, generator, rank, world)
        if rank==0:
            write_json(output/"summary.json", dict(
                status="completed" if completed==opt["updates"] else "stopped_at_requested_update",
                completed_updates=completed, configured_updates=opt["updates"], smoke=smoke,
                world_size=world, teacher_update=config["algorithm"]["teacher_update"],
                teacher_endpoint=config["algorithm"]["teacher_endpoint"], stage_selection=config["algorithm"]["stage_selection"],
                seconds=time.perf_counter()-started, data_report=data_report,
                peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
                result_scope="Execution/semantics evidence only; benchmark scores require separate evaluation"))
    except Exception as exc:
        if rank==0:
            write_json(output/"failure.json",dict(error=type(exc).__name__,message=str(exc)))
        raise
    finally:
        if lock is not None:
            lock.close()
        if world>1 and dist.is_initialized():
            dist.destroy_process_group()
