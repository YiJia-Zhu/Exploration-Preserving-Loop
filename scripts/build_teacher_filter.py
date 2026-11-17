"""Optional once-per-problem correctness cache from a frozen native-exit teacher."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    from ep_loop.config import DEFAULT_CONFIG
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.limit is not None and args.limit<1:
        p.error('--limit must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    from ep_loop.observability import RunLogger
    with RunLogger(args.output.with_suffix('.events.jsonl'), args.output.with_suffix('.console.log'), 'teacher_filter') as observer:
        with observer.stage('teacher_filter_run', echo=True, config_path=args.config):
            build_filter(args, observer)


def build_filter(args, observer):
    import torch
    from transformers import AutoTokenizer
    from ep_loop.config import load_config, merge, validate_config
    from ep_loop.data import filter_policy, prepare_training_records, canonical_hash
    from ep_loop.inference import generate, integer_answer, verify_integer_response
    from ep_loop.backends import load_backend
    from ep_loop.reproducibility import execution_provenance, sample_seed, seed_everything
    from ep_loop.observability import model_progress
    root = Path(__file__).resolve().parents[1]
    config = validate_config(load_config(args.config))
    seed_everything(config["seed"])
    model_dir = (root/config["model"]["path"]).resolve()
    with observer.stage('model_load', echo=True, model_path=str(model_dir), device=args.device), model_progress(observer.show_progress):
        teacher, _ = load_backend(config["model"],model_dir,args.device,config["optimization"]["mixed_precision"],False)
    teacher.model.eval()  # generate() is no_grad; retain autocast-cache eligibility.
    with observer.stage('tokenizer_load', echo=True):
        tokenizer = AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    with observer.stage('data_prepare', echo=True):
        records, _ = prepare_training_records(merge(config,{"data":{"correctness_cache":None},
            'distillation':{'question_filter':'all'}}),tokenizer,root,observer=observer)
    if args.limit:
        records = records[:args.limit]
    if any(integer_answer(r["answer"]) is None for r in records):
        raise ValueError("This filter verifier requires integer ground truth")
    policy = filter_policy(config)
    reproducibility = execution_provenance(root, teacher.device, config["seed"], phase="teacher_filter")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    accepted_count, token_loops, generation_seconds = 0, 0, 0.
    with args.output.open("x") as f:
        f.write(json.dumps(dict(kind="metadata",policy=policy,reproducibility=reproducibility,
            scope="Frozen initial teacher, one answer per problem; reuse correctness only"))+"\n")
        for record in observer.progress(records, desc='Teacher filter', unit='problem', position=0):
            sample_span = observer.start('filter_sample', id=record['id'])
            completion_seed = sample_seed(config["seed"], record["problem_sha256"], 0)
            seed_everything(completion_seed)
            rng = torch.Generator(device=teacher.device).manual_seed(completion_seed)
            response = generate(teacher, record["input_ids"], policy["max_loops"],
                policy["sampling"]["max_new_tokens"], policy["sampling"]["temperature"],
                policy["sampling"]["top_p"], rng, endpoint="native", exit_threshold=policy["exit_threshold"],
                observer=observer, log_context=dict(id=record['id']), progress=True,
                progress_desc=f"Filter {record['id']}", progress_position=1)
            text = tokenizer.decode(response.token_ids)
            verdict = verify_integer_response(text,record["answer"],record["messages"])
            row = dict(id=record["id"],problem_sha256=record["problem_sha256"],text=text,
                answer_sha256=canonical_hash(record['answer']),verifier_correct=verdict['correct'],
                sampling_seed=completion_seed,
                response_tokens=len(response.token_ids),reached_eos=response.reached_eos,
                executed_token_loops=response.executed_token_loops,**verdict)
            row['correct'] = verdict['correct'] and (response.reached_eos or not policy['require_eos'])
            accepted_count += int(row['correct'])
            token_loops += response.executed_token_loops
            generation_seconds += response.seconds
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
            f.flush()
            observer.finish(sample_span, response_tokens=len(response.token_ids), teacher_filter_accepted=row['correct'], **verdict)
    args.output.with_suffix('.summary.json').write_text(json.dumps(dict(problems=len(records),
        accepted_problems=accepted_count,acceptance_rate=accepted_count/len(records) if records else 0,
        executed_token_loops=token_loops,generation_seconds=generation_seconds,
        policy=policy,observability=observer.metadata()),ensure_ascii=False,indent=2)+'\n')


if __name__ == "__main__":
    main()
