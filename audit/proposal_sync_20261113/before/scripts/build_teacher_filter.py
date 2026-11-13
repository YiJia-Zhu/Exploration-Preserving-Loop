"""Optional once-per-problem correctness cache from a frozen native-exit teacher."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from transformers import AutoTokenizer
    from ep_loop.config import load_config, merge, validate_config
    from ep_loop.data import filter_policy, prepare_training_records
    from ep_loop.inference import generate, integer_answer, verify_integer_response
    from ep_loop.ouro import load_ouro
    root = Path(__file__).resolve().parents[1]
    config = validate_config(load_config(args.config))
    model_dir = (root/config["model"]["path"]).resolve()
    teacher, _ = load_ouro(model_dir,args.device,config["optimization"]["mixed_precision"],False)
    teacher.model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    records, _ = prepare_training_records(merge(config,{"data":{"correctness_cache":None}}),tokenizer,root)
    if args.limit:
        records = records[:args.limit]
    if any(integer_answer(r["answer"]) is None for r in records):
        raise ValueError("This filter verifier requires integer ground truth")
    policy = filter_policy(config)
    rng = torch.Generator(device=teacher.device).manual_seed(config["seed"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(dict(kind="metadata",policy=policy,scope="Frozen initial teacher, one answer per problem; reuse correctness only"))+"\n")
        for record in records:
            response = generate(teacher, record["input_ids"], policy["max_loops"],
                policy["sampling"]["max_new_tokens"], policy["sampling"]["temperature"],
                policy["sampling"]["top_p"], rng, endpoint="native", exit_threshold=policy["exit_threshold"])
            text = tokenizer.decode(response.token_ids)
            verdict = verify_integer_response(text,record["answer"])
            row = dict(id=record["id"],problem_sha256=record["problem_sha256"],text=text,
                response_tokens=len(response.token_ids),reached_eos=response.reached_eos,
                executed_token_loops=response.executed_token_loops,**verdict)
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
            f.flush()
            print(json.dumps({k:row[k] for k in ("id","correct","response_tokens")}),flush=True)


if __name__ == "__main__":
    main()
