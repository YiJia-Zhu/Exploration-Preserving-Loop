"""Fixed-depth math evaluation with the paper's 16-sample decoding settings."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--loops", type=int, action="append")
    p.add_argument("--samples-per-problem", type=int, help="Use 1 for the paper's OlympiadBench/BeyondAIME protocol")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from transformers import AutoTokenizer
    from ep_loop.config import load_config,validate_config
    from ep_loop.data import normalize_record,prompt_ids,read_jsonl
    from ep_loop.inference import generate,integer_answer,math_metrics,verify_integer_response
    from ep_loop.backends import load_backend
    config = validate_config(load_config(args.config))
    records = [normalize_record(row,i) for i,row in enumerate(read_jsonl(args.data))]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Evaluation problem IDs must be unique")
    if any(integer_answer(r["answer"]) is None for r in records):
        raise ValueError("Use the official external evaluator for noninteger/symbolic ground truth")
    backend,_ = load_backend(config["model"],args.checkpoint,args.device,config["optimization"]["mixed_precision"],False)
    backend.model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    settings = dict(config["evaluation"])
    if args.samples_per_problem is not None:
        if args.samples_per_problem < 1:
            p.error("--samples-per-problem must be positive")
        settings["samples_per_problem"] = args.samples_per_problem
    args.output.parent.mkdir(parents=True,exist_ok=True)
    summaries = {}
    with args.output.open("x") as f:
        for loops in args.loops or settings["student_loops"]:
            rng = torch.Generator(device=backend.device).manual_seed(config["seed"])
            correctness = {}
            for record in records:
                ids = prompt_ids(tokenizer,record,config["model"]["enable_thinking"],config["model"]["repair_template_newlines"])
                correctness[record["id"]] = []
                for sample in range(settings["samples_per_problem"]):
                    response = generate(backend,ids,loops,settings["max_new_tokens"],settings["temperature"],settings["top_p"],rng)
                    text = tokenizer.decode(response.token_ids)
                    verdict = verify_integer_response(text,record["answer"])
                    correctness[record["id"]].append(verdict["correct"])
                    f.write(json.dumps(dict(id=record["id"],loops=loops,sample=sample,text=text,
                        response_tokens=len(response.token_ids),reached_eos=response.reached_eos,**verdict),ensure_ascii=False)+"\n")
                    f.flush()
            summaries[str(loops)] = math_metrics(correctness)
    report = dict(settings=settings,results=summaries,
        verifier="strict_boxed_integer_v1; narrower than original numerical/symbolic equivalence",
        scope="Observed fixed-depth samples only; benchmark identity must be checked separately")
    args.output.with_suffix(".summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
