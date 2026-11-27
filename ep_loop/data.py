"""Preserve source prompt messages; canonicalize IDs and response-only masks."""
from pathlib import Path
import hashlib
import json
import random


MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
PROMPT_SAMPLING = 'uniform_without_replacement_within_batch_independent_across_updates'


def sample_rollout_indices(rng, pool_size, prompts_per_batch):
    """Sample distinct rows within an update and allow revisits across updates."""
    if not 1 <= prompts_per_batch <= pool_size:
        raise ValueError('Training pool must contain at least prompts_per_batch distinct questions')
    return rng.sample(range(pool_size), prompts_per_batch)


def split_training_records(records, train_count, dev_count, eval_count, seed):
    """None selects all length-valid questions except the held-out dev suffix.

    Numeric train_count preserves the historical subset split and its ordering.
    """
    count = len(records)-dev_count if train_count is None else train_count
    if count < 1 or dev_count < 1 or not 1 <= eval_count <= dev_count:
        raise ValueError('Invalid train/dev/eval split sizes')
    if count+dev_count > len(records):
        raise ValueError('Not enough unique length-valid examples for TRAIN_EXAMPLES + DEV_EXAMPLES')
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    dev = shuffled[count:count+dev_count]
    return {'train.jsonl':shuffled[:count], 'dev.jsonl':dev, 'eval.jsonl':dev[:eval_count]}


def split_manifest_matches(saved, current):
    """Match the split and its input hashes across different storage locations."""
    def without_paths(manifest):
        result = dict(manifest)
        result.pop('source', None)
        for key in ('evaluation_dataset', 'training_decontamination_dataset'):
            if isinstance(result.get(key), dict):
                result[key] = {field: value for field, value in result[key].items() if field != 'path'}
        return result

    return isinstance(saved, dict) and without_paths(saved) == without_paths(current)


def canonical_hash(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_record(row, index=0):
    messages = row.get("messages", row.get("prompt"))
    if messages is None:
        problem = row.get("problem", row.get("question"))
        if not isinstance(problem, str):
            raise ValueError(f"Row {index} needs messages, prompt, problem, or question")
        messages = [{"role": "user", "content": problem + MATH_SUFFIX}]
    elif isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Invalid prompt messages at row {index}")
    if any(m.get("role") not in {"system", "user", "assistant"} or not isinstance(m.get("content"), str)
           for m in messages):
        raise ValueError(f"Only text chat messages are supported, row {index}")
    reward = row.get("reward_model") or {}
    if isinstance(reward, str):
        reward = json.loads(reward)
    answer = row.get("answer", row.get("ground_truth", reward.get("ground_truth")))
    fingerprint = canonical_hash(messages)
    ident = row.get("id", row.get("unique_id", (row.get("extra_info") or {}).get("index", fingerprint)))
    if ident is None:
        ident = fingerprint
    record = dict(id=str(ident), messages=messages, answer=None if answer is None else str(answer),
                  problem_sha256=fingerprint)
    if row.get('benchmark'):
        record['benchmark'] = str(row['benchmark'])
        record['problem'] = row.get('problem', messages[-1]['content'].removesuffix(MATH_SUFFIX))
    return record


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class OfflineResponses:
    """Index pre-generated teacher trajectories without retaining all tokens in RAM."""

    def __init__(self, path, records, responses_per_prompt):
        self.path = Path(path)
        prompts = {record['problem_sha256'] for record in records}
        self.offsets = {}
        digest = hashlib.sha256()
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt, sample = row['problem_sha256'], row['sample']
                if type(sample) is not int or sample < 0:
                    raise ValueError('Offline response sample must be a nonnegative integer')
                if prompt in prompts and sample < responses_per_prompt:
                    key = prompt, sample
                    if key in self.offsets or not isinstance(row.get('tokens'), list) or not row['tokens']:
                        raise ValueError(f'Duplicate or empty offline response for {key}')
                    self.offsets[key] = offset
        expected = len(prompts) * responses_per_prompt
        if len(self.offsets) != expected:
            raise ValueError(f'Offline responses cover {len(self.offsets)}/{expected} required question/sample pairs')
        self.sha256 = digest.hexdigest()

    def tokens(self, problem_hash, sample):
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[problem_hash, sample])
            row = json.loads(stream.readline())
        if (row['problem_sha256'], row['sample']) != (problem_hash, sample):
            raise ValueError('Offline response file changed after indexing')
        return row['tokens']


def prompt_ids(tokenizer, record, enable_thinking=True, repair_template_newlines=True):
    # Repair only the Jinja template string, never the user's mathematical text.
    old = getattr(tokenizer, "chat_template", None)
    try:
        if old and repair_template_newlines:
            tokenizer.chat_template = old.replace("\\\\n", "\\n")
        ids = tokenizer.apply_chat_template(record["messages"], tokenize=True,
                                            add_generation_prompt=True,
                                            enable_thinking=enable_thinking)
    finally:
        if old is not None:
            tokenizer.chat_template = old
    return list(ids)


def filter_policy(config):
    """Only correctness acceptance is reusable; per-token depths are not labels."""
    from .reproducibility import SAMPLE_SEED_SCHEME
    policy = dict(model_revision=config["model"]["revision"],
                schema_version=2,require_eos=config.get('distillation',{}).get('require_eos',True),
                seed=config["seed"], sampling_seed_scheme=SAMPLE_SEED_SCHEME,
                max_loops=config["algorithm"]["teacher_max_loops"],
                exit_rule="ouro_native_cdf", exit_threshold=config["algorithm"]["exit_threshold"],
                enable_thinking=config["model"]["enable_thinking"],
                repair_template_newlines=config["model"]["repair_template_newlines"],
                max_prompt_tokens=config["data"]["max_prompt_tokens"],
                sampling={k:config["rollout"][k] for k in ("temperature", "top_p", "max_new_tokens")},
                verifier="prompt_aligned_integer_v2")
    if config["model"]["family"] == "huginn":
        policy.update(model_family="huginn", exit_rule="huginn_native_kl_previous_current",
                      cache_lookup_strategy="latest-m4", prefill="native_whole_prompt",
                      initial_state="native_random_per_forward")
    return policy


def prepare_training_records(config, tokenizer, root, smoke=False, observer=None):
    parquet = None
    if smoke:
        raw = [{"id":f"smoke-{i}", "problem":f"What is {i}+1?", "answer":str(i+1)} for i in range(8)]
    else:
        path = Path(config["data"]["train_path"])
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise FileNotFoundError(f"Training data missing: {path}")
        if path.suffix == '.parquet':
            from pyarrow.parquet import ParquetFile
            parquet = ParquetFile(path)
            batches = parquet.iter_batches(batch_size=4096, columns=['prompt', 'reward_model'])
            raw = (row for batch in batches for row in batch.to_pylist())
        else:
            raw = read_jsonl(path)
    source_count = parquet.metadata.num_rows if parquet is not None else len(raw)
    expected = config["data"]["expected_examples"]
    if expected is not None and source_count != expected:
        raise ValueError(f"Source dataset has {source_count} rows; configuration expects {expected}")
    encoded = []
    seen_ids = set()
    # Published DAPO repeats prompts. Reuse tokenization, preserving each source row.
    encoded_prompts = {} if parquet is not None else None
    overlong = 0
    source_records = (observer.progress(raw, total=source_count, desc="Tokenize prompts", unit="row", leave=False)
                      if observer else raw)
    for index, row in enumerate(source_records):
        record = None
        if encoded_prompts is not None:
            key = (tuple((msg['role'], msg['content']) for msg in row['prompt']),
                   (row.get('reward_model') or {}).get('ground_truth'))
            record = encoded_prompts.get(key)
        if record is None:
            record = normalize_record(row, index)
            record['input_ids'] = prompt_ids(tokenizer, record, config['model']['enable_thinking'],
                                             config['model']['repair_template_newlines'])
            if encoded_prompts is not None:
                encoded_prompts[key] = record
        if encoded_prompts is not None:
            record = dict(record, id=str(index))
        else:
            if record['id'] in seen_ids:
                raise ValueError('Training IDs are not unique')
            seen_ids.add(record['id'])
        ids = record['input_ids']
        if len(ids) > config["data"]["max_prompt_tokens"]:
            overlong += 1
            continue
        if not ids:
            raise ValueError("Empty prompt")
        encoded.append(record)
    filter_path = config["data"].get("correctness_cache")
    question_filter = config.get('distillation',{}).get('question_filter','teacher_correct' if filter_path else 'all')
    filter_rejected = 0
    if filter_path:
        cache = read_jsonl(root / filter_path)
        expected_policy = filter_policy(config)
        if not cache or cache[0].get("kind") != "metadata" or cache[0].get("policy") != expected_policy:
            raise ValueError("Correctness cache does not match the frozen initial dynamic-teacher policy")
        labels = {row["id"]:row for row in cache[1:]}
        if len(labels)!=len(cache)-1:
            raise ValueError('Correctness cache contains duplicate problem IDs')
        kept = []
        for record in encoded:
            label = labels.get(record["id"])
            if label is None or label["problem_sha256"] != record["problem_sha256"]:
                raise ValueError(f"Missing or mismatched cached correctness for {record['id']}")
            if label.get('answer_sha256') != canonical_hash(record['answer']):
                raise ValueError(f"Correctness cache ground truth changed for {record['id']}")
            if not isinstance(label["correct"], bool):
                raise ValueError("Correctness cache labels must be booleans")
            if label["correct"] is True:
                kept.append(record)
        filter_rejected = len(encoded) - len(kept)
        encoded = (random.Random(config['seed']).sample(encoded,len(kept)) if question_filter=='random_matched' else kept)
    if not encoded:
        raise ValueError("No usable training prompts after filtering/length checks")
    return encoded, dict(source_examples=source_count, correctness_rejected=filter_rejected,
                         prompt_sampling=PROMPT_SAMPLING,
                         question_filter=question_filter,
                         overlong_prompts_dropped=overlong, usable_examples=len(encoded),
                         usable_prompt_hash=canonical_hash([r["problem_sha256"] for r in encoded]))


class ByteTokenizer:
    """Test-only tokenizer for the random 260-token official tiny architecture."""
    eos_token_id = 2
    chat_template = None

    def encode(self, text, **kwargs):
        return [byte+4 for byte in text.encode()]

    def decode(self, ids, **kwargs):
        return bytes(i-4 for i in ids if 4 <= int(i) < 260).decode(errors="replace")

    def apply_chat_template(self, messages, **kwargs):
        text = "\n".join(m["role"]+": "+m["content"] for m in messages)+"\nassistant: "
        return [1]+self.encode(text)
