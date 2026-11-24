"""Freeze the BBH frontier inputs from pinned public questions and CoT prompts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / '.cache' / 'huggingface_datasets'
FRONTIER_DIR = 'frontier_135_seed1234'
DEFAULT_FRONTIER_DIR = ROOT / 'data' / 'bbh_frontier'
DATASET = 'lukaemon/bbh'
DATASET_REVISION = '982bb89fd79532a8ac676a61fc42eb1aeec63f99'
PROMPTS = 'suzgunmirac/BIG-Bench-Hard'
PROMPT_REVISION = '9ee07bd481feebf959a6b59d61ea57bdcf30964d'
SEED = 1234
PER_TASK = 5
TASKS = tuple('''
boolean_expressions causal_judgement date_understanding disambiguation_qa
dyck_languages formal_fallacies geometric_shapes hyperbaton
logical_deduction_five_objects logical_deduction_seven_objects logical_deduction_three_objects
movie_recommendation multistep_arithmetic_two navigate object_counting penguins_in_a_table
reasoning_about_colored_objects ruin_names salient_translation_error_detection snarks
sports_understanding temporal_sequences tracking_shuffled_objects_five_objects
tracking_shuffled_objects_seven_objects tracking_shuffled_objects_three_objects web_of_lies word_sorting
'''.split())


def cached_download(url, path):
    if path.is_file():
        return path.read_bytes()
    with urlopen(url, timeout=30) as response:
        contents = response.read()
    write_frozen(path, contents)
    return contents


def build_task(task, cache_dir):
    import pyarrow.parquet as pq

    dataset_file = f'{task}/test-00000-of-00001.parquet'
    prompt_name = f'{task}.txt'
    dataset_url = f'https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/{dataset_file}'
    prompt_url = f'https://raw.githubusercontent.com/{PROMPTS}/{PROMPT_REVISION}/cot-prompts/{prompt_name}'
    dataset = cached_download(dataset_url, cache_dir / DATASET_REVISION / dataset_file)
    prompt_file = cached_download(prompt_url, cache_dir / 'cot-prompts' / PROMPT_REVISION / prompt_name)
    rows = pq.read_table(io.BytesIO(dataset), columns=['input', 'target']).to_pylist()
    _, delimiter, demonstrations = prompt_file.decode().partition('\n-----\n')
    demonstrations = demonstrations.strip()
    if not delimiter or len(re.findall(r'^Q:', demonstrations, re.M)) != 3:
        raise ValueError(f'Expected three official CoT demonstrations in {task}')

    # E.1 gives the seed and sample count. Restart the documented seed per task.
    indices = sorted(random.Random(SEED).sample(range(len(rows)), PER_TASK))
    examples = []
    for index in indices:
        question, target = rows[index]['input'].strip(), rows[index]['target'].strip()
        if not question or not target or f'\nQ: {question}\nA:' in demonstrations:
            raise ValueError(f'Invalid or duplicated demonstration question in {task}/{index}')
        prompt = f"{demonstrations}\n\nQ: {question}\nA: Let's think step by step."
        examples.append(dict(id=f'{task}/{index}', task=task, source_index=index, target=target,
                             prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    evidence = dict(task=task, source_examples=len(rows), sampled_indices=indices,
                    dataset_url=dataset_url, dataset_sha256=hashlib.sha256(dataset).hexdigest(),
                    prompt_url=prompt_url, prompt_source_sha256=hashlib.sha256(prompt_file).hexdigest())
    return examples, evidence


def write_frozen(path, contents):
    if path.exists():
        if path.read_bytes() != contents:
            raise FileExistsError(f'Existing frozen file differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix='.download-') as temporary:
        staged = Path(temporary) / path.name
        staged.write_bytes(contents)
        try:
            path.hardlink_to(staged)
        except FileExistsError:
            if path.read_bytes() != contents:
                raise FileExistsError(f'Existing frozen file differs: {path}') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, help='Root for raw inputs (defaults to .cache/huggingface_datasets)')
    parser.add_argument('--output-dir', type=Path,
                        help=f'Defaults to data/bbh_frontier, or BBH/{FRONTIER_DIR} with --cache-dir')
    args = parser.parse_args()
    cache_dir = (args.cache_dir or DEFAULT_CACHE_DIR) / 'BBH'
    output_dir = args.output_dir or (cache_dir / FRONTIER_DIR if args.cache_dir else DEFAULT_FRONTIER_DIR)
    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = list(executor.map(lambda task: build_task(task, cache_dir), TASKS))
    examples = [example for task_examples, _ in tasks for example in task_examples]
    if Counter(row['task'] for row in examples) != dict.fromkeys(TASKS, PER_TASK):
        raise ValueError('Unexpected BBH frontier task coverage')

    data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in examples).encode()
    manifest = dict(dataset=DATASET, dataset_revision=DATASET_REVISION, split='test',
                    prompts=PROMPTS, prompt_revision=PROMPT_REVISION, examples=len(examples),
                    sample_seed=SEED, samples_per_task=PER_TASK,
                    sample_method='Python Random(seed).sample independently per task, then sort source indices',
                    exact_paper_subset_verified=False, output_sha256=hashlib.sha256(data).hexdigest(),
                    tasks=[evidence for _, evidence in tasks])
    write_frozen(output_dir / 'eval.jsonl', data)
    write_frozen(output_dir / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    print(f'Frozen {len(examples)} questions across {len(tasks)} BBH tasks in {output_dir}')


if __name__ == '__main__':
    main()
