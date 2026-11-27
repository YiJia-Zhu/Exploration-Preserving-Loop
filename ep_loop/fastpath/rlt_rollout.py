"""Generate one RAD-OPD update from a snapshot of the current student."""
from pathlib import Path
import time

import torch

from ep_loop.config import student_budget
from ep_loop.inference import GeneratedResponse
from ep_loop.ragged import sample_training_threshold
from .rlt_actor import RltActor


def generate_rlt_rollouts(student, snapshot, config, records, update, generator, output, gpu, observer):
    rollout = config['rollout']
    loops = student_budget(config)
    samples = rollout['responses_per_prompt']
    cap = rollout['max_new_tokens']
    seeds = rollout.get('sample_seeds')
    if seeds is None:
        seeds = torch.randint(0, 2**63 - 1, (len(records) * samples,), generator=generator,
                              device=generator.device, dtype=torch.int64).tolist()
    else:
        seeds = seeds * len(records)

    jobs, requests = [], []
    for record in records:
        threshold = sample_training_threshold(config['algorithm']['rad_tau'], config['seed'],
                                              record['problem_sha256'], update)
        for sample in range(samples):
            seed = seeds[len(jobs)]
            jobs.append((record['id'], sample, seed))
            requests.append(dict(input_ids=record['input_ids'], min_loops=1, exit_threshold=threshold,
                                 max_new_tokens=cap, temperature=rollout['temperature'],
                                 top_p=rollout['top_p'], seed=seed))

    responses = [None] * len(jobs)
    with observer.stage('rlt_model_load', echo=True, update=update+1, weight_version=update):
        actor = RltActor.from_release(
            snapshot, gpu, Path(output) / 'rlt_rollout.log', loops=loops,
            max_model_len=max(len(request['input_ids']) for request in requests) + cap,
            max_num_batched_tokens=4096, prefill_chunk_size=128, kv_block_size=64)
    with actor:
        if actor.runtime['kv_layout'] != 'last_exited' or actor.runtime['executed_loops'] != loops:
            raise RuntimeError('Unexpected RAD-OPD rollout KV layout or prefill depth')
        observer.event('rlt_rollout_runtime', echo=True, update=update+1,
                       weight_version=update, runtime=actor.runtime)
        start = time.perf_counter()
        with observer.stage('rlt_rollout_batch', echo=True, update=update+1, requests=len(jobs),
                            weight_version=update):
            for index, result, seconds in actor.iter_responses(requests):
                tokens, depths = result['token_ids'], result['exit_depths']
                ident, sample, seed = jobs[index]
                stop = result['finish_reason'] == 'stop'
                token_loops = len(requests[index]['input_ids']) * loops + sum(depths[1:])
                if (not 0 < len(tokens) == len(depths) <= cap or depths[0] != loops
                        or any(not 1 <= depth <= loops for depth in depths)
                        or result['finish_reason'] not in {'length', 'stop'}
                        or stop != (tokens[-1] in student.stop_token_ids)
                        or (not stop and len(tokens) != cap)
                        or result['executed_token_loops'] != token_loops):
                    raise RuntimeError(f'Invalid vLLM-RLT trajectory for {ident}/{sample}')
                response = GeneratedResponse(tokens, depths, stop, token_loops, seconds,
                                             execution_depths=depths, cache_policy='rlt_last_exited')
                responses[index] = (ident, sample, response, seed)
                observer.event('rlt_rollout_ready', update=update+1, id=ident, sample=sample,
                               weight_version=update, sampling_seed=seed,
                               exit_threshold=requests[index]['exit_threshold'], response_tokens=len(tokens))
        observer.event('rlt_rollout_batch_metrics', echo=True, update=update+1, weight_version=update,
                       batch_seconds=time.perf_counter()-start, responses=len(responses),
                       response_tokens=sum(len(response.token_ids) for _, _, response, _ in responses))
    return responses
