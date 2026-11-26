"""Run vLLM-RLT on one visible GPU and stream each completed request."""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import json
import sys
import time
import traceback


def execution_token_loops(output, params, prefill_loops):
    depths = output.exit_depths
    if (not depths or len(depths) != len(output.token_ids) or depths[0] != prefill_loops
            or any(not params.min_loops <= depth <= prefill_loops for depth in depths[1:])
            or (params.exit_threshold == 1.0 and any(depth != prefill_loops for depth in depths))):
        raise RuntimeError('vLLM-RLT returned an invalid prefill or decode exit depth')
    # The first output comes from prefill. The final sampled token has no KV row.
    return prefill_loops * len(output.prompt_token_ids) + sum(depths[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--loops', required=True, type=int)
    parser.add_argument('--max-model-len', required=True, type=int)
    parser.add_argument('--max-num-seqs', type=int)
    parser.add_argument('--max-num-batched-tokens', type=int)
    parser.add_argument('--prefill-chunk-size', type=int)
    parser.add_argument('--kv-block-size', type=int)
    parser.add_argument('--attention-backend', default='triton')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--async-scheduling', action='store_true')
    parser.add_argument('--enable-prefix-caching', action='store_true')
    args = parser.parse_args()
    protocol = sys.stdout

    def emit(message):
        print(json.dumps(message), file=protocol, flush=True)

    engine = None
    try:
        with redirect_stdout(sys.stderr):
            from ep_loop.cpu_runtime import configure_cpu_threads
            configure_cpu_threads()
            import torch
            import vllm_rlt
            from vllm_rlt import SamplingParams
            from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
            from vllm_rlt.engine.stage_engine import StageEngine
            from vllm_rlt.models import AutoModelForCausalLM, prepare_model

            source, revision, config = prepare_model(args.model)
            if config.model_type != 'ouro' or not 1 <= args.loops <= config.total_ut_steps:
                raise ValueError('vLLM-RLT evaluation requires Ouro and a supported prefill depth')
            if args.max_model_len > config.max_position_embeddings:
                raise ValueError('Evaluation exceeds the checkpoint context limit')
            # RLT runs the full configured depth at prefill, including the first
            # sampled token. Align the model config as well as the request bounds.
            config = replace(config, model_specific_config=replace(
                config.model_specific_config, total_ut_steps=args.loops))
            model = AutoModelForCausalLM.from_pretrained(
                source, revision=revision, model_config=config, device='cuda:0', dtype=torch.bfloat16)
            scheduler = SchedulerConfig(**{
                name: getattr(args, name)
                for name in ('max_num_seqs', 'max_num_batched_tokens', 'prefill_chunk_size')
                if getattr(args, name) is not None
            })
            cache_options = {} if args.kv_block_size is None else {'block_size': args.kv_block_size}
            cache_config = CacheConfig(
                layout='last_exited', gpu_memory_utilization=0.9,
                enable_prefix_caching=args.enable_prefix_caching, **cache_options)
            engine = StageEngine(
                model,
                cache_config=cache_config,
                scheduler_config=scheduler,
                attention_backend=args.attention_backend,
                exit_config=ExitConfig(mode='ouro_delayed' if args.async_scheduling else 'ouro'),
                execution_config=ExecutionConfig(cuda_graphs=not args.eager,
                    async_scheduling=args.async_scheduling, static_buffers=args.async_scheduling,
                    pad_to_power_of_two=args.async_scheduling),
            )
        effective = engine.scheduler.config
        runtime = dict(
            ready=True, executed_loops=args.loops, backend='vllm-rlt',
            module_path=vllm_rlt.__file__,
            attention_backend=args.attention_backend, async_scheduling=args.async_scheduling,
            kv_layout=cache_config.layout, torch_version=torch.__version__, cuda_version=torch.version.cuda,
            gpu_memory_utilization=cache_config.gpu_memory_utilization,
            cuda_graphs=not args.eager,
            enable_prefix_caching=args.enable_prefix_caching,
            max_num_seqs=effective.max_num_seqs,
            max_num_batched_tokens=effective.max_num_batched_tokens,
            prefill_chunk_size=effective.prefill_chunk_size,
            kv_block_size=engine.cache_manager.block_size,
            kv_tokens=engine.cache_manager.num_blocks * engine.cache_manager.block_size,
            speculative_config=None,
        )
        print(json.dumps(dict(event='rlt_engine_ready', **runtime)), file=sys.stderr, flush=True)
        emit(runtime)
        for line in sys.stdin:
            message = json.loads(line)
            if message['op'] == 'close':
                break
            if message['op'] != 'generate':
                raise ValueError(f"Unsupported operation {message['op']}")
            cached_before = engine.cache_manager.prefix_hits
            start = time.perf_counter()
            first_token_times = {}
            params_by_id = {}
            with redirect_stdout(sys.stderr):
                for index, request in enumerate(message['requests']):
                    if len(request['input_ids']) + request['max_new_tokens'] > args.max_model_len:
                        raise ValueError('Request exceeds the evaluation context limit')
                    stop_options = {'ignore_eos': True} if request.get('ignore_eos') else {}
                    params = SamplingParams(
                        max_tokens=request['max_new_tokens'], temperature=request['temperature'],
                        top_p=request['top_p'], seed=request['seed'],
                        min_loops=request.get('min_loops', args.loops), max_loops=args.loops,
                        exit_threshold=request.get('exit_threshold', 1.0), **stop_options)
                    params_by_id[str(index)] = params
                    engine.add_request(str(index), request['input_ids'], params)
                while engine.has_unfinished_requests():
                    for output in engine.step():
                        elapsed = time.perf_counter() - start
                        if output.token_ids:
                            first_token_times.setdefault(output.request_id, elapsed)
                        if not output.finished:
                            continue
                        ids = output.token_ids
                        token_loops = execution_token_loops(output, params_by_id[output.request_id], args.loops)
                        first_token = first_token_times.get(output.request_id)
                        timing = (dict(time_to_first_token_seconds=first_token,
                                       first_to_last_token_seconds=elapsed-first_token)
                                  if first_token is not None else {})
                        emit(dict(ok=True, index=int(output.request_id), seconds=elapsed,
                                  response=dict(token_ids=ids, exit_depths=output.exit_depths,
                                      finish_reason=output.finish_reason, request_timing=timing,
                                      executed_token_loops=token_loops)))
            prompt_tokens = sum(len(request['input_ids']) for request in message['requests'])
            print(json.dumps(dict(event='rlt_batch_finished', seconds=time.perf_counter()-start,
                requests=len(message['requests']), prompt_tokens=prompt_tokens,
                cached_prompt_tokens=engine.cache_manager.prefix_hits-cached_before)),
                file=sys.stderr, flush=True)
            emit(dict(ok=True, done=True))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit(dict(ok=False, error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        if engine is not None:
            with redirect_stdout(sys.stderr):
                engine.close()


if __name__ == '__main__':
    main()
