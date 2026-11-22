"""Share one tensor-parallel inference engine between adjacent HF DDP ranks.

Object collectives use a separate CPU/Gloo group: waiting HF ranks must not
allocate CUDA communication buffers while vLLM owns their GPUs. Each rank
retains its original questions, sampling RNG and response order for training.
"""
from datetime import timedelta
import os

import torch.distributed as dist


def gpu_groups(devices, tensor_parallel_size):
    if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
        raise ValueError('vllm_tensor_parallel_size must be a positive integer')
    if not devices or len(devices) % tensor_parallel_size:
        raise ValueError('GPU count must be divisible by vllm_tensor_parallel_size')
    if any(not device.strip() for device in devices) or len(set(devices))!=len(devices):
        raise ValueError('Distinct nonempty GPU IDs required')
    return [','.join(devices[start:start+tensor_parallel_size])
            for start in range(0, len(devices), tensor_parallel_size)]


def split_responses(result, counts):
    """Partition by original rank order, including duplicate sampled questions."""
    if len(result['responses']) != sum(counts):
        raise RuntimeError('Inference returned a different number of responses')
    offset = 0
    parts = []
    for count in counts:
        parts.append(dict(result, responses=result['responses'][offset:offset+count]))
        offset += count
    return parts


class InferenceGroup:
    def __init__(self, gpu, ranks=(0,), rank=0, process_group=None):
        self.gpu = gpu
        self.ranks = tuple(ranks)
        self.rank = rank
        self.leader = self.ranks[0]
        self.size = len(self.ranks)
        self.is_leader = rank == self.leader
        self.process_group = process_group

    def execute(self, requests, operation):
        """One merged engine call; return only this HF rank's responses.

        Gathering also ensures every group member has offloaded HF state.
        Operation includes sleeping the engine before results are scattered,
        so a follower cannot restore HF tensors onto an active TP worker.
        """
        if self.size == 1:
            return split_responses(operation(requests), [len(requests)])[0]
        gathered = [None] * self.size if self.is_leader else None
        dist.gather_object(requests, gathered, dst=self.leader, group=self.process_group)
        envelopes = None
        if self.is_leader:
            try:
                merged = [request for rows in gathered for request in rows]
                parts = split_responses(operation(merged), [len(rows) for rows in gathered])
                envelopes = [dict(ok=True, result=part) for part in parts]
            except Exception as error:
                # Propagate normal engine/weight/trace failures to followers;
                # do not leave them waiting for the multi-hour rollout timeout.
                envelopes = [dict(ok=False, error=f'{type(error).__name__}: {error}')
                             for _ in self.ranks]
        received = [None]
        dist.scatter_object_list(received, envelopes, src=self.leader, group=self.process_group)
        if not received[0]['ok']:
            raise RuntimeError(f'TP inference group {self.ranks} failed: {received[0]["error"]}')
        return received[0]['result']


def create_inference_group(config, local_rank):
    """Called once, in the same order on every trainer rank after dist init."""
    size = config['implementation'].get('vllm_tensor_parallel_size', 1)
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    devices = visible if visible != [''] else [str(i) for i in range(world)]
    if size == 1:
        return InferenceGroup(devices[local_rank], (rank,), rank)
    if (world != int(os.environ.get('LOCAL_WORLD_SIZE', str(world))) or rank != local_rank
            or len(devices) != world):
        raise ValueError('Paired TP inference currently requires a single-node DDP job')
    groups = gpu_groups(devices, size)
    if not dist.is_initialized():
        raise ValueError('Initialize the trainer process group before paired inference')
    selected = None
    timeout = timedelta(seconds=config['implementation'].get('distributed_timeout_seconds', 21600))
    for index, gpu in enumerate(groups):
        ranks = tuple(range(index*size, (index+1)*size))
        process_group = dist.new_group(ranks=list(ranks), backend='gloo', timeout=timeout)
        if rank in ranks:
            selected = InferenceGroup(gpu, ranks, rank, process_group)
    return selected
