import time
from collections import defaultdict

import torch.distributed as dist


def _tensor_bytes(tensor: "torch.Tensor") -> int:
    # rough size in bytes, good enough for logging
    return tensor.numel() * tensor.element_size()


class CommLogger:
    """helper to track collective volume and optionally fake bandwidth"""

    def __init__(self, bandwidth_gbps: float | None = None):
        self.bandwidth_bytes_per_sec = (
            bandwidth_gbps * 1e9 if bandwidth_gbps is not None else None
        )
        self.bytes = defaultdict(int)

    def _throttle(self, num_bytes: int) -> None:
        # pretend the link is slower by sleeping a bit
        if self.bandwidth_bytes_per_sec:
            time.sleep(num_bytes / self.bandwidth_bytes_per_sec)

    def record(self, kind: str, num_bytes: int) -> None:
        self.bytes[kind] += int(num_bytes)
        self._throttle(num_bytes)

    def summary(self) -> dict:
        return dict(self.bytes)

    def reset(self) -> None:
        self.bytes = defaultdict(int)


def all_reduce(tensor, comm: CommLogger | None = None, group=None):
    dist.all_reduce(tensor, group=group)
    if comm is not None:
        world_size = dist.get_world_size(group)
        comm.record("all_reduce", _tensor_bytes(tensor) * (world_size - 1))


def reduce_scatter(output, input_list, comm: CommLogger | None = None, group=None):
    dist.reduce_scatter(output, input_list, group=group)
    if comm is not None:
        # rough comm cost per rank for reduce-scatter
        world_size = dist.get_world_size(group)
        input_bytes = sum(_tensor_bytes(t) for t in input_list)
        comm.record("reduce_scatter", input_bytes * (world_size - 1) // world_size)


def all_gather(output_list, input_tensor, comm: CommLogger | None = None, group=None):
    dist.all_gather(output_list, input_tensor, group=group)
    if comm is not None:
        world_size = dist.get_world_size(group)
        comm.record("all_gather", _tensor_bytes(input_tensor) * (world_size - 1))


def broadcast(tensor, src, comm: CommLogger | None = None, group=None):
    dist.broadcast(tensor, src=src, group=group)
    if comm is not None:
        world_size = dist.get_world_size(group)
        comm.record("broadcast", _tensor_bytes(tensor) * (world_size - 1))
