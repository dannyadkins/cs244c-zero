import torch
import torch.distributed as dist

from .comm import all_gather, all_reduce, reduce_scatter


def allreduce_grads(params, comm=None, group=None):
    # classic ddp-style gradient sync
    world_size = dist.get_world_size(group)
    for p in params:
        if p.grad is None:
            continue
        all_reduce(p.grad, comm=comm, group=group)
        p.grad.div_(world_size)


def reduce_scatter_grads(params, comm=None, group=None):
    # shard grads across ranks with reduce-scatter
    world_size = dist.get_world_size(group)
    for p in params:
        if p.grad is None:
            continue
        shard_slice = getattr(p, "_shard_slice", None)
        if p.ndim == 0 or shard_slice is None:
            # tiny or unsharded params just do the normal all-reduce
            all_reduce(p.grad, comm=comm, group=group)
            p.grad.div_(world_size)
            p.grad_shard = p.grad
            continue

        # split grad into per-rank chunks, then reduce-scatter to sum + keep our shard
        chunks = list(torch.chunk(p.grad, world_size, dim=0))
        shard = torch.zeros_like(chunks[0])
        reduce_scatter(shard, chunks, comm=comm, group=group)
        shard.div_(world_size)

        p.grad_shard = shard
        p.grad = None


def all_gather_params(params, comm=None, group=None):
    # pull full params back together from shards
    world_size = dist.get_world_size(group)
    for p in params:
        if p.ndim == 0:
            continue

        shard_slice = getattr(p, "_shard_slice", None)
        if shard_slice is None:
            continue

        shard = p.data[shard_slice].contiguous()
        gather_list = [torch.empty_like(shard) for _ in range(world_size)]
        all_gather(gather_list, shard, comm=comm, group=group)
        full = torch.cat(gather_list, dim=0)
        p.data.copy_(full)
