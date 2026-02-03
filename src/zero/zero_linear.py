import math

import torch
import torch.distributed as dist
from torch import nn

from .comm import all_gather, reduce_scatter


class ShardLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_shard, bias_shard, group, comm):
        world_size = dist.get_world_size(group)

        # gather full weight from shards
        weight_list = [torch.empty_like(weight_shard) for _ in range(world_size)]
        all_gather(weight_list, weight_shard, comm=comm, group=group)
        weight_full = torch.cat(weight_list, dim=0)

        # same deal for bias
        bias_list = [torch.empty_like(bias_shard) for _ in range(world_size)]
        all_gather(bias_list, bias_shard, comm=comm, group=group)
        bias_full = torch.cat(bias_list, dim=0)

        # flatten for the matmul, then restore shape
        input_2d = input.reshape(-1, input.shape[-1])
        out_2d = input_2d.matmul(weight_full.t()) + bias_full
        out = out_2d.reshape(*input.shape[:-1], weight_full.shape[0])

        # save for backward
        ctx.save_for_backward(input, weight_shard, bias_shard)
        ctx.group = group
        ctx.comm = comm
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_shard, bias_shard = ctx.saved_tensors
        group = ctx.group
        comm = ctx.comm
        world_size = dist.get_world_size(group)

        # gather full weights again for backward math
        weight_list = [torch.empty_like(weight_shard) for _ in range(world_size)]
        all_gather(weight_list, weight_shard, comm=comm, group=group)
        weight_full = torch.cat(weight_list, dim=0)

        # flatten for easier math
        input_2d = input.reshape(-1, input.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        # grad for input is matmul with weights
        grad_input_2d = grad_output_2d.matmul(weight_full)
        grad_input = grad_input_2d.reshape_as(input)

        # full grad for weights
        grad_weight_full = grad_output_2d.t().matmul(input_2d)
        # split into per-rank chunks
        grad_weight_chunks = list(torch.chunk(grad_weight_full, world_size, dim=0))
        grad_weight_shard = torch.empty_like(weight_shard)
        # sum + scatter so each rank keeps its shard
        reduce_scatter(grad_weight_shard, grad_weight_chunks, comm=comm, group=group)
        grad_weight_shard.div_(world_size)

        # same for bias
        grad_bias_full = grad_output_2d.sum(dim=0)
        grad_bias_chunks = list(torch.chunk(grad_bias_full, world_size, dim=0))
        grad_bias_shard = torch.empty_like(bias_shard)
        reduce_scatter(grad_bias_shard, grad_bias_chunks, comm=comm, group=group)
        grad_bias_shard.div_(world_size)

        # return grads for input + sharded params only
        return grad_input, grad_weight_shard, grad_bias_shard, None, None


class ZeroLinear(nn.Module):
    """linear layer with sharded params and on-the-fly all-gather"""

    def __init__(self, in_features, out_features, bias=True, *, group=None, comm=None):
        super().__init__()
        # keep bias for now to keep the math simple
        if not bias:
            raise ValueError("ZeroLinear keeps bias for now to keep the math tidy.")

        # default to using the whole world if no group is passed in
        if group is None:
            group = dist.group.WORLD

        world_size = dist.get_world_size(group)
        # make sure output splits evenly across ranks
        if out_features % world_size != 0:
            raise ValueError(
                f"out_features={out_features} must be divisible by world_size={world_size}"
            )

        # figure out how many outputs this rank owns
        shard_out = out_features // world_size
        self.weight_shard = nn.Parameter(torch.empty(shard_out, in_features))
        self.bias_shard = nn.Parameter(torch.empty(shard_out))
        self.weight_shard._is_sharded = True
        self.bias_shard._is_sharded = True

        self.in_features = in_features
        self.out_features = out_features
        self.shard_out = shard_out
        self.group = group
        self.comm = comm

        # same init as the pytorch linear layer
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_shard, -bound, bound)
        nn.init.uniform_(self.bias_shard, -bound, bound)

    def forward(self, x):
        # run through the custom sharded linear op
        return ShardLinearFunction.apply(x, self.weight_shard, self.bias_shard, self.group, self.comm)
