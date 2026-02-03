import math

import torch
import torch.distributed as dist
from torch import nn

from .zero_linear import ZeroLinear


def init_sharded_linear_params(model, group=None):
    """init ZeroLinear shards by scattering from rank 0."""
    if group is None:
        group = dist.group.WORLD

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    for module in model.modules():
        if not isinstance(module, ZeroLinear):
            continue

        # rank 0 builds full weights, then scatters shards out
        bound = 1 / math.sqrt(module.in_features)
        if rank == 0:
            full_w = torch.empty(
                (module.out_features, module.in_features),
                device=module.weight_shard.device,
                dtype=module.weight_shard.dtype,
            )
            full_b = torch.empty(
                (module.out_features,),
                device=module.bias_shard.device,
                dtype=module.bias_shard.dtype,
            )
            nn.init.uniform_(full_w, -bound, bound)
            nn.init.uniform_(full_b, -bound, bound)
            # split into one chunk per rank
            w_chunks = list(full_w.chunk(world_size, dim=0))
            b_chunks = list(full_b.chunk(world_size, dim=0))
        else:
            w_chunks = None
            b_chunks = None

        # scatter the shards to all ranks
        dist.scatter(module.weight_shard.data, scatter_list=w_chunks, src=0, group=group)
        dist.scatter(module.bias_shard.data, scatter_list=b_chunks, src=0, group=group)
