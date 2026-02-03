import math

import torch


def assign_param_shards(params, world_size: int, rank: int, *, strict: bool = True):
    # mark which slice this rank owns (dim 0 only)
    for p in params:
        if p.ndim == 0:
            # tiny scalars stay replicated
            p._shard_slice = None
            continue
        dim0 = p.shape[0]
        if dim0 % world_size != 0:
            msg = (
                f"param dim0={dim0} not divisible by world_size={world_size}. "
                "Pick model dims that divide cleanly (d_model, n_heads, vocab, etc)."
            )
            if strict:
                raise ValueError(msg)
            else:
                # if not strict, keep the full param here
                p._shard_slice = None
                continue
        shard_size = dim0 // world_size
        start = rank * shard_size
        end = start + shard_size
        p._shard_slice = slice(start, end)

def _get_shard_view(p, grad, grads_are_sharded: bool):
    # pick the chunk this rank should update
    shard_slice = getattr(p, "_shard_slice", None)
    if shard_slice is None:
        # param isn't sharded, use it all
        return p.data, grad

    if grads_are_sharded:
        # grad already matches our shard
        return p.data[shard_slice], grad

    # otherwise slice the full grad
    return p.data[shard_slice], grad[shard_slice]

class ShardAdamW:
    """adamw that only stores state for the local shard"""
    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

    def zero_grad(self):
        # clear both full grads and shard grads
        for p in self.params:
            p.grad = None
            if hasattr(p, "grad_shard"):
                p.grad_shard = None

    @torch.no_grad()
    def step(self, *, grads_are_sharded: bool):
        # grads_are_sharded means we already did reduce-scatter
        beta1, beta2 = self.betas

        for p in self.params:
            # pick the right grad buffer
            grad = p.grad_shard if grads_are_sharded else p.grad
            if grad is None:
                continue

            # get the correct view for this shard
            data_view, grad_view = _get_shard_view(p, grad, grads_are_sharded)

            # set up state on first use
            state = self.state.setdefault(p, {})
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(data_view)
                state["exp_avg_sq"] = torch.zeros_like(data_view)

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # decoupled weight decay like adamw
            if self.weight_decay != 0:
                data_view.add_(data_view, alpha=-self.lr * self.weight_decay)

            # standard adam moments, but only for our shard
            exp_avg.mul_(beta1).add_(grad_view, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_view, grad_view, value=1 - beta2)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

            denom = exp_avg_sq.sqrt().add_(self.eps)
            data_view.addcdiv_(exp_avg, denom, value=-step_size)
