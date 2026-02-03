import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from data import make_batch
from model import TransformerModel
from zero.comm import CommLogger
from zero.init import init_sharded_linear_params
from zero.ops import all_gather_params, allreduce_grads, reduce_scatter_grads
from zero.shard_optimizer import ShardAdamW, assign_param_shards
from zero.zero_linear import ZeroLinear


def setup_distributed(backend: str):
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        # handy for single-process runs without torchrun
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend)


def build_linear_factory(stage, comm, group):
    # stage 3 swaps in the sharded linear layer
    if stage == 3:
        return lambda in_f, out_f, bias=True: ZeroLinear(
            in_f, out_f, bias=bias, group=group, comm=comm
        )
    return lambda in_f, out_f, bias=True: torch.nn.Linear(in_f, out_f, bias=bias)


def split_sharded_params(params):
    # keep track of which params are sharded vs replicated
    sharded = []
    replicated = []
    for p in params:
        if getattr(p, "_is_sharded", False):
            sharded.append(p)
        else:
            replicated.append(p)
    return sharded, replicated


def main():
    parser = argparse.ArgumentParser(description="ZeRO stages 1–3 in tiny PyTorch")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dim-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bandwidth-gbps", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()

    # pick a backend if one wasn't provided
    backend = args.backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    setup_distributed(backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # map each process to its local gpu
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # same seed so replicas line up; stage 3 re-inits sharded weights
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    comm = CommLogger(args.bandwidth_gbps)
    group = dist.group.WORLD

    # build the model, swapping in sharded linears if needed
    linear_factory = build_linear_factory(args.stage, comm, group)
    model = TransformerModel(
        vocab_size=args.vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_ff=args.dim_ff,
        seq_len=args.seq_len,
        dropout=args.dropout,
        linear_factory=linear_factory,
    ).to(device)

    if args.stage == 3:
        # stage 3 starts with sharded weights, so we scatter init from rank 0
        init_sharded_linear_params(model, group=group)

    params = list(model.parameters())
    sharded_params, replicated_params = split_sharded_params(params)

    # stages 1/2 shard optimizer state across ranks
    if args.stage in (1, 2):
        assign_param_shards(params, world_size, rank, strict=False)
        optimizer = ShardAdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    model.train()
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    # main train loop
    step_times = []
    for step in range(args.steps):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if isinstance(optimizer, ShardAdamW):
            optimizer.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)

        # reset comm counters each step so logs are easy to read
        comm.reset()

        x, y = make_batch(args.batch_size, args.seq_len, args.vocab, device, generator)

        start = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab), y.reshape(-1))
        loss.backward()

        # pick the right collective pattern for each stage
        if args.stage == 0:
            allreduce_grads(params, comm=comm, group=group)
            optimizer.step()
        elif args.stage == 1:
            allreduce_grads(params, comm=comm, group=group)
            optimizer.step(grads_are_sharded=False)
            all_gather_params(params, comm=comm, group=group)
        elif args.stage == 2:
            reduce_scatter_grads(params, comm=comm, group=group)
            optimizer.step(grads_are_sharded=True)
            all_gather_params(params, comm=comm, group=group)
        elif args.stage == 3:
            allreduce_grads(replicated_params, comm=comm, group=group)
            optimizer.step()

        if isinstance(optimizer, ShardAdamW):
            optimizer.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_time = time.perf_counter() - start
        if step >= args.warmup:
            step_times.append(step_time)

        # keep logs to rank 0 so output isn't spammy
        if rank == 0 and (step % args.log_every == 0):
            mem_alloc = 0.0
            mem_reserved = 0.0
            if device.type == "cuda":
                mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
                mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9
            comm_stats = comm.summary()
            print(
                f"step {step:04d} | loss {loss.item():.4f} | time {step_time*1000:.1f} ms "
                f"| mem {mem_alloc:.2f} GB alloc / {mem_reserved:.2f} GB res "
                f"| comm {comm_stats}"
            )

    if rank == 0 and step_times:
        avg = sum(step_times) / len(step_times)
        print(f"avg step time (post-warmup): {avg*1000:.2f} ms")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
