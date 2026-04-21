from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import distributed as dist


@dataclass(frozen=True)
class DistEnv:
    world_size: int
    global_rank: int
    local_rank: int
    is_main: bool
    enabled: bool


def setup_distributed() -> DistEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistEnv(world_size=1, global_rank=0, local_rank=0, is_main=True, enabled=False)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistEnv(
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        is_main=(global_rank == 0),
        enabled=True,
    )


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_int(value: int, src: int = 0, device: torch.device | None = None) -> int:
    if not dist.is_initialized():
        return value
    target = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([value], dtype=torch.long, device=target)
    dist.broadcast(t, src=src)
    return int(t.item())


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())
