from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_distributed: bool
    is_main: bool


def setup_distributed(device_preference: str = "cuda") -> DistEnv:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    use_cuda = device_preference == "cuda" and torch.cuda.is_available()
    is_distributed = world_size > 1

    if is_distributed:
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    else:
        device = torch.device("cuda" if use_cuda else "cpu")

    return DistEnv(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_distributed=is_distributed,
        is_main=(rank == 0),
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(value: int | float, device: torch.device) -> int | float:
    if not (dist.is_available() and dist.is_initialized()):
        return value

    if isinstance(value, int):
        tensor = torch.tensor(value, device=device, dtype=torch.long)
    else:
        tensor = torch.tensor(value, device=device, dtype=torch.float64)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()