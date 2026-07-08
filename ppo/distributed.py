from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ppo.config import PPOConfig


@dataclass(frozen=True)
class DistributedState:
    rank: int
    local_rank: int
    world_size: int

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(args: PPOConfig) -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    state = DistributedState(rank=rank, local_rank=local_rank, world_size=world_size)
    if not state.enabled:
        return state

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("Distributed PPO requires CUDA. Run without torchrun for CPU training.")
    if not dist.is_nccl_available():
        raise RuntimeError("Distributed PPO requires NCCL support in PyTorch.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return state


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(agent: nn.Module) -> nn.Module:
    return agent.module if isinstance(agent, DDP) else agent


def distributed_sum(value: float, device: torch.device, state: DistributedState) -> float:
    total = torch.tensor(value, dtype=torch.float64, device=device)
    if state.enabled:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float(total.item())


def distributed_mean(value: float, device: torch.device, state: DistributedState) -> float:
    total = distributed_sum(value, device, state)
    return total / state.world_size


def make_run_name(args: PPOConfig, device: torch.device, state: DistributedState) -> str:
    timestamp = int(time.time())
    if state.enabled:
        timestamp_tensor = torch.tensor(timestamp, dtype=torch.int64, device=device)
        dist.broadcast(timestamp_tensor, src=0)
        timestamp = int(timestamp_tensor.item())
    return f"{args.exp_name}__{args.seed}__{timestamp}"
