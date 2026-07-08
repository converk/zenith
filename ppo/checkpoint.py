from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn, optim

from ppo.config import PPOConfig
from ppo.distributed import unwrap_model


def checkpoint_path(checkpoint_dir: str, run_name: str, global_step: int) -> Path:
    return Path(checkpoint_dir) / run_name / f"checkpoint_{global_step}.pt"


def save_checkpoint(
    agent: nn.Module,
    optimizer: optim.Optimizer,
    args: PPOConfig,
    global_step: int,
    iteration: int,
    completed_episodes: int,
    run_name: str,
) -> Path:
    path = checkpoint_path(args.checkpoint_dir, run_name, global_step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(agent).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": asdict(args),
            "global_step": global_step,
            "iteration": iteration,
            "completed_episodes": completed_episodes,
        },
        path,
    )
    return path


def load_checkpoint(
    agent: nn.Module,
    optimizer: optim.Optimizer,
    resume_from: str,
    device: torch.device,
) -> tuple[int, int, int]:
    # torch.load uses pickle internally; only load checkpoints you trust.
    checkpoint = torch.load(resume_from, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    if next(iter(model_state_dict), "").startswith("module."):
        model_state_dict = {
            key.removeprefix("module."): value for key, value in model_state_dict.items()
        }
    unwrap_model(agent).load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    global_step = int(checkpoint.get("global_step", 0))
    start_iteration = int(checkpoint.get("iteration", 0)) + 1
    completed_episodes = int(checkpoint.get("completed_episodes", 0))
    return global_step, start_iteration, completed_episodes
