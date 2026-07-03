from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

from riichi_parallel_env import AGENTS


@dataclass
class PPOConfig:
    exp_name: str = os.path.basename("ppo.py").rstrip(".py")
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "mahjong-ai"
    wandb_entity: str | None = None
    capture_video: bool = False

    target_iterations: int = 5_000
    learning_rate: float = 3e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    model_size: str = "medium"
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 10
    resume_from: str | None = None
    eval_interval: int = 10
    eval_episodes: int = 8

    batch_size: int = 0
    minibatch_size: int = 0


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser()
    defaults = PPOConfig()
    for field_name, field_value in asdict(defaults).items():
        option = f"--{field_name.replace('_', '-')}"
        if isinstance(field_value, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=field_value)
        else:
            parser.add_argument(option, type=_arg_type(field_value), default=field_value)
    return finalize_config(PPOConfig(**vars(parser.parse_args())))


def finalize_config(config: PPOConfig) -> PPOConfig:
    if config.target_iterations <= 0:
        raise ValueError("target_iterations must be greater than 0")
    config.batch_size = config.num_envs * len(AGENTS) * config.num_steps
    config.minibatch_size = config.batch_size // config.num_minibatches
    return config


def _arg_type(value: object):
    if value is None:
        return str
    return type(value)
