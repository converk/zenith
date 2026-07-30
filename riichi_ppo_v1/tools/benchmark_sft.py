"""Benchmark a bounded number of real SFT optimizer steps."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import random
import socket
import time

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.sft.data import iter_split_samples
from riichi_ppo_v1.sft.train import (
    collate_samples,
    length_bucketed_batches,
    load_config,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(
    rank: int,
    world_size: int,
    config: dict[str, object],
    dataset: str,
    steps: int,
) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    seed = int(config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    preset = ModelConfig.preset(str(config["model_size"]))
    model_values = asdict(preset)
    model_values["context_tokens"] = int(config["context_tokens"])
    model = KyokuTransformerActorCritic(ModelConfig(**model_values)).to(device)
    for name, parameter in model.named_parameters():
        if name.split(".", 1)[0] in {
            "critic_embedding", "critic_backbone", "value_head", "value_query",
        }:
            parameter.requires_grad_(False)
    model = DistributedDataParallel(
        model, device_ids=[rank], broadcast_buffers=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        eps=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
        fused=True,
    )
    local_batch = max(1, int(config["batch_size"]) // world_size)
    samples = iter_split_samples(
        Path(dataset),
        "train",
        gamma=float(config["gamma"]),
        seed=int(config["seed"]),
        shuffle=True,
        shuffle_buffer_kyokus=int(config["shuffle_buffer_kyokus"]),
        rank=rank,
        world_size=world_size,
        include_critic=False,
    )
    batches = iter(length_bucketed_batches(
        samples,
        local_batch,
        window_batches=int(config["length_bucket_window_batches"]),
    ))
    use_bf16 = bool(
        str(config["inference_dtype"]).lower() == "bf16"
        and torch.cuda.is_bf16_supported()
    )
    rows_out: list[tuple[float, float, float, float, float]] = []
    model.train()
    for step in range(1, int(steps) + 1):
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        rows = next(batches)
        batch = collate_samples(rows, device, include_critic=False)
        effective_tokens = sum(sample.token_length + 1 for sample in rows)
        padded_tokens = len(rows) * (max(sample.token_length for sample in rows) + 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16,
        ):
            output = model(
                batch["token_factors"],
                batch["token_numeric"],
                batch["legal_mask"],
                batch["token_lengths"],
                policy_only=True,
            )
            loss = F.cross_entropy(
                output["policy_logits"].float(), batch["actions"],
            )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), float(config["max_grad_norm"]),
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        values = torch.tensor(
            [
                elapsed,
                float(len(rows)),
                float(effective_tokens),
                float(padded_tokens),
                float(torch.cuda.max_memory_allocated(device) / 2**20),
            ],
            dtype=torch.float64,
            device=device,
        )
        elapsed_max = values[0].clone()
        dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(values[1:4], op=dist.ReduceOp.SUM)
        dist.all_reduce(values[4], op=dist.ReduceOp.MAX)
        loss_value = loss.detach()
        grad_value = torch.as_tensor(grad_norm, device=device)
        dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_value, op=dist.ReduceOp.SUM)
        if rank == 0:
            elapsed_s = float(elapsed_max)
            global_samples = float(values[1])
            global_effective = float(values[2])
            global_padded = float(values[3])
            samples_per_s = global_samples / max(elapsed_s, 1e-9)
            tokens_per_s = global_effective / max(elapsed_s, 1e-9)
            padding = 1.0 - global_effective / max(global_padded, 1.0)
            rows_out.append((
                elapsed_s, samples_per_s, tokens_per_s, padding, float(values[4]),
            ))
            print(
                f"step={step} elapsed_s={elapsed_s:.4f} "
                f"samples={global_samples:.0f} samples_per_s={samples_per_s:.2f} "
                f"effective_tokens_per_s={tokens_per_s:.2f} "
                f"padding_fraction={padding:.4f} "
                f"loss={float(loss_value) / world_size:.5f} "
                f"grad_norm={float(grad_value) / world_size:.5f} "
                f"peak_gpu_memory_mb={float(values[4]):.1f}",
                flush=True,
            )

    if rank == 0:
        warm = rows_out[1:] if len(rows_out) > 1 else rows_out
        elapsed_total = sum(row[0] for row in warm)
        samples_total = int(config["batch_size"]) * len(warm)
        rates = np.asarray([row[1] for row in warm], dtype=np.float64)
        paddings = np.asarray([row[3] for row in warm], dtype=np.float64)
        token_rates = np.asarray([row[2] for row in warm], dtype=np.float64)
        print(
            f"summary measured_steps={len(warm)} warmup_steps={len(rows_out)-len(warm)} "
            f"elapsed_s={elapsed_total:.4f} samples={samples_total} "
            f"samples_per_s={samples_total / max(elapsed_total, 1e-9):.2f} "
            f"step_samples_per_s_mean={rates.mean():.2f} "
            f"step_samples_per_s_std={rates.std(ddof=1) if len(rates) > 1 else 0.0:.2f} "
            f"step_samples_per_s_min={rates.min():.2f} "
            f"step_samples_per_s_max={rates.max():.2f} "
            f"effective_tokens_per_s_mean={token_rates.mean():.2f} "
            f"padding_fraction_mean={paddings.mean():.4f} "
            f"padding_fraction_min={paddings.min():.4f} "
            f"padding_fraction_max={paddings.max():.4f}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("datasets/tenhou_sft_2024_2025"),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    config = load_config(args.config)
    world_size = int(config["learner_gpus"])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, config, str(args.dataset), int(args.steps)),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
