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
    group_classification_loss,
    length_bucketed_batches,
    load_config,
    rule_teacher_loss,
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
    model_values["critic_layers"] = int(config.get("critic_layers", preset.critic_layers))
    model_values["policy_head_type"] = str(config.get("policy_head_type", "isolated_action_query"))
    model = KyokuTransformerActorCritic(ModelConfig(**model_values)).to(device)
    train_value = bool(config.get("train_public_value", True))
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        if root == "critic_embedding" or (not train_value and root in {
            "critic_backbone", "value_head", "value_query",
        }):
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
    rows_out: list[tuple[float, ...]] = []
    model.train()
    for step in range(1, int(steps) + 1):
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        loader_started = time.perf_counter()
        rows = next(batches)
        batch = collate_samples(rows, device, include_critic=False)
        loader_s = time.perf_counter() - loader_started
        effective_tokens = sum(sample.token_length for sample in rows)
        padded_tokens = len(rows) * max(sample.token_length for sample in rows)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16,
        ):
            forward_started = torch.cuda.Event(enable_timing=True)
            forward_finished = torch.cuda.Event(enable_timing=True)
            forward_started.record()
            output = model(
                batch["token_factors"],
                batch["token_numeric"],
                batch["legal_mask"],
                batch["token_lengths"],
                policy_only=not train_value,
            )
            forward_finished.record()
            policy_loss = F.cross_entropy(
                output["policy_logits"].float(), batch["actions"],
            )
            group_loss = group_classification_loss(output["policy_logits"].float(), batch["actions"])
            rule_loss = rule_teacher_loss(output["policy_logits"].float(), batch["teacher_masks"])
            loss = policy_loss + float(config.get("group_coef", 0.0)) * group_loss + float(config.get("rule_coef", 0.0)) * rule_loss
            if train_value:
                loss = loss + float(config.get("public_value_coef", 0.0)) * F.huber_loss(
                    output["value"].float(), batch["value_targets"],
                )
        backward_started = torch.cuda.Event(enable_timing=True)
        backward_finished = torch.cuda.Event(enable_timing=True)
        backward_started.record()
        loss.backward()
        backward_finished.record()
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), float(config["max_grad_norm"]),
        )
        optimizer_started = torch.cuda.Event(enable_timing=True)
        optimizer_finished = torch.cuda.Event(enable_timing=True)
        optimizer_started.record()
        optimizer.step()
        optimizer_finished.record()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        forward_ms = float(forward_started.elapsed_time(forward_finished))
        backward_ms = float(backward_started.elapsed_time(backward_finished))
        optimizer_ms = float(optimizer_started.elapsed_time(optimizer_finished))
        model.eval()
        infer_started = torch.cuda.Event(enable_timing=True)
        infer_finished = torch.cuda.Event(enable_timing=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            infer_started.record()
            model.module.forward_policy(
                batch["token_factors"], batch["token_numeric"], batch["legal_mask"], batch["token_lengths"],
            )
            infer_finished.record()
        torch.cuda.synchronize(device)
        inference_ms = float(infer_started.elapsed_time(infer_finished))
        model.train()

        values = torch.tensor(
            [
                elapsed,
                float(len(rows)),
                float(effective_tokens),
                float(padded_tokens),
                float(torch.cuda.memory_allocated(device) / 2**20),
                float(torch.cuda.memory_reserved(device) / 2**20),
                float(torch.cuda.max_memory_allocated(device) / 2**20),
                forward_ms,
                inference_ms,
                backward_ms,
                optimizer_ms,
                loader_s * 1000.0,
            ],
            dtype=torch.float64,
            device=device,
        )
        elapsed_max = values[0].clone()
        dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(values[1:4], op=dist.ReduceOp.SUM)
        dist.all_reduce(values[4:], op=dist.ReduceOp.MAX)
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
                elapsed_s, samples_per_s, tokens_per_s, padding,
                float(values[4]), float(values[5]), float(values[6]),
                float(values[7]), float(values[8]), float(values[9]),
                float(values[10]), float(values[11]),
            ))
            print(
                f"step={step} elapsed_s={elapsed_s:.4f} "
                f"samples={global_samples:.0f} samples_per_s={samples_per_s:.2f} "
                f"effective_tokens_per_s={tokens_per_s:.2f} "
                f"padding_fraction={padding:.4f} "
                f"loss={float(loss_value) / world_size:.5f} "
                f"grad_norm={float(grad_value) / world_size:.5f} "
                f"gpu_allocated_mb={float(values[4]):.1f} "
                f"gpu_reserved_mb={float(values[5]):.1f} "
                f"peak_gpu_memory_mb={float(values[6]):.1f}",
                f" forward_ms={float(values[7]):.3f} backward_ms={float(values[9]):.3f} "
                f"optimizer_ms={float(values[10]):.3f} loader_ms={float(values[11]):.3f} "
                f"inference_batch_ms={float(values[8]):.3f}",
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
            f"step_time_mean_s={np.mean([row[0] for row in warm]):.4f} "
            f"step_time_p95_s={np.quantile([row[0] for row in warm], 0.95):.4f} "
            f"samples_per_s={samples_total / max(elapsed_total, 1e-9):.2f} "
            f"per_rank_samples_per_s={samples_total / max(elapsed_total, 1e-9) / world_size:.2f} "
            f"step_samples_per_s_mean={rates.mean():.2f} "
            f"step_samples_per_s_std={rates.std(ddof=1) if len(rates) > 1 else 0.0:.2f} "
            f"step_samples_per_s_min={rates.min():.2f} "
            f"step_samples_per_s_max={rates.max():.2f} "
            f"effective_tokens_per_s_mean={token_rates.mean():.2f} "
            f"padding_fraction_mean={paddings.mean():.4f} "
            f"padding_fraction_min={paddings.min():.4f} "
            f"padding_fraction_max={paddings.max():.4f}",
            f" forward_ms_mean={np.mean([row[7] for row in warm]):.3f} "
            f"backward_ms_mean={np.mean([row[9] for row in warm]):.3f} "
            f"optimizer_ms_mean={np.mean([row[10] for row in warm]):.3f} "
            f"loader_ms_mean={np.mean([row[11] for row in warm]):.3f} "
            f"inference_batch_ms_mean={np.mean([row[8] for row in warm]):.3f}",
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
    parser.add_argument("--steps", type=int, default=3)
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
