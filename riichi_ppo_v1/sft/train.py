"""Joint supervised policy/value training over prepared MJAI kyoku shards."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time
from typing import Any, Iterable, Iterator

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import yaml

from ..model import KyokuTransformerActorCritic, ModelConfig
from ..model.schema import TOKEN_SCHEMA_VERSION
from .data import SftSample, iter_split_samples
from .heuristic_evaluation import evaluate_against_heuristics
from .tensorboard import SftMetricWindow, write_sft_scalars


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1,
    "device": "cuda",
    "learner_gpus": 2,
    "model_size": "mid",
    "context_tokens": 4096,
    "critic_layers": 2,
    "epochs": 1,
    "batch_size": 512,
    "learning_rate": 1.5e-4,
    "min_learning_rate": 2e-5,
    "warmup_fraction": 0.02,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 0.5,
    "train_critic": False,
    "value_coef": 0.5,
    "gamma": 0.99,
    "inference_dtype": "bf16",
    "shuffle_buffer_kyokus": 8192,
    "length_bucket_window_batches": 32,
    "checkpoint_interval_steps": 5000,
    "log_interval_steps": 100,
    "validation_max_samples": 0,
    "validation_interval_steps": 5000,
    "validation_samples_per_run": 32768,
    "heuristic_evaluation_enabled": True,
    "heuristic_evaluation_interval_steps": 25000,
    "heuristic_evaluation_hanchan_count": 128,
    "heuristic_evaluation_parallel_hanchan_count": 1,
    "heuristic_evaluation_final_hanchan_count": 128,
    "heuristic_evaluation_seed_base": 20260717,
    "heuristic_evaluation_game_mode": "4p-red-half",
    "heuristic_evaluation_max_steps": 4000,
    "checkpoint_dir": "checkpoints/checkpoints/train_riichi_v10_sft",
    "tensorboard_enabled": True,
    "tensorboard_dirname": "tensorboard",
    "resume": None,
}


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        with path.open(encoding="utf-8") as file:
            overlay = yaml.safe_load(file)
        if not isinstance(overlay, dict):
            raise ValueError("SFT config must be a mapping")
        config.update(overlay)
    return config


def dataset_manifest_hash(dataset: Path) -> str:
    return hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()


def collate_samples(
    samples: list[SftSample],
    device: torch.device,
    *,
    include_critic: bool = True,
) -> dict[str, torch.Tensor]:
    batch = len(samples)
    max_tokens = max(sample.token_length for sample in samples)
    factors = torch.zeros((batch, max_tokens, 10), dtype=torch.uint8)
    numeric = torch.zeros((batch, max_tokens, 8), dtype=torch.float32)
    lengths = torch.empty(batch, dtype=torch.long)
    legal = torch.empty((batch, 241), dtype=torch.bool)
    actions = torch.empty(batch, dtype=torch.long)
    for row, sample in enumerate(samples):
        factors[row, :sample.token_length] = torch.from_numpy(sample.token_factors)
        numeric[row, :sample.token_length] = torch.from_numpy(sample.token_numeric)
        lengths[row] = sample.token_length
        legal[row] = torch.from_numpy(sample.legal_mask)
        actions[row] = sample.action
    result = {
        "token_factors": factors.to(device, non_blocking=True),
        "token_numeric": numeric.to(device, non_blocking=True),
        "token_lengths": lengths.to(device, non_blocking=True),
        "legal_mask": legal.to(device, non_blocking=True),
        "actions": actions.to(device, non_blocking=True),
    }
    if include_critic:
        max_critic = max(sample.critic_length for sample in samples)
        critic = torch.zeros((batch, max_critic, 10), dtype=torch.uint8)
        critic_lengths = torch.empty(batch, dtype=torch.long)
        targets = torch.empty(batch, dtype=torch.float32)
        for row, sample in enumerate(samples):
            if sample.critic_length:
                critic[row, :sample.critic_length] = torch.from_numpy(sample.critic_factors)
            critic_lengths[row] = sample.critic_length
            targets[row] = sample.value_target
        result.update({
            "critic_factors": critic.to(device, non_blocking=True),
            "critic_lengths": critic_lengths.to(device, non_blocking=True),
            "value_targets": targets.to(device, non_blocking=True),
        })
    return result


def length_bucketed_batches(
    samples: Iterable[SftSample],
    batch_size: int,
    *,
    window_batches: int,
    rng: random.Random | None = None,
) -> Iterator[list[SftSample]]:
    """Make low-padding batches without exposing a sorted curriculum.

    Inputs are randomized before reaching this function.  We sort only a
    bounded window, form similarly sized batches, then shuffle those batches
    again.  This keeps padding low while consecutive optimizer steps still see
    varied sequence lengths.
    """
    window: list[SftSample] = []
    capacity = max(batch_size, batch_size * window_batches)

    def drain(rows: list[SftSample]) -> Iterator[list[SftSample]]:
        rows.sort(key=lambda item: item.token_length)
        batches = [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]
        if rng is not None:
            rng.shuffle(batches)
        yield from batches

    for sample in samples:
        window.append(sample)
        if len(window) < capacity:
            continue
        yield from drain(window)
        window = []
    if window:
        yield from drain(window)


def _action_group(action_id: int) -> str:
    if action_id == 0:
        return "pass"
    if 1 <= action_id <= 74:
        return "discard"
    if action_id == 75:
        return "reach"
    if 76 <= action_id <= 132:
        return "chi"
    if 133 <= action_id <= 169:
        return "pon"
    if action_id == 170:
        return "daiminkan"
    if 171 <= action_id <= 204:
        return "ankan"
    if 205 <= action_id <= 238:
        return "kakan"
    if action_id == 239:
        return "hora"
    return "ryukyoku"


def _model_config(config: dict[str, Any]) -> ModelConfig:
    base = ModelConfig.preset(str(config["model_size"]))
    values = asdict(base)
    values["context_tokens"] = int(config["context_tokens"])
    values["critic_layers"] = int(config.get("critic_layers", base.critic_layers))
    return ModelConfig(**values)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    config: dict[str, Any],
    manifest_hash: str,
    epoch: int,
    global_step: int,
    rank_steps: list[int],
    best_validation_loss: float = float("inf"),
    best_heuristic_point_delta: float = float("-inf"),
    metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save({
        "model": {name: value.detach().cpu() for name, value in module.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": asdict(module.config),
        "sft_config": config,
        "training_stage": "sft",
        "training_mode": "joint_actor_critic" if bool(config["train_critic"]) else "actor_only",
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "dataset_manifest_hash": manifest_hash,
        "epoch": epoch,
        "global_step": global_step,
        "rank_steps": rank_steps,
        "best_validation_loss": float(best_validation_loss),
        "best_heuristic_point_delta": float(best_heuristic_point_delta),
        "metrics": dict(metrics or {}),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, path)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Path,
    config: dict[str, Any],
    device: torch.device,
    *,
    max_samples: int | None = None,
) -> dict[str, float]:
    model.eval()
    train_critic = bool(config["train_critic"])
    batch_size = max(1, int(config["batch_size"]) // max(1, int(config["learner_gpus"])))
    samples = iter_split_samples(
        dataset,
        "validation",
        gamma=float(config["gamma"]),
        seed=int(config["seed"]),
        shuffle=False,
        include_critic=train_critic,
    )
    batches = length_bucketed_batches(
        samples,
        batch_size,
        window_batches=int(config["length_bucket_window_batches"]),
    )
    totals = {
        "samples": 0.0, "policy_ce_sum": 0.0, "value_huber_sum": 0.0,
        "value_abs_sum": 0.0, "value_sq_sum": 0.0, "top1": 0.0, "top3": 0.0,
    }
    group_correct: dict[str, list[int]] = {}
    maximum = (
        int(config.get("validation_max_samples", 0))
        if max_samples is None else int(max_samples)
    )
    use_bf16 = bool(
        str(config.get("inference_dtype", "bf16")).lower() == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    for rows in batches:
        if maximum:
            remaining = maximum - int(totals["samples"])
            if remaining <= 0:
                break
            rows = rows[:remaining]
        batch = collate_samples(rows, device, include_critic=train_critic)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            if train_critic:
                output = model(
                    batch["token_factors"], batch["token_numeric"], batch["legal_mask"],
                    batch["token_lengths"], critic_factors=batch["critic_factors"],
                    critic_lengths=batch["critic_lengths"],
                )
            else:
                output = model.forward_policy(
                    batch["token_factors"], batch["token_numeric"], batch["legal_mask"],
                    batch["token_lengths"],
                )
        logits = output["policy_logits"].float()
        targets = batch["actions"]
        ce = F.cross_entropy(logits, targets, reduction="none")
        top = logits.topk(3, dim=-1).indices
        totals["samples"] += len(rows)
        totals["policy_ce_sum"] += float(ce.sum())
        if train_critic:
            values = output["value"].float()
            value_targets = batch["value_targets"]
            huber = F.huber_loss(values, value_targets, reduction="none")
            error = values - value_targets
            totals["value_huber_sum"] += float(huber.sum())
            totals["value_abs_sum"] += float(error.abs().sum())
            totals["value_sq_sum"] += float(error.square().sum())
        totals["top1"] += float((top[:, 0] == targets).sum())
        totals["top3"] += float((top == targets[:, None]).any(-1).sum())
        for row, target, predictions in zip(rows, targets.tolist(), top.tolist(), strict=True):
            group = _action_group(row.action)
            counts = group_correct.setdefault(group, [0, 0, 0])
            counts[0] += int(target == predictions[0])
            counts[1] += int(target in predictions)
            counts[2] += 1
    count = max(totals["samples"], 1.0)
    result = {
        "validation/samples": totals["samples"],
        "validation/policy_ce": totals["policy_ce_sum"] / count,
        "validation/top1": totals["top1"] / count,
        "validation/top3": totals["top3"] / count,
    }
    if train_critic:
        result.update({
            "validation/value_huber": totals["value_huber_sum"] / count,
            "validation/value_mae": totals["value_abs_sum"] / count,
            "validation/value_rmse": math.sqrt(totals["value_sq_sum"] / count),
        })
    for group, (correct, top3_correct, group_count) in sorted(group_correct.items()):
        result[f"validation/top1_{group}"] = correct / max(group_count, 1)
        result[f"validation/top3_{group}"] = top3_correct / max(group_count, 1)
    result["validation/loss"] = result["validation/policy_ce"]
    if train_critic:
        result["validation/loss"] += (
            float(config["value_coef"]) * result["validation/value_huber"]
        )
    return result


def _rank_steps(world_size: int, local_steps: int) -> list[int]:
    if world_size == 1:
        return [local_steps]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, int(local_steps))
    return [int(value) for value in gathered]


def _train_worker_impl(
    rank: int,
    world_size: int,
    config: dict[str, Any],
    dataset: Path,
    output: Path,
    writers: list[SummaryWriter],
) -> None:
    seed = int(config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    distributed = world_size > 1
    device = torch.device(f"cuda:{rank}" if str(config["device"]).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    model = KyokuTransformerActorCritic(_model_config(config)).to(device)
    train_critic = bool(config["train_critic"])
    critic_roots = {"critic_embedding", "critic_backbone", "value_head", "value_query"}
    if not train_critic:
        for name, parameter in model.named_parameters():
            if name.split(".", 1)[0] in critic_roots:
                parameter.requires_grad_(False)
    optimized_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=float(config["learning_rate"]),
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        eps=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
        fused=device.type == "cuda",
    )
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    train_decisions = int(manifest["counts"]["train_decisions"])
    estimated_steps = max(
        1,
        math.ceil(train_decisions / int(config["batch_size"])) * int(config["epochs"]),
    )
    warmup = max(1, int(estimated_steps * float(config["warmup_fraction"])))

    def lr_scale(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = min(1.0, (step - warmup) / max(estimated_steps - warmup, 1))
        ratio = float(config["min_learning_rate"]) / float(config["learning_rate"])
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    manifest_hash = dataset_manifest_hash(dataset)
    start_epoch = 0
    global_step = 0
    skip_steps = 0
    best_validation_loss = float("inf")
    best_heuristic_point_delta = float("-inf")
    if config.get("resume"):
        payload = torch.load(config["resume"], map_location=device, weights_only=False)
        if int(payload.get("token_schema_version", 0)) != TOKEN_SCHEMA_VERSION:
            raise RuntimeError("SFT resume checkpoint has an incompatible token schema")
        if payload.get("dataset_manifest_hash") != manifest_hash:
            raise RuntimeError("SFT resume checkpoint belongs to a different dataset manifest")
        expected_mode = "joint_actor_critic" if train_critic else "actor_only"
        if payload.get("training_mode", expected_mode) != expected_mode:
            raise RuntimeError("SFT resume checkpoint training mode differs from current config")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("global_step", 0))
        rank_progress = payload.get("rank_steps", [])
        skip_steps = int(rank_progress[rank]) if rank < len(rank_progress) else 0
        best_validation_loss = float(payload.get("best_validation_loss", float("inf")))
        best_heuristic_point_delta = float(
            payload.get("best_heuristic_point_delta", float("-inf"))
        )
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"].cpu())
            np.random.set_state(payload["numpy_rng"])
            random.setstate(payload["python_rng"])
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    writer: SummaryWriter | None = None
    if rank == 0 and bool(config.get("tensorboard_enabled", True)):
        tensorboard_path = output / str(config.get("tensorboard_dirname", "tensorboard"))
        writer = SummaryWriter(
            str(tensorboard_path),
            purge_step=global_step if config.get("resume") else None,
        )
        writers.append(writer)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[rank], broadcast_buffers=False
        )
    local_batch = max(1, int(config["batch_size"]) // world_size)
    use_bf16 = bool(
        str(config.get("inference_dtype", "bf16")).lower() == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    local_steps = 0
    local_samples = 0
    local_effective_tokens = 0
    local_padded_tokens = 0
    started = time.perf_counter()
    last_step_end = started
    metric_window = SftMetricWindow() if rank == 0 else None
    model.train()
    join_context = model.join if isinstance(model, DistributedDataParallel) else nullcontext
    with join_context():
        for epoch in range(start_epoch, int(config["epochs"])):
            sample_stream = iter_split_samples(
                dataset,
                "train",
                gamma=float(config["gamma"]),
                seed=int(config["seed"]) + epoch,
                shuffle=True,
                shuffle_buffer_kyokus=int(config["shuffle_buffer_kyokus"]),
                rank=rank,
                world_size=world_size,
                include_critic=train_critic,
            )
            batches = length_bucketed_batches(
                sample_stream,
                local_batch,
                window_batches=int(config["length_bucket_window_batches"]),
                rng=random.Random(int(config["seed"]) + epoch * 1_000_003 + rank),
            )
            for batch_index, rows in enumerate(batches):
                if epoch == start_epoch and batch_index < skip_steps:
                    continue
                step_started = last_step_end
                batch = collate_samples(rows, device, include_critic=train_critic)
                effective_tokens = sum(sample.token_length + 1 for sample in rows)
                padded_tokens = len(rows) * (max(sample.token_length for sample in rows) + 1)
                local_samples += len(rows)
                local_effective_tokens += effective_tokens
                local_padded_tokens += padded_tokens
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    if train_critic:
                        model_output = model(
                            batch["token_factors"], batch["token_numeric"], batch["legal_mask"],
                            batch["token_lengths"], critic_factors=batch["critic_factors"],
                            critic_lengths=batch["critic_lengths"],
                        )
                    else:
                        model_output = model(
                            batch["token_factors"], batch["token_numeric"], batch["legal_mask"],
                            batch["token_lengths"], policy_only=True,
                        )
                    policy_loss = F.cross_entropy(model_output["policy_logits"].float(), batch["actions"])
                    if train_critic:
                        value_loss = F.huber_loss(
                            model_output["value"].float(), batch["value_targets"]
                        )
                        loss = policy_loss + float(config["value_coef"]) * value_loss
                    else:
                        value_loss = None
                        loss = policy_loss
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                step_finished = time.perf_counter()
                local_steps += 1
                global_step += 1
                if metric_window is not None:
                    metric_window.update(
                        logits=model_output["policy_logits"].float(),
                        actions=batch["actions"],
                        legal_mask=batch["legal_mask"],
                        token_lengths=batch["token_lengths"],
                        loss=loss,
                        policy_ce=policy_loss,
                        value_huber=value_loss,
                        effective_tokens=effective_tokens,
                        padded_tokens=padded_tokens,
                        step_seconds=step_finished - step_started,
                    )
                log_interval = max(1, int(config["log_interval_steps"]))
                if rank == 0 and global_step % log_interval == 0:
                    elapsed = time.perf_counter() - started
                    assert metric_window is not None
                    log_metrics = metric_window.scalars()
                    log_metrics.update({
                        "optimizer/learning_rate": float(scheduler.get_last_lr()[0]),
                        "optimizer/grad_norm_pre_clip": float(grad_norm),
                        "optimizer/grad_norm_post_clip": min(
                            float(grad_norm), float(config["max_grad_norm"])
                        ),
                        "performance/cumulative_samples_per_s": (
                            local_samples / max(elapsed, 1e-9)
                        ),
                    })
                    if device.type == "cuda":
                        log_metrics.update({
                            "system/gpu_memory_allocated_mb": (
                                torch.cuda.memory_allocated(device) / 2**20
                            ),
                            "system/gpu_memory_reserved_mb": (
                                torch.cuda.memory_reserved(device) / 2**20
                            ),
                            "system/gpu_memory_peak_mb": (
                                torch.cuda.max_memory_allocated(device) / 2**20
                            ),
                        })
                    if writer is not None:
                        write_sft_scalars(writer, log_metrics, global_step)
                    print(
                        f"epoch={epoch + 1} step={global_step} "
                        f"loss={log_metrics['train/loss']:.5f} "
                        f"policy_ce={log_metrics['train/policy_ce']:.5f} "
                        + (
                            f"value_huber={log_metrics['train/value_huber']:.5f} "
                            if "train/value_huber" in log_metrics else ""
                        )
                        +
                        f"top1={log_metrics['train/top1']:.5f} "
                        f"top3={log_metrics['train/top3']:.5f} "
                        f"grad_norm={float(grad_norm):.5f} "
                        f"lr={scheduler.get_last_lr()[0]:.8f} "
                        f"samples_per_s={log_metrics['performance/window_samples_per_s']:.2f}",
                        flush=True,
                    )
                    metric_window.reset()
                checkpoint_interval = int(config["checkpoint_interval_steps"])
                validation_interval = int(config.get("validation_interval_steps", 0))
                heuristic_interval = int(
                    config.get("heuristic_evaluation_interval_steps", 0)
                )
                checkpoint_due = (
                    checkpoint_interval > 0 and global_step % checkpoint_interval == 0
                )
                validation_due = (
                    validation_interval > 0 and global_step % validation_interval == 0
                )
                heuristic_due = (
                    bool(config.get("heuristic_evaluation_enabled", True))
                    and heuristic_interval > 0
                    and global_step % heuristic_interval == 0
                )
                if checkpoint_due or validation_due or heuristic_due:
                    progress = _rank_steps(world_size, local_steps)
                    if rank == 0:
                        if validation_due:
                            module = (
                                model.module
                                if isinstance(model, DistributedDataParallel) else model
                            )
                            validation = evaluate(
                                module,
                                dataset,
                                config,
                                device,
                                max_samples=int(config["validation_samples_per_run"]),
                            )
                            if writer is not None:
                                write_sft_scalars(writer, validation, global_step)
                                writer.flush()
                            candidate = float(validation["validation/loss"])
                            if candidate < best_validation_loss:
                                best_validation_loss = candidate
                                _save_checkpoint(
                                    output / "best.pt", model, optimizer, scheduler,
                                    config=config, manifest_hash=manifest_hash, epoch=epoch,
                                    global_step=global_step, rank_steps=progress,
                                    best_validation_loss=best_validation_loss,
                                    best_heuristic_point_delta=best_heuristic_point_delta,
                                    metrics=validation,
                                )
                            model.train()
                        if heuristic_due:
                            module = (
                                model.module
                                if isinstance(model, DistributedDataParallel) else model
                            )
                            heuristic_metrics = evaluate_against_heuristics(
                                module,
                                device,
                                config,
                                # Checkpoint selection must compare every
                                # candidate on exactly the same deterministic
                                # games. Rotating seeds belong in a separate
                                # generalization evaluation, not this score.
                                cycle=0,
                            )
                            if writer is not None:
                                write_sft_scalars(
                                    writer, heuristic_metrics, global_step,
                                )
                                writer.flush()
                            point_delta = float(
                                heuristic_metrics[
                                    "heuristic_eval/kyoku/point_delta_mean"
                                ]
                            )
                            if point_delta > best_heuristic_point_delta:
                                best_heuristic_point_delta = point_delta
                                _save_checkpoint(
                                    output / "best_heuristic.pt",
                                    model,
                                    optimizer,
                                    scheduler,
                                    config=config,
                                    manifest_hash=manifest_hash,
                                    epoch=epoch,
                                    global_step=global_step,
                                    rank_steps=progress,
                                    best_validation_loss=best_validation_loss,
                                    best_heuristic_point_delta=best_heuristic_point_delta,
                                    metrics=heuristic_metrics,
                                )
                            with (output / "heuristic_evaluation.jsonl").open(
                                "a", encoding="utf-8",
                            ) as file:
                                file.write(json.dumps({
                                    "global_step": global_step,
                                    "metrics": heuristic_metrics,
                                }, ensure_ascii=False) + "\n")
                            print(
                                f"heuristic_evaluation step={global_step} "
                                f"kyokus={heuristic_metrics.get('heuristic_eval/kyoku/count', 0):.0f} "
                                f"point_delta_mean={point_delta:.4f} "
                                f"mean_rank={heuristic_metrics.get('heuristic_eval/match/mean_rank', 0.0):.4f} "
                                f"win_rate={heuristic_metrics.get('heuristic_eval/kyoku/win_rate', 0.0):.4f} "
                                f"deal_in_rate={heuristic_metrics.get('heuristic_eval/kyoku/deal_in_rate', 0.0):.4f}",
                                flush=True,
                            )
                            model.train()
                        if checkpoint_due:
                            _save_checkpoint(
                                output / "latest.pt", model, optimizer, scheduler,
                                config=config, manifest_hash=manifest_hash, epoch=epoch,
                                global_step=global_step, rank_steps=progress,
                                best_validation_loss=best_validation_loss,
                                best_heuristic_point_delta=best_heuristic_point_delta,
                            )
                    if distributed:
                        # Keep every DDP rank at the same safe boundary while
                        # rank 0 owns validation/evaluation inference.
                        dist.barrier()
                    last_step_end = time.perf_counter()
                else:
                    last_step_end = step_finished
    if distributed:
        dist.barrier()
    progress = _rank_steps(world_size, local_steps)
    throughput_values = torch.tensor(
        [local_samples, local_effective_tokens, local_padded_tokens],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(throughput_values, op=dist.ReduceOp.SUM)
    metrics: dict[str, float] = {}
    if rank == 0:
        module = model.module if isinstance(model, DistributedDataParallel) else model
        metrics = evaluate(module, dataset, config, device)
        if writer is not None:
            write_sft_scalars(writer, metrics, global_step)
        metrics["training/elapsed_s"] = time.perf_counter() - started
        metrics["training/global_step"] = float(global_step)
        metrics["training/samples"] = float(throughput_values[0])
        metrics["training/samples_per_s"] = float(throughput_values[0]) / max(
            metrics["training/elapsed_s"], 1e-9
        )
        metrics["training/effective_tokens"] = float(throughput_values[1])
        metrics["training/padded_tokens"] = float(throughput_values[2])
        metrics["training/padding_fraction"] = 1.0 - float(throughput_values[1]) / max(
            float(throughput_values[2]), 1.0
        )
        if device.type == "cuda":
            metrics["training/gpu_peak_allocated_mb"] = (
                torch.cuda.max_memory_allocated(device) / 2**20
            )
        if bool(config.get("heuristic_evaluation_enabled", True)):
            final_heuristic = evaluate_against_heuristics(
                module,
                device,
                config,
                hanchan_count=int(
                    config.get("heuristic_evaluation_final_hanchan_count", 128)
                ),
                cycle=0,
            )
            metrics.update(final_heuristic)
            if writer is not None:
                write_sft_scalars(writer, final_heuristic, global_step)
            final_point_delta = float(
                final_heuristic["heuristic_eval/kyoku/point_delta_mean"]
            )
            if final_point_delta > best_heuristic_point_delta:
                best_heuristic_point_delta = final_point_delta
                _save_checkpoint(
                    output / "best_heuristic.pt",
                    model,
                    optimizer,
                    scheduler,
                    config=config,
                    manifest_hash=manifest_hash,
                    epoch=int(config["epochs"]),
                    global_step=global_step,
                    rank_steps=progress,
                    best_validation_loss=best_validation_loss,
                    best_heuristic_point_delta=best_heuristic_point_delta,
                    metrics=final_heuristic,
                )
        _save_checkpoint(
            output / "latest.pt", model, optimizer, scheduler,
            config=config, manifest_hash=manifest_hash, epoch=int(config["epochs"]),
            global_step=global_step, rank_steps=progress,
            best_validation_loss=min(
                best_validation_loss, float(metrics["validation/loss"])
            ),
            best_heuristic_point_delta=best_heuristic_point_delta,
            metrics=metrics,
        )
        if float(metrics["validation/loss"]) < best_validation_loss:
            best_validation_loss = float(metrics["validation/loss"])
            _save_checkpoint(
                output / "best.pt", model, optimizer, scheduler,
                config=config, manifest_hash=manifest_hash, epoch=int(config["epochs"]),
                global_step=global_step, rank_steps=progress,
                best_validation_loss=best_validation_loss, metrics=metrics,
                best_heuristic_point_delta=best_heuristic_point_delta,
            )
        (output / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if writer is not None:
            writer.flush()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def train_worker(
    rank: int, world_size: int, config: dict[str, Any], dataset: Path, output: Path,
) -> None:
    writers: list[SummaryWriter] = []
    try:
        _train_worker_impl(rank, world_size, config, dataset, output, writers)
    finally:
        for writer in writers:
            writer.flush()
            writer.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_output(config: dict[str, Any], override: Path | None) -> Path:
    return override if override is not None else Path(str(config["checkpoint_dir"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--learner-gpus", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume:
        config["resume"] = str(args.resume)
    if args.device:
        config["device"] = args.device
    if args.learner_gpus is not None:
        config["learner_gpus"] = args.learner_gpus
    output = resolve_output(config, args.output)
    config["checkpoint_dir"] = str(output)
    world_size = int(config["learner_gpus"]) if str(config["device"]).startswith("cuda") else 1
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        torch.multiprocessing.spawn(
            train_worker,
            args=(world_size, config, args.dataset, output),
            nprocs=world_size,
            join=True,
        )
    else:
        train_worker(0, 1, config, args.dataset, output)


if __name__ == "__main__":
    main()
