"""V16 actor-only SFT 从零训练入口。

V16 输入为 Objective Facts + Compact Snapshot + 每动作一对 Query,网络为
symmetric_action_query 策略头。节奏键(每 3000 steps 验证/保存、
最终 96 半庄)只引用 ``sft/contract.py`` 的机制常量,禁止实验配置复制。
"""

from __future__ import annotations

from contextlib import nullcontext
import itertools
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
from ..model.schema import NUM_ACTIONS
from .checkpoint import checkpoint_payload, load_exact_resume
from .contract import (
    SFT_CADENCE_STEPS,
    SFT_FINAL_EVAL_HANCHAN_COUNT,
    dataset_manifest_hash,
    load_manifest,
    training_mode,
    validate_v16_manifest,
)
from .data import V16Sample, iter_split_samples
from .tensorboard import SftMetricWindow, write_sft_scalars


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1,
    "device": "cuda",
    "learner_gpus": 2,
    "model_size": "v16",
    "context_tokens": 4096,
    "policy_head_type": "symmetric_action_query",
    "epochs": 1,
    "train_critic": False,
    "train_public_value": False,
    "batch_size": 512,
    "learning_rate": 1.5e-4,
    "min_learning_rate": 2e-5,
    "warmup_fraction": 0.02,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "inference_dtype": "bf16",
    "shuffle_buffer_kyokus": 8192,
    "length_bucket_window_batches": 32,
    "log_interval_steps": 100,
    "validation_max_samples": 0,
    "validation_samples_per_run": 150000,
    "max_train_steps": 0,
    "stop_after_steps": 0,
    "tensorboard_enabled": True,
    "tensorboard_dirname": "tensorboard",
    "resume": None,
    "init_model": None,
}

# 节奏键单一来源(sft/contract.py 常量),实验配置出现这些键即拒绝。
_CADENCE_KEYS = (
    "validation_interval_steps",
    "checkpoint_interval_steps",
)


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        with path.open(encoding="utf-8") as file:
            overlay = yaml.safe_load(file)
        if not isinstance(overlay, dict):
            raise ValueError("V16 SFT config must be a mapping")
        config.update(overlay)
    return config


def validate_config(config: dict[str, Any]) -> None:
    duplicated = set(_CADENCE_KEYS) & set(config)
    if duplicated:
        raise ValueError(
            "v16 SFT cadence keys must stay single-sourced in sft/contract.py: "
            + ", ".join(sorted(duplicated))
        )
    if str(config.get("policy_head_type")) != "symmetric_action_query":
        raise ValueError("v16 SFT requires policy_head_type=symmetric_action_query")
    if str(config.get("model_size")) != "v16":
        raise ValueError("v16 SFT requires model_size=v16")
    if bool(config.get("train_critic", False)) or bool(config.get("train_public_value", False)):
        raise ValueError("v16 SFT is actor-only; critic training is not supported here")
    dataset = Path(str(config["dataset"]))
    if not (dataset / "manifest.json").is_file():
        raise FileNotFoundError(f"v16 SFT dataset manifest does not exist: {dataset}")
    world_size = int(config["learner_gpus"]) if str(config["device"]).startswith("cuda") else 1
    if world_size <= 0:
        raise ValueError("learner_gpus must be positive")
    if int(config["batch_size"]) <= 0 or int(config["batch_size"]) % world_size:
        raise ValueError("global batch_size must be positive and divisible by learner_gpus")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive")
    if int(config["context_tokens"]) <= 0:
        raise ValueError("context_tokens must be positive")


def _v16_model_config(config: dict[str, Any]) -> ModelConfig:
    base = ModelConfig.preset(str(config["model_size"]))
    values = {
        **base.__dict__,
        "context_tokens": int(config["context_tokens"]),
    }
    return ModelConfig(**values)


def collate_v16(samples: list[V16Sample], device: torch.device) -> dict[str, torch.Tensor]:
    batch = len(samples)
    history_max = max(sample.history_length for sample in samples)
    snapshot_max = max(sample.snapshot_length for sample in samples)
    action_max = max(sample.query_pair_count for sample in samples)
    history_factors = torch.zeros((batch, history_max, 10), dtype=torch.uint8)
    history_numeric = torch.zeros((batch, history_max, 8), dtype=torch.float32)
    history_lengths = torch.empty(batch, dtype=torch.long)
    snapshot_kinds = torch.zeros((batch, snapshot_max), dtype=torch.long)
    snapshot_cat = torch.zeros((batch, snapshot_max, 4), dtype=torch.long)
    snapshot_num = torch.zeros((batch, snapshot_max, 7), dtype=torch.float32)
    snapshot_lengths = torch.empty(batch, dtype=torch.long)
    query_rows = torch.zeros((batch, 2 * action_max, 15), dtype=torch.long)
    action_ids = torch.zeros((batch, action_max), dtype=torch.long)
    pair_counts = torch.empty(batch, dtype=torch.long)
    legal = torch.zeros((batch, NUM_ACTIONS), dtype=torch.bool)
    actions = torch.empty(batch, dtype=torch.long)
    for row, sample in enumerate(samples):
        history_factors[row, : sample.history_length] = torch.from_numpy(sample.history_factors)
        history_numeric[row, : sample.history_length] = torch.from_numpy(sample.history_numeric)
        history_lengths[row] = sample.history_length
        snapshot_kinds[row, : sample.snapshot_length] = torch.from_numpy(sample.snapshot_kinds)
        snapshot_cat[row, : sample.snapshot_length] = torch.from_numpy(sample.snapshot_cat)
        snapshot_num[row, : sample.snapshot_length] = torch.from_numpy(sample.snapshot_num)
        snapshot_lengths[row] = sample.snapshot_length
        query_rows[row, : sample.query_rows.shape[0]] = torch.from_numpy(sample.query_rows)
        action_ids[row, : sample.query_pair_count] = torch.from_numpy(sample.action_ids)
        pair_counts[row] = sample.query_pair_count
        legal[row] = torch.from_numpy(sample.legal_mask)
        actions[row] = sample.action
    return {
        "history_factors": history_factors.to(device, non_blocking=True),
        "history_numeric": history_numeric.to(device, non_blocking=True),
        "history_lengths": history_lengths.to(device, non_blocking=True),
        "snapshot_kinds": snapshot_kinds.to(device, non_blocking=True),
        "snapshot_cat": snapshot_cat.to(device, non_blocking=True),
        "snapshot_num": snapshot_num.to(device, non_blocking=True),
        "snapshot_lengths": snapshot_lengths.to(device, non_blocking=True),
        "query_rows": query_rows.to(device, non_blocking=True),
        "action_ids": action_ids.to(device, non_blocking=True),
        "pair_counts": pair_counts.to(device, non_blocking=True),
        "legal_mask": legal.to(device, non_blocking=True),
        "actions": actions.to(device, non_blocking=True),
    }


def length_bucketed_batches_v16(
    samples: Iterable[V16Sample],
    batch_size: int,
    *,
    window_batches: int,
    rng: random.Random | None = None,
) -> Iterator[list[V16Sample]]:
    window: list[V16Sample] = []
    capacity = max(batch_size, batch_size * window_batches)

    def drain(rows: list[V16Sample]) -> Iterator[list[V16Sample]]:
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


def _v16_forward(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # 统一走 __call__/forward 分发,使 DistributedDataParallel 也能正确触发
    # 梯度同步;DDP 包装后没有 forward_v16 方法。
    return model(
        history_factors=batch["history_factors"],
        history_numeric=batch["history_numeric"],
        history_lengths=batch["history_lengths"],
        snapshot_kinds=batch["snapshot_kinds"],
        snapshot_cat=batch["snapshot_cat"],
        snapshot_num=batch["snapshot_num"],
        snapshot_lengths=batch["snapshot_lengths"],
        query_rows=batch["query_rows"],
        query_action_ids=batch["action_ids"],
        query_pair_counts=batch["pair_counts"],
        legal_mask=batch["legal_mask"],
        policy_only=True,
    )


@torch.no_grad()
def evaluate_v16(
    model: nn.Module,
    dataset: Path,
    config: dict[str, Any],
    device: torch.device,
    *,
    max_samples: int | None = None,
) -> dict[str, float]:
    model.eval()
    local_batch = max(1, int(config["batch_size"]) // max(1, int(config["learner_gpus"])))
    samples = iter_split_samples(
        dataset, "validation", gamma=float(config["gamma"]),
        seed=int(config["seed"]), shuffle=False, include_critic=False,
    )
    maximum = int(config.get("validation_max_samples", 0)) if max_samples is None else int(max_samples)
    if maximum:
        samples = itertools.islice(samples, maximum)
    batches = length_bucketed_batches_v16(
        samples, local_batch, window_batches=int(config["length_bucket_window_batches"]),
    )
    use_bf16 = bool(
        str(config.get("inference_dtype", "bf16")).lower() == "bf16"
        and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    total = ce_sum = top1 = top3 = 0
    for rows in batches:
        batch = collate_v16(rows, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            output = _v16_forward(model, batch)
        logits = output["policy_logits"].float()
        targets = batch["actions"]
        ce_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
        top = logits.topk(3, dim=-1).indices
        top1 += int((top[:, 0] == targets).sum())
        top3 += int((top == targets[:, None]).any(-1).sum())
        total += len(rows)
    count = max(total, 1)
    return {
        "validation/samples": float(total),
        "validation/policy_ce": ce_sum / count,
        "validation/top1": top1 / count,
        "validation/top3": top3 / count,
        "validation/loss": ce_sum / count,
    }


def _save_checkpoint_v16(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    config: dict[str, Any],
    manifest_hash: str,
    epoch: int,
    global_step: int,
    rank_batches_consumed: list[int],
    best_validation_loss: float,
    metrics: dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model, optimizer, scheduler, config=config, manifest_hash=manifest_hash,
        mode=training_mode(config), epoch=epoch, global_step=global_step,
        rank_batches_consumed=rank_batches_consumed,
        best_validation_loss=best_validation_loss,
        metrics=dict(metrics or {}),
        rank_rng_states=[_local_rng_state()],
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _local_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _train_v16_worker_impl(
    rank: int,
    world_size: int,
    config: dict[str, Any],
    dataset: Path,
    output: Path,
    writers: list[SummaryWriter],
) -> None:
    validate_config(config)
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
    model = KyokuTransformerActorCritic(_v16_model_config(config)).to(device)
    if config.get("init_model"):
        initialized = torch.load(str(config["init_model"]), map_location="cpu")
        payload = initialized.get("model", initialized)
        model.load_state_dict(payload, strict=True)
        del initialized
    critic_roots = {"critic_embedding", "critic_backbone", "value_head", "value_query"}
    for name, parameter in model.named_parameters():
        if name.split(".", 1)[0] in critic_roots:
            parameter.requires_grad_(False)
    optimized = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not optimized:
        raise RuntimeError("v16 SFT configuration leaves no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimized,
        lr=float(config["learning_rate"]),
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        eps=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
        fused=device.type == "cuda",
    )
    manifest = load_manifest(dataset)
    validate_v16_manifest(manifest)
    manifest_hash = dataset_manifest_hash(dataset)
    train_decisions = int(manifest["counts"]["train_decisions"])
    estimated_steps = max(1, math.ceil(train_decisions / int(config["batch_size"])) * int(config["epochs"]))
    if int(config.get("max_train_steps", 0)) > 0:
        estimated_steps = min(estimated_steps, int(config["max_train_steps"]))
    warmup = max(1, int(estimated_steps * float(config["warmup_fraction"])))

    def lr_scale(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = min(1.0, (step - warmup) / max(estimated_steps - warmup, 1))
        ratio = float(config["min_learning_rate"]) / float(config["learning_rate"])
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    global_step = 0
    skip_steps = 0
    start_epoch = 0
    steps_in_epoch = 0
    best_validation_loss = float("inf")
    if config.get("resume"):
        payload = load_exact_resume(
            config["resume"], model_config=model.config, training_mode="actor_only",
            dataset_manifest_hash=manifest_hash, world_size=world_size,
            trainable_scope="full_actor",
        )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["data_cursor"]["epoch"])
        skip_steps = int(payload["data_cursor"]["rank_batches_consumed"][rank])
        global_step = int(payload["global_step"])
        best_validation_loss = float(payload.get("best_validation_loss", float("inf")))
        rng_state = payload["rank_rng_states"][rank]
        torch.set_rng_state(rng_state["torch"].cpu())
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["python"])
        if rng_state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"].cpu(), device=device)
    writer: SummaryWriter | None = None
    if rank == 0 and bool(config.get("tensorboard_enabled", True)):
        writer = SummaryWriter(str(output / str(config.get("tensorboard_dirname", "tensorboard"))))
        writers.append(writer)
    if distributed:
        # SFT 只训练 Actor,value/Q scorer 等 Critic 参数不参与前向/loss;
        # 必须允许未使用参数,否则 DDP 会在第二次迭代报 reduction 失败。
        model = DistributedDataParallel(
            model, device_ids=[rank], broadcast_buffers=False,
            find_unused_parameters=True,
        )
    local_batch = max(1, int(config["batch_size"]) // world_size)
    use_bf16 = bool(
        str(config.get("inference_dtype", "bf16")).lower() == "bf16"
        and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    started = time.perf_counter()
    metric_window = SftMetricWindow() if rank == 0 else None
    model.train()
    stop_training = False
    join_context = model.join if isinstance(model, DistributedDataParallel) else nullcontext
    with join_context():
        for epoch in range(start_epoch, int(config["epochs"])):
            steps_in_epoch = skip_steps if epoch == start_epoch else 0
            sample_stream = iter_split_samples(
                dataset, "train", gamma=float(config["gamma"]),
                seed=int(config["seed"]) + epoch, shuffle=True,
                shuffle_buffer_kyokus=int(config["shuffle_buffer_kyokus"]),
                rank=rank, world_size=world_size, include_critic=False,
            )
            batches = length_bucketed_batches_v16(
                sample_stream, local_batch,
                window_batches=int(config["length_bucket_window_batches"]),
                rng=random.Random(int(config["seed"]) + epoch * 1_000_003 + rank),
            )
            for batch_index, rows in enumerate(batches):
                if int(config.get("stop_after_steps", 0)) > 0 and global_step >= int(config["stop_after_steps"]):
                    stop_training = True
                    break
                if int(config.get("max_train_steps", 0)) > 0 and global_step >= int(config["max_train_steps"]):
                    stop_training = True
                    break
                if epoch == start_epoch and batch_index < skip_steps:
                    continue
                step_started = time.perf_counter()
                batch = collate_v16(rows, device)
                effective_tokens = sum(sample.token_length for sample in rows)
                padded_tokens = len(rows) * max(sample.token_length for sample in rows)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    model_output = _v16_forward(model, batch)
                    loss = F.cross_entropy(model_output["policy_logits"].float(), batch["actions"])
                    policy_ce = loss.detach()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(config["max_grad_norm"]),
                )
                optimizer.step()
                scheduler.step()
                global_step += 1
                steps_in_epoch += 1
                if metric_window is not None:
                    total_lengths = (
                        batch["history_lengths"] + batch["snapshot_lengths"] + 2 * batch["pair_counts"]
                    )
                    metric_window.update(
                        logits=model_output["policy_logits"].detach(),
                        actions=batch["actions"],
                        legal_mask=batch["legal_mask"],
                        token_lengths=total_lengths,
                        loss=loss.detach(),
                        policy_ce=policy_ce,
                        value_huber=None,
                        effective_tokens=effective_tokens,
                        padded_tokens=padded_tokens,
                        step_seconds=time.perf_counter() - step_started,
                    )
                if global_step % int(config["log_interval_steps"]) == 0 and rank == 0:
                    print(f"epoch={epoch} step={global_step} loss={float(loss):.4f}", flush=True)
                if global_step % SFT_CADENCE_STEPS == 0 and rank == 0 and metric_window is not None:
                    metrics = metric_window.scalars()
                    validation = evaluate_v16(
                        model.module if isinstance(model, DistributedDataParallel) else model,
                        dataset, config, device,
                        max_samples=int(config["validation_samples_per_run"]),
                    )
                    metrics.update(validation)
                    metrics["validation/loss"] = validation["validation/policy_ce"]
                    progress = [steps_in_epoch] * world_size
                    if validation["validation/policy_ce"] < best_validation_loss:
                        best_validation_loss = float(validation["validation/policy_ce"])
                    _save_checkpoint_v16(
                        output / "latest.pt", model, optimizer, scheduler,
                        config=config, manifest_hash=manifest_hash, epoch=epoch,
                        global_step=global_step, rank_batches_consumed=progress,
                        best_validation_loss=best_validation_loss, metrics=metrics,
                    )
                    if float(validation["validation/policy_ce"]) <= best_validation_loss:
                        _save_checkpoint_v16(
                            output / "best.pt", model, optimizer, scheduler,
                            config=config, manifest_hash=manifest_hash, epoch=epoch,
                            global_step=global_step, rank_batches_consumed=progress,
                            best_validation_loss=best_validation_loss, metrics=metrics,
                        )
                    (output / "metrics.json").write_text(
                        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                    )
                    if writer is not None:
                        write_sft_scalars(writer, metrics, global_step)
                        writer.flush()
                    metric_window = SftMetricWindow()
                    print(json.dumps(metrics, ensure_ascii=False), flush=True)
                if distributed and global_step % SFT_CADENCE_STEPS == 0:
                    dist.barrier()
            if stop_training:
                break
        if rank == 0:
            metrics = metric_window.scalars() if metric_window is not None and metric_window.steps else {}
            final_metrics = evaluate_v16(
                model.module if isinstance(model, DistributedDataParallel) else model,
                dataset, config, device,
                max_samples=int(config["validation_samples_per_run"]),
            )
            metrics.update(final_metrics)
            metrics["validation/loss"] = final_metrics["validation/policy_ce"]
            _save_checkpoint_v16(
                output / "latest.pt", model, optimizer, scheduler,
                config=config, manifest_hash=manifest_hash, epoch=start_epoch,
                global_step=global_step, rank_batches_consumed=[steps_in_epoch] * world_size,
                best_validation_loss=best_validation_loss, metrics=metrics,
            )
            (output / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            if writer is not None:
                write_sft_scalars(writer, metrics, global_step)
                writer.flush()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    elapsed = time.perf_counter() - started
    print(f"rank={rank} v16 SFT finished steps={global_step} elapsed={elapsed:.1f}s", flush=True)


def train_v16_worker(
    rank: int, world_size: int, config: dict[str, Any], dataset: Path, output: Path,
) -> None:
    writers: list[SummaryWriter] = []
    try:
        _train_v16_worker_impl(rank, world_size, config, dataset, output, writers)
    finally:
        for writer in writers:
            writer.flush()
            writer.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(config: dict[str, Any], dataset: Path | None = None, output: Path | None = None) -> None:
    dataset = dataset if dataset is not None else Path(str(config["dataset"]))
    output = output if output is not None else Path(str(config["checkpoint_dir"]))
    validate_config(config)
    config["checkpoint_dir"] = str(output)
    if config.get("resume"):
        if not Path(str(config["resume"])).is_file():
            raise FileNotFoundError(f"SFT resume checkpoint does not exist: {config['resume']}")
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"refusing to overwrite non-empty fresh-training output: {output}"
        )
    world_size = int(config["learner_gpus"]) if str(config["device"]).startswith("cuda") else 1
    if str(config["device"]).startswith("cuda") and torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"learner_gpus={world_size}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        torch.multiprocessing.spawn(
            train_v16_worker,
            args=(world_size, config, dataset, output),
            nprocs=world_size,
            join=True,
        )
    else:
        train_v16_worker(0, 1, config, dataset, output)
