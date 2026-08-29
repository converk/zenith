"""V18 actor-only SFT 从零训练入口。

V18 输入为当前局面状态快照（Shared 公共前缀 + Opponent Analysis + 每动作
Offense/Defense Query），网络为 current_state_snapshot 策略头。节奏键
(每 3000 steps 验证/保存、最终 96 半庄)只引用 ``sft/contract.py`` 的机制常量,
禁止实验配置复制。
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
from .actor_bc import actor_parameters, freeze_critic
from .contract import (
    SFT_CADENCE_STEPS,
    dataset_manifest_hash,
    load_manifest,
    training_mode,
    validate_manifest,
)
from .data import EncodedSample, iter_split_samples
from .tensorboard import SftMetricWindow, write_sft_scalars


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1,
    "device": "cuda",
    "learner_gpus": 2,
    "model_size": "v18",
    "context_tokens": 256,
    "policy_head_type": "current_state_snapshot",
    "dense_slot_dim": 32,
    "dense_fusion_dim": 512,
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
    "inference_dtype": "bf16",
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
            raise ValueError("V18 SFT config must be a mapping")
        config.update(overlay)
    return config


def validate_config(config: dict[str, Any]) -> None:
    duplicated = set(_CADENCE_KEYS) & set(config)
    if duplicated:
        raise ValueError(
            "V18 SFT cadence keys must stay single-sourced in sft/contract.py: "
            + ", ".join(sorted(duplicated))
        )
    if str(config.get("policy_head_type")) != "current_state_snapshot":
        raise ValueError("V18 SFT requires policy_head_type=current_state_snapshot")
    if str(config.get("model_size")) != "v18":
        raise ValueError("V18 SFT requires model_size=v18")
    if bool(config.get("train_critic", False)) or bool(config.get("train_public_value", False)):
        raise ValueError("V18 SFT is actor-only; critic training is not supported here")
    dataset = Path(str(config["dataset"]))
    if not (dataset / "manifest.json").is_file():
        raise FileNotFoundError(f"V18 SFT dataset manifest does not exist: {dataset}")
    world_size = int(config["learner_gpus"]) if str(config["device"]).startswith("cuda") else 1
    if world_size <= 0:
        raise ValueError("learner_gpus must be positive")
    if int(config["batch_size"]) <= 0 or int(config["batch_size"]) % world_size:
        raise ValueError("global batch_size must be positive and divisible by learner_gpus")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive")
    if int(config["context_tokens"]) <= 0:
        raise ValueError("context_tokens must be positive")


def _model_config(config: dict[str, Any]) -> ModelConfig:
    base = ModelConfig.preset(str(config["model_size"]))
    values = {
        **base.__dict__,
        "context_tokens": int(config["context_tokens"]),
        "dense_slot_dim": int(config["dense_slot_dim"]),
        "dense_fusion_dim": int(config["dense_fusion_dim"]),
    }
    return ModelConfig(**values)


def collate_samples(
    samples: list[EncodedSample],
    device: torch.device,
    *,
    validate_semantics: bool = True,
) -> dict[str, torch.Tensor]:
    batch = len(samples)
    actor_max = max(sample.token_length for sample in samples)
    action_max = max(sample.query_pair_count for sample in samples)
    from ..model.encoding_protocol import TOKEN_NUMERIC_WIDTH, TOKEN_ROW_WIDTH

    actor_factors_np = np.zeros((batch, actor_max, TOKEN_ROW_WIDTH), dtype=np.int64)
    actor_numeric_np = np.zeros((batch, actor_max, TOKEN_NUMERIC_WIDTH), dtype=np.float32)
    actor_lengths_np = np.empty(batch, dtype=np.int64)
    query_rows_np = np.zeros((batch, 2 * action_max, 15), dtype=np.int64)
    action_ids_np = np.zeros((batch, action_max), dtype=np.int64)
    pair_counts_np = np.empty(batch, dtype=np.int64)
    legal_np = np.zeros((batch, NUM_ACTIONS), dtype=np.bool_)
    actions_np = np.empty(batch, dtype=np.int64)
    for row, sample in enumerate(samples):
        actor_factors_np[row, : sample.token_length] = sample.actor_factors
        actor_numeric_np[row, : sample.token_length] = sample.actor_numeric
        actor_lengths_np[row] = sample.token_length
        query_rows_np[row, : sample.query_rows.shape[0]] = sample.query_rows
        action_ids_np[row, : sample.query_pair_count] = sample.action_ids
        pair_counts_np[row] = sample.query_pair_count
        legal_np[row] = sample.legal_mask
        actions_np[row] = sample.action
    if validate_semantics:
        from ..model.semantic_validation import assert_actor_input_semantics

        assert_actor_input_semantics(
            actor_factors_np,
            actor_numeric_np,
            actor_lengths_np,
            query_rows_np,
            action_ids_np,
            pair_counts_np,
            legal_np,
        )
    actor_factors = torch.from_numpy(actor_factors_np).to(device, non_blocking=True)
    actor_numeric = torch.from_numpy(actor_numeric_np).to(device, non_blocking=True)
    actor_lengths = torch.from_numpy(actor_lengths_np).to(device, non_blocking=True)
    query_rows = torch.from_numpy(query_rows_np).to(device, non_blocking=True)
    action_ids = torch.from_numpy(action_ids_np).to(device, non_blocking=True)
    pair_counts = torch.from_numpy(pair_counts_np).to(device, non_blocking=True)
    legal = torch.from_numpy(legal_np).to(device, non_blocking=True)
    actions = torch.from_numpy(actions_np).to(device, non_blocking=True)
    return {
        "actor_factors": actor_factors,
        "actor_numeric": actor_numeric,
        "actor_lengths": actor_lengths,
        "query_rows": query_rows,
        "action_ids": action_ids,
        "pair_counts": pair_counts,
        "legal_mask": legal,
        "actions": actions,
    }


def length_bucketed_batches(
    samples: Iterable[EncodedSample],
    batch_size: int,
    *,
    window_batches: int,
    rng: random.Random | None = None,
) -> Iterator[list[EncodedSample]]:
    window: list[EncodedSample] = []
    capacity = max(batch_size, batch_size * window_batches)

    def drain(rows: list[EncodedSample]) -> Iterator[list[EncodedSample]]:
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


def _forward_actor(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # 统一走 __call__/forward 分发,使 DistributedDataParallel 也能正确触发
    # 梯度同步;统一走 forward 分发。
    return model(
        actor_factors=batch["actor_factors"],
        actor_numeric=batch["actor_numeric"],
        actor_lengths=batch["actor_lengths"],
        query_action_ids=batch["action_ids"],
        query_pair_counts=batch["pair_counts"],
        legal_mask=batch["legal_mask"],
        policy_only=True,
    )


def _assert_targets_legal(actions: torch.Tensor, legal_mask: torch.Tensor) -> None:
    """BC 损失前重验目标动作 ∈ legal_mask（fail closed，拒绝损坏样本）。"""
    if not torch.all(legal_mask.gather(1, actions.view(-1, 1))):
        raise RuntimeError("BC target action is not present in legal_mask (corrupt sample)")


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
    local_batch = max(1, int(config["batch_size"]) // max(1, int(config["learner_gpus"])))
    samples = iter_split_samples(
        dataset, "validation",
        seed=int(config["seed"]), shuffle=False, include_critic=False,
    )
    maximum = int(config.get("validation_max_samples", 0)) if max_samples is None else int(max_samples)
    if maximum:
        samples = itertools.islice(samples, maximum)
    batches = length_bucketed_batches(
        samples, local_batch, window_batches=int(config["length_bucket_window_batches"]),
    )
    use_bf16 = bool(
        str(config.get("inference_dtype", "bf16")).lower() == "bf16"
        and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    total = ce_sum = top1 = top3 = 0
    for rows in batches:
        batch = collate_samples(
            rows, device, validate_semantics=bool(config.get("validate_semantics", True)),
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            output = _forward_actor(model, batch)
        logits = output["policy_logits"].float()
        targets = batch["actions"]
        _assert_targets_legal(targets, batch["legal_mask"])
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


def _train_worker_impl(
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
    model = KyokuTransformerActorCritic(_model_config(config)).to(device)
    if config.get("init_model"):
        initialized = torch.load(str(config["init_model"]), map_location="cpu")
        payload = initialized.get("model", initialized)
        model.load_state_dict(payload, strict=True)
        del initialized
    freeze_critic(model)
    # 可选的 torch.compile 快速路径：在 DDP 包装前编译原始模块
    # （配置 torch_compile: true 开启；首次迭代编译较慢，需重启训练生效）。
    if bool(config.get("torch_compile", False)):
        model = torch.compile(model)
    optimized = list(actor_parameters(model))
    if not optimized:
        raise RuntimeError("V18 SFT configuration leaves no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimized,
        lr=float(config["learning_rate"]),
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        eps=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
        fused=device.type == "cuda",
    )
    manifest = load_manifest(dataset)
    validate_manifest(manifest)
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
                dataset, "train",
                seed=int(config["seed"]) + epoch, shuffle=True,
                rank=rank, world_size=world_size, include_critic=False,
            )
            batches = length_bucketed_batches(
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
                batch = collate_samples(
                    rows, device, validate_semantics=bool(config.get("validate_semantics", True)),
                )
                effective_tokens = sum(sample.token_length for sample in rows)
                padded_tokens = len(rows) * max(sample.token_length for sample in rows)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    model_output = _forward_actor(model, batch)
                    _assert_targets_legal(batch["actions"], batch["legal_mask"])
                    loss = F.cross_entropy(model_output["policy_logits"].float(), batch["actions"])
                    policy_ce = loss.detach()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(config["max_grad_norm"]),
                )
                optimizer.step()
                scheduler.step()
                global_step += 1
                steps_in_epoch += 1
                if metric_window is not None:
                    total_lengths = batch["actor_lengths"]
                    metric_window.update(
                        logits=model_output["policy_logits"].detach(),
                        actions=batch["actions"],
                        legal_mask=batch["legal_mask"],
                        token_lengths=total_lengths,
                        loss=loss.detach(),
                        policy_ce=policy_ce,
                        effective_tokens=effective_tokens,
                        padded_tokens=padded_tokens,
                        step_seconds=time.perf_counter() - step_started,
                    )
                if global_step % int(config["log_interval_steps"]) == 0 and rank == 0:
                    print(f"epoch={epoch} step={global_step} loss={float(loss):.4f}", flush=True)
                if global_step % SFT_CADENCE_STEPS == 0 and rank == 0 and metric_window is not None:
                    metrics = metric_window.scalars()
                    validation = evaluate(
                        model.module if isinstance(model, DistributedDataParallel) else model,
                        dataset, config, device,
                        max_samples=int(config["validation_samples_per_run"]),
                    )
                    metrics.update(validation)
                    progress = [steps_in_epoch] * world_size
                    if validation["validation/policy_ce"] < best_validation_loss:
                        best_validation_loss = float(validation["validation/policy_ce"])
                    _save_checkpoint(
                        output / "latest.pt", model, optimizer, scheduler,
                        config=config, manifest_hash=manifest_hash, epoch=epoch,
                        global_step=global_step, rank_batches_consumed=progress,
                        best_validation_loss=best_validation_loss, metrics=metrics,
                    )
                    if float(validation["validation/policy_ce"]) <= best_validation_loss:
                        _save_checkpoint(
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
            final_metrics = evaluate(
                model.module if isinstance(model, DistributedDataParallel) else model,
                dataset, config, device,
                max_samples=int(config["validation_samples_per_run"]),
            )
            metrics.update(final_metrics)
            _save_checkpoint(
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
    print(f"rank={rank} V18 SFT finished steps={global_step} elapsed={elapsed:.1f}s", flush=True)


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
            train_worker,
            args=(world_size, config, dataset, output),
            nprocs=world_size,
            join=True,
        )
    else:
        train_worker(0, 1, config, dataset, output)
