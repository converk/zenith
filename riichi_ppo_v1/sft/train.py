"""Joint supervised policy/value training over prepared MJAI kyoku shards."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
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
from ..model.feature_schema import (
    DECISION_ANALYSIS_VERSION, RUST_ANALYSIS_VERSION, feature_schema_sha256,
    legacy_encoder_sha256,
)
from ..model.schema import TOKEN_SCHEMA_VERSION
from .data import SftSample, iter_split_samples
from .heuristic_evaluation import evaluate_against_heuristics
from .tensorboard import SftMetricWindow, write_sft_scalars


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1,
    "token_schema_version": TOKEN_SCHEMA_VERSION,
    "device": "cuda",
    "learner_gpus": 2,
    "model_size": "mid",
    "context_tokens": 4096,
    "critic_layers": 2,
    "epochs": 1,
    "max_train_steps": 0,
    "batch_size": 512,
    "learning_rate": 1.5e-4,
    "min_learning_rate": 2e-5,
    "warmup_fraction": 0.02,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "train_critic": False,
    "train_public_value": False,
    "policy_head_type": "isolated_action_query",
    "group_coef": 0.25,
    "public_value_coef": 0.0,
    "rule_coef": 0.05,
    "rule_decay_fraction": 0.20,
    "gamma": 0.99,
    "inference_dtype": "bf16",
    "shuffle_buffer_kyokus": 8192,
    "length_bucket_window_batches": 32,
    "checkpoint_interval_steps": 5000,
    "log_interval_steps": 100,
    "validation_max_samples": 0,
    "validation_interval_steps": 6000,
    "validation_samples_per_run": 150000,
    "heuristic_evaluation_enabled": True,
    "heuristic_evaluation_interval_steps": 18000,
    "heuristic_evaluation_hanchan_count": 96,
    "heuristic_evaluation_parallel_hanchan_count": 24,
    "heuristic_evaluation_final_hanchan_count": 96,
    "heuristic_evaluation_seed_base": 20260717,
    "heuristic_evaluation_game_mode": "4p-red-half",
    "heuristic_evaluation_max_steps": 4000,
    "checkpoint_dir": "checkpoints/train_riichi_v13_sft",
    "tensorboard_enabled": True,
    "tensorboard_dirname": "tensorboard",
    "resume": None,
    "ablation_aligned_batches": False,
    "ablation_identity_reference_dataset": None,
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


def validate_config(config: dict[str, Any]) -> None:
    """Reject launch-time settings whose effective semantics are ambiguous."""
    world_size = (
        int(config["learner_gpus"])
        if str(config["device"]).startswith("cuda") else 1
    )
    if world_size <= 0:
        raise ValueError("learner_gpus must be positive")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")
    if int(config["batch_size"]) % world_size:
        raise ValueError("global batch_size must be divisible by learner_gpus")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive")
    if int(config["context_tokens"]) <= 0:
        raise ValueError("context_tokens must be positive")
    train_value = bool(config["train_critic"]) or bool(config["train_public_value"])
    if not train_value and float(config["public_value_coef"]) != 0.0:
        raise ValueError("actor-only SFT requires public_value_coef=0")
    if int(config.get("validation_samples_per_run", 0)) < 0:
        raise ValueError("validation_samples_per_run must be nonnegative")


def dataset_manifest_hash(dataset: Path) -> str:
    return hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()


def assert_ablation_cache_alignment(dataset: Path, reference: Path) -> None:
    """Require paired v11/v13 caches to contain the same exact sample stream."""
    manifests = []
    for path in (dataset, reference):
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"aligned ablation cache manifest is missing: {manifest_path}")
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    current, paired = manifests
    schemas = {
        int(current.get("token_schema_version", -1)),
        int(paired.get("token_schema_version", -1)),
    }
    if schemas != {11, TOKEN_SCHEMA_VERSION}:
        raise RuntimeError(
            f"aligned ablation requires one schema-11 and one schema-{TOKEN_SCHEMA_VERSION} cache"
        )
    if current.get("selection_manifest_sha256") != paired.get("selection_manifest_sha256"):
        raise RuntimeError("aligned ablation caches use different selected kyoku sequences")
    current_identity = current.get("sample_identity_contract")
    paired_identity = paired.get("sample_identity_contract")
    if not isinstance(current_identity, dict) or not isinstance(paired_identity, dict):
        raise RuntimeError(
            "aligned ablation caches lack the complete sample identity contract; re-encode both caches"
        )
    for split in ("train", "validation"):
        left = current_identity.get(split)
        right = paired_identity.get(split)
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise RuntimeError(f"aligned ablation caches lack {split} sample identities")
        required = (
            "samples", "sequence_sha256", "sharded_sequence_sha256",
            "supervision_sequence_sha256",
        )
        if any(left.get(field) != right.get(field) for field in required):
            raise RuntimeError(
                f"aligned ablation {split} identity, supervision, or chunk layout differs"
            )


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
    targets = torch.empty(batch, dtype=torch.float32)
    teachers = torch.zeros((batch, 241), dtype=torch.bool)
    for row, sample in enumerate(samples):
        factors[row, :sample.token_length] = torch.from_numpy(sample.token_factors)
        numeric[row, :sample.token_length] = torch.from_numpy(sample.token_numeric)
        lengths[row] = sample.token_length
        legal[row] = torch.from_numpy(sample.legal_mask)
        actions[row] = sample.action
        targets[row] = sample.value_target
        if sample.teacher_mask is not None:
            teachers[row] = torch.from_numpy(sample.teacher_mask)
    result = {
        "token_factors": factors.to(device, non_blocking=True),
        "token_numeric": numeric.to(device, non_blocking=True),
        "token_lengths": lengths.to(device, non_blocking=True),
        "legal_mask": legal.to(device, non_blocking=True),
        "actions": actions.to(device, non_blocking=True),
        "value_targets": targets.to(device, non_blocking=True),
        "teacher_masks": teachers.to(device, non_blocking=True),
    }
    if include_critic:
        max_critic = max(sample.critic_length for sample in samples)
        critic = torch.zeros((batch, max_critic, 10), dtype=torch.uint8)
        critic_lengths = torch.empty(batch, dtype=torch.long)
        for row, sample in enumerate(samples):
            if sample.critic_length:
                critic[row, :sample.critic_length] = torch.from_numpy(sample.critic_factors)
            critic_lengths[row] = sample.critic_length
        result.update({
            "critic_factors": critic.to(device, non_blocking=True),
            "critic_lengths": critic_lengths.to(device, non_blocking=True),
        })
    return result


def length_bucketed_batches(
    samples: Iterable[SftSample],
    batch_size: int,
    *,
    window_batches: int,
    rng: random.Random | None = None,
    align_across_schemas: bool = False,
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
        if not align_across_schemas:
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
        return "kan"
    if 171 <= action_id <= 204:
        return "kan"
    if 205 <= action_id <= 238:
        return "kan"
    if action_id == 239:
        return "hora"
    return "ryukyoku"


_GROUP_IDS = tuple(_action_group(action) for action in range(241))
_GROUP_NAMES = ("pass", "discard", "reach", "chi", "pon", "kan", "hora", "ryukyoku")
_GROUP_SLICES = (
    slice(0, 1), slice(1, 75), slice(75, 76), slice(76, 133),
    slice(133, 170), slice(170, 239), slice(239, 240), slice(240, 241),
)
_ACTION_GROUP_INDEX = torch.tensor(
    [_GROUP_NAMES.index(value) for value in _GROUP_IDS], dtype=torch.long,
)


def group_classification_loss(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """CE over logsumexp action groups, only where >=2 groups are legal."""
    grouped, available = grouped_action_logits(logits)
    targets = _ACTION_GROUP_INDEX.to(actions.device)[actions]
    eligible = available.sum(dim=1) >= 2
    if not bool(eligible.any()):
        return logits.new_zeros(())
    return F.cross_entropy(grouped[eligible], targets[eligible])


def grouped_action_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate fixed actions into the eight strategic action groups."""
    group_logits = []
    legal_groups = []
    for columns in _GROUP_SLICES:
        values = logits[:, columns]
        group_logits.append(torch.logsumexp(values, dim=1))
        legal_groups.append(torch.isfinite(values).any(dim=1))
    grouped = torch.stack(group_logits, dim=1)
    available = torch.stack(legal_groups, dim=1)
    return grouped, available


def rule_teacher_loss(logits: torch.Tensor, teacher_mask: torch.Tensor) -> torch.Tensor:
    """Uniform soft-target CE over tied best rule candidates."""
    counts = teacher_mask.sum(dim=1)
    eligible = (counts >= 1) & (torch.isfinite(logits).sum(dim=1) >= 2)
    if not bool(eligible.any()):
        return logits.new_zeros(())
    log_probs = F.log_softmax(logits[eligible], dim=1)
    targets = teacher_mask[eligible].float() / counts[eligible, None]
    return -(targets * log_probs.masked_fill(~teacher_mask[eligible], 0.0)).sum(dim=1).mean()


def _model_config(config: dict[str, Any]) -> ModelConfig:
    base = ModelConfig.preset(str(config["model_size"]))
    values = asdict(base)
    values["context_tokens"] = int(config["context_tokens"])
    values["critic_layers"] = int(config.get("critic_layers", base.critic_layers))
    values["policy_head_type"] = str(config.get("policy_head_type", "isolated_action_query"))
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
        "training_mode": (
            "joint_actor_critic" if bool(config["train_critic"])
            else ("actor_public_value" if bool(config.get("train_public_value", True)) else "actor_only")
        ),
        "token_schema_version": int(config.get("token_schema_version", TOKEN_SCHEMA_VERSION)),
        "feature_schema_sha256": config.get("feature_schema_sha256"),
        "rust_analysis_version": config.get("rust_analysis_version"),
        "decision_analysis_version": config.get("decision_analysis_version"),
        "legacy_encoder_sha256": config.get("legacy_encoder_sha256"),
        "policy_head_type": module.config.policy_head_type,
        "dataset_manifest_hash": manifest_hash,
        "epoch": epoch,
        "global_step": global_step,
        "rank_steps": rank_steps,
        "data_cursor": {
            "version": 1,
            "epoch": int(epoch),
            "rank_batches_consumed": [int(value) for value in rank_steps],
            "world_size": len(rank_steps),
        },
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
    train_value = train_critic or bool(config.get("train_public_value", True))
    batch_size = max(1, int(config["batch_size"]) // max(1, int(config["learner_gpus"])))
    samples = iter_split_samples(
        dataset,
        "validation",
        gamma=float(config["gamma"]),
        seed=int(config["seed"]),
        shuffle=False,
        include_critic=train_critic,
    )
    totals = {
        "samples": 0.0, "policy_ce_sum": 0.0, "value_huber_sum": 0.0,
        "value_abs_sum": 0.0, "value_sq_sum": 0.0, "top1": 0.0, "top3": 0.0,
        "group_top1": 0.0, "group_samples": 0.0,
        "reach_brier_sum": 0.0, "reach_opportunities": 0.0,
        "rule_samples": 0.0, "optimal_shanten": 0.0,
        "optimal_ukeire_samples": 0.0, "optimal_ukeire": 0.0,
        "call_pass_samples": 0.0, "call_pass_correct": 0.0,
    }
    group_correct: dict[str, list[int]] = {}
    maximum = (
        int(config.get("validation_max_samples", 0))
        if max_samples is None else int(max_samples)
    )
    if maximum:
        # Select the fixed validation identities before length bucketing.
        # Truncating after a sorted bucket preferentially retained the shortest
        # rows from the final, partially consumed window.
        samples = itertools.islice(samples, maximum)
    batches = length_bucketed_batches(
        samples,
        batch_size,
        window_batches=int(config["length_bucket_window_batches"]),
        align_across_schemas=bool(config.get("ablation_aligned_batches", False)),
    )
    use_bf16 = bool(
        str(config.get("inference_dtype", "bf16")).lower() == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    for rows in batches:
        batch = collate_samples(rows, device, include_critic=train_critic)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            if train_value:
                output = model(
                    batch["token_factors"], batch["token_numeric"], batch["legal_mask"],
                    batch["token_lengths"],
                    critic_factors=batch.get("critic_factors"),
                    critic_lengths=batch.get("critic_lengths"),
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
        if train_value:
            values = output["value"].float()
            value_targets = batch["value_targets"]
            huber = F.huber_loss(values, value_targets, reduction="none")
            error = values - value_targets
            totals["value_huber_sum"] += float(huber.sum())
            totals["value_abs_sum"] += float(error.abs().sum())
            totals["value_sq_sum"] += float(error.square().sum())
        totals["top1"] += float((top[:, 0] == targets).sum())
        totals["top3"] += float((top == targets[:, None]).any(-1).sum())
        if int(config.get("token_schema_version", TOKEN_SCHEMA_VERSION)) == TOKEN_SCHEMA_VERSION:
            for sample, predicted in zip(rows, top[:, 0].tolist(), strict=True):
                offense = (sample.token_factors[:, 0] == 7) & (sample.token_factors[:, 9] == 1)
                action_ids = sample.token_factors[offense, 2].astype(np.int64) - 1
                structural = sample.token_factors[offense, 4].astype(np.int64)
                ukeire = sample.token_numeric[offense, 3]
                comparable = structural > 0
                if comparable.any() and int(predicted) in action_ids[comparable]:
                    best_shanten = int(structural[comparable].min())
                    chosen = int(np.flatnonzero(action_ids == int(predicted))[0])
                    totals["rule_samples"] += 1
                    shanten_ok = int(structural[chosen]) == best_shanten
                    totals["optimal_shanten"] += float(shanten_ok)
                    if shanten_ok:
                        best_ukeire = float(ukeire[structural == best_shanten].max())
                        totals["optimal_ukeire_samples"] += 1
                        totals["optimal_ukeire"] += float(np.isclose(float(ukeire[chosen]), best_ukeire))
        grouped, available_groups = grouped_action_logits(logits)
        group_targets = _ACTION_GROUP_INDEX.to(targets.device)[targets]
        group_eligible = available_groups.sum(dim=1) >= 2
        totals["group_top1"] += float((grouped.argmax(dim=1)[group_eligible] == group_targets[group_eligible]).sum())
        totals["group_samples"] += float(group_eligible.sum())
        call_groups = torch.tensor([3, 4, 5], device=targets.device)
        call_or_pass = available_groups[:, 0] & available_groups[:, 3:6].any(dim=1)
        if bool(call_or_pass.any()):
            predicted_groups = grouped.argmax(dim=1)
            predicted_call = (predicted_groups[:, None] == call_groups).any(dim=1)
            target_call = (group_targets[:, None] == call_groups).any(dim=1)
            totals["call_pass_samples"] += float(call_or_pass.sum())
            totals["call_pass_correct"] += float((predicted_call[call_or_pass] == target_call[call_or_pass]).sum())
        reach_opportunity = batch["legal_mask"][:, 75]
        if bool(reach_opportunity.any()):
            reach_probability = logits.softmax(dim=1)[:, 75]
            reach_target = targets.eq(75).float()
            totals["reach_brier_sum"] += float(
                (reach_probability[reach_opportunity] - reach_target[reach_opportunity]).square().sum()
            )
            totals["reach_opportunities"] += float(reach_opportunity.sum())
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
        "validation/action_group_top1": totals["group_top1"] / max(totals["group_samples"], 1.0),
        "validation/reach_opportunity_brier": totals["reach_brier_sum"] / max(totals["reach_opportunities"], 1.0),
        "validation/reach_opportunities": totals["reach_opportunities"],
        "validation/optimal_shanten_rate": totals["optimal_shanten"] / max(totals["rule_samples"], 1.0),
        "validation/optimal_ukeire_rate": totals["optimal_ukeire"] / max(totals["optimal_ukeire_samples"], 1.0),
        "validation/optimal_ukeire_samples": totals["optimal_ukeire_samples"],
        "validation/rule_samples": totals["rule_samples"],
        "validation/call_pass_accuracy": totals["call_pass_correct"] / max(totals["call_pass_samples"], 1.0),
    }
    if train_value:
        result.update({
            "validation/value_huber": totals["value_huber_sum"] / count,
            "validation/value_mae": totals["value_abs_sum"] / count,
            "validation/value_rmse": math.sqrt(totals["value_sq_sum"] / count),
        })
    for group, (correct, top3_correct, group_count) in sorted(group_correct.items()):
        result[f"validation/top1_{group}"] = correct / max(group_count, 1)
        result[f"validation/top3_{group}"] = top3_correct / max(group_count, 1)
    result["validation/loss"] = result["validation/policy_ce"]
    if train_value:
        result["validation/loss"] += (
            float(config["public_value_coef"]) * result["validation/value_huber"]
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
    train_critic = bool(config["train_critic"])
    train_value = train_critic or bool(config.get("train_public_value", True))
    frozen_roots: set[str] = set()
    if not train_critic:
        frozen_roots.add("critic_embedding")
    if not train_value:
        frozen_roots.update({"critic_backbone", "value_head", "value_query"})
    if frozen_roots:
        for name, parameter in model.named_parameters():
            if name.split(".", 1)[0] in frozen_roots:
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
    raw_dataset = manifest.get("format") == "riichi-sft-kyoku-v1"
    if distributed and raw_dataset:
        raise RuntimeError(
            "distributed SFT requires a precomputed encoded dataset so rank sample counts can be balanced"
        )
    dataset_schema = int(manifest.get("token_schema_version", TOKEN_SCHEMA_VERSION if raw_dataset else -1))
    if dataset_schema not in {11, TOKEN_SCHEMA_VERSION}:
        raise RuntimeError(f"unsupported SFT dataset schema {dataset_schema}; schema 12 must be re-encoded")
    requested_schema = int(config.get("token_schema_version", dataset_schema))
    if requested_schema != dataset_schema:
        raise RuntimeError(
            f"SFT config requests schema {requested_schema}, but dataset is schema {dataset_schema}"
        )
    policy_head_type = str(config.get("policy_head_type", "isolated_action_query"))
    if dataset_schema == 11 and policy_head_type != "legacy_fixed":
        raise RuntimeError("schema-11 data is supported only by the legacy_fixed baseline")
    config["token_schema_version"] = dataset_schema
    config["feature_schema_sha256"] = manifest.get(
        "feature_schema_sha256", feature_schema_sha256() if raw_dataset else None,
    )
    config["rust_analysis_version"] = manifest.get(
        "rust_analysis_version", RUST_ANALYSIS_VERSION if raw_dataset else None,
    )
    config["decision_analysis_version"] = manifest.get(
        "decision_analysis_version", DECISION_ANALYSIS_VERSION if raw_dataset else None,
    )
    config["legacy_encoder_sha256"] = manifest.get(
        "legacy_encoder_sha256", legacy_encoder_sha256() if dataset_schema == 11 and raw_dataset else None,
    )
    train_decisions = int(manifest["counts"]["train_decisions"])
    if distributed:
        samples_per_rank, extra_samples = divmod(train_decisions, world_size)
        rank_sample_counts = [
            samples_per_rank + int(rank_index < extra_samples)
            for rank_index in range(world_size)
        ]
        configured_local_batch = int(config["batch_size"]) // world_size
        rank_batch_counts = [
            math.ceil(count / configured_local_batch) for count in rank_sample_counts
        ]
        if len(set(rank_batch_counts)) != 1:
            raise RuntimeError(
                "the final global batch cannot be split into a non-empty batch on every DDP rank; "
                "choose a batch_size/world_size combination with a larger final remainder"
            )
    estimated_steps = max(
        1,
        math.ceil(train_decisions / int(config["batch_size"])) * int(config["epochs"]),
    )
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
    manifest_hash = dataset_manifest_hash(dataset)
    start_epoch = 0
    global_step = 0
    skip_steps = 0
    best_validation_loss = float("inf")
    best_heuristic_point_delta = float("-inf")
    if config.get("resume"):
        payload = torch.load(config["resume"], map_location=device, weights_only=False)
        if int(payload.get("token_schema_version", 0)) != dataset_schema:
            raise RuntimeError("SFT resume checkpoint has an incompatible token schema")
        if payload.get("feature_schema_sha256") != config.get("feature_schema_sha256"):
            raise RuntimeError("SFT resume checkpoint has an incompatible feature schema hash")
        if payload.get("rust_analysis_version") != config.get("rust_analysis_version"):
            raise RuntimeError("SFT resume checkpoint has an incompatible Rust analysis version")
        if payload.get("decision_analysis_version") != config.get("decision_analysis_version"):
            raise RuntimeError("SFT resume checkpoint has an incompatible decision-analysis version")
        if payload.get("legacy_encoder_sha256") != config.get("legacy_encoder_sha256"):
            raise RuntimeError("SFT resume checkpoint has an incompatible legacy encoder hash")
        checkpoint_head = payload.get("policy_head_type", payload.get("model_config", {}).get("policy_head_type"))
        if checkpoint_head != model.config.policy_head_type:
            raise RuntimeError("SFT resume checkpoint has an incompatible policy head type")
        if payload.get("dataset_manifest_hash") != manifest_hash:
            raise RuntimeError("SFT resume checkpoint belongs to a different dataset manifest")
        expected_mode = (
            "joint_actor_critic" if train_critic
            else ("actor_public_value" if train_value else "actor_only")
        )
        if payload.get("training_mode", expected_mode) != expected_mode:
            raise RuntimeError("SFT resume checkpoint training mode differs from current config")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        cursor = payload.get("data_cursor")
        if cursor is not None:
            if int(cursor.get("version", -1)) != 1:
                raise RuntimeError("SFT resume checkpoint has an unsupported data cursor")
            if int(cursor.get("world_size", world_size)) != world_size:
                raise RuntimeError("SFT resume checkpoint data cursor uses a different world size")
            start_epoch = int(cursor.get("epoch", 0))
            rank_progress = cursor.get("rank_batches_consumed", [])
        else:
            # Backward compatibility for checkpoints written before the
            # versioned cursor was introduced.  Their rank_steps are valid for
            # the first resume from an uninterrupted run.
            start_epoch = int(payload.get("epoch", 0))
            rank_progress = payload.get("rank_steps", [])
        global_step = int(payload.get("global_step", 0))
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
    cursor_epoch = start_epoch
    cursor_steps_in_epoch = skip_steps
    stop_training = False
    with join_context():
        for epoch in range(start_epoch, int(config["epochs"])):
            steps_in_epoch = skip_steps if epoch == start_epoch else 0
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
                align_across_schemas=bool(config.get("ablation_aligned_batches", False)),
            )
            for batch_index, rows in enumerate(batches):
                if int(config.get("max_train_steps", 0)) > 0 and global_step >= int(config["max_train_steps"]):
                    stop_training = True
                    break
                if epoch == start_epoch and batch_index < skip_steps:
                    continue
                step_started = last_step_end
                batch = collate_samples(rows, device, include_critic=train_critic)
                query_extra = int(str(config.get("policy_head_type")) == "legacy_fixed")
                effective_tokens = sum(sample.token_length + query_extra for sample in rows)
                padded_tokens = len(rows) * (max(sample.token_length for sample in rows) + query_extra)
                local_samples += len(rows)
                local_effective_tokens += effective_tokens
                local_padded_tokens += padded_tokens
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    if train_value:
                        model_output = model(
                            batch["token_factors"], batch["token_numeric"], batch["legal_mask"],
                            batch["token_lengths"],
                            critic_factors=batch.get("critic_factors"),
                            critic_lengths=batch.get("critic_lengths"),
                        )
                    else:
                        model_output = model(
                            batch["token_factors"], batch["token_numeric"], batch["legal_mask"],
                            batch["token_lengths"], policy_only=True,
                        )
                    policy_loss = F.cross_entropy(model_output["policy_logits"].float(), batch["actions"])
                    group_loss = group_classification_loss(
                        model_output["policy_logits"].float(), batch["actions"],
                    )
                    rule_loss = rule_teacher_loss(
                        model_output["policy_logits"].float(), batch["teacher_masks"],
                    )
                    rule_progress = global_step / max(
                        estimated_steps * float(config["rule_decay_fraction"]), 1.0,
                    )
                    rule_weight = float(config["rule_coef"]) * max(0.0, 1.0 - rule_progress)
                    loss = policy_loss + float(config["group_coef"]) * group_loss + rule_weight * rule_loss
                    if train_value:
                        value_loss = F.huber_loss(
                            model_output["value"].float(), batch["value_targets"]
                        )
                        loss = loss + float(config["public_value_coef"]) * value_loss
                    else:
                        value_loss = None
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                step_finished = time.perf_counter()
                local_steps += 1
                steps_in_epoch += 1
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
                    progress = _rank_steps(world_size, steps_in_epoch)
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
            if stop_training:
                cursor_epoch = epoch
                cursor_steps_in_epoch = steps_in_epoch
                break
            # The next checkpoint starts at the following epoch rather than
            # replaying the just-completed one.  Reset the resume skip after the
            # first resumed epoch so it cannot leak into later epochs.
            cursor_epoch = epoch + 1
            cursor_steps_in_epoch = 0
            skip_steps = 0
    if distributed:
        dist.barrier()
    progress = _rank_steps(world_size, cursor_steps_in_epoch)
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
                    epoch=cursor_epoch,
                    global_step=global_step,
                    rank_steps=progress,
                    best_validation_loss=best_validation_loss,
                    best_heuristic_point_delta=best_heuristic_point_delta,
                    metrics=final_heuristic,
                )
        _save_checkpoint(
            output / "latest.pt", model, optimizer, scheduler,
            config=config, manifest_hash=manifest_hash, epoch=cursor_epoch,
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
                config=config, manifest_hash=manifest_hash, epoch=cursor_epoch,
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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "sft.yaml",
        help="SFT config (defaults to the repository's canonical sft.yaml)",
    )
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
    validate_config(config)
    output = resolve_output(config, args.output)
    config["checkpoint_dir"] = str(output)
    if config.get("resume"):
        resume_path = Path(str(config["resume"]))
        if not resume_path.is_file():
            raise FileNotFoundError(f"SFT resume checkpoint does not exist: {resume_path}")
    elif output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        raise FileExistsError(
            f"refusing to overwrite non-empty fresh-training output: {output}; "
            "choose a new --output directory"
        )
    if bool(config.get("ablation_aligned_batches", False)):
        reference_value = config.get("ablation_identity_reference_dataset")
        if not reference_value:
            raise RuntimeError(
                "ablation_aligned_batches requires ablation_identity_reference_dataset"
            )
        assert_ablation_cache_alignment(args.dataset, Path(str(reference_value)))
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
            args=(world_size, config, args.dataset, output),
            nprocs=world_size,
            join=True,
        )
    else:
        train_worker(0, 1, config, args.dataset, output)


if __name__ == "__main__":
    main()
