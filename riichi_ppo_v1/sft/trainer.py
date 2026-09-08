"""V19 actor-only SFT（Actor BC + 信念五头监督联合）从零训练入口。

V19 输入为当前局面状态快照（Shared 公共前缀 + Opponent Analysis + 信念 token + 每动作
Offense/Defense Query），网络为 current_state_snapshot 策略头 + 信念网络。节奏键
(每 3000 steps 验证/保存、最终 96 半庄)只引用 ``sft/contract.py`` 的机制常量,
禁止实验配置复制。
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import socket
import time
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from ..model import KyokuTransformerActorCritic, ModelConfig
from ..training.belief import (
    NORM_BASELINE_FLOOR,
    flatten_grads_cosine,
    is_belief_private_parameter,
)
from ..model.belief_network import (
    LOSS_BUCKET_CLASSES,
    LOSS_NORM_MAX,
    loss_bucket_centers,
    loss_bucket_targets,
)
from ..model.checkpoint import strip_compile_prefix
from ..model.schema import NUM_ACTIONS
from .actor_bc import actor_parameters, freeze_critic
from .checkpoint import checkpoint_payload, load_exact_resume
from .contract import (
    BELIEF_LABEL_SHAPES,
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
    "model_size": "v19",
    "context_tokens": 320,
    "policy_head_type": "current_state_snapshot",
    "dense_slot_dim": 32,
    "dense_fusion_dim": 512,
    "epochs": 1,
    "train_critic": False,
    "train_public_value": False,
    "batch_size": 4096,
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
    "log_interval_steps": 10,
    "validation_max_samples": 0,
    "validation_samples_per_run": 150000,
    "max_train_steps": 0,
    "stop_after_steps": 0,
    "tensorboard_enabled": True,
    "tensorboard_dirname": "tensorboard",
    "resume": None,
    "init_model": None,
    # 训练前向跳过 GPU 侧重复结构校验（输入由 Rust 编码器 fail-closed 生成 +
    # SFT 契约校验覆盖），与 torch_compile 配合稳定编译。
    "validate_structure": False,
    # V19 信念监督联合（训练分册 §4.4/§6）：唯一现行配置为 v19_sft.yaml。
    # 2026-09-07 模糊化：五头损失按标签熵/基线归一化，λ_k 默认 1.0 即均衡。
    "belief_sft_coef": 1.0,
    "belief_head_weight_hand": 1.0,
    "belief_head_weight_wait": 1.0,
    "belief_head_weight_danger": 1.0,
    "belief_head_weight_loss_bucket": 1.0,
    # 信念共享层梯度耦合（与 critic 同构，实施方案 §5.1）。
    # 2026-09-08 用户决策置 0：信念监督不再更新公共权重，只更新私有的
    # 信念网络（backbone/query/头），公共主干只由 BC 梯度塑造（残差式隔离）。
    "belief_public_grad_scale": 0.0,
    # 逐动作信念读出：SFT 阶段开启并 detach 特征（信念头只由标签校准）。
    "belief_readout_enabled": True,
    "belief_readout_detach": True,
    # P1 诊断：反事实「开闸②」梯度余弦的触发节奏（rank 0、每 N 步一次；
    # 0 关闭）。诊断为独立 eager 前向，不影响训练梯度。
    "grad_cosine_interval_steps": 500,
    # 条件/加权损失（实施方案 §4.1，模糊化后适配）。
    "belief_danger_pos_weight": 5.0,
    "belief_loss_positive_weight": 20.0,
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
            raise ValueError("V19 SFT config must be a mapping")
        config.update(overlay)
    return config


def validate_config(config: dict[str, Any]) -> None:
    duplicated = set(_CADENCE_KEYS) & set(config)
    if duplicated:
        raise ValueError(
            "V19 SFT cadence keys must stay single-sourced in sft/contract.py: "
            + ", ".join(sorted(duplicated))
        )
    if str(config.get("policy_head_type")) != "current_state_snapshot":
        raise ValueError("V19 SFT requires policy_head_type=current_state_snapshot")
    if str(config.get("model_size")) != "v19":
        raise ValueError("V19 SFT requires model_size=v19")
    if bool(config.get("train_critic", False)) or bool(config.get("train_public_value", False)):
        raise ValueError("V19 SFT is actor-only; critic training is not supported here")
    dataset = Path(str(config["dataset"]))
    if not (dataset / "manifest.json").is_file():
        raise FileNotFoundError(f"V19 SFT dataset manifest does not exist: {dataset}")
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
    belief_hand_np = np.zeros((batch, *BELIEF_LABEL_SHAPES["hand"]), dtype=np.uint8)
    belief_shanten_np = np.zeros((batch, *BELIEF_LABEL_SHAPES["shanten"]), dtype=np.uint8)
    belief_wait_np = np.zeros((batch, *BELIEF_LABEL_SHAPES["wait"]), dtype=np.uint8)
    belief_danger_np = np.zeros((batch, *BELIEF_LABEL_SHAPES["danger"]), dtype=np.uint8)
    belief_loss_np = np.zeros((batch, *BELIEF_LABEL_SHAPES["loss"]), dtype=np.float32)
    for row, sample in enumerate(samples):
        actor_factors_np[row, : sample.token_length] = sample.actor_factors
        actor_numeric_np[row, : sample.token_length] = sample.actor_numeric
        actor_lengths_np[row] = sample.token_length
        query_rows_np[row, : sample.query_rows.shape[0]] = sample.query_rows
        action_ids_np[row, : sample.query_pair_count] = sample.action_ids
        pair_counts_np[row] = sample.query_pair_count
        legal_np[row] = sample.legal_mask
        actions_np[row] = sample.action
        for name, array, target in (
            ("belief_hand", sample.belief_hand, belief_hand_np[row]),
            ("belief_shanten", sample.belief_shanten, belief_shanten_np[row]),
            ("belief_wait", sample.belief_wait, belief_wait_np[row]),
            ("belief_danger", sample.belief_danger, belief_danger_np[row]),
            ("belief_loss", sample.belief_loss, belief_loss_np[row]),
        ):
            values = np.asarray(array)
            if values.shape != target.shape:
                raise ValueError(
                    f"{name} shape {values.shape} != {target.shape} in SFT sample"
                )
            target[:] = values
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
    # host 侧容量/类别行表预计算：透传给 forward 消除 GPU max().item() 与
    # embedding argsort/tolist 同步，同时稳定 torch.compile（实施方案 §4.3）。
    from ..model.dense_embedding import compute_kind_row_plan
    from ..model.encoding_protocol import SEGMENT_SHARED

    shared_per_row = (actor_factors_np[..., 0] == SEGMENT_SHARED).sum(axis=-1)
    shared_capacity = int(shared_per_row.max(initial=0))
    kind_row_plan = compute_kind_row_plan(actor_factors_np)
    actor_factors = torch.from_numpy(actor_factors_np).to(device, non_blocking=True)
    actor_numeric = torch.from_numpy(actor_numeric_np).to(device, non_blocking=True)
    actor_lengths = torch.from_numpy(actor_lengths_np).to(device, non_blocking=True)
    query_rows = torch.from_numpy(query_rows_np).to(device, non_blocking=True)
    action_ids = torch.from_numpy(action_ids_np).to(device, non_blocking=True)
    pair_counts = torch.from_numpy(pair_counts_np).to(device, non_blocking=True)
    legal = torch.from_numpy(legal_np).to(device, non_blocking=True)
    actions = torch.from_numpy(actions_np).to(device, non_blocking=True)
    belief_hand = torch.from_numpy(belief_hand_np).to(device, non_blocking=True)
    belief_shanten = torch.from_numpy(belief_shanten_np).to(device, non_blocking=True)
    belief_wait = torch.from_numpy(belief_wait_np).to(device, non_blocking=True)
    belief_danger = torch.from_numpy(belief_danger_np).to(device, non_blocking=True)
    belief_loss = torch.from_numpy(belief_loss_np).to(device, non_blocking=True)
    return {
        "actor_factors": actor_factors,
        "actor_numeric": actor_numeric,
        "actor_lengths": actor_lengths,
        "query_rows": query_rows,
        "action_ids": action_ids,
        "pair_counts": pair_counts,
        "legal_mask": legal,
        "actions": actions,
        "belief_hand": belief_hand,
        "belief_shanten": belief_shanten,
        "belief_wait": belief_wait,
        "belief_danger": belief_danger,
        "belief_loss": belief_loss,
        # host 侧非张量字段：_forward_actor 取出后传给模型 forward。
        "shared_capacity": shared_capacity,
        "kind_row_plan": kind_row_plan,
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


def _forward_actor(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    belief_summary_detach: bool = True,
    belief_readout_detach_override: bool | None = None,
) -> dict[str, torch.Tensor]:
    # 统一走 __call__/forward 分发,使 DistributedDataParallel 也能正确触发
    # 梯度同步;统一走 forward 分发。host 预计算的容量/行表在 collate 时已
    # 放入 batch dict，此处取出透传（避免 GPU 同步；torch.compile 稳定）。
    # 两个覆写关键字仅供反事实梯度诊断使用（默认保持训练契约不变）。
    return model(
        actor_factors=batch["actor_factors"],
        actor_numeric=batch["actor_numeric"],
        actor_lengths=batch["actor_lengths"],
        query_action_ids=batch["action_ids"],
        query_pair_counts=batch["pair_counts"],
        legal_mask=batch["legal_mask"],
        policy_only=True,
        validate_structure=bool(config.get("validate_structure", False)),
        belief_public_grad_scale=float(config.get("belief_public_grad_scale", 0.25)),
        belief_readout_enabled=bool(config.get("belief_readout_enabled", True)),
        belief_readout_detach=(
            bool(config.get("belief_readout_detach", True))
            if belief_readout_detach_override is None
            else belief_readout_detach_override
        ),
        belief_summary_detach=belief_summary_detach,
        shared_capacity=batch.get("shared_capacity"),
        kind_row_plan=batch.get("kind_row_plan"),
    )


def _unwrap_diagnostic_model(model: nn.Module) -> nn.Module:
    """解 DDP/compile 包装取裸模块（诊断前向用 eager,不进 compile 缓存）。"""
    module = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(module, "_orig_mod", module)


def _belief_gate_cosine_scalars(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float]:
    """P1 诊断:反事实「开闸②」的 BC/信念监督梯度方向一致性(纯测量)。

    完整梯度隔离下,BC 损失与信念监督在信念私有参数上没有共同图路径,
    LIVE 图互梯度恒为 None,余弦无定义。本函数做一次 eager 反事实前向
    (解除 token/readout 路径 detach),测「若开闸②」BC 梯度与信念监督梯度
    在信念私有参数(belief_query/backbone/四头,不含 token_matrix)上的余弦:
    持续负相关 → 开闸②必然互相干扰;接近正相关 → 开闸②「安全」(但不
    保证有益)。不触碰 ``param.grad``,由调用方 try/except 降级。
    """
    raw = _unwrap_diagnostic_model(model)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        output = _forward_actor(
            raw, batch, config,
            belief_summary_detach=False, belief_readout_detach_override=False,
        )
        policy_ce = F.cross_entropy(output["policy_logits"].float(), batch["actions"])
        belief_total = _belief_losses(output, batch, config)["belief_loss_total"]
    belief_parameters = [
        parameter for name, parameter in raw.named_parameters()
        if is_belief_private_parameter(name)
    ]
    bc_grads = torch.autograd.grad(
        policy_ce, belief_parameters, allow_unused=True, retain_graph=True,
    )
    belief_grads = torch.autograd.grad(
        belief_total, belief_parameters, allow_unused=True, retain_graph=True,
    )
    measured = flatten_grads_cosine(bc_grads, belief_grads)
    scalars: dict[str, float] = {}
    if measured is not None:
        cosine, bc_norm, belief_norm = measured
        scalars["train/grad_cos_bc_belief_private"] = cosine
        scalars["train/grad_cos_bc_norm"] = bc_norm
        scalars["train/grad_cos_belief_norm"] = belief_norm
    return scalars


def _assert_targets_legal(actions: torch.Tensor, legal_mask: torch.Tensor) -> None:
    """BC 损失前重验目标动作 ∈ legal_mask（fail closed，拒绝损坏样本）。"""
    if not torch.all(legal_mask.gather(1, actions.view(-1, 1))):
        raise RuntimeError("BC target action is not present in legal_mask (corrupt sample)")


_BELIEF_OUTPUT_KEYS = (
    "belief_hand_logits",
    "belief_wait_logits",
    "belief_danger_logits",
    "belief_loss_bucket_logits",
)


def _require_belief_outputs(output: dict[str, torch.Tensor]) -> None:
    missing = sorted(set(_BELIEF_OUTPUT_KEYS) - set(output))
    if missing:
        raise RuntimeError(
            "model forward did not emit belief outputs (missing: "
            + ", ".join(missing) + "); V19 SFT requires the belief network"
        )


def _label_entropy(labels: Tensor, classes: int) -> Tensor:
    """标签分布的香农熵，用作 CE 头损失的归一化基线。"""
    counts = torch.bincount(labels.reshape(-1).long(), minlength=classes)
    p = counts.float().clamp_min(1e-8)
    p = p / p.sum()
    return -(p * p.log()).sum()


def _weighted_bce_constant_baseline(positive_fraction: Tensor, pos_weight: float) -> Tensor:
    """Danger 正例加权 BCE 的最优常数预测基线。"""
    p = positive_fraction.float().clamp(1e-8, 1.0 - 1e-8)
    w = float(pos_weight)
    s = (w * p) / (w * p + (1.0 - p))
    baseline = -(w * p * s.log() + (1.0 - p) * (1.0 - s).log())
    return baseline.clamp_min(1e-6)


def _belief_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """V19 模糊化信念联合损失：四头监督（均为逐头均值后的归一化损失）。

    2026-09-08 设计（用户决策）：Hand = 16 组计数桶 CE、Wait = 听牌+宽度桶
    CE（类别 0 = 非听，即是否听牌的精确信号）、Danger 正例加权 BCE、
    Loss = 铳点损失**分桶分类 CE**（边界 1000/5000/9000/13000/17000，共
    6 类，仅在危险正例子集上加权 (1 + pos_weight·I(raw>0))——精确预测
    铳点压力过大，分桶后 4000 与 6000 同档，损失贡献不放大细粒度差异）。
    每头原始损失除以**标签分布基线**（熵 / 最优常数 BCE，一律过下限
    ``NORM_BASELINE_FLOOR``；loss_bucket 基线在其损失实际所在的危险正例
    子集上计算），因此 λ_k=1.0 时四头贡献同量级；`belief_head_weight_*`
    仍可进一步微调。
    """
    _require_belief_outputs(output)
    hand_logits = output["belief_hand_logits"].float()
    wait_logits = output["belief_wait_logits"].float()
    danger_logits = output["belief_danger_logits"].float()
    bucket_logits = output["belief_loss_bucket_logits"].float()
    hand_labels = batch["belief_hand"].long().view(-1, 3, 16)
    wait_labels = batch["belief_wait"].long().view(-1, 3)
    danger_labels = batch["belief_danger"].float().view(-1, 3, 34)
    loss_raw = batch["belief_loss"].float().view(-1, 3, 34)
    bucket_labels = loss_bucket_targets(loss_raw)

    hand_loss = F.cross_entropy(
        hand_logits.reshape(-1, 3), hand_labels.reshape(-1),
    )
    wait_loss = F.cross_entropy(
        wait_logits.reshape(-1, 5), wait_labels.reshape(-1),
    )

    # danger：正例加权 BCE（pos_weight=5.0）。
    danger_pos_weight = float(config.get("belief_danger_pos_weight", 5.0))
    danger_loss = F.binary_cross_entropy_with_logits(
        danger_logits,
        danger_labels,
        pos_weight=torch.full_like(danger_labels, danger_pos_weight),
    )

    # loss：铳点分桶 CE，仅在危险正例子集上加权 (1 + w·I(raw>0))。
    positive_weight = float(config.get("belief_loss_positive_weight", 20.0))
    loss_ce_none = (
        F.cross_entropy(
            bucket_logits.reshape(-1, LOSS_BUCKET_CLASSES),
            bucket_labels.reshape(-1),
            reduction="none",
        ).view(danger_labels.shape)
        * (1.0 + positive_weight * (loss_raw > 0.0).float())
    )
    pos_mask = danger_labels > 0.0
    if bool(pos_mask.any()):
        loss_bucket = (loss_ce_none * pos_mask).sum() / pos_mask.sum()
    else:
        loss_bucket = loss_ce_none.mean()

    # 每头标签分布基线（保证 λ=1 时贡献均衡）。
    # 归一化基线一律过下限（见 training/belief.NORM_BASELINE_FLOOR 注释）。
    hand_scale = _label_entropy(hand_labels, 3).clamp_min(NORM_BASELINE_FLOOR)
    wait_scale = _label_entropy(wait_labels, 5).clamp_min(NORM_BASELINE_FLOOR)
    danger_scale = _weighted_bce_constant_baseline(
        danger_labels.float().mean(), danger_pos_weight,
    ).clamp_min(NORM_BASELINE_FLOOR)
    # loss_bucket 损失只在危险正例子集上计算,基线必须测同一分布(全量标签
    # 几乎全为桶 0,熵≈0);空正例批回退全量熵。
    if bool(pos_mask.any()):
        loss_scale = _label_entropy(bucket_labels[pos_mask], LOSS_BUCKET_CLASSES)
    else:
        loss_scale = _label_entropy(bucket_labels, LOSS_BUCKET_CLASSES)
    loss_scale = loss_scale.clamp_min(NORM_BASELINE_FLOOR)

    normalized = {
        "hand": hand_loss / hand_scale,
        "wait": wait_loss / wait_scale,
        "danger": danger_loss / danger_scale,
        "loss_bucket": loss_bucket / loss_scale,
    }
    weights = {
        "hand": float(config.get("belief_head_weight_hand", 1.0)),
        "wait": float(config.get("belief_head_weight_wait", 1.0)),
        "danger": float(config.get("belief_head_weight_danger", 1.0)),
        "loss_bucket": float(config.get("belief_head_weight_loss_bucket", 1.0)),
    }
    weighted = sum(weights[key] * normalized[key] for key in weights)
    coef = float(config.get("belief_sft_coef", 1.0))
    return {
        "belief_hand_loss": hand_loss,
        "belief_wait_loss": wait_loss,
        "belief_danger_loss": danger_loss,
        "belief_loss_bucket_loss": loss_bucket,
        "belief_hand_loss_norm": normalized["hand"],
        "belief_wait_loss_norm": normalized["wait"],
        "belief_danger_loss_norm": normalized["danger"],
        "belief_loss_bucket_loss_norm": normalized["loss_bucket"],
        "belief_loss_weighted": weighted,
        "belief_loss_total": coef * weighted,
    }


def _binary_auc(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """手动 rank-based AUC（含平局平均秩；单类别时返回 0.5）。

    只在 SFT 验证 cadence 调用（性能红线：训练步内禁止 CPU AUC）。
    """
    probs = probabilities.detach().float().reshape(-1).cpu().numpy()
    truth = labels.detach().float().reshape(-1).cpu().numpy()
    total = int(truth.shape[0])
    positive = int(truth.sum())
    negative = total - positive
    if total == 0 or positive == 0 or negative == 0:
        return probabilities.new_tensor(0.5)
    order = np.argsort(probs, kind="mergesort")
    sorted_probs = probs[order]
    ranks = np.empty(total, dtype=np.float64)
    start = 0
    while start < total:
        end = start + 1
        while end < total and sorted_probs[end] == sorted_probs[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_ranks = float(ranks[truth == 1].sum())
    auc = (positive_ranks - positive * (positive + 1) / 2.0) / (
        positive * negative
    )
    return probabilities.new_tensor(float(auc))


def _belief_metrics(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    include_auc: bool = False,
) -> dict[str, torch.Tensor]:
    """SFT 模糊化信念指标。

    训练步只返回 GPU 纯 torch 指标（acc/topk/mean/recall），绝不调用
    ``_binary_auc``；``include_auc=True`` 时（仅验证 cadence）追加 Danger AUC。
    Wait 头已改为 5 类宽度桶：指标为听牌二判精度与宽度桶 top-1/MAE。
    """
    _require_belief_outputs(output)
    hand_logits = output["belief_hand_logits"].float()
    hand_labels = batch["belief_hand"].long().view(-1, 3, 16)
    hand_acc = (hand_logits.argmax(-1) == hand_labels).float().mean()

    wait_logits = output["belief_wait_logits"].float()
    wait_labels = batch["belief_wait"].long().view(-1, 3)
    wait_top1 = (wait_logits.argmax(-1) == wait_labels).float().mean()
    # 桶 0 = 非听；听牌二判精度 = 预测桶 0 与否与标签一致。
    wait_tenpai_acc = (
        (wait_logits.argmax(-1) == 0).float() == (wait_labels == 0).float()
    ).float().mean()
    wait_probs = torch.softmax(wait_logits, dim=-1)
    wait_index = torch.arange(5, device=wait_probs.device, dtype=wait_probs.dtype)
    wait_width_expected = (wait_probs * wait_index).sum(dim=-1)
    wait_width_mae = (wait_width_expected - wait_labels.float()).abs().mean()

    danger_logits = output["belief_danger_logits"].float()
    danger_labels = batch["belief_danger"].float().view(-1, 3, 34)
    danger_probs = torch.sigmoid(danger_logits)
    sorted_idx = danger_probs.topk(34, dim=-1).indices
    sorted_correct = danger_labels.gather(-1, sorted_idx)
    k_true = danger_labels.sum(-1, keepdim=True)
    positions = torch.arange(
        34, device=danger_probs.device, dtype=danger_probs.dtype,
    ).view(1, 1, -1)
    danger_recall_at_topk = (
        (sorted_correct * (positions < k_true)).sum(-1, keepdim=True)
        / k_true.clamp_min(1)
    ).float().mean()

    # 铳点分桶：逐格 top-1 命中率 + 桶期望（可解释粗粒度 MAE，归一化单位）。
    bucket_logits = output["belief_loss_bucket_logits"].float()
    loss_raw = batch["belief_loss"].float().view(-1, 3, 34)
    bucket_labels = loss_bucket_targets(loss_raw)
    loss_bucket_accuracy = (
        (bucket_logits.argmax(-1) == bucket_labels).float().mean()
    )
    loss_prob = torch.softmax(bucket_logits, dim=-1)
    centers = loss_bucket_centers(bucket_logits.device).view(1, 1, 1, LOSS_BUCKET_CLASSES)
    loss_expected = (loss_prob * centers).sum(dim=-1)
    loss_target = (
        torch.clamp(loss_raw, max=LOSS_NORM_MAX) / LOSS_NORM_MAX
    )
    loss_expected_mae = (loss_expected - loss_target).abs().mean()

    result = {
        "belief_hand_acc": hand_acc,
        "belief_wait_top1": wait_top1,
        "belief_wait_tenpai_acc": wait_tenpai_acc,
        "belief_wait_width_mae": wait_width_mae,
        "belief_danger_recall_at_topk": danger_recall_at_topk,
        "belief_loss_bucket_accuracy": loss_bucket_accuracy,
        "belief_loss_expected_mae": loss_expected_mae,
    }
    if include_auc:
        result["belief_danger_auc"] = _binary_auc(danger_probs, danger_labels)
    return result


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
    belief_sums: dict[str, float] = {}
    for rows in batches:
        batch = collate_samples(
            rows, device, validate_semantics=bool(config.get("validate_semantics", True)),
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            output = _forward_actor(model, batch, config)
        logits = output["policy_logits"].float()
        targets = batch["actions"]
        _assert_targets_legal(targets, batch["legal_mask"])
        ce_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
        top = logits.topk(3, dim=-1).indices
        top1 += int((top[:, 0] == targets).sum())
        top3 += int((top == targets[:, None]).any(-1).sum())
        belief_metrics = _belief_metrics(output, batch, include_auc=True)
        # 验证 cadence 追加 per-head loss（含条件/加权式）。
        belief_metrics.update({
            key: value for key, value in _belief_losses(output, batch, config).items()
            if key.startswith("belief_")
        })
        for name, value in belief_metrics.items():
            belief_sums[name] = belief_sums.get(name, 0.0) + float(value) * len(rows)
        total += len(rows)
    count = max(total, 1)
    result = {
        "validation/samples": float(total),
        "validation/policy_ce": ce_sum / count,
        "validation/top1": top1 / count,
        "validation/top3": top3 / count,
        "validation/loss": ce_sum / count,
    }
    for name, accumulated in belief_sums.items():
        result[f"validation/{name}"] = accumulated / count
    return result


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
        model.load_state_dict(strip_compile_prefix(payload), strict=True)
        del initialized
    freeze_critic(model)
    # 可选的 torch.compile 快速路径：在 DDP 包装前编译原始模块
    # （配置 torch_compile: true 开启；首次迭代编译较慢，需重启训练生效）。
    if bool(config.get("torch_compile", False)):
        model = torch.compile(model)
    optimized = list(actor_parameters(model))
    if not optimized:
        raise RuntimeError("V19 SFT configuration leaves no trainable parameters")
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
        # resume 时 model 已被 compile 包装:加载到解包后的模块,键不带前缀;
        # strip 兼容旧版带 _orig_mod. 前缀的 checkpoint。
        getattr(model, "_orig_mod", model).load_state_dict(
            strip_compile_prefix(payload["model"]),
        )
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
    # P1 诊断降级标记:反事实余弦失败后本进程内禁用,绝不中断训练。
    gate_cosine_failed = False
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
                    model_output = _forward_actor(model, batch, config)
                    _assert_targets_legal(batch["actions"], batch["legal_mask"])
                    policy_ce = F.cross_entropy(model_output["policy_logits"].float(), batch["actions"])
                    belief_parts = _belief_losses(model_output, batch, config)
                    loss = policy_ce + belief_parts["belief_loss_total"]
                    policy_ce = policy_ce.detach()
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
                    belief_metrics = _belief_metrics(model_output, batch)
                    # 训练步 per-head loss 日志（GPU 标量，与指标同批累计）。
                    belief_metrics.update({
                        key: value for key, value in belief_parts.items()
                        if key.startswith("belief_")
                    })
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
                        belief_metrics=belief_metrics,
                    )
                if global_step % int(config["log_interval_steps"]) == 0 and rank == 0:
                    print(f"epoch={epoch} step={global_step} loss={float(loss):.4f}", flush=True)
                if global_step % SFT_CADENCE_STEPS == 0 and rank == 0 and metric_window is not None:
                    metrics = metric_window.scalars()
                    # P1 监控:信念接口使用度——token_matrix 权重范数(零初始化
                    # 起步,增长曲线直接回答「策略用了多少信念」)。
                    diagnostic_model = _unwrap_diagnostic_model(model)
                    metrics["train/belief_token_matrix_weight_norm"] = float(
                        diagnostic_model.belief_network.token_matrix.weight
                        .detach().float().norm()
                    )
                    metrics["train/belief_token_matrix_bias_norm"] = float(
                        diagnostic_model.belief_network.token_matrix.bias
                        .detach().float().norm()
                    )
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
                # P1 诊断:反事实开闸②梯度余弦(独立节奏,rank 0;step 后
                # 快照当前权重;不触碰训练梯度;失败一次性禁用)。
                gate_interval = int(config.get("grad_cosine_interval_steps", 0))
                if (
                    gate_interval > 0
                    and rank == 0
                    and global_step % gate_interval == 0
                    and not gate_cosine_failed
                ):
                    try:
                        gate_scalars = _belief_gate_cosine_scalars(
                            model, batch, config, device=device, use_bf16=use_bf16,
                        )
                        if writer is not None:
                            write_sft_scalars(writer, gate_scalars, global_step)
                            writer.flush()
                    except Exception as exc:  # noqa: BLE001 - 诊断降级
                        gate_cosine_failed = True
                        print(
                            f"gate-cosine diagnostics failed and disabled: {exc!r}",
                            flush=True,
                        )
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
    print(f"rank={rank} V19 SFT finished steps={global_step} elapsed={elapsed:.1f}s", flush=True)


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
