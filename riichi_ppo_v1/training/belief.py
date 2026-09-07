"""V19 信念五头监督的共享损失与指标（PPO learner 与 1v3 评测共用）。

2026-09-07 模糊化：Hand = 花色×段位 16 组 × {0,1,≥2} 桶（CE）、
Wait = 听牌 + 待牌宽度桶 5 类（CE）、Shanten/Danger 不变、Loss 仅在危险
正例子集上加权 Huber。每头原始损失除以标签分布基线（熵/最优常数 BCE/
正例中位偏差），因此 λ_k=1.0 时五头贡献天然同量级。

与 ``sft/trainer.py`` 内的信念训练函数同构（同一套标签定义），但本模块
面向 PPO 的逐批/逐样本聚合与评测面，并提供无 CPU 同步的纯 torch AUC
近似（训练更新不引入逐 minibatch GPU→CPU 往返）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

BELIEF_OUTPUT_KEYS = (
    "belief_hand_logits",
    "belief_shanten_logits",
    "belief_wait_logits",
    "belief_danger_logits",
    "belief_loss_pred",
)

HAND_CLASSES = 3
SHANTEN_CLASSES = 9
WAIT_CLASSES = 5
DANGER_CLASSES = 34


def require_belief_outputs(output: dict[str, torch.Tensor]) -> None:
    """模型前向必须包含信念五头输出（V19 拓扑 fail closed）。"""
    missing = sorted(set(BELIEF_OUTPUT_KEYS) - set(output))
    if missing:
        raise RuntimeError(
            "model forward did not emit belief outputs (missing: "
            + ", ".join(missing) + "); V19 training requires the belief network"
        )


def loss_target_norm(raw_loss: torch.Tensor) -> torch.Tensor:
    """Loss 目标归一化：min(raw, 24000) / 24000，与 sigmoid 预测同尺度。"""
    return torch.clamp(raw_loss.float(), max=24000.0) / 24000.0


def binary_auc(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """纯 torch rank-based AUC（含平局平均秩；单类别时返回 0.5）。

    每次返回标量；训练/评测小批内调用一次，避免逐元素 CPU 同步。
    """
    probs = probabilities.detach().float().reshape(-1)
    truth = labels.detach().float().reshape(-1)
    total = int(truth.numel())
    positive = int(truth.sum().item())
    negative = total - positive
    if total == 0 or positive == 0 or negative == 0:
        return probs.new_tensor(0.5)
    order = torch.argsort(probs, stable=True)
    sorted_probs = probs[order]
    left = torch.searchsorted(sorted_probs, sorted_probs, right=False)
    right = torch.searchsorted(sorted_probs, sorted_probs, right=True)
    average_ranks = (left.float() + right.float() + 1.0) / 2.0
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = average_ranks
    positive_ranks = ranks[truth == 1.0].sum()
    return (positive_ranks - float(positive) * (positive + 1) / 2.0) / (
        float(positive) * float(negative)
    )


def _label_entropy(labels: Tensor, classes: int) -> Tensor:
    counts = torch.bincount(labels.reshape(-1).long(), minlength=classes)
    p = counts.float().clamp_min(1e-8)
    p = p / p.sum()
    return -(p * p.log()).sum()


def _weighted_bce_constant_baseline(positive_fraction: Tensor, pos_weight: float) -> Tensor:
    p = positive_fraction.float().clamp(1e-8, 1.0 - 1e-8)
    w = float(pos_weight)
    s = (w * p) / (w * p + (1.0 - p))
    return (-(w * p * s.log() + (1.0 - p) * (1.0 - s).log())).clamp_min(1e-6)


def _loss_positive_baseline(target: Tensor, mask: Tensor) -> Tensor:
    positives = target[mask]
    if positives.numel() == 0:
        return target.new_tensor(1.0)
    median = positives.median()
    return (positives - median).abs().mean().clamp_min(1e-4)


def _belief_loss_components(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    danger_pos_weight: float = 5.0,
    loss_positive_weight: float = 20.0,
) -> dict[str, torch.Tensor]:
    """模糊化五头监督的基础损失项（原始 + 归一化；不含 λ_k 加权）。"""
    hand_labels = batch["belief_hand"].long().reshape(*output["belief_hand_logits"].shape[:-1])
    shanten_labels = batch["belief_shanten"].long()
    wait_labels = batch["belief_wait"].long().reshape(output["belief_wait_logits"].shape[0], 3)
    danger_labels = batch["belief_danger"].float().reshape(*output["belief_danger_logits"].shape)
    target = loss_target_norm(batch["belief_loss"]).reshape(*output["belief_loss_pred"].shape)

    hand_loss = F.cross_entropy(
        output["belief_hand_logits"].float().reshape(-1, HAND_CLASSES),
        hand_labels.reshape(-1),
    )
    shanten_loss = F.cross_entropy(
        output["belief_shanten_logits"].float().reshape(-1, SHANTEN_CLASSES),
        shanten_labels.reshape(-1),
    )
    wait_loss = F.cross_entropy(
        output["belief_wait_logits"].float().reshape(-1, WAIT_CLASSES),
        wait_labels.reshape(-1),
    )

    danger_loss = F.binary_cross_entropy_with_logits(
        output["belief_danger_logits"].float(),
        danger_labels,
        pos_weight=torch.full_like(danger_labels, float(danger_pos_weight)),
    )

    loss_huber_none = (
        F.huber_loss(output["belief_loss_pred"].float(), target, reduction="none")
        * (1.0 + float(loss_positive_weight) * (target > 0.0).float())
    )
    pos_mask = danger_labels > 0.0
    if bool(pos_mask.any()):
        loss_huber = (loss_huber_none * pos_mask).sum() / pos_mask.sum()
    else:
        loss_huber = loss_huber_none.mean()

    hand_scale = _label_entropy(hand_labels, HAND_CLASSES)
    shanten_scale = _label_entropy(shanten_labels, SHANTEN_CLASSES)
    wait_scale = _label_entropy(wait_labels, WAIT_CLASSES)
    danger_scale = _weighted_bce_constant_baseline(
        danger_labels.float().mean(), danger_pos_weight,
    )
    loss_scale = _loss_positive_baseline(target, pos_mask)

    return {
        "belief/hand_loss": hand_loss,
        "belief/shanten_loss": shanten_loss,
        "belief/wait_loss": wait_loss,
        "belief/danger_loss": danger_loss,
        "belief/loss_loss": loss_huber,
        "belief/hand_loss_norm": hand_loss / hand_scale,
        "belief/shanten_loss_norm": shanten_loss / shanten_scale,
        "belief/wait_loss_norm": wait_loss / wait_scale,
        "belief/danger_loss_norm": danger_loss / danger_scale,
        "belief/loss_loss_norm": loss_huber / loss_scale,
    }


def belief_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    head_weights: dict[str, float] | None = None,
    danger_pos_weight: float = 5.0,
    loss_positive_weight: float = 20.0,
) -> dict[str, torch.Tensor]:
    """V19 模糊化信念联合损失：五头监督（归一化）+ λ_k 加权。"""
    require_belief_outputs(output)
    components = _belief_loss_components(
        output,
        batch,
        danger_pos_weight=danger_pos_weight,
        loss_positive_weight=loss_positive_weight,
    )
    weights = {
        "hand": 1.0,
        "shanten": 1.0,
        "wait": 1.0,
        "danger": 1.0,
        "loss": 1.0,
    }
    if head_weights:
        weights.update({key: float(value) for key, value in head_weights.items()})
    weighted = (
        weights["hand"] * components["belief/hand_loss_norm"]
        + weights["shanten"] * components["belief/shanten_loss_norm"]
        + weights["wait"] * components["belief/wait_loss_norm"]
        + weights["danger"] * components["belief/danger_loss_norm"]
        + weights["loss"] * components["belief/loss_loss_norm"]
    )
    return {
        **components,
        "belief_loss_total": weighted,
    }


def belief_metrics_per_sample(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """返回逐样本模糊化信念指标（shape [B]），供 learner 的样本均值聚合使用。

    对 AUC 类全局标量扩展为 [B]（每样本携带同一批值，样本加权平均后仍为
    批量 AUC 的合理近似；评测侧使用批量版更精确）。训练步内只做 GPU 纯 torch
    计算：argsort/searchsorted 仅用于 PPO 侧（红线的 SFT 训练步不调用本函数）。
    """
    require_belief_outputs(output)
    hand_logits = output["belief_hand_logits"].float()
    hand_labels = batch["belief_hand"].long().reshape(*hand_logits.shape[:-1])
    hand_acc = (hand_logits.argmax(-1) == hand_labels).float().mean(dim=(-1, -2))

    shanten_logits = output["belief_shanten_logits"].float()
    shanten_labels = batch["belief_shanten"].long()
    shanten_top1 = (shanten_logits.argmax(-1) == shanten_labels).float().mean(dim=-1)

    wait_logits = output["belief_wait_logits"].float()
    wait_labels = batch["belief_wait"].long().reshape(wait_logits.shape[0], 3)
    wait_top1 = (wait_logits.argmax(-1) == wait_labels).float().mean(dim=-1)
    wait_tenpai_acc = (
        (wait_logits.argmax(-1) == 0).float() == (wait_labels == 0).float()
    ).float().mean(dim=-1)
    wait_probs = torch.softmax(wait_logits, dim=-1)
    wait_index = torch.arange(WAIT_CLASSES, device=wait_probs.device, dtype=wait_probs.dtype)
    wait_width_expected = (wait_probs * wait_index).sum(dim=-1)
    wait_width_mae = (wait_width_expected - wait_labels.float()).abs().mean(dim=-1)

    danger_logits = output["belief_danger_logits"].float()
    danger_labels = batch["belief_danger"].float().reshape(*danger_logits.shape)
    danger_prob = torch.sigmoid(danger_logits)
    danger_auc = binary_auc(danger_prob, danger_labels)
    # top-k 召回（k=真值可荣数）：纯 torch 排序取前 k 命中比例。
    sorted_idx = danger_prob.topk(danger_prob.shape[-1], dim=-1).indices
    sorted_correct = danger_labels.gather(-1, sorted_idx)
    k_true = danger_labels.sum(-1, keepdim=True)
    positions = torch.arange(
        danger_prob.shape[-1], device=danger_prob.device, dtype=danger_prob.dtype,
    ).view(1, 1, -1)
    danger_recall_at_topk = (
        (sorted_correct * (positions < k_true)).sum(-1, keepdim=True)
        / k_true.clamp_min(1)
    ).mean(dim=(-1, -2))

    target = loss_target_norm(batch["belief_loss"]).reshape(*danger_logits.shape)
    loss_mae = (output["belief_loss_pred"].float() - target).abs().mean(dim=(-1, -2))
    loss_conditional_mae = (
        (output["belief_loss_pred"].float() - target).abs() * danger_labels
    ).sum(dim=(-1, -2)) / danger_labels.sum(dim=(-1, -2)).clamp_min(1)

    batch_size = hand_acc.shape[0]
    return {
        "belief/hand_accuracy": hand_acc,
        "belief/shanten_top1": shanten_top1,
        "belief/wait_top1": wait_top1,
        "belief/wait_tenpai_acc": wait_tenpai_acc,
        "belief/wait_width_mae": wait_width_mae,
        "belief/danger_auc": danger_auc.expand(batch_size),
        "belief/danger_recall_at_topk": danger_recall_at_topk,
        "belief/loss_mae": loss_mae,
        "belief/loss_conditional_mae": loss_conditional_mae,
    }


def belief_metrics_batch(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """评测/汇总用的整批标量模糊化信念指标。"""
    require_belief_outputs(output)
    per_sample = belief_metrics_per_sample(output, batch)
    return {
        "hand_accuracy": float(per_sample["belief/hand_accuracy"].mean()),
        "shanten_top1": float(per_sample["belief/shanten_top1"].mean()),
        "wait_top1": float(per_sample["belief/wait_top1"].mean()),
        "wait_tenpai_acc": float(per_sample["belief/wait_tenpai_acc"].mean()),
        "wait_width_mae": float(per_sample["belief/wait_width_mae"].mean()),
        "danger_auc": float(per_sample["belief/danger_auc"].mean()),
        "danger_recall_at_topk": float(per_sample["belief/danger_recall_at_topk"].mean()),
        "loss_mae": float(per_sample["belief/loss_mae"].mean()),
        "loss_conditional_mae": float(per_sample["belief/loss_conditional_mae"].mean()),
    }


def belief_metric_keys() -> tuple[str, ...]:
    """PPO learner 注册的全部信念指标键（供 DDP 聚合集合维护）。"""
    return (
        "belief/hand_accuracy",
        "belief/shanten_top1",
        "belief/wait_top1",
        "belief/wait_tenpai_acc",
        "belief/wait_width_mae",
        "belief/danger_auc",
        "belief/danger_recall_at_topk",
        "belief/loss_mae",
        "belief/loss_conditional_mae",
        "belief/hand_loss",
        "belief/shanten_loss",
        "belief/wait_loss",
        "belief/danger_loss",
        "belief/loss_loss",
        "belief/hand_loss_norm",
        "belief/shanten_loss_norm",
        "belief/wait_loss_norm",
        "belief/danger_loss_norm",
        "belief/loss_loss_norm",
        "belief/total_loss",
    )
