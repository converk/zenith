"""V19 信念四头监督的共享损失与指标（PPO learner 与 1v3 评测共用）。

2026-09-07 模糊化：Hand = 花色×段位 16 组 × {0,1,≥2} 桶（CE）、
Wait = 听牌 + 待牌宽度桶 5 类（CE）、Danger 逐牌加权 BCE、Loss 逐牌 Huber。
2026-09-08 信念头精简（用户决策）：
- 删除 Shanten 头——是否听牌由 Wait 头类别 0（非听）精确提供；
- Loss 头由逐牌精确 Huber 回归改为**分桶分类**（桶边界
  1000/5000/9000/13000/17000 原始点数，共 6 类）：精确预测铳点损失压力
  过大且对局面影响小（4000 与 6000 差不多），分桶模糊化后损失贡献均匀；
- 四头（hand/wait/danger/loss_bucket）原始损失除以标签分布基线
  （熵 / 最优常数 BCE），λ_k=1.0 时四头贡献天然同量级。

与 ``sft/trainer.py`` 内的信念训练函数同构（同一套标签定义），但本模块
面向 PPO 的逐批/逐样本聚合与评测面，并提供无 CPU 同步的纯 torch AUC
近似（训练更新不引入逐 minibatch GPU→CPU 往返）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ..model.belief_network import (
    LOSS_BUCKET_CLASSES,
    LOSS_NORM_MAX,
    loss_bucket_centers,
    loss_bucket_targets,
)

BELIEF_OUTPUT_KEYS = (
    "belief_hand_logits",
    "belief_wait_logits",
    "belief_danger_logits",
    "belief_loss_bucket_logits",
)

HAND_CLASSES = 3
WAIT_CLASSES = 5
DANGER_CLASSES = 34


def require_belief_outputs(output: dict[str, torch.Tensor]) -> None:
    """模型前向必须包含信念四头输出（V19 拓扑 fail closed）。"""
    missing = sorted(set(BELIEF_OUTPUT_KEYS) - set(output))
    if missing:
        raise RuntimeError(
            "model forward did not emit belief outputs (missing: "
            + ", ".join(missing) + "); V19 training requires the belief network"
        )


def loss_target_norm(raw_loss: torch.Tensor) -> torch.Tensor:
    """Loss 目标归一化：min(raw, 24000) / 24000，与桶中心同尺度。"""
    return torch.clamp(raw_loss.float(), max=LOSS_NORM_MAX) / LOSS_NORM_MAX


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


def _belief_loss_components(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    danger_pos_weight: float = 5.0,
    loss_positive_weight: float = 20.0,
) -> dict[str, torch.Tensor]:
    """模糊化四头监督的基础损失项（原始 + 归一化；不含 λ_k 加权）。

    Loss 分桶 CE 沿用旧 Huber 的条件/加权结构：仅危险正例（danger>0）子集
    计入，且桶 CE 乘 (1 + loss_positive_weight·1[raw>0]) 放大有真实铳点
    样本的贡献。
    """
    hand_labels = batch["belief_hand"].long().reshape(*output["belief_hand_logits"].shape[:-1])
    wait_labels = batch["belief_wait"].long().reshape(output["belief_wait_logits"].shape[0], 3)
    danger_labels = batch["belief_danger"].float().reshape(*output["belief_danger_logits"].shape)
    raw_loss = batch["belief_loss"].reshape(*output["belief_loss_bucket_logits"].shape[:-1])
    bucket_labels = loss_bucket_targets(raw_loss)

    hand_loss = F.cross_entropy(
        output["belief_hand_logits"].float().reshape(-1, HAND_CLASSES),
        hand_labels.reshape(-1),
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

    loss_ce_none = (
        F.cross_entropy(
            output["belief_loss_bucket_logits"].float().reshape(-1, LOSS_BUCKET_CLASSES),
            bucket_labels.reshape(-1),
            reduction="none",
        ).view(danger_labels.shape)
        * (1.0 + float(loss_positive_weight) * (raw_loss > 0.0).float())
    )
    pos_mask = danger_labels > 0.0
    if bool(pos_mask.any()):
        loss_bucket = (loss_ce_none * pos_mask).sum() / pos_mask.sum()
    else:
        loss_bucket = loss_ce_none.mean()

    hand_scale = _label_entropy(hand_labels, HAND_CLASSES)
    wait_scale = _label_entropy(wait_labels, WAIT_CLASSES)
    danger_scale = _weighted_bce_constant_baseline(
        danger_labels.float().mean(), danger_pos_weight,
    )
    loss_scale = _label_entropy(bucket_labels, LOSS_BUCKET_CLASSES)

    return {
        "belief/hand_loss": hand_loss,
        "belief/wait_loss": wait_loss,
        "belief/danger_loss": danger_loss,
        "belief/loss_bucket_loss": loss_bucket,
        "belief/hand_loss_norm": hand_loss / hand_scale,
        "belief/wait_loss_norm": wait_loss / wait_scale,
        "belief/danger_loss_norm": danger_loss / danger_scale,
        "belief/loss_bucket_loss_norm": loss_bucket / loss_scale,
    }


def belief_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    head_weights: dict[str, float] | None = None,
    danger_pos_weight: float = 5.0,
    loss_positive_weight: float = 20.0,
) -> dict[str, torch.Tensor]:
    """V19 模糊化信念联合损失：四头监督（归一化）+ λ_k 加权。"""
    require_belief_outputs(output)
    components = _belief_loss_components(
        output,
        batch,
        danger_pos_weight=danger_pos_weight,
        loss_positive_weight=loss_positive_weight,
    )
    weights = {
        "hand": 1.0,
        "wait": 1.0,
        "danger": 1.0,
        "loss_bucket": 1.0,
    }
    if head_weights:
        weights.update({key: float(value) for key, value in head_weights.items()})
    weighted = (
        weights["hand"] * components["belief/hand_loss_norm"]
        + weights["wait"] * components["belief/wait_loss_norm"]
        + weights["danger"] * components["belief/danger_loss_norm"]
        + weights["loss_bucket"] * components["belief/loss_bucket_loss_norm"]
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

    wait_logits = output["belief_wait_logits"].float()
    wait_labels = batch["belief_wait"].long().reshape(wait_logits.shape[0], 3)
    wait_top1 = (wait_logits.argmax(-1) == wait_labels).float().mean(dim=-1)
    # 是否听牌的精确命中率（类别 0 = 非听）——不模糊的高信息量信号。
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

    # 铳点分桶：逐格 top-1 命中率 + 桶期望（可解释粗粒度 MAE，归一化单位）。
    bucket_logits = output["belief_loss_bucket_logits"].float()
    raw_loss = batch["belief_loss"].reshape(*bucket_logits.shape[:-1])
    bucket_labels = loss_bucket_targets(raw_loss)
    loss_bucket_accuracy = (
        (bucket_logits.argmax(-1) == bucket_labels).float().mean(dim=(-1, -2))
    )
    loss_prob = torch.softmax(bucket_logits, dim=-1)
    centers = loss_bucket_centers(bucket_logits.device).view(
        1, 1, 1, LOSS_BUCKET_CLASSES,
    )
    loss_expected = (loss_prob * centers).sum(dim=-1)
    loss_target = loss_target_norm(raw_loss)
    loss_expected_mae = (loss_expected - loss_target).abs().mean(dim=(-1, -2))

    batch_size = hand_acc.shape[0]
    return {
        "belief/hand_accuracy": hand_acc,
        "belief/wait_top1": wait_top1,
        "belief/wait_tenpai_acc": wait_tenpai_acc,
        "belief/wait_width_mae": wait_width_mae,
        "belief/danger_auc": danger_auc.expand(batch_size),
        "belief/danger_recall_at_topk": danger_recall_at_topk,
        "belief/loss_bucket_accuracy": loss_bucket_accuracy,
        "belief/loss_expected_mae": loss_expected_mae,
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
        "wait_top1": float(per_sample["belief/wait_top1"].mean()),
        "wait_tenpai_acc": float(per_sample["belief/wait_tenpai_acc"].mean()),
        "wait_width_mae": float(per_sample["belief/wait_width_mae"].mean()),
        "danger_auc": float(per_sample["belief/danger_auc"].mean()),
        "danger_recall_at_topk": float(per_sample["belief/danger_recall_at_topk"].mean()),
        "loss_bucket_accuracy": float(per_sample["belief/loss_bucket_accuracy"].mean()),
        "loss_expected_mae": float(per_sample["belief/loss_expected_mae"].mean()),
    }


def belief_metric_keys() -> tuple[str, ...]:
    """PPO learner 注册的全部信念指标键（供 DDP 聚合集合维护）。"""
    return (
        "belief/hand_accuracy",
        "belief/wait_top1",
        "belief/wait_tenpai_acc",
        "belief/wait_width_mae",
        "belief/danger_auc",
        "belief/danger_recall_at_topk",
        "belief/loss_bucket_accuracy",
        "belief/loss_expected_mae",
        "belief/hand_loss",
        "belief/wait_loss",
        "belief/danger_loss",
        "belief/loss_bucket_loss",
        "belief/hand_loss_norm",
        "belief/wait_loss_norm",
        "belief/danger_loss_norm",
        "belief/loss_bucket_loss_norm",
        "belief/total_loss",
    )
