"""V19 信念五头监督的共享损失与指标(PPO learner 与 1v3 评测共用)。

与 ``sft/trainer.py`` 内的信念训练函数同构(同一套 D17-D19 标签定义),但
本模块面向 PPO 的逐批/逐样本聚合与评测面,并提供无 CPU 同步的纯 torch
AUC 近似(训练更新不引入逐 minibatch GPU→CPU 往返)。
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

BELIEF_OUTPUT_KEYS = (
    "belief_hand_logits",
    "belief_shanten_logits",
    "belief_wait_logits",
    "belief_danger_logits",
    "belief_loss_pred",
)


def require_belief_outputs(output: dict[str, torch.Tensor]) -> None:
    """模型前向必须包含信念五头输出(V19 拓扑 fail closed)。"""
    missing = sorted(set(BELIEF_OUTPUT_KEYS) - set(output))
    if missing:
        raise RuntimeError(
            "model forward did not emit belief outputs (missing: "
            + ", ".join(missing) + "); V19 training requires the belief network"
        )


def loss_target_norm(raw_loss: torch.Tensor) -> torch.Tensor:
    """Loss 目标归一化:min(raw, 24000) / 24000,与 sigmoid 预测同尺度(D18)。"""
    return torch.clamp(raw_loss.float(), max=24000.0) / 24000.0


def binary_auc(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """纯 torch rank-based AUC(含平局平均秩;单类别时返回 0.5)。

    每次返回标量;训练/评测小批内调用一次,避免逐元素 CPU 同步。
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


def belief_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    head_weights: dict[str, float] | None = None,
    wait_danger_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    """V19 信念联合损失:五头监督 + Wait-Danger 软约束。

    返回逐项损失与加总 ``belief_loss_total``;调用方把该总项加入 PPO 总损失。
    """
    require_belief_outputs(output)
    hand_labels = batch["belief_hand"].long().reshape(*output["belief_hand_logits"].shape[:-1])
    shanten_labels = batch["belief_shanten"].long()
    wait_labels = batch["belief_wait"].float().reshape(*output["belief_wait_logits"].shape)
    danger_labels = batch["belief_danger"].float().reshape(*output["belief_danger_logits"].shape)
    target = loss_target_norm(batch["belief_loss"]).reshape(
        *output["belief_loss_pred"].shape,
    )

    hand_loss = F.cross_entropy(
        output["belief_hand_logits"].float().reshape(-1, 5), hand_labels.reshape(-1),
    )
    shanten_loss = F.cross_entropy(
        output["belief_shanten_logits"].float().reshape(-1, 9), shanten_labels.reshape(-1),
    )
    wait_loss = F.binary_cross_entropy_with_logits(
        output["belief_wait_logits"].float(), wait_labels,
    )
    danger_loss = F.binary_cross_entropy_with_logits(
        output["belief_danger_logits"].float(), danger_labels,
    )
    loss_pred = output["belief_loss_pred"].float()
    loss_huber = F.huber_loss(loss_pred, target, reduction="mean")
    wait_prob = torch.sigmoid(output["belief_wait_logits"].float())
    danger_prob = torch.sigmoid(output["belief_danger_logits"].float())
    # Wait 头含第 35 位 N/A 列,而 Danger 只覆盖 34 种牌;软约束在 34 种
    # 可荣/危险牌子集上逐格对齐(任务公示的 λ_c 公式按可荣合法掩码语义)。
    wait_danger = torch.relu(
        danger_prob - wait_prob[..., : danger_prob.shape[-1]] * danger_labels
    ).mean()

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
        weights["hand"] * hand_loss
        + weights["shanten"] * shanten_loss
        + weights["wait"] * wait_loss
        + weights["danger"] * danger_loss
        + weights["loss"] * loss_huber
    )
    return {
        "belief_hand_loss": hand_loss,
        "belief_shanten_loss": shanten_loss,
        "belief_wait_loss": wait_loss,
        "belief_danger_loss": danger_loss,
        "belief_loss_loss": loss_huber,
        "belief_wait_danger": wait_danger,
        "belief_loss_total": weighted + float(wait_danger_weight) * wait_danger,
    }


def belief_metrics_per_sample(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """返回逐样本信念指标(shape [B]),供 learner 的样本均值聚合使用。

    对 AUC 类全局标量扩展为 [B](每样本携带同一批值,样本加权平均后仍为
    批量 AUC 的合理近似;评测侧使用批量版更精确)。
    """
    require_belief_outputs(output)
    hand_logits = output["belief_hand_logits"].float()
    hand_labels = batch["belief_hand"].long().reshape(*hand_logits.shape[:-1])
    hand_acc = (hand_logits.argmax(-1) == hand_labels).float().mean(dim=(-1, -2))

    shanten_logits = output["belief_shanten_logits"].float()
    shanten_labels = batch["belief_shanten"].long()
    shanten_top1 = (shanten_logits.argmax(-1) == shanten_labels).float().mean(dim=-1)

    wait_logits = output["belief_wait_logits"].float()
    wait_labels = batch["belief_wait"].float().reshape(*wait_logits.shape)
    wait_probs = torch.sigmoid(wait_logits)
    wait_top5 = wait_probs.topk(5, dim=-1).indices
    wait_precision = wait_labels.gather(-1, wait_top5).float().mean(dim=(-1, -2))

    danger_logits = output["belief_danger_logits"].float()
    danger_labels = batch["belief_danger"].float().reshape(*danger_logits.shape)
    danger_auc = binary_auc(torch.sigmoid(danger_logits), danger_labels)
    wait_auc = binary_auc(wait_probs, wait_labels)

    target = loss_target_norm(batch["belief_loss"]).reshape(*danger_logits.shape)
    loss_mae = (output["belief_loss_pred"].float() - target).abs().mean(dim=(-1, -2))

    batch_size = hand_acc.shape[0]
    return {
        "belief/hand_accuracy": hand_acc,
        "belief/shanten_top1": shanten_top1,
        "belief/wait_auc": wait_auc.expand(batch_size),
        "belief/wait_precision_at_5": wait_precision,
        "belief/danger_auc": danger_auc.expand(batch_size),
        "belief/loss_mae": loss_mae,
    }


def belief_metrics_batch(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """评测/汇总用的整批标量信念指标。"""
    require_belief_outputs(output)
    per_sample = belief_metrics_per_sample(output, batch)
    return {
        "hand_accuracy": float(per_sample["belief/hand_accuracy"].mean()),
        "shanten_top1": float(per_sample["belief/shanten_top1"].mean()),
        "wait_auc": float(per_sample["belief/wait_auc"].mean()),
        "wait_precision_at_5": float(per_sample["belief/wait_precision_at_5"].mean()),
        "danger_auc": float(per_sample["belief/danger_auc"].mean()),
        "loss_mae": float(per_sample["belief/loss_mae"].mean()),
    }


def belief_metric_keys() -> tuple[str, ...]:
    """PPO learner 注册的全部信念指标键(供 DDP 聚合集合维护)。"""
    return (
        "belief/hand_accuracy",
        "belief/shanten_top1",
        "belief/wait_auc",
        "belief/wait_precision_at_5",
        "belief/danger_auc",
        "belief/loss_mae",
        "belief/total_loss",
    )
