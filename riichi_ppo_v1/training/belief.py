"""V19 信念五头监督的共享损失与指标(PPO learner 与 1v3 评测共用)。

与 ``sft/trainer.py`` 内的信念训练函数同构(同一套 D17-D19 标签定义),但
本模块面向 PPO 的逐批/逐样本聚合与评测面,并提供无 CPU 同步的纯 torch
AUC 近似(训练更新不引入逐 minibatch GPU→CPU 往返)。

v19 60% 方案(D20-D22 修订):wait 拆「N/A 二判 + 仅听牌行 34 牌」两级、
danger 用 pos_weight=5.0 的正例加权、loss 用「(1 + 20·I(target>0))」逐格
加权 huber;新增条件指标与 per-head loss 日志。
2026-09-06:wait_tile BCE 默认关闭(`wait_tile_weight=0.0`,仅保留 N/A 听牌
二判),原始 tile BCE 指标仍上报供监控。
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

# 听牌行判定：wait 标签第 35 位（N/A）为 0 表示该家听牌。
WAIT_N_A_INDEX = 34


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


def _belief_loss_components(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    wait_tenpai_weight: float = 1.0,
    wait_tile_weight: float = 0.0,
    danger_pos_weight: float = 5.0,
    loss_positive_weight: float = 20.0,
) -> dict[str, torch.Tensor]:
    """五头监督的基础损失项（不含 λ_k 加权与 wait_danger 项）。"""
    hand_labels = batch["belief_hand"].long().reshape(*output["belief_hand_logits"].shape[:-1])
    shanten_labels = batch["belief_shanten"].long()
    wait_labels = batch["belief_wait"].float().reshape(*output["belief_wait_logits"].shape)
    danger_labels = batch["belief_danger"].float().reshape(*output["belief_danger_logits"].shape)
    target = loss_target_norm(batch["belief_loss"]).reshape(*output["belief_loss_pred"].shape)

    hand_loss = F.cross_entropy(
        output["belief_hand_logits"].float().reshape(-1, 5), hand_labels.reshape(-1),
    )
    shanten_loss = F.cross_entropy(
        output["belief_shanten_logits"].float().reshape(-1, 9), shanten_labels.reshape(-1),
    )

    # wait：N/A 位二判（全样本）+ 34 牌逐格 BCE（仅听牌行）。
    wait_logits = output["belief_wait_logits"].float()
    tenpai_mask = wait_labels[..., WAIT_N_A_INDEX].eq(0.0)
    wait_tenpai_loss = F.binary_cross_entropy_with_logits(
        wait_logits[..., WAIT_N_A_INDEX], wait_labels[..., WAIT_N_A_INDEX],
    )
    wait_tile_bce = F.binary_cross_entropy_with_logits(
        wait_logits[..., :wait_logits.shape[-1] - 1],
        wait_labels[..., :wait_logits.shape[-1] - 1],
        reduction="none",
    )
    selected = tenpai_mask.unsqueeze(-1)
    wait_tile_loss = (
        (wait_tile_bce * selected).sum() / selected.sum().clamp_min(1)
    )
    wait_loss = float(wait_tenpai_weight) * wait_tenpai_loss + float(
        wait_tile_weight,
    ) * wait_tile_loss

    # danger：正例加权 BCE，供决策校准（pos_weight=5.0）。
    danger_loss = F.binary_cross_entropy_with_logits(
        output["belief_danger_logits"].float(),
        danger_labels,
        pos_weight=torch.full_like(danger_labels, float(danger_pos_weight)),
    )

    # loss：逐格加权 huber，(1 + 20·I(target>0))。
    loss_pred = output["belief_loss_pred"].float()
    loss_huber_none = F.huber_loss(loss_pred, target, reduction="none")
    loss_huber = (
        loss_huber_none * (1.0 + float(loss_positive_weight) * (target > 0.0).float())
    ).mean()

    wait_prob = torch.sigmoid(wait_logits)
    danger_prob = torch.sigmoid(output["belief_danger_logits"].float())
    # Wait 头含第 35 位 N/A 列,而 Danger 只覆盖 34 种牌;软约束在 34 种
    # 可荣/危险牌子集上逐格对齐(任务公示的 λ_c 公式按可荣合法掩码语义)。
    wait_danger = torch.relu(
        danger_prob - wait_prob[..., : danger_prob.shape[-1]] * danger_labels
    ).mean()

    return {
        "belief/hand_loss": hand_loss,
        "belief/shanten_loss": shanten_loss,
        "belief/wait_tenpai_loss": wait_tenpai_loss,
        "belief/wait_tile_loss": wait_tile_loss,
        "belief/wait_loss": wait_loss,
        "belief/danger_loss": danger_loss,
        "belief/loss_loss": loss_huber,
        "belief/wait_danger": wait_danger,
    }


def belief_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    head_weights: dict[str, float] | None = None,
    wait_danger_weight: float = 0.05,
    wait_tenpai_weight: float = 1.0,
    wait_tile_weight: float = 0.0,
    danger_pos_weight: float = 5.0,
    loss_positive_weight: float = 20.0,
) -> dict[str, torch.Tensor]:
    """V19 信念联合损失:五头监督(条件/加权式) + Wait-Danger 软约束。

    返回逐项损失与加总 ``belief_loss_total``;调用方把该总项加入 PPO 总损失。
    """
    require_belief_outputs(output)
    components = _belief_loss_components(
        output,
        batch,
        wait_tenpai_weight=wait_tenpai_weight,
        wait_tile_weight=wait_tile_weight,
        danger_pos_weight=danger_pos_weight,
        loss_positive_weight=loss_positive_weight,
    )
    weights = {
        "hand": 0.7,
        "shanten": 0.8,
        "wait": 1.5,
        "danger": 5.0,
        "loss": 5.0,
    }
    if head_weights:
        weights.update({key: float(value) for key, value in head_weights.items()})
    weighted = (
        weights["hand"] * components["belief/hand_loss"]
        + weights["shanten"] * components["belief/shanten_loss"]
        + weights["wait"] * components["belief/wait_loss"]
        + weights["danger"] * components["belief/danger_loss"]
        + weights["loss"] * components["belief/loss_loss"]
    )
    return {
        **components,
        "belief_loss_total": weighted + float(wait_danger_weight) * components["belief/wait_danger"],
    }


def belief_metrics_per_sample(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """返回逐样本信念指标(shape [B]),供 learner 的样本均值聚合使用。

    对 AUC 类全局标量扩展为 [B](每样本携带同一批值,样本加权平均后仍为
    批量 AUC 的合理近似;评测侧使用批量版更精确)。训练步内只做 GPU 纯 torch
    计算:argsort/searchsorted 仅用于 PPO 侧(红线的 SFT 训练步不调用本函数)。
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
    # N/A 位二判：预测非听牌位（=1）与标签是否一致，等价于听牌/非听二分。
    tenpai_mask = wait_labels[..., WAIT_N_A_INDEX].eq(0.0)
    wait_tenpai_acc = (
        (wait_probs[..., WAIT_N_A_INDEX] >= 0.5).float()
        == wait_labels[..., WAIT_N_A_INDEX]
    ).float().mean(dim=-1)
    # 仅听牌行、34 牌的 precision@2（无听牌行时返回 0）。
    wait_top2 = wait_probs[..., :wait_probs.shape[-1] - 1].topk(2, dim=-1).indices
    wait_correct2 = wait_labels[..., :wait_labels.shape[-1] - 1].gather(-1, wait_top2).float()
    wait_precision_at_2 = (
        (wait_correct2 * tenpai_mask.unsqueeze(-1)).sum(dim=(-1, -2))
        / (2 * tenpai_mask.sum(dim=-1)).clamp_min(1)
    )

    danger_logits = output["belief_danger_logits"].float()
    danger_labels = batch["belief_danger"].float().reshape(*danger_logits.shape)
    danger_prob = torch.sigmoid(danger_logits)
    danger_auc = binary_auc(danger_prob, danger_labels)
    wait_auc = binary_auc(wait_probs, wait_labels)
    wait_conditional_auc = binary_auc(
        wait_probs[..., :wait_probs.shape[-1] - 1][tenpai_mask],
        wait_labels[..., :wait_labels.shape[-1] - 1][tenpai_mask],
    )
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

    wait_danger_violation = (
        danger_prob > (wait_probs[..., :danger_prob.shape[-1]] * danger_labels)
    ).float().mean(dim=(-1, -2))

    batch_size = hand_acc.shape[0]
    return {
        "belief/hand_accuracy": hand_acc,
        "belief/shanten_top1": shanten_top1,
        "belief/wait_auc": wait_auc.expand(batch_size),
        "belief/wait_conditional_auc": wait_conditional_auc.expand(batch_size),
        "belief/wait_precision_at_5": wait_precision,
        "belief/wait_precision_at_2": wait_precision_at_2,
        "belief/wait_tenpai_acc": wait_tenpai_acc,
        "belief/danger_auc": danger_auc.expand(batch_size),
        "belief/danger_recall_at_topk": danger_recall_at_topk,
        "belief/loss_mae": loss_mae,
        "belief/loss_conditional_mae": loss_conditional_mae,
        "belief/wait_danger_violation": wait_danger_violation,
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
        "wait_conditional_auc": float(per_sample["belief/wait_conditional_auc"].mean()),
        "wait_precision_at_5": float(per_sample["belief/wait_precision_at_5"].mean()),
        "wait_precision_at_2": float(per_sample["belief/wait_precision_at_2"].mean()),
        "wait_tenpai_acc": float(per_sample["belief/wait_tenpai_acc"].mean()),
        "danger_auc": float(per_sample["belief/danger_auc"].mean()),
        "danger_recall_at_topk": float(per_sample["belief/danger_recall_at_topk"].mean()),
        "loss_mae": float(per_sample["belief/loss_mae"].mean()),
        "loss_conditional_mae": float(per_sample["belief/loss_conditional_mae"].mean()),
        "wait_danger_violation": float(per_sample["belief/wait_danger_violation"].mean()),
    }


def belief_metric_keys() -> tuple[str, ...]:
    """PPO learner 注册的全部信念指标键(供 DDP 聚合集合维护)。"""
    return (
        "belief/hand_accuracy",
        "belief/shanten_top1",
        "belief/wait_auc",
        "belief/wait_conditional_auc",
        "belief/wait_precision_at_5",
        "belief/wait_precision_at_2",
        "belief/wait_tenpai_acc",
        "belief/danger_auc",
        "belief/danger_recall_at_topk",
        "belief/loss_mae",
        "belief/loss_conditional_mae",
        "belief/wait_danger_violation",
        "belief/hand_loss",
        "belief/shanten_loss",
        "belief/wait_loss",
        "belief/wait_tenpai_loss",
        "belief/wait_tile_loss",
        "belief/danger_loss",
        "belief/loss_loss",
        "belief/wait_danger",
        "belief/total_loss",
    )
