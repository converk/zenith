"""Low-overhead TensorBoard metrics for supervised riichi training."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol

import torch
from torch import Tensor


class ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...


ACTION_GROUPS = (
    "pass", "discard", "reach", "chi", "pon", "kan", "hora", "ryukyoku",
)

ACTION_GROUP_LABELS = {
    "pass": "过",
    "discard": "打牌",
    "reach": "立直",
    "chi": "吃",
    "pon": "碰",
    "kan": "杠",
    "hora": "和牌",
    "ryukyoku": "流局",
}

SCALAR_TAGS = {
    "train/loss": "SFT/训练/总损失 (loss)",
    "train/policy_ce": "SFT/训练/策略交叉熵 (policy_ce)",
    "train/value_huber": "SFT/训练/价值 Huber (value_huber)",
    "train/top1": "SFT/训练/Top-1 准确率 (top1)",
    "train/top3": "SFT/训练/Top-3 准确率 (top3)",
    "optimizer/learning_rate": "SFT/优化器/学习率 (learning_rate)",
    "optimizer/grad_norm_pre_clip": "SFT/优化器/裁剪前梯度范数 (grad_norm_pre_clip)",
    "optimizer/grad_norm_post_clip": "SFT/优化器/裁剪后梯度范数 (grad_norm_post_clip)",
    "model/offense_projection_weight_norm": "SFT/模型/Offense Projection 权重范数",
    "model/offense_projection_grad_norm": "SFT/模型/Offense Projection 梯度范数",
    "model/v13_logit_drift_rms": "SFT/模型/相对 V13 Logit Drift RMS",
    "model/v13_logit_drift_max": "SFT/模型/相对 V13 Logit Drift Max",
    "data/legal_actions_mean": "SFT/数据/平均合法动作数 (legal_actions_mean)",
    "data/token_length_mean": "SFT/数据/平均序列长度 (token_length_mean)",
    "data/token_length_max": "SFT/数据/最大序列长度 (token_length_max)",
    "data/padding_fraction": "SFT/数据/Padding 比例 (padding_fraction)",
    "performance/window_samples_per_s": "SFT/性能/窗口样本每秒 (window_samples_per_s)",
    "performance/cumulative_samples_per_s": "SFT/性能/累计样本每秒 (cumulative_samples_per_s)",
    "performance/effective_tokens_per_s": "SFT/性能/有效 Token 每秒 (effective_tokens_per_s)",
    "performance/step_time_s": "SFT/性能/平均 Step 耗时·秒 (step_time_s)",
    "system/gpu_memory_allocated_mb": "SFT/系统/Rank0 已分配显存·MB (allocated)",
    "system/gpu_memory_reserved_mb": "SFT/系统/Rank0 已保留显存·MB (reserved)",
    "system/gpu_memory_peak_mb": "SFT/系统/Rank0 峰值显存·MB (peak)",
}


def _group_masks(actions: Tensor) -> dict[str, Tensor]:
    return {
        "pass": actions.eq(0),
        "discard": actions.ge(1) & actions.le(74),
        "reach": actions.eq(75),
        "chi": actions.ge(76) & actions.le(132),
        "pon": actions.ge(133) & actions.le(169),
        "kan": actions.ge(170) & actions.le(238),
        "hora": actions.eq(239),
        "ryukyoku": actions.eq(240),
    }


@dataclass
class SftMetricWindow:
    """Accumulate detached GPU counters and synchronize only when logging."""

    sums: dict[str, Tensor] = field(default_factory=dict)
    steps: int = 0
    step_seconds: float = 0.0
    padded_tokens: int = 0
    effective_tokens: int = 0

    def _add(self, name: str, value: Tensor) -> None:
        detached = value.detach()
        if name in self.sums:
            self.sums[name].add_(detached)
        else:
            self.sums[name] = detached.clone()

    def update(
        self,
        *,
        logits: Tensor,
        actions: Tensor,
        legal_mask: Tensor,
        token_lengths: Tensor,
        loss: Tensor,
        policy_ce: Tensor,
        value_huber: Tensor | None,
        effective_tokens: int,
        padded_tokens: int,
        step_seconds: float,
    ) -> None:
        batch = int(actions.numel())
        top = logits.detach().topk(3, dim=-1).indices
        top1 = top[:, 0].eq(actions)
        top3 = top.eq(actions[:, None]).any(-1)
        self._add("samples", actions.new_tensor(batch, dtype=torch.float32))
        self._add("loss_sum", loss.detach().float() * batch)
        self._add("policy_ce_sum", policy_ce.detach().float() * batch)
        if value_huber is not None:
            self._add("value_huber_sum", value_huber.detach().float() * batch)
        self._add("top1", top1.float().sum())
        self._add("top3", top3.float().sum())
        self._add("legal_actions", legal_mask.detach().sum().float())
        self._add("token_lengths", token_lengths.detach().sum().float())
        maximum = token_lengths.detach().max().float()
        if "token_length_max" in self.sums:
            self.sums["token_length_max"] = torch.maximum(
                self.sums["token_length_max"], maximum
            )
        else:
            self.sums["token_length_max"] = maximum.clone()
        for group, mask in _group_masks(actions).items():
            self._add(f"{group}/count", mask.float().sum())
            self._add(f"{group}/top1", (top1 & mask).float().sum())
            self._add(f"{group}/top3", (top3 & mask).float().sum())
        self.steps += 1
        self.step_seconds += float(step_seconds)
        self.effective_tokens += int(effective_tokens)
        self.padded_tokens += int(padded_tokens)

    def scalars(self) -> dict[str, float]:
        if "samples" not in self.sums:
            return {}
        values = {name: float(value) for name, value in self.sums.items()}
        samples = max(values["samples"], 1.0)
        seconds = max(self.step_seconds, 1e-9)
        result = {
            "train/loss": values["loss_sum"] / samples,
            "train/policy_ce": values["policy_ce_sum"] / samples,
            "train/top1": values["top1"] / samples,
            "train/top3": values["top3"] / samples,
            "data/legal_actions_mean": values["legal_actions"] / samples,
            "data/token_length_mean": values["token_lengths"] / samples,
            "data/token_length_max": values["token_length_max"],
            "data/padding_fraction": 1.0 - self.effective_tokens / max(self.padded_tokens, 1),
            "performance/window_samples_per_s": samples / seconds,
            "performance/effective_tokens_per_s": self.effective_tokens / seconds,
            "performance/step_time_s": seconds / max(self.steps, 1),
        }
        if "value_huber_sum" in values:
            result["train/value_huber"] = values["value_huber_sum"] / samples
        for group in ACTION_GROUPS:
            count = values[f"{group}/count"]
            result[f"train/action/{group}/count"] = count
            if count > 0:
                result[f"train/action/{group}/top1"] = values[f"{group}/top1"] / count
                result[f"train/action/{group}/top3"] = values[f"{group}/top3"] / count
        return result

    def reset(self) -> None:
        self.sums.clear()
        self.steps = 0
        self.step_seconds = 0.0
        self.padded_tokens = 0
        self.effective_tokens = 0


def _display_tag(name: str) -> str | None:
    if name in SCALAR_TAGS:
        return SCALAR_TAGS[name]
    if name.startswith("optimizer/grad_norm_branch/"):
        return f"SFT/优化器/分支梯度范数/{name.rsplit('/', 1)[-1]}"
    parts = name.split("/")
    if len(parts) == 4 and parts[:2] == ["train", "action"]:
        group, metric = parts[2], parts[3]
        if group in ACTION_GROUP_LABELS and metric in {"top1", "top3", "count"}:
            labels = {"top1": "Top-1 准确率", "top3": "Top-3 准确率", "count": "样本数"}
            return f"SFT/训练动作/{ACTION_GROUP_LABELS[group]}/{labels[metric]} ({group}_{metric})"
    if name.startswith("validation/"):
        suffix = name.removeprefix("validation/")
        if suffix in {"loss", "policy_ce", "value_huber", "value_mae", "value_rmse", "top1", "top3", "samples"}:
            return f"SFT/验证/{suffix}"
        for group in ACTION_GROUPS:
            if suffix in {f"top1_{group}", f"top3_{group}"}:
                metric = suffix.split("_", 1)[0]
                return f"SFT/验证动作/{ACTION_GROUP_LABELS[group]}/{metric.upper()} ({suffix})"
    if name.startswith("heuristic_eval/"):
        suffix = name.removeprefix("heuristic_eval/")
        return f"SFT/启发式对局评测/{suffix}"
    return None


def write_sft_scalars(
    writer: ScalarWriter, metrics: Mapping[str, float], step: int,
) -> None:
    for name, value in metrics.items():
        tag = _display_tag(name)
        if tag is not None and math.isfinite(float(value)):
            writer.add_scalar(tag, float(value), int(step))
