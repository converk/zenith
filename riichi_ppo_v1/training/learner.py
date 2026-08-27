"""V18 PPO 优化器:V18 批 collate、PPO clip + value Huber + GAE value advantage。

``PPOLearner`` 同时支持单卡(默认)与双卡 DDP:传 ``rank``/``world_size``
后模型由 ``DistributedDataParallel`` 包装,update 内所有梯度 collective 与
early-stop/KL 判定都在进程组内同步。
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from ..model import KyokuTransformerActorCritic, ModelConfig
from ..model.schema import TOKEN_SCHEMA_VERSION
from .profiling import StageProfiler
from .rollout_buffer import RolloutBuffer

# V18 参数分支:actor/critic/shared 三组独立调度,gradient 按参数根分发。
ACTOR_ROOTS = {"actor_backbone", "query_embedding", "action_fusion", "policy_mlp"}
CRITIC_ROOTS = {"critic_embedding", "critic_backbone", "value_head", "value_query"}
SHARED_ROOTS = {"token_embedding", "snapshot_embeddings", "public_backbone"}


def validate_fresh_model_checkpoint_contract(payload: dict[str, Any]) -> None:
    """校验用作 V18 PPO 初始化的 SFT/模型 checkpoint 契约。"""
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("V18 PPO requires a checkpoint with model weights")
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict) or raw_config.get("policy_head_type") != "current_state_snapshot":
        raise RuntimeError("V18 PPO requires a current_state_snapshot model checkpoint")


def scheduled_learning_rate(
    base: float,
    update: int,
    total_updates: int,
    warmup_fraction: float,
    min_lr: float | None = None,
) -> float:
    """学习率调度:1-based updates,warmup 线性升到 base,之后线性衰减。

    ``min_lr`` 为 None 时保持历史 Exp 风格(衰减终点约为 base/剩余步数);
    给定数值时从 base 线性衰减到 min_lr,``min_lr=0.0`` 时终点为 0。
    """
    total = max(1, int(total_updates))
    step = max(0, int(update))
    warmup = int(total * float(warmup_fraction))
    if warmup > 0 and step <= warmup:
        return float(base) * float(step) / float(warmup)
    decay_updates = max(1, total - warmup)
    if min_lr is None:
        return float(base) * max(0.0, float(total - step + 1) / float(decay_updates))
    progress = min(max(0.0, float(step - warmup) / float(decay_updates)), 1.0)
    return float(base) - (float(base) - float(min_lr)) * progress


def scheduled_entropy_coefficient(
    start: float,
    end: float,
    update: int,
    total_updates: int,
    *,
    middle: float | None = None,
    middle_fraction: float = 0.5,
) -> float:
    """entropy 系数调度:默认线性退火,配置 middle 时三点分段线性。"""
    if middle is not None:
        return scheduled_piecewise_coefficient(
            start, float(middle), end, update, total_updates, middle_fraction,
        )
    total = max(1, int(total_updates))
    progress = min(max(float(update) / float(total), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * progress


def scheduled_piecewise_coefficient(
    start: float,
    middle: float,
    end: float,
    update: int,
    total_updates: int,
    middle_fraction: float,
) -> float:
    """三点分段单调调度,精确命中首/中/末锚点。"""
    if not 0.0 < float(middle_fraction) < 1.0:
        raise ValueError("middle_fraction must be in (0, 1)")
    total = max(1, int(total_updates))
    step = min(max(int(update), 1), total)
    middle_update = min(total, max(1, round(total * float(middle_fraction))))
    if step <= middle_update:
        local = (step - 1) / max(middle_update - 1, 1)
        return float(start) + (float(middle) - float(start)) * local
    local = (step - middle_update) / max(total - middle_update, 1)
    return float(middle) + (float(end) - float(middle)) * local


def scheduled_u_coefficient(
    start: float,
    middle: float,
    end: float,
    update: int,
    total_updates: int,
    middle_fraction: float = 0.5,
) -> float:
    """分段线性 U 形调度,精确锚定 start/middle/end。"""
    if not 0.0 < float(middle_fraction) < 1.0:
        raise ValueError("middle_fraction must be in (0, 1)")
    total = max(1, int(total_updates))
    step = min(max(int(update), 1), total)
    middle_update = min(total, max(1, round(total * float(middle_fraction))))
    if step <= middle_update:
        local = (step - 1) / max(middle_update - 1, 1)
        return float(start) + (float(middle) - float(start)) * local
    local = (step - middle_update) / max(total - middle_update, 1)
    return float(middle) + (float(end) - float(middle)) * local


def value_loss_values(predicted: torch.Tensor, returns: torch.Tensor, loss_name: str) -> torch.Tensor:
    """按样本返回配置的 PPO value 目标损失。"""
    normalized = str(loss_name).lower()
    if normalized == "huber":
        return F.huber_loss(predicted, returns, reduction="none")
    if normalized == "mse":
        return F.mse_loss(predicted, returns, reduction="none")
    raise ValueError(f"unknown value_loss {loss_name!r}; expected 'huber' or 'mse'")


def normalize_value_targets(
    predicted: torch.Tensor,
    returns: torch.Tensor,
    *,
    mode: str,
    mean: float,
    std: float,
    std_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 critic 预测与目标放入配置的损失空间(rollout 值本身不变)。"""
    normalized_mode = str(mode).lower()
    if normalized_mode == "none":
        return predicted, returns
    if normalized_mode == "batch_std":
        scale = max(float(std), float(std_floor))
        return (predicted - float(mean)) / scale, (returns - float(mean)) / scale
    raise ValueError("value_target_normalization must be one of 'none' or 'batch_std'")


def optimizer_branch_parameters(
    optimizer: torch.optim.Optimizer,
) -> dict[str, list[nn.Parameter]]:
    """从 optimizer 参数组读取 actor/critic/shared 参数列表。"""
    grouped: dict[str, list[nn.Parameter]] = {"actor": [], "critic": [], "shared": []}
    for group in optimizer.param_groups:
        branch = str(group.get("branch", ""))
        if branch not in grouped:
            raise ValueError(f"unclassified optimizer branch: {branch}")
        grouped[branch].extend(group["params"])
    return grouped


def clip_branch_grad_norms(
    branch_parameters: dict[str, list[nn.Parameter]],
    max_norms: dict[str, float],
) -> dict[str, torch.Tensor]:
    """独立裁剪 actor/shared/critic,返回每个分支的裁剪前后与缩放比例。

    裁剪用 ``clip_grad_norm_(..., foreach=True)`` 的融合内核;post 范数由
    代数恒等式 ``post = pre × clamp(max/(pre+1e-6), 1)`` 直接导出(与
    clip_grad_norm_ 内部实际应用的缩放系数一致),免去逐参数二次范数扫描;
    整函数无 ``.item()``/``bool()`` 的 GPU→CPU 同步点。
    """
    device = next(
        parameter.device
        for parameters in branch_parameters.values()
        for parameter in parameters
    )
    metrics: dict[str, torch.Tensor] = {}
    for branch in ("actor", "shared", "critic"):
        parameters = branch_parameters[branch]
        max_norm = float(max_norms[branch])
        pre_clip = nn.utils.clip_grad_norm_(parameters, max_norm, foreach=True)
        pre_clip_device = pre_clip.to(device=device)
        # 与 clip_grad_norm_ 内部一致的缩放系数(max/(total+1e-6),clamp 到 1);
        # pre==0 时系数被 clamp 为 1,post=0,无需分支判断。
        scale = (max_norm / (pre_clip_device.detach() + 1e-6)).clamp(max=1.0)
        post_clip = pre_clip_device * scale
        metrics[f"grad_norm_{branch}_pre_clip"] = pre_clip_device
        metrics[f"grad_norm_{branch}_post_clip"] = post_clip
        metrics[f"grad_norm_{branch}_clip_scale"] = scale
        metrics[f"grad_norm_{branch}_clipped"] = (
            pre_clip_device > max_norm
        ).float()
    return metrics


def accumulation_group_size(
    planned_minibatches: int,
    accumulation_steps: int,
    minibatch_index: int,
) -> int:
    """返回当前 minibatch 所属累积组的实际大小。"""
    planned = int(planned_minibatches)
    steps = max(1, int(accumulation_steps))
    index = int(minibatch_index)
    if planned <= 0:
        raise ValueError("planned_minibatches must be positive")
    if index < 0 or index >= planned:
        raise ValueError("minibatch_index is out of range")
    group_start = (index // steps) * steps
    return min(steps, planned - group_start)


def _rollout_values(
    transitions: RolloutBuffer,
    field: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """从 rollout SoA 读取标量字段。"""
    return np.asarray(getattr(transitions, field), dtype=dtype)


def discounted_empirical_returns(
    transitions: RolloutBuffer, gamma: float,
) -> np.ndarray:
    """Monte Carlo reward-to-go,每局终局时重置(向量化)。

    语义与旧的逐 transition Python 循环一致:在 float64 上累积(与旧实现的
    Python float 累积同精度),每段(以 done 为右边界,含 done 自身奖励)
    ``returns[i] = Σ_{j=i}^{end} rewards[j]·γ^(j-i)``,最后统一转 float32。
    γ=1.0 时退化为段内前缀和差,除法为精确 1.0。O(N),无 Python 逐元素循环。
    """
    count = len(transitions)
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    rewards64 = _rollout_values(
        transitions, "rewards", np.dtype(np.float32),
    ).astype(np.float64)
    done = _rollout_values(transitions, "done", np.dtype(np.bool_))
    discount_powers = np.power(float(gamma), np.arange(count, dtype=np.float64))
    scaled = rewards64 * discount_powers
    cumulative = np.concatenate(([0.0], np.cumsum(scaled)))
    ends = np.flatnonzero(done)
    starts_ext = np.concatenate(([0], ends + 1, [count]))
    segment_ids = np.repeat(
        np.arange(len(ends) + 1, dtype=np.int64), np.diff(starts_ext),
    )
    segment_end_cumulative = cumulative[starts_ext[1:]]
    index_range = np.arange(count, dtype=np.int64)
    returns64 = (
        segment_end_cumulative[segment_ids] - cumulative[index_range]
    ) / discount_powers
    return returns64.astype(np.float32)


def rollout_update_targets(
    transitions: RolloutBuffer,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """完整 rollout 的 advantage 归一化与 λ-return(供 DDP 分片复用)。"""
    source_advantages = _rollout_values(
        transitions, "advantages", np.dtype(np.float32),
    )
    advantages = (
        (source_advantages - source_advantages.mean(dtype=np.float64))
        / (source_advantages.std(dtype=np.float64) + 1e-8)
    ).astype(np.float32)
    del gamma
    returns = (
        _rollout_values(transitions, "values", np.dtype(np.float32))
        + source_advantages
    ).astype(np.float32)
    return (
        advantages,
        returns,
        float(returns.mean(dtype=np.float64)),
        float(returns.std(dtype=np.float64)),
    )


def approximate_kl_values(new_logprob: torch.Tensor, old_logprob: torch.Tensor) -> torch.Tensor:
    """返回每条样本的 PPO 近似 KL 估计。"""
    log_ratio = new_logprob - old_logprob
    ratio = log_ratio.exp()
    return (ratio - 1.0) - log_ratio


def categorical_kl_values(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """返回 KL(policy || 冻结 SFT reference),非法 -inf 位置按 0 处理。"""
    policy_logprob = F.log_softmax(policy_logits.float(), dim=-1)
    reference_logprob = F.log_softmax(reference_logits.float(), dim=-1)
    finite = torch.isfinite(policy_logprob) & torch.isfinite(reference_logprob)
    probability = torch.where(finite, policy_logprob.exp(), torch.zeros_like(policy_logprob))
    safe_policy = torch.where(finite, policy_logprob, torch.zeros_like(policy_logprob))
    safe_reference = torch.where(finite, reference_logprob, torch.zeros_like(reference_logprob))
    return (probability * (safe_policy - safe_reference)).sum(-1)


def policy_entropy_values(
    logprobabilities: torch.Tensor,
    probabilities: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按合法动作掩码计算 raw 与 normalized 策略熵。

    normalized = H / log(max(num_legal_actions, 2))。非法动作 logits 为
    -inf,必须在乘法前把 log_prob 与概率都替换为 0;仅在乘积上 masked_fill
    只救前向,反向仍会因 -inf × 0 回传 NaN 梯度。
    """
    safe_logprobabilities = torch.where(
        legal_mask, logprobabilities, torch.zeros_like(logprobabilities),
    )
    safe_probabilities = torch.where(
        legal_mask, probabilities, torch.zeros_like(probabilities),
    )
    entropy_values = -(safe_logprobabilities * safe_probabilities).sum(-1)
    legal_action_counts = legal_mask.sum(-1).float().clamp_min(2.0)
    normalized_entropy_values = entropy_values / legal_action_counts.log()
    return entropy_values, normalized_entropy_values


def transition_length_metrics(
    transitions: RolloutBuffer,
    prefix: str = "update/buffer",
) -> dict[str, float]:
    """描述 V18 语义序列长度与全量 padding 基线。"""
    lengths = np.asarray(transitions.sequence_lengths, dtype=np.int64)
    if np.any(lengths < 0):
        raise ValueError("sequence length cannot be negative")
    global_padded_input_tokens = int(lengths.max()) * len(transitions)
    effective_input_tokens = int(lengths.sum())
    return {
        f"{prefix}_transition_tokens_mean": float(lengths.mean()),
        f"{prefix}_transition_input_tokens_mean": float(lengths.mean()),
        f"{prefix}_transition_input_tokens_max": float(lengths.max()),
        f"{prefix}_effective_input_tokens": float(effective_input_tokens),
        f"{prefix}_global_padded_input_tokens": float(global_padded_input_tokens),
        f"{prefix}_global_padding_input_tokens": float(global_padded_input_tokens - effective_input_tokens),
        f"{prefix}_global_padding_fraction_of_padded_input_tokens": float(
            (global_padded_input_tokens - effective_input_tokens) / max(global_padded_input_tokens, 1)
        ),
    }


def rollout_target_metrics(
    transitions: RolloutBuffer,
) -> dict[str, float]:
    """全量 rollout 的序列长度与 advantage 统计(host 侧,供 learner 与汇总共用)。"""
    length_metrics = transition_length_metrics(transitions)
    advantages = np.asarray(transitions.advantages, dtype=np.float64)
    values = np.asarray(transitions.values, dtype=np.float64)
    length_metrics.update({
        "buffer/advantage_mean": float(advantages.mean()),
        "buffer/advantage_std": float(advantages.std()),
        "buffer/value_mean": float(values.mean()),
        "buffer/value_std": float(values.std()),
    })
    raw_advantages = _rollout_values(
        transitions, "advantages", np.dtype(np.float64),
    )
    length_metrics.update({
        "advantage_mean": float(raw_advantages.mean()),
        "advantage_std": float(raw_advantages.std()),
    })
    return length_metrics


def transfer_batch_to_device(
    host_batch: dict[str, torch.Tensor],
    device: torch.device,
    profiler: StageProfiler | None = None,
) -> dict[str, torch.Tensor]:
    """把 CPU minibatch 传输到 learner 设备。"""
    profile = profiler or StageProfiler(enabled=False)
    with profile.stage("update/collate_h2d"):
        return {
            name: value.to(device=device)
            for name, value in host_batch.items()
        }


class PPOLearner:
    """V18 Actor-Critic 的 PPO 优化器(单卡或双卡 DDP)。"""

    def __init__(
        self,
        model_size: str,
        device: str,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        **hyperparameters: Any,
    ) -> None:
        if model_size != "v18":
            raise ValueError("PPOLearner only supports model_size='v18'")
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = int(world_size) if world_size is not None else 1
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.world_size > 1:
            if self.rank is None:
                raise ValueError("rank is required when world_size > 1")
            if self.device.type != "cuda":
                raise ValueError("distributed PPO learner requires a CUDA device")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.hp = hyperparameters
        preset = ModelConfig.preset("v18")
        self.config = replace(
            preset,
            context_tokens=int(hyperparameters.get("context_tokens", preset.context_tokens)),
            critic_layers=int(hyperparameters.get("critic_layers", preset.critic_layers)),
        )
        self.model = KyokuTransformerActorCritic(self.config).to(self.device)
        requested_bf16 = str(hyperparameters.get("inference_dtype", "bf16")).lower() == "bf16"
        self.use_bf16 = bool(
            requested_bf16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        parameter_groups: dict[str, list[nn.Parameter]] = {
            "shared": [], "actor": [], "critic": [],
        }
        for name, parameter in self.model.named_parameters():
            root = name.split(".", 1)[0]
            if root in ACTOR_ROOTS:
                parameter_groups["actor"].append(parameter)
            elif root in CRITIC_ROOTS:
                parameter_groups["critic"].append(parameter)
            elif root in SHARED_ROOTS:
                parameter_groups["shared"].append(parameter)
            else:
                raise ValueError(f"unclassified optimizer parameter: {name}")
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": parameters,
                    "branch": branch,
                    "lr": float(
                        hyperparameters.get(
                            f"{branch}_learning_rate", hyperparameters["learning_rate"]
                        )
                    ),
                }
                for branch, parameters in parameter_groups.items()
            ],
            lr=float(hyperparameters["learning_rate"]),
            betas=(
                float(hyperparameters.get("adam_beta1", 0.9)),
                float(hyperparameters.get("adam_beta2", 0.999)),
            ),
            eps=float(hyperparameters.get("adam_epsilon", 1e-5)),
            weight_decay=float(hyperparameters.get("weight_decay", 0.01)),
            fused=self.use_bf16,
        )
        self.branch_parameters = optimizer_branch_parameters(self.optimizer)
        self.model_ddp: DistributedDataParallel | None = (
            DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                broadcast_buffers=False,
                # 移除 Q 后必须允许未使用参数:bootstrap 期只训 critic(actor/
                # shared 不进 loss);DDP
                # 会在每次 backward 额外扫描未使用参数,模型仅约 3M 参数,
                # 开销可忽略。
                find_unused_parameters=True,
            )
            if self.world_size > 1
            else None
        )
        self.reference_model: KyokuTransformerActorCritic | None = None
        self.iteration = 0
        self.profiler = StageProfiler(enabled=bool(hyperparameters.get("profile_enabled", True)))
        self.profile_cuda_sync = bool(hyperparameters.get("profile_cuda_sync", False))
        self.update_batch_mode = str(hyperparameters.get("update_batch_mode", "streaming")).lower()
        if self.update_batch_mode not in {"streaming", "auto"}:
            raise ValueError("update_batch_mode must be one of streaming or auto")
        self.value_target_normalization = str(
            hyperparameters.get("value_target_normalization", "batch_std")
        ).lower()
        if self.value_target_normalization not in {"none", "batch_std"}:
            raise ValueError("value_target_normalization must be one of 'none' or 'batch_std'")
        self.value_target_std_floor = float(hyperparameters.get("value_target_std_floor", 1e-2))
        if self.value_target_std_floor <= 0:
            raise ValueError("value_target_std_floor must be positive")
        self.critic_public_grad_scale = float(hyperparameters.get("critic_public_grad_scale", 1.0))
        if not 0.0 <= self.critic_public_grad_scale <= 1.0:
            raise ValueError("critic_public_grad_scale must be in [0, 1]")
        self.critic_private_embedding_grad_scale = float(
            hyperparameters.get("critic_private_embedding_grad_scale", 1.0)
        )
        if not 0.0 <= self.critic_private_embedding_grad_scale <= 1.0:
            raise ValueError("critic_private_embedding_grad_scale must be in [0, 1]")
        self.branch_max_grad_norms = {
            "actor": float(hyperparameters["actor_max_grad_norm"]),
            "shared": float(hyperparameters["shared_max_grad_norm"]),
            "critic": float(hyperparameters["critic_max_grad_norm"]),
        }
        if any(value <= 0.0 for value in self.branch_max_grad_norms.values()):
            raise ValueError("branch max grad norms must be positive")
        self.entropy_loss_mode = str(
            hyperparameters["entropy_loss_mode"]
        ).lower()
        if self.entropy_loss_mode != "normalized":
            raise ValueError("V18 PPO requires entropy_loss_mode=normalized")
        self.gradient_accumulation_steps = max(
            1, int(hyperparameters.get("gradient_accumulation_steps", 1))
        )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")

    def _sync_cuda(self) -> None:
        if self.profile_cuda_sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def _gpu_stage(self, name: str):
        self._sync_cuda()
        with self.profiler.stage(name):
            try:
                yield
            finally:
                self._sync_cuda()

    def weights(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.model.state_dict().items()}

    def rng_state(self) -> dict[str, Any]:
        """当前 rank 的 Python/numpy/torch/CUDA RNG 状态(供 checkpoint 保存)。"""
        return {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    def shutdown(self) -> None:
        """单卡 learner 无需额外清理(与双卡 LearnerDDP 的接口对齐)。"""

    def _model_forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        critic_bootstrap: bool,
    ) -> dict[str, torch.Tensor]:
        """统一走 ``__call__`` 分发,DDP 包装后 forward 才能触发梯度同步。"""
        model = self.model_ddp if self.model_ddp is not None else self.model
        return model(
            actor_factors=batch["actor_factors"],
            actor_numeric=batch["actor_numeric"],
            actor_lengths=batch["actor_lengths"],
            query_action_ids=batch["query_action_ids"],
            query_pair_counts=batch["query_pair_counts"],
            legal_mask=batch["legal_mask"],
            critic_factors=batch["critic_factors"],
            critic_lengths=batch["critic_lengths"],
            detach_critic_public=critic_bootstrap,
            critic_public_grad_scale=self.critic_public_grad_scale,
            critic_private_embedding_grad_scale=self.critic_private_embedding_grad_scale,
        )

    def update(
        self,
        transitions: RolloutBuffer,
        *,
        shuffle_seed: int | None = None,
        advantages: np.ndarray | None = None,
        returns: np.ndarray | None = None,
    ) -> dict[str, float]:
        if not isinstance(transitions, RolloutBuffer):
            raise TypeError("PPOLearner.update requires a RolloutBuffer")
        if not transitions:
            raise ValueError("cannot update from an empty rollout")
        self.profiler.reset()
        length_metrics = rollout_target_metrics(transitions)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if advantages is None or returns is None:
            if (advantages is None) != (returns is None):
                raise ValueError("advantages and returns must be provided together")
            with self.profiler.stage("update/advantage_normalize"):
                advantages, returns, return_mean, return_std = rollout_update_targets(
                    transitions, float(self.hp.get("gamma", 1.0)),
                )
        else:
            advantages = np.asarray(advantages, dtype=np.float32)
            returns = np.asarray(returns, dtype=np.float32)
            if (
                advantages.shape != (len(transitions),)
                or returns.shape != (len(transitions),)
            ):
                raise ValueError(
                    "provided advantages/returns must have one value per transition"
                )
            return_mean = float(returns.mean(dtype=np.float64))
            return_std = float(returns.std(dtype=np.float64))

        mc_returns = discounted_empirical_returns(
            transitions, float(self.hp.get("gamma", 1.0)),
        )
        raw_advantages = _rollout_values(transitions, "advantages", np.dtype(np.float64))
        raw_values = _rollout_values(
            transitions, "values", np.dtype(np.float64),
        )
        lambda_var = float(np.var(returns, dtype=np.float64))
        mc_var = float(np.var(mc_returns, dtype=np.float64))
        lambda_ev = (
            0.0
            if lambda_var <= 1e-12
            else 1.0 - float(np.var(returns - raw_values, dtype=np.float64)) / lambda_var
        )
        mc_ev = (
            0.0
            if mc_var <= 1e-12
            else 1.0 - float(np.var(mc_returns - raw_values, dtype=np.float64)) / mc_var
        )
        length_metrics.update({
            "value_explained_variance": lambda_ev,
            "value_explained_variance_lambda": lambda_ev,
            "value_explained_variance_mc": mc_ev,
            "raw_advantage_mean": float(raw_advantages.mean(dtype=np.float64)),
            "raw_advantage_std": float(raw_advantages.std(dtype=np.float64)),
            "normalized_advantage_mean": float(advantages.mean(dtype=np.float64)),
            "normalized_advantage_std": float(advantages.std(dtype=np.float64)),
            "lambda_return_mean": float(returns.mean(dtype=np.float64)),
            "lambda_return_std": float(returns.std(dtype=np.float64)),
            "mc_return_mean": float(mc_returns.mean(dtype=np.float64)),
            "mc_return_std": float(mc_returns.std(dtype=np.float64)),
        })

        # 每个 epoch / minibatch 复用同一份紧凑缓冲,避免恢复 Transition 对象。
        transitions.advantages = np.asarray(advantages, dtype=np.float32)

        metric_sample_sums: dict[str, torch.Tensor] = {}
        metric_sample_count = 0
        step_metric_totals: dict[str, torch.Tensor] = {}
        ratio_samples: list[torch.Tensor] = []
        updates = 0
        optimizer_steps = 0
        count = len(transitions)
        configured_epochs = int(self.hp["update_epochs"])
        minibatch_size = int(self.hp["minibatch_size"])
        planned_minibatches_per_epoch = (count + minibatch_size - 1) // minibatch_size
        update_number = self.iteration + 1
        total_updates = int(self.hp.get("total_updates", self.hp.get("iterations", 1)))
        bootstrap_updates = max(0, int(self.hp.get("critic_bootstrap_updates", 0)))
        critic_bootstrap = update_number <= bootstrap_updates
        policy_update_number = max(0, update_number - bootstrap_updates)
        total_policy_updates = max(1, total_updates - bootstrap_updates)
        branch_bases = {
            "actor": float(self.hp.get("actor_learning_rate", self.hp["learning_rate"])),
            "shared": float(self.hp.get("shared_learning_rate", self.hp["learning_rate"])),
            "critic": float(self.hp.get("critic_learning_rate", self.hp["learning_rate"])),
        }
        # 各参数组可选的 LR 下限;未配置时沿用历史 Exp 风格衰减终点。
        branch_mins = {
            "actor": self.hp.get("actor_learning_rate_min"),
            "shared": self.hp.get("shared_learning_rate_min"),
            "critic": self.hp.get("critic_learning_rate_min"),
        }
        if critic_bootstrap:
            # critic 预热:先冻结 Actor/shared,只让 value 在特权输入上收敛,
            # 避免随机初始化的值函数扰动策略。
            branch_learning_rates = {
                "actor": 0.0,
                "shared": 0.0,
                "critic": float(
                    self.hp.get(
                        "critic_bootstrap_learning_rate",
                        self.hp.get("critic_learning_rate", self.hp["learning_rate"]),
                    )
                ),
            }
        else:
            branch_learning_rates = {
                branch: scheduled_learning_rate(
                    base,
                    policy_update_number,
                    total_policy_updates,
                    float(self.hp.get("warmup_fraction", 0.0)),
                    min_lr=None if branch_mins[branch] is None else float(branch_mins[branch]),
                )
                for branch, base in branch_bases.items()
            }
        for group in self.optimizer.param_groups:
            group["lr"] = branch_learning_rates[str(group["branch"])]
        if critic_bootstrap:
            entropy_coef = 0.0
        else:
            entropy_coef = scheduled_entropy_coefficient(
                float(self.hp["entropy_start"]),
                float(self.hp["entropy_end"]),
                policy_update_number,
                total_policy_updates,
                middle=float(self.hp["entropy_middle"]),
                middle_fraction=float(self.hp["entropy_middle_fraction"]),
            )
        if critic_bootstrap:
            sft_kl_coef = 0.0
        elif "sft_kl_coef_middle" in self.hp:
            sft_kl_coef = scheduled_u_coefficient(
                float(self.hp.get("sft_kl_coef_start", 0.0)),
                float(self.hp["sft_kl_coef_middle"]),
                float(self.hp.get("sft_kl_coef_end", 0.0)),
                policy_update_number,
                total_policy_updates,
                float(self.hp.get("sft_kl_middle_fraction", 0.5)),
            )
        else:
            sft_kl_coef = 0.0
        if sft_kl_coef > 0.0 and self.reference_model is None:
            raise RuntimeError(
                "SFT KL anchor is enabled but no frozen reference model was loaded; "
                "start PPO with --init-model or resume an anchored PPO checkpoint"
            )
        value_coef = float(self.hp.get("value_coef", 0.5))

        self.model.train()
        stop_early = False
        rank_seed = (
            None
            if shuffle_seed is None
            else int(shuffle_seed) + int(self.rank or 0) * 1_000_003
        )
        rng = np.random.default_rng(rank_seed) if rank_seed is not None else None
        epochs_started = 0
        epochs_completed = 0
        executed_samples = 0
        executed_tokens = 0
        executed_padded_input_tokens = 0
        running_kl_sum: torch.Tensor | None = None
        running_kl_count = 0
        for _ in range(configured_epochs):
            epochs_started += 1
            with self.profiler.stage("update/length_bucket"):
                minibatches = transitions.bucketed_minibatches(
                    minibatch_size,
                    rng=rng,
                    bucket_window_multiplier=int(
                        self.hp.get("bucket_window_multiplier", 1)
                    ),
                )
            for batch_number, indices in enumerate(minibatches, start=1):
                with self.profiler.stage("update/collate_soa_gather"):
                    host_batch = transitions.collate(indices)
                batch = transfer_batch_to_device(host_batch, self.device, self.profiler)
                legal_mask = batch["legal_mask"]
                actions = batch["actions"]
                old_logprobs = batch["old_logprobs"]
                adv = batch["advantages"]
                batch_returns = torch.as_tensor(returns[indices], device=self.device)
                executed_samples += len(indices)
                executed_tokens += int(transitions.sequence_lengths[indices].sum())
                executed_padded_input_tokens += int(
                    len(indices) * int(transitions.sequence_lengths[indices].max())
                )
                with self._gpu_stage("update/model_forward"):
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_bf16,
                    ):
                        output = self._model_forward(
                            batch, critic_bootstrap=critic_bootstrap,
                        )
                        reference_output = None
                        if sft_kl_coef > 0.0:
                            assert self.reference_model is not None
                            with torch.no_grad():
                                reference_output = self.reference_model(
                                    batch["actor_factors"],
                                    batch["actor_numeric"],
                                    batch["actor_lengths"],
                                    batch["query_action_ids"],
                                    batch["query_pair_counts"],
                                    legal_mask,
                                    policy_only=True,
                                )
                        # PPO 数值路径离开模型前统一提升为 FP32。
                        logits = output["policy_logits"].float()
                        logprobabilities = F.log_softmax(logits, dim=-1)
                        probabilities = logprobabilities.exp()
                with self._gpu_stage("update/distribution_and_loss"):
                    logprob = logprobabilities.gather(1, actions[:, None]).squeeze(1)
                    old_logprobs = old_logprobs.float()
                    adv = adv.float()
                    ratio = (logprob - old_logprobs).exp()
                    clipped = ratio.clamp(
                        1 - float(self.hp["ppo_clip"]), 1 + float(self.hp["ppo_clip"]),
                    ) * adv
                    policy_loss_values = -torch.minimum(ratio * adv, clipped)
                    value = output["value"].float()
                    normalized_value, normalized_returns = normalize_value_targets(
                        value,
                        batch_returns,
                        mode=self.value_target_normalization,
                        mean=return_mean,
                        std=return_std,
                        std_floor=self.value_target_std_floor,
                    )
                    value_loss_values_ = value_loss_values(
                        normalized_value, normalized_returns, str(self.hp.get("value_loss", "huber")),
                    )
                    raw_value_loss = value_loss_values(
                        value, batch_returns, str(self.hp.get("value_loss", "huber")),
                    )
                    # 非法动作 logits 为 -inf:必须在乘法前把 log_prob 与概率
                    # 都替换为 0。仅在乘积上 masked_fill 只救前向,反向仍会因
                    # -inf × 0 回传 NaN 梯度。
                    entropy_values, normalized_entropy_values = policy_entropy_values(
                        logprobabilities, probabilities, legal_mask,
                    )
                    if reference_output is None:
                        sft_reference_kl_values = torch.zeros_like(policy_loss_values)
                    else:
                        sft_reference_kl_values = categorical_kl_values(
                            output["policy_logits"],
                            reference_output["policy_logits"],
                        )
                    if critic_bootstrap:
                        # 预热期只训 critic:policy/entropy/KL 不进入损失,
                        # actor/shared 学习率同时为 0。
                        entropy_loss_values = entropy_values
                        loss = value_coef * value_loss_values_.mean()
                    else:
                        entropy_loss_values = normalized_entropy_values
                        loss = (
                            policy_loss_values.mean()
                            + value_coef * value_loss_values_.mean()
                            - entropy_coef * entropy_loss_values.mean()
                            + sft_kl_coef * sft_reference_kl_values.mean()
                        )
                    evaluated_loss = loss
                loss_is_finite = torch.isfinite(evaluated_loss)
                loss_detail = (
                    f"policy={float(policy_loss_values.mean())} "
                    f"value={float(value_loss_values_.mean())} "
                    f"entropy={float(entropy_values.mean())} "
                    f"sft_kl={float(sft_reference_kl_values.mean())}"
                )
                if self.world_size == 1 and not loss_is_finite:
                    raise RuntimeError("non-finite PPO loss: " + loss_detail)
                with self._gpu_stage("update/backward"):
                    # Gradient accumulation:配置 >1 时,loss 除以累积步数后
                    # 反向,只在最后一累积步执行梯度裁剪 + optimizer.step,
                    # 保持 global effective batch ≈ minibatch_size × 累积步数。
                    accumulation_steps = self.gradient_accumulation_steps
                    planned_minibatches = (
                        configured_epochs * planned_minibatches_per_epoch
                    )
                    actual_group_size = accumulation_group_size(
                        planned_minibatches,
                        accumulation_steps,
                        updates,
                    )
                    scaled_loss = evaluated_loss / float(max(actual_group_size, 1))
                    scaled_loss.backward()
                    # ``updates`` 为累计已处理 minibatch 数(从 0 开始),
                    # 每组 accumulation_steps 的最后一步才 step;最后一个
                    # minibatch 无论是否整组都强制 step,避免挂起梯度丢失。
                    step_within_group = (
                        (updates + 1) % accumulation_steps == 0
                        if accumulation_steps > 1
                        else True
                    )
                    is_last_batch = updates + 1 == planned_minibatches
                    should_step = (
                        accumulation_steps <= 1
                        or step_within_group
                        or is_last_batch
                    )
                if self.world_size > 1:
                    # 必须等所有 rank 完成 backward 后再做全局有限性判定,
                    # 避免单 rank 提前异常导致 NCCL collective 失配。
                    finite = torch.tensor(
                        float(loss_is_finite), device=self.device,
                    )
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if finite.item() == 0.0:
                        raise RuntimeError(
                            "non-finite PPO loss on one of the DDP ranks: "
                            + loss_detail
                        )
                if should_step:
                    with self._gpu_stage("update/gradient_clip"):
                        branch_clip_metrics = clip_branch_grad_norms(
                            self.branch_parameters,
                            self.branch_max_grad_norms,
                        )
                    with self._gpu_stage("update/optimizer_step"):
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                else:
                    parameter_device = next(self.model.parameters()).device
                    branch_clip_metrics = {
                        f"grad_norm_{branch}_{suffix}": torch.zeros(
                            (), device=parameter_device,
                        )
                        for branch in ("actor", "shared", "critic")
                        for suffix in ("pre_clip", "post_clip", "clip_scale", "clipped")
                    }
                    for branch in ("actor", "shared", "critic"):
                        branch_clip_metrics[f"grad_norm_{branch}_clip_scale"] = torch.ones(
                            (), device=parameter_device,
                        )
                with self._gpu_stage("update/diagnostic_metrics"):
                    with torch.no_grad():
                        kl_values = approximate_kl_values(logprob, old_logprobs)
                        clipfrac_values = (
                            (ratio - 1).abs() > float(self.hp["ppo_clip"])
                        ).float()
                        if float(self.hp["target_kl"]) > 0 and not critic_bootstrap:
                            kl_sum = kl_values.sum()
                            if running_kl_sum is None:
                                running_kl_sum = kl_sum.detach().clone()
                            else:
                                running_kl_sum.add_(kl_sum.detach())
                            running_kl_count += len(indices)
                            interval = max(
                                1,
                                int(self.hp.get("target_kl_check_interval", 0)) or 1,
                            )
                            if should_step and optimizer_steps % interval == 0:
                                if self.world_size > 1:
                                    kl_total = running_kl_sum.detach().clone()
                                    kl_count = torch.tensor(
                                        float(running_kl_count), device=self.device,
                                    )
                                    dist.all_reduce(kl_total, op=dist.ReduceOp.SUM)
                                    dist.all_reduce(kl_count, op=dist.ReduceOp.SUM)
                                    checked_kl = kl_total / kl_count.clamp_min(1.0)
                                else:
                                    checked_kl = running_kl_sum / max(running_kl_count, 1)
                                if float(checked_kl) > float(self.hp["target_kl"]):
                                    stop_early = True
                for name, values in (
                    (
                        "loss",
                        (
                            value_coef * value_loss_values_
                            if critic_bootstrap
                            else (
                                policy_loss_values
                                + value_coef * value_loss_values_
                                - entropy_coef * entropy_loss_values
                                + sft_kl_coef * sft_reference_kl_values
                            )
                        ),
                    ),
                    ("policy_loss", policy_loss_values),
                    ("value_loss", value_loss_values_),
                    ("value_loss_raw", raw_value_loss),
                    ("value_prediction", value),
                    ("entropy", entropy_values),
                    ("entropy_normalized", normalized_entropy_values),
                    ("sft_reference_kl", sft_reference_kl_values),
                    ("approx_kl", kl_values),
                    ("clipfrac", clipfrac_values),
                    ("ratio", ratio),
                ):
                    detached = values.detach().sum()
                    if name in metric_sample_sums:
                        metric_sample_sums[name].add_(detached)
                    else:
                        metric_sample_sums[name] = detached.clone()
                metric_sample_count += len(indices)
                if len(indices):
                    ratio_samples.append(ratio.detach())
                if should_step:
                    for name, value in branch_clip_metrics.items():
                        detached_value = value.detach()
                        if name in step_metric_totals:
                            step_metric_totals[name].add_(detached_value)
                        else:
                            step_metric_totals[name] = detached_value.clone()
                    pre_values = [
                        branch_clip_metrics[f"grad_norm_{branch}_pre_clip"]
                        for branch in ("actor", "shared", "critic")
                    ]
                    post_values = [
                        branch_clip_metrics[f"grad_norm_{branch}_post_clip"]
                        for branch in ("actor", "shared", "critic")
                    ]
                    for name, value in (
                        ("grad_norm", torch.stack(pre_values).square().sum().sqrt()),
                        ("grad_norm_post_clip", torch.stack(post_values).square().sum().sqrt()),
                    ):
                        detached_value = value.detach()
                        if name in step_metric_totals:
                            step_metric_totals[name].add_(detached_value)
                        else:
                            step_metric_totals[name] = detached_value.clone()
                    for branch in ("actor", "shared", "critic"):
                        name = f"grad_norm_{branch}"
                        detached_value = branch_clip_metrics[
                            f"grad_norm_{branch}_pre_clip"
                        ].detach()
                        if name in step_metric_totals:
                            step_metric_totals[name].add_(detached_value)
                        else:
                            step_metric_totals[name] = detached_value.clone()
                updates += 1
                if stop_early:
                    break
            if stop_early:
                break
            epochs_completed += 1
        self.iteration += 1
        if not executed_samples:
            raise RuntimeError("PPO update completed without a minibatch")
        length_metrics.update({
            "update/configured_epochs": float(configured_epochs),
            "update/epochs_started": float(epochs_started),
            "update/epochs_completed": float(epochs_completed),
            "update/planned_minibatches": float(configured_epochs * planned_minibatches_per_epoch),
            "update/executed_minibatches": float(updates),
            "update/early_stop": float(stop_early),
            "update/executed_transition_samples": float(executed_samples),
            "update/executed_transition_tokens_mean": float(executed_tokens / executed_samples),
            "update/executed_transition_input_tokens_mean": float(executed_tokens / executed_samples),
            "update/executed_padded_input_tokens": float(executed_padded_input_tokens),
            "update/executed_padding_input_tokens": float(
                executed_padded_input_tokens - executed_tokens
            ),
            "update/executed_padding_fraction_of_padded_input_tokens": float(
                (executed_padded_input_tokens - executed_tokens) / max(executed_padded_input_tokens, 1)
            ),
            "system/learning_rate": float(branch_learning_rates["actor"]),
            "system/actor_learning_rate": float(branch_learning_rates["actor"]),
            "system/shared_learning_rate": float(branch_learning_rates["shared"]),
            "system/critic_learning_rate": float(branch_learning_rates["critic"]),
            "system/entropy_coef": float(entropy_coef),
            "system/sft_kl_coef": float(sft_kl_coef),
            "system/critic_public_grad_scale": float(
                0.0 if critic_bootstrap else self.critic_public_grad_scale
            ),
            "system/critic_private_embedding_grad_scale": float(
                self.critic_private_embedding_grad_scale
            ),
            "training/critic_bootstrap": float(critic_bootstrap),
            "training/policy_update": float(policy_update_number),
        })
        sample_count_tensor = torch.tensor(float(metric_sample_count), device=self.device)
        sample_names = tuple(metric_sample_sums)
        sample_values = torch.stack([metric_sample_sums[name] for name in sample_names]).div(
            sample_count_tensor.clamp_min(1.0)
        ).tolist()
        step_names = tuple(step_metric_totals)
        step_values = torch.stack(
            [step_metric_totals[name] for name in step_names]
        ).div(max(optimizer_steps, 1)).tolist()
        result = (
            dict(zip(sample_names, sample_values))
            | dict(zip(step_names, step_values))
            | {"transitions": float(count)}
            | length_metrics
        )
        if ratio_samples:
            result["ratio_p95"] = float(
                torch.quantile(torch.cat(ratio_samples), 0.95).item()
            )
        result.update(self.profiler.delta({}, prefix="timing"))
        if self.device.type == "cuda":
            result.update({
                "gpu/torch_memory_allocated_mb": float(torch.cuda.memory_allocated(self.device) / 2**20),
                "gpu/torch_memory_reserved_mb": float(torch.cuda.memory_reserved(self.device) / 2**20),
                "gpu/torch_memory_peak_allocated_mb": float(torch.cuda.max_memory_allocated(self.device) / 2**20),
                "gpu/torch_memory_peak_reserved_mb": float(torch.cuda.max_memory_reserved(self.device) / 2**20),
            })
        return result

    def save(
        self,
        path: str | Path,
        train_config: dict[str, Any],
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        """原子写 checkpoint:model + optimizer + model_config(+ RNG/iteration)。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        merged_extra = dict(extra_state or {})
        # 单卡路径也保存 rank_rng_states,使旧格式与双卡 resume 共享同一入口。
        merged_extra.setdefault("rank_rng_states", [self.rng_state()])
        payload = {
            "ppo_format_version": 4,
            "model": self.weights(),
            "optimizer": self.optimizer.state_dict(),
            "model_config": asdict(self.config),
            "train_config": train_config,
            "iteration": self.iteration,
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "extra_state": merged_extra,
        }
        if self.reference_model is not None:
            payload["sft_reference_model"] = {
                name: value.detach().cpu()
                for name, value in self.reference_model.state_dict().items()
            }
        destination = Path(path)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    def load(self, path: str | Path) -> None:
        """精确 resume:校验 V18 契约后恢复 model/optimizer/iteration/RNG。"""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        required = {
            "model", "optimizer", "model_config", "iteration", "torch_rng",
            "cuda_rng", "python_rng", "numpy_rng", "token_schema_version",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise RuntimeError("PPO exact resume checkpoint is missing: " + ", ".join(missing))
        if int(payload.get("ppo_format_version", 0)) != 4:
            raise RuntimeError("only V18 PPO checkpoints (format 4) can be resumed")
        if int(payload.get("token_schema_version", 0)) != TOKEN_SCHEMA_VERSION:
            raise RuntimeError(
                f"checkpoint token schema {payload.get('token_schema_version')} is "
                f"incompatible with required schema {TOKEN_SCHEMA_VERSION}"
            )
        try:
            checkpoint_config = ModelConfig.from_mapping(dict(payload["model_config"]))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("PPO checkpoint has an invalid model_config") from exc
        if checkpoint_config != self.config:
            raise RuntimeError(
                "PPO exact resume model_config differs from the active V18 topology"
            )
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload["iteration"])
        reference_state = payload.get("sft_reference_model")
        if reference_state is not None:
            self.reference_model = KyokuTransformerActorCritic(self.config).to(self.device)
            self.reference_model.load_state_dict(reference_state, strict=True)
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()
        elif float(self.hp.get("sft_kl_coef_start", 0.0)) > 0.0:
            raise RuntimeError(
                "PPO checkpoint does not contain the frozen SFT reference required "
                "by the configured KL anchor"
            )
        rank_states = (payload.get("extra_state") or {}).get("rank_rng_states")
        if (
            self.rank is not None
            and isinstance(rank_states, list)
            and len(rank_states) == self.world_size
            and all(isinstance(item, dict) for item in rank_states)
        ):
            # 双卡 checkpoint:每个 rank 恢复自己的 RNG。
            state = rank_states[self.rank]
            torch.set_rng_state(state["torch"].cpu())
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            if state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(state["cuda"].cpu(), device=self.device)
        else:
            # 单卡 checkpoint:两个 rank 都恢复同一份 RNG(继续作为双卡训练)。
            torch.set_rng_state(payload["torch_rng"].cpu())
            random.setstate(payload["python_rng"])
            np.random.set_state(payload["numpy_rng"])
            if payload["cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([
                    state.cpu() for state in payload["cuda_rng"]
                ])

    def load_model_weights(self, path: str | Path) -> None:
        """从 V18 SFT checkpoint 初始化全新 PPO(iteration 归零、optimizer 全新)。"""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        validate_fresh_model_checkpoint_contract(payload)
        checkpoint_config = ModelConfig.from_mapping(dict(payload["model_config"]))
        if checkpoint_config != self.config:
            raise RuntimeError(
                "V18 SFT checkpoint model_config differs from the active PPO topology"
            )
        self.model.load_state_dict(payload["model"], strict=True)
        self.reference_model = KyokuTransformerActorCritic(self.config).to(self.device)
        self.reference_model.load_state_dict(self.model.state_dict(), strict=True)
        self.reference_model.requires_grad_(False)
        self.reference_model.eval()
        self.iteration = 0
