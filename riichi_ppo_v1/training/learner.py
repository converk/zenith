"""V16 PPO 优化器:V16 批 collate、PPO clip + value Huber + Top-3 Q loss。"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..model import KyokuTransformerActorCritic, ModelConfig
from ..model.schema import NUM_ACTIONS, TOKEN_SCHEMA_VERSION
from .metrics import ppo_buffer_metrics
from .profiling import StageProfiler
from .trajectory import Transition, transition_sequence_length

# V16 参数分支:Q loss 只经 critic_hidden 更新 q_scorer/critic,动作表示已在
# q_scores_v16 内 detach,保证 Q loss 不直接更新 Actor 分支。
ACTOR_ROOTS = {"actor_backbone", "query_embedding", "action_fusion", "policy_mlp"}
CRITIC_ROOTS = {"critic_embedding", "critic_backbone", "value_head", "value_query", "q_scorer"}
SHARED_ROOTS = {"token_embedding", "snapshot_embeddings", "public_backbone"}


def select_top3_candidates(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    behavior_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-3 Q-boosting 的候选集(契约 reward-v16.md §4)。

    - boost 候选 = 合法动作中的 Top-3;
    - 训练候选 = Top-3 ∪ 实际 rollout 行为动作(最多 4 个,去重)。
    返回 ``(boost_ids [B,3], training_ids [B,4])``,右侧以 -1 补齐。
    """
    batch = policy_logits.shape[0]
    if policy_logits.ndim != 2 or legal_mask.shape != policy_logits.shape:
        raise ValueError("policy_logits and legal_mask must be [batch, actions]")
    if behavior_actions.shape != (batch,):
        raise ValueError("behavior_actions must be [batch]")
    masked = policy_logits.masked_fill(~legal_mask, float("-inf"))
    top_count = min(3, int(legal_mask.sum(dim=1).min()))
    top3 = masked.topk(top_count, dim=-1).indices
    boost_ids = top3.new_full((batch, 3), -1)
    boost_ids[:, :top_count] = top3
    training_ids = top3.new_full((batch, 4), -1)
    training_ids[:, :top_count] = top3
    training_ids[:, top_count] = behavior_actions
    cleaned = top3.new_full((batch, 4), -1)
    for row in range(batch):
        unique: list[int] = []
        seen: set[int] = set()
        for value in training_ids[row].tolist():
            if int(value) < 0 or int(value) in seen:
                continue
            seen.add(int(value))
            unique.append(int(value))
        cleaned[row, : len(unique)] = top3.new_tensor(unique)
    return boost_ids, cleaned


def candidate_q_loss(
    q_scores: torch.Tensor,
    q_targets: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """训练候选的 Q 回归(Huber);无效候选不参与。"""
    if q_scores.shape != q_targets.shape or q_scores.shape != candidate_valid.shape:
        raise ValueError("q_scores/q_targets/candidate_valid shapes differ")
    if not bool(candidate_valid.any()):
        return q_scores.new_zeros(())
    return F.huber_loss(
        q_scores[candidate_valid],
        q_targets[candidate_valid],
        reduction="mean",
    )


def validate_fresh_model_checkpoint_contract(payload: dict[str, Any]) -> None:
    """校验用作 V16 PPO 初始化的 SFT/模型 checkpoint 契约。"""
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("V16 PPO requires a checkpoint with model weights")
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict) or raw_config.get("policy_head_type") != "symmetric_action_query":
        raise RuntimeError("V16 PPO requires a symmetric_action_query model checkpoint")


def scheduled_learning_rate(base: float, update: int, total_updates: int, warmup_fraction: float) -> float:
    """Exp 风格 warmup 后线性衰减,采用 1-based updates。"""
    total = max(1, int(total_updates))
    step = max(0, int(update))
    warmup = int(total * float(warmup_fraction))
    if warmup > 0 and step <= warmup:
        return float(base) * float(step) / float(warmup)
    decay_updates = max(1, total - warmup)
    return float(base) * max(0.0, float(total - step + 1) / float(decay_updates))


def scheduled_entropy_coefficient(start: float, end: float, update: int, total_updates: int) -> float:
    """entropy 系数从 start 线性退火到 end。"""
    total = max(1, int(total_updates))
    progress = min(max(float(update) / float(total), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * progress


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


def branch_grad_norms(model: nn.Module) -> dict[str, torch.Tensor]:
    """返回裁剪前的 actor/critic/shared 三组 L2 梯度范数。"""
    parameter_device = next(model.parameters()).device
    squared_sums = {
        "actor": torch.zeros((), device=parameter_device),
        "critic": torch.zeros((), device=parameter_device),
        "shared": torch.zeros((), device=parameter_device),
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        root = name.split(".", 1)[0]
        if root in ACTOR_ROOTS:
            branch = "actor"
        elif root in CRITIC_ROOTS:
            branch = "critic"
        elif root in SHARED_ROOTS:
            branch = "shared"
        else:  # 防止未来新增参数被静默漏报。
            raise ValueError(f"unclassified model parameter for gradient metrics: {name}")
        squared_sums[branch].add_(parameter.grad.detach().float().square().sum())
    return {name: value.sqrt() for name, value in squared_sums.items()}


def discounted_empirical_returns(transitions: list[Transition], gamma: float) -> np.ndarray:
    """Monte Carlo reward-to-go,每局终局时重置。"""
    returns = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        if item.done:
            running = 0.0
        running = float(item.reward) + float(gamma) * running
        returns[index] = np.float32(running)
    return returns


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


def transition_length_metrics(transitions: list[Transition], prefix: str = "update/buffer") -> dict[str, float]:
    """描述 V16 语义序列长度与全量 padding 基线。"""
    lengths = np.asarray([transition_sequence_length(item) for item in transitions], dtype=np.int64)
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


def length_bucketed_minibatches(
    transitions: list[Transition], minibatch_size: int, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    """返回序列长度相近的随机顺序 minibatch,避免长尾 padding。"""
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    lengths = np.asarray([transition_sequence_length(item) for item in transitions], dtype=np.int64)
    if not len(lengths):
        raise ValueError("cannot bucket an empty rollout")
    if np.any(lengths < 0):
        raise ValueError("sequence length cannot be negative")
    permutation = rng.permutation if rng is not None else np.random.permutation
    shuffled = permutation(len(transitions))
    sorted_indices = shuffled[np.argsort(lengths[shuffled], kind="stable")]
    batches = tuple(
        sorted_indices[start : start + minibatch_size]
        for start in range(0, len(sorted_indices), minibatch_size)
    )
    batch_order = permutation(len(batches))
    return tuple(batches[index] for index in batch_order)


def materialize_host_batch(
    transitions: list[Transition],
    profiler: StageProfiler | None = None,
    *,
    advantages: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """把一个长度分桶 minibatch 按 V16 分段 padding 成 CPU 张量。"""
    if not transitions:
        raise ValueError("cannot collate an empty rollout")
    profile = profiler or StageProfiler(enabled=False)
    with profile.stage("update/collate_shape_and_allocate"):
        batch = len(transitions)
        max_history = max(item.history_length for item in transitions)
        max_snapshot = max(item.snapshot_length for item in transitions)
        max_pairs = max(item.query_pair_counts for item in transitions)
        max_critic = max(int(item.critic_length) for item in transitions)
        history_factors = torch.zeros((batch, max_history, 10), dtype=torch.uint8)
        history_numeric = torch.zeros((batch, max_history, 8), dtype=torch.float32)
        history_lengths = torch.empty(batch, dtype=torch.long)
        snapshot_kinds = torch.zeros((batch, max_snapshot), dtype=torch.uint8)
        snapshot_cat = torch.zeros((batch, max_snapshot, 4), dtype=torch.uint8)
        snapshot_num = torch.zeros((batch, max_snapshot, 7), dtype=torch.float32)
        snapshot_lengths = torch.empty(batch, dtype=torch.long)
        query_rows = torch.zeros((batch, 2 * max_pairs, 15), dtype=torch.int32)
        query_action_ids = torch.zeros((batch, max_pairs), dtype=torch.int32)
        query_pair_counts = torch.empty(batch, dtype=torch.long)
        legal = torch.empty((batch, NUM_ACTIONS), dtype=torch.bool)
        critic_factors = torch.zeros((batch, max_critic, 10), dtype=torch.uint8)
        critic_lengths = torch.empty(batch, dtype=torch.long)
        actions = torch.empty(batch, dtype=torch.long)
        old_logprobs = torch.empty(batch, dtype=torch.float32)
        q_targets = torch.empty(batch, dtype=torch.float32)
        advantage_values = torch.empty(batch, dtype=torch.float32)
    with profile.stage("update/collate_host_padding_copy"):
        source_advantages = (
            np.asarray([item.advantage for item in transitions], dtype=np.float32)
            if advantages is None
            else np.asarray(advantages, dtype=np.float32)
        )
        if source_advantages.shape != (batch,):
            raise ValueError("advantages must have one value per transition")
        for row, item in enumerate(transitions):
            # Ray 反序列化出的 numpy 数组只读,torch.as_tensor 会零拷贝包装并
            # 触发非可写警告;torch.tensor 总是拷贝,统一走这条安全路径。
            history_factors[row, : item.history_length] = torch.tensor(item.history_factors)
            history_numeric[row, : item.history_length] = torch.tensor(item.history_numeric)
            history_lengths[row] = int(item.history_length)
            snapshot_kinds[row, : item.snapshot_length] = torch.tensor(item.snapshot_kinds)
            snapshot_cat[row, : item.snapshot_length] = torch.tensor(item.snapshot_cat)
            snapshot_num[row, : item.snapshot_length] = torch.tensor(item.snapshot_num)
            snapshot_lengths[row] = int(item.snapshot_length)
            query_rows[row, : item.query_rows.shape[0]] = torch.tensor(item.query_rows)
            query_action_ids[row, : item.query_pair_counts] = torch.tensor(item.query_action_ids)
            query_pair_counts[row] = int(item.query_pair_counts)
            legal[row] = torch.tensor(item.legal_mask)
            critic_length = int(item.critic_length)
            if critic_length:
                if item.critic_factors is None:
                    raise ValueError("critic transition length requires critic arrays")
                critic_factors[row, :critic_length] = torch.tensor(item.critic_factors[:critic_length])
            critic_lengths[row] = critic_length
            actions[row] = int(item.action)
            old_logprobs[row] = float(item.logprob)
            q_targets[row] = float(item.q_target)
            advantage_values[row] = float(source_advantages[row])
    return {
        "history_factors": history_factors,
        "history_numeric": history_numeric,
        "history_lengths": history_lengths,
        "snapshot_kinds": snapshot_kinds,
        "snapshot_cat": snapshot_cat,
        "snapshot_num": snapshot_num,
        "snapshot_lengths": snapshot_lengths,
        "query_rows": query_rows,
        "query_action_ids": query_action_ids,
        "query_pair_counts": query_pair_counts,
        "legal_mask": legal,
        "critic_factors": critic_factors,
        "critic_lengths": critic_lengths,
        "actions": actions,
        "old_logprobs": old_logprobs,
        "advantages": advantage_values,
        "q_targets": q_targets,
    }


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


def collate(
    transitions: list[Transition],
    device: torch.device,
    profiler: StageProfiler | None = None,
    *,
    advantages: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Padding 一个长度分桶 minibatch 并转移到 learner 设备。"""
    host_batch = materialize_host_batch(transitions, profiler, advantages=advantages)
    return transfer_batch_to_device(host_batch, device, profiler)


class PPOLearner:
    """V16 Actor-Critic 的单卡 PPO 优化器。"""

    def __init__(self, model_size: str, device: str, **hyperparameters: Any) -> None:
        if model_size != "v16":
            raise ValueError("PPOLearner only supports model_size='v16'")
        self.device = torch.device(device)
        self.hp = hyperparameters
        preset = ModelConfig.preset("v16")
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

    def update(
        self,
        transitions: list[Transition],
        *,
        shuffle_seed: int | None = None,
    ) -> dict[str, float]:
        if not transitions:
            raise ValueError("cannot update from an empty rollout")
        self.profiler.reset()
        length_metrics = transition_length_metrics(transitions)
        length_metrics.update(ppo_buffer_metrics(transitions))
        raw_q_targets = np.asarray([item.q_target for item in transitions], dtype=np.float64)
        raw_q_taken = np.asarray([item.q_taken for item in transitions], dtype=np.float64)
        raw_expected_q = np.asarray([item.expected_q for item in transitions], dtype=np.float64)
        raw_advantages = np.asarray([item.advantage for item in transitions], dtype=np.float64)
        q_target_mean = float(raw_q_targets.mean())
        q_target_std = float(raw_q_targets.std())
        length_metrics.update({
            "q_target_mean": q_target_mean,
            "q_target_std": q_target_std,
            "q_taken_mean": float(raw_q_taken.mean()),
            "q_taken_std": float(raw_q_taken.std()),
            "expected_q_mean": float(raw_expected_q.mean()),
            "expected_q_std": float(raw_expected_q.std()),
            "qboost_advantage_mean": float(raw_advantages.mean()),
            "qboost_advantage_std": float(raw_advantages.std()),
            "q_explained_variance": float(length_metrics["q_explained_variance"]),
        })
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        source_advantages = np.asarray(
            [item.advantage for item in transitions], dtype=np.float32,
        )
        with self.profiler.stage("update/advantage_normalize"):
            advantages = (
                (source_advantages - source_advantages.mean(dtype=np.float64))
                / (source_advantages.std(dtype=np.float64) + 1e-8)
            ).astype(np.float32)
        returns = discounted_empirical_returns(
            transitions, float(self.hp.get("gamma", 1.0)),
        )
        return_mean = float(returns.mean(dtype=np.float64))
        return_std = float(returns.std(dtype=np.float64))

        metric_sample_sums: dict[str, torch.Tensor] = {}
        metric_sample_count = 0
        step_metric_totals: dict[str, torch.Tensor] = {}
        ratio_samples: list[torch.Tensor] = []
        updates = 0
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
        if critic_bootstrap:
            # critic 预热:先冻结 Actor/shared,只让 value + Q scorer 在特权
            # 输入上收敛,避免随机初始化的 Q 自举目标扰动策略。
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
                )
                for branch, base in branch_bases.items()
            }
        for group in self.optimizer.param_groups:
            group["lr"] = branch_learning_rates[str(group["branch"])]
        if critic_bootstrap:
            entropy_coef = 0.0
        else:
            entropy_coef = scheduled_entropy_coefficient(
                float(self.hp.get("entropy_start", self.hp.get("entropy_coef", 0.0))),
                float(self.hp.get("entropy_end", self.hp.get("entropy_coef", 0.0))),
                policy_update_number,
                total_policy_updates,
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
        q_coef = float(self.hp.get("q_coef", 1.0))

        self.model.train()
        stop_early = False
        rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
        epochs_started = 0
        epochs_completed = 0
        executed_samples = 0
        executed_tokens = 0
        executed_padded_input_tokens = 0
        for _ in range(configured_epochs):
            epochs_started += 1
            epoch_kl_sum: torch.Tensor | None = None
            epoch_kl_count = 0
            with self.profiler.stage("update/length_bucket"):
                minibatches = length_bucketed_minibatches(transitions, minibatch_size, rng=rng)
            for indices in minibatches:
                selected = [transitions[int(index)] for index in indices]
                batch = collate(
                    selected,
                    self.device,
                    self.profiler,
                    advantages=advantages[indices],
                )
                legal_mask = batch["legal_mask"]
                actions = batch["actions"]
                old_logprobs = batch["old_logprobs"]
                adv = batch["advantages"]
                q_targets = batch["q_targets"]
                batch_returns = torch.as_tensor(returns[indices], device=self.device)
                executed_samples += len(selected)
                executed_tokens += sum(transition_sequence_length(item) for item in selected)
                executed_padded_input_tokens += len(selected) * max(
                    transition_sequence_length(item) for item in selected
                )
                with self._gpu_stage("update/model_forward"):
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_bf16,
                    ):
                        output = self.model.forward_v16(
                            batch["history_factors"],
                            batch["history_numeric"],
                            batch["history_lengths"],
                            batch["snapshot_kinds"],
                            batch["snapshot_cat"],
                            batch["snapshot_num"],
                            batch["snapshot_lengths"],
                            batch["query_rows"],
                            batch["query_action_ids"],
                            batch["query_pair_counts"],
                            legal_mask,
                            critic_factors=batch["critic_factors"],
                            critic_lengths=batch["critic_lengths"],
                            detach_critic_public=critic_bootstrap,
                            critic_public_grad_scale=self.critic_public_grad_scale,
                        )
                        reference_output = None
                        if sft_kl_coef > 0.0:
                            assert self.reference_model is not None
                            with torch.no_grad():
                                reference_output = self.reference_model.forward_v16(
                                    batch["history_factors"],
                                    batch["history_numeric"],
                                    batch["history_lengths"],
                                    batch["snapshot_kinds"],
                                    batch["snapshot_cat"],
                                    batch["snapshot_num"],
                                    batch["snapshot_lengths"],
                                    batch["query_rows"],
                                    batch["query_action_ids"],
                                    batch["query_pair_counts"],
                                    legal_mask,
                                    policy_only=True,
                                )
                        # Q scorer 的输入 critic_hidden/action_hiddens 仍为 BF16,
                        # 必须在 autocast 内执行;其余 PPO 数值路径离开模型前已
                        # 提升为 FP32。
                        _boost_ids, training_ids = select_top3_candidates(
                            output["policy_logits"].float(), legal_mask, actions,
                        )
                        candidate_q = self.model.q_scores_v16(
                            output["critic_hidden"],
                            output["action_hiddens"],
                            batch["query_action_ids"],
                            batch["query_pair_counts"],
                            training_ids.long(),
                        )
                        q_valid = training_ids.ge(0) & torch.isfinite(candidate_q)
                        q_loss = candidate_q_loss(
                            candidate_q,
                            q_targets.float()[:, None].expand_as(candidate_q),
                            q_valid,
                        )
                with self._gpu_stage("update/distribution_and_loss"):
                    logits = output["policy_logits"].float()
                    logprobabilities = F.log_softmax(logits, dim=-1)
                    logprob = logprobabilities.gather(1, actions[:, None]).squeeze(1)
                    old_logprobs = old_logprobs.float()
                    adv = adv.float()
                    q_targets = q_targets.float()
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
                    probabilities = logprobabilities.exp()
                    safe_logprobabilities = torch.where(
                        legal_mask, logprobabilities, torch.zeros_like(logprobabilities),
                    )
                    safe_probabilities = torch.where(
                        legal_mask, probabilities, torch.zeros_like(probabilities),
                    )
                    entropy_values = -(safe_logprobabilities * safe_probabilities).sum(-1)
                    legal_action_counts = legal_mask.sum(-1).float().clamp_min(2.0)
                    normalized_entropy_values = entropy_values / legal_action_counts.log()
                    behavior_match = training_ids == actions[:, None]
                    q_prediction = torch.zeros(len(selected), device=self.device)
                    matched_rows = behavior_match.any(dim=1)
                    q_prediction[matched_rows] = candidate_q[behavior_match]
                    if reference_output is None:
                        sft_reference_kl_values = torch.zeros_like(policy_loss_values)
                    else:
                        sft_reference_kl_values = categorical_kl_values(
                            output["policy_logits"],
                            reference_output["policy_logits"],
                        )
                    if critic_bootstrap:
                        # 预热期只训 critic/Q:policy/entropy/KL 不进入损失,
                        # actor/shared 学习率同时为 0。
                        loss = (
                            value_coef * value_loss_values_.mean()
                            + q_coef * q_loss
                        )
                    else:
                        loss = (
                            policy_loss_values.mean()
                            + value_coef * value_loss_values_.mean()
                            + q_coef * q_loss
                            - entropy_coef * entropy_values.mean()
                            + sft_kl_coef * sft_reference_kl_values.mean()
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "non-finite PPO loss: "
                            f"policy={float(policy_loss_values.mean())} "
                            f"value={float(value_loss_values_.mean())} "
                            f"q={float(q_loss)} entropy={float(entropy_values.mean())} "
                            f"sft_kl={float(sft_reference_kl_values.mean())}"
                        )
                with self._gpu_stage("update/zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
                with self._gpu_stage("update/backward"):
                    loss.backward()
                with self._gpu_stage("update/gradient_clip"):
                    branch_norms = branch_grad_norms(self.model)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.hp["max_grad_norm"]),
                    )
                    grad_norm_post_clip = grad_norm.clamp(max=float(self.hp["max_grad_norm"]))
                with self._gpu_stage("update/optimizer_step"):
                    self.optimizer.step()
                with self._gpu_stage("update/diagnostic_metrics"):
                    with torch.no_grad():
                        kl_values = approximate_kl_values(logprob, old_logprobs)
                        clipfrac_values = (
                            (ratio - 1).abs() > float(self.hp["ppo_clip"])
                        ).float()
                        if float(self.hp["target_kl"]) > 0:
                            kl_sum = kl_values.sum()
                            if epoch_kl_sum is None:
                                epoch_kl_sum = kl_sum.detach().clone()
                            else:
                                epoch_kl_sum.add_(kl_sum.detach())
                            epoch_kl_count += len(selected)
                for name, values in (
                    (
                        "loss",
                        (
                            value_coef * value_loss_values_ + q_coef * q_loss
                            if critic_bootstrap
                            else (
                                policy_loss_values
                                + value_coef * value_loss_values_
                                + q_coef * q_loss
                                - entropy_coef * entropy_values
                                + sft_kl_coef * sft_reference_kl_values
                            )
                        ),
                    ),
                    ("policy_loss", policy_loss_values),
                    ("value_loss", value_loss_values_),
                    ("value_loss_raw", raw_value_loss),
                    ("value_prediction", value),
                    ("q_loss", q_loss.expand(len(selected))),
                    ("q_prediction", q_prediction),
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
                metric_sample_count += len(selected)
                if len(selected):
                    ratio_samples.append(ratio.detach())
                for name, value in (
                    ("grad_norm", grad_norm),
                    ("grad_norm_post_clip", grad_norm_post_clip),
                ):
                    detached = value.detach()
                    if name in step_metric_totals:
                        step_metric_totals[name].add_(detached)
                    else:
                        step_metric_totals[name] = detached.clone()
                for branch, value in branch_norms.items():
                    name = f"grad_norm_{branch}"
                    detached_value = value.detach()
                    if name in step_metric_totals:
                        step_metric_totals[name].add_(detached_value)
                    else:
                        step_metric_totals[name] = detached_value.clone()
                updates += 1
            epochs_completed += 1
            if float(self.hp["target_kl"]) > 0 and epoch_kl_sum is not None:
                epoch_kl = epoch_kl_sum / max(epoch_kl_count, 1)
                if float(epoch_kl) > float(self.hp["target_kl"]):
                    stop_early = True
                    break
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
        ).div(max(updates, 1)).tolist()
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
        payload = {
            "ppo_format_version": 3,
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
            "extra_state": dict(extra_state or {}),
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
        """精确 resume:校验 v16 契约后恢复 model/optimizer/iteration/RNG。"""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        required = {
            "model", "optimizer", "model_config", "iteration", "torch_rng",
            "cuda_rng", "python_rng", "numpy_rng", "token_schema_version",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise RuntimeError("PPO exact resume checkpoint is missing: " + ", ".join(missing))
        if int(payload.get("ppo_format_version", 0)) != 3:
            raise RuntimeError("only V16 PPO checkpoints (format 3) can be resumed")
        if int(payload.get("token_schema_version", 0)) != TOKEN_SCHEMA_VERSION:
            raise RuntimeError(
                f"checkpoint token schema {payload.get('token_schema_version')} is "
                f"incompatible with required schema {TOKEN_SCHEMA_VERSION}"
            )
        try:
            checkpoint_config = ModelConfig(**dict(payload["model_config"]))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("PPO checkpoint has an invalid model_config") from exc
        if checkpoint_config != self.config:
            raise RuntimeError(
                "PPO exact resume model_config differs from the active V16 topology"
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
        torch.set_rng_state(payload["torch_rng"].cpu())
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        if payload["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])

    def load_model_weights(self, path: str | Path) -> None:
        """从 v16 SFT checkpoint 初始化全新 PPO(iteration 归零、optimizer 全新)。"""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        validate_fresh_model_checkpoint_contract(payload)
        checkpoint_config = ModelConfig(**dict(payload["model_config"]))
        if checkpoint_config != self.config:
            raise RuntimeError(
                "V16 SFT checkpoint model_config differs from the active PPO topology"
            )
        self.model.load_state_dict(payload["model"], strict=True)
        self.reference_model = KyokuTransformerActorCritic(self.config).to(self.device)
        self.reference_model.load_state_dict(self.model.state_dict(), strict=True)
        self.reference_model.requires_grad_(False)
        self.reference_model.eval()
        self.iteration = 0
