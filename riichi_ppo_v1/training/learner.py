"""V18 PPO 优化器:V18 批 collate、PPO clip + value Huber + GAE value advantage。

``PPOLearner`` 同时支持单卡(默认)与双卡 DDP:传 ``rank``/``world_size``
后模型由 ``DistributedDataParallel`` 包装,update 内所有梯度 collective 与
early-stop/KL 判定都在进程组内同步。
"""

from __future__ import annotations

import gc
import os
import queue as _queue
import random
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
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
ACTOR_ROOTS = {"actor_backbone", "action_fusion", "policy_mlp"}
CRITIC_ROOTS = {"critic_embedding", "critic_backbone", "value_head", "value_query"}
SHARED_ROOTS = {"token_embedding", "public_backbone"}


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
    """独立裁剪 actor/shared/critic,返回每个分支的裁剪前后与缩放比例。"""
    device = next(
        parameter.device
        for parameters in branch_parameters.values()
        for parameter in parameters
    )
    metrics: dict[str, torch.Tensor] = {}
    for branch in ("actor", "shared", "critic"):
        parameters = branch_parameters[branch]
        max_norm = float(max_norms[branch])
        pre_clip = nn.utils.clip_grad_norm_(parameters, max_norm)
        post_clip = torch.zeros((), device=device)
        for parameter in parameters:
            if parameter.grad is not None:
                post_clip.add_(parameter.grad.detach().float().square().sum())
        post_clip = post_clip.sqrt()
        scale = torch.ones((), device=device)
        if float(pre_clip.detach()) > 0.0:
            scale = (post_clip / pre_clip.to(device=device)).clamp(max=1.0)
        metrics[f"grad_norm_{branch}_pre_clip"] = pre_clip.to(device=device)
        metrics[f"grad_norm_{branch}_post_clip"] = post_clip
        metrics[f"grad_norm_{branch}_clip_scale"] = scale
        metrics[f"grad_norm_{branch}_clipped"] = (
            pre_clip.to(device=device) > max_norm
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


def _discounted_empirical_returns_loop(
    transitions: RolloutBuffer, gamma: float,
) -> np.ndarray:
    """gamma≠1 的 MC reward-to-go:保留原 Python 循环。

    gamma≠1 的 ``reward + gamma * running`` 递推是 Horner 求值,每一步的
    乘加依赖前一步结果,``np.add.accumulate`` 后缀和无法逐位复现其浮点
    结合顺序,故按参数分派保留此路径(通用性:按 gamma 值判断,不硬编码)。
    """
    returns = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    rewards = _rollout_values(transitions, "rewards", np.dtype(np.float32))
    done = _rollout_values(transitions, "done", np.dtype(np.bool_))
    for index in range(len(transitions) - 1, -1, -1):
        if done[index]:
            running = 0.0
        running = float(rewards[index]) + float(gamma) * running
        returns[index] = np.float32(running)
    return returns


def _discounted_empirical_returns_gamma1(
    rewards: np.ndarray, done: np.ndarray,
) -> np.ndarray:
    """gamma=1.0 的 MC reward-to-go:按局分段做 float64 反向后缀和。

    与原 Python 循环逐位一致:每段(以 done 行结尾)内
    ``np.add.accumulate`` 从段末起以 float64 顺序累加,与旧循环逐步
    ``running = reward + 1.0 * running`` 的浮点加法序列完全相同
    (IEEE 加法交换律保证操作数顺序无关);旧循环逐元素 np.float32 截断
    与末端一次性 cast 等价。
    """
    total = len(rewards)
    returns = np.zeros(total, dtype=np.float32)
    # 段以 done 行结尾;末尾若残留未终局的尾段(rollout 截断),从其起点单独成段。
    segment_ends = np.flatnonzero(done).tolist()
    if not segment_ends or segment_ends[-1] != total - 1:
        segment_ends.append(total - 1)
    start = 0
    for end in segment_ends:
        segment = rewards[start:end + 1].astype(np.float64)
        returns[start:end + 1] = np.add.accumulate(segment[::-1])[::-1]
        start = end + 1
    return returns


def discounted_empirical_returns(
    transitions: RolloutBuffer, gamma: float,
) -> np.ndarray:
    """Monte Carlo reward-to-go,每局终局时重置。

    gamma=1.0(生产值)走分段向量化后缀和,与原 Python 循环逐位一致
    (单测以 ``np.array_equal`` 断言);gamma≠1 保留原循环路径。
    """
    if float(gamma) == 1.0:
        rewards = _rollout_values(transitions, "rewards", np.dtype(np.float32))
        done = _rollout_values(transitions, "done", np.dtype(np.bool_))
        return _discounted_empirical_returns_gamma1(rewards, done)
    return _discounted_empirical_returns_loop(transitions, gamma)


def rollout_update_targets(
    transitions: RolloutBuffer,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """完整 rollout 的 advantage 归一化与 λ-return(供 DDP 分片复用)。"""
    source_advantages = _rollout_values(
        transitions, "advantages", np.dtype(np.float32),
    )
    advantages = (
        (source_advantages - source_advantages.mean(dtype=np.float64))
        / (source_advantages.std(dtype=np.float64) + 1e-8)
    ).astype(np.float32)
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
    host_batch: dict[str, Any],
    device: torch.device,
    profiler: StageProfiler | None = None,
) -> dict[str, Any]:
    """把 CPU minibatch 传输到 learner 设备。

    索引类张量(host 侧为 uint8/紧凑 dtype)在 GPU 侧统一转 ``long``,与旧
    int64 直传数值逐位一致;浮点/bool/长度字段原样传输。CUDA 上对整批
    做一次 ``pin_memory`` 后以 ``non_blocking`` 异步发起 H2D,消除逐字段
    pageable 传输的隐式同步等待(纯数据通路优化,不改任何数值)。非张量
    字段(host 侧标量/类别行表)原样返回。
    """
    tensors = {
        name: value for name, value in host_batch.items()
        if isinstance(value, torch.Tensor)
    }
    others = {
        name: value for name, value in host_batch.items()
        if not isinstance(value, torch.Tensor)
    }
    profile = StageProfiler(enabled=False) if profiler is None else profiler
    with profile.stage("update/collate_h2d"):
        if device.type != "cuda":
            transferred = {
                name: (
                    value.to(device=device).long()
                    if _is_compact_index_tensor(value) else value.to(device=device)
                )
                for name, value in tensors.items()
            }
            return {**transferred, **others}
        pinned = {
            name: value.pin_memory() if value.device.type == "cpu" else value
            for name, value in tensors.items()
        }
        transferred = {
            name: value.to(device=device, non_blocking=True)
            for name, value in pinned.items()
        }
        promoted = {
            name: value.long() if _is_compact_index_tensor(value) else value
            for name, value in transferred.items()
        }
        return {**promoted, **others}


def _is_compact_index_tensor(value: torch.Tensor) -> bool:
    """判断 host minibatch 字段是否为需在 GPU 侧恢复 long 的紧凑索引。"""
    return (
        value.dtype in (torch.uint8, torch.int8, torch.int16)
        and value.ndim >= 1
    )


class _PrefetchAborted(Exception):  # noqa: N818 -- 故意的控制流信号而非错误,docstring 已注明,不加 Error 后缀
    """collate 预取被正常停止信号中断(early-stop 路径,非错误)。"""


@dataclass
class _PrefetchState:
    """一次 update 的 collate 预取状态(线程/队列/计时)。"""

    queue: _queue.Queue[Any]
    stop_event: threading.Event
    finished_event: threading.Event
    errors: list[BaseException]
    times: list[float]
    thread: threading.Thread | None = None
    # 预取线程在队列 put 上被消费端反压阻塞的累计时长(队列满/主线程取慢)。
    blocked_s: float = 0.0


def _prefetch_collate_worker(
    transitions: RolloutBuffer,
    epoch_plans: list[tuple[np.ndarray, ...]],
    state: _PrefetchState,
) -> None:
    """预取线程:按主线程预计算的 minibatch 顺序逐批 collate,入有界队列。

    只做纯 CPU numpy 工作,不触碰 CUDA;``put`` 带超时并反复检查
    ``stop_event``,consumer 提前停止时线程在 ~0.2s 内自行退出,不会永久挂起。
    """
    counter = 0
    try:
        for plan in epoch_plans:
            for indices in plan:
                if state.stop_event.is_set():
                    return
                started = time.perf_counter()
                host_batch = transitions.collate(indices)
                state.times[counter] = time.perf_counter() - started
                counter += 1
                # put 的总耗时(含队满重试)即线程被消费端反压阻塞的时长,
                # 与 state.times(collate 纯计算时长)互补,构成线程 busy 全景。
                put_started = time.perf_counter()
                while not state.stop_event.is_set():
                    try:
                        state.queue.put((indices, host_batch), timeout=0.2)
                        break
                    except _queue.Full:
                        continue
                state.blocked_s += time.perf_counter() - put_started
    except BaseException as exc:  # noqa: BLE001 - 交由主线程统一抛出
        state.errors.append(exc)
    finally:
        state.finished_event.set()


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
        # torch.compile 快速路径(△ 级:浮点归约顺序改变,训练轨迹漂移;
        # 维护者 2026-09-01 批准,配置键 torch_compile 默认开启,置 false
        # 回退 eager)。必须在 DDP 包装**之前**编译原始模块(SFT trainer
        # 同做法);后续 DDP/权重读取经 state_dict 代理到 _orig_mod 正常工作。
        self.torch_compile = bool(hyperparameters.get("torch_compile", True))
        if self.torch_compile:
            self.model = torch.compile(self.model)
        # C:冻结 SFT reference 的 policy-only 前向编译(△ 级,浮点归约顺序
        # 改变;默认跟随 torch_compile,置 false 回退 eager reference)。实现
        # 上只编译 ``forward_actor`` 这个独立 code object(见 architecture 文档),
        # 与训练模型 forward 的 dynamo 缓存互不挤占;模块本体不包装,
        # state_dict 键与 eager 完全一致,checkpoint 契约不受影响。
        self.torch_compile_reference = bool(
            hyperparameters.get("torch_compile_reference", self.torch_compile)
        )
        self.model_ddp: DistributedDataParallel | None = (
            DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                broadcast_buffers=False,
                # bootstrap 期只训 critic(actor/shared 不进 loss),必须
                # find_unused_parameters=True;动态重建 DDP 关闭该旗标存在
                # 旧 hook 清理风险,列为 P2 观察项(实测 dummy 0 系数接入为
                # 负优化,见 bootstrap loss 注释)。
                find_unused_parameters=True,
            )
            if self.world_size > 1
            else None
        )
        self.reference_model: KyokuTransformerActorCritic | None = None
        self.iteration = 0
        self.profiler = StageProfiler(enabled=bool(hyperparameters.get("profile_enabled", True)))
        self.profile_cuda_sync = bool(hyperparameters.get("profile_cuda_sync", False))
        self.collate_prefetch = bool(hyperparameters.get("update_collate_prefetch", True))
        # 训练期跳过前向 GPU 侧结构校验:输入由 Rust 编码器 fail-closed 生成 +
        # SFT 契约校验 + 单测覆盖,校验本身每次 forward 引入十余次 GPU→CPU
        # 同步。默认 True 保持历史行为;关闭仅移除重复检查,不改数值。
        self.update_validate_structure = bool(
            hyperparameters.get("update_validate_structure", True)
        )
        # B2:SFT reference logits 预计算的分块行数(大批量提升 GPU 利用率)。
        self.reference_precompute_batch_size = max(
            1, int(hyperparameters.get("update_reference_precompute_batch_size", 8192))
        )
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

    def release_cache(self) -> None:
        """评测前释放缓存显存:gc + ``empty_cache`` 归还未分配的缓存块。

        update 结束后缓存分配器保留大量 cached-unallocated 块(参考 logits
        预计算与激活缓存),与 learner 同卡的 1v3 评测分片无法使用而被挤爆
        (实测 GPU0 44.42GiB 仅余 20MiB,分片 OOM);显式释放后评测分片可用
        显存恢复到接近空卡水平。模型/优化器等活跃张量不受影响。
        """
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _build_reference_model(self, state_dict: dict[str, torch.Tensor]) -> None:
        """按给定权重构造冻结 SFT reference(必填键严格校验,compile 可选)。

        ``torch_compile_reference`` 开启时仅把 ``forward_actor`` 换成编译包装
        (独立 code object,不与训练模型 forward 抢缓存;模块本体不包装,
        state_dict 键与 eager 完全一致)。冻结与 eval 语义与原两处构造一致。
        """
        model = KyokuTransformerActorCritic(self.config).to(self.device)
        if self.torch_compile_reference:
            model.forward_actor = torch.compile(model.forward_actor)
        model.load_state_dict(state_dict, strict=True)
        model.requires_grad_(False)
        model.eval()
        self.reference_model = model

    def _sync_cuda(self) -> None:
        if self.profile_cuda_sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _rebuild_ddp_without_unused_when_ready(self, critic_bootstrap: bool) -> None:
        """bootstrap 结束后重建 DDP,关闭 ``find_unused_parameters``。

        V18 PPO 只在 critic bootstrap 期存在未参与 loss 的 actor/shared 参数,
        因此必须保留 ``find_unused_parameters=True`` 才能通过 DDP 训练;bootstrap
        结束后所有参数都进入 loss,DDP 官方告警明确该旗标每次 backward 都会多走
        一次 autograd 图遍历,对双卡大更新(≈2000 minibatch)是可观的固定开销。
        这里在首次非 bootstrap update 时重建一次(旧 wrapper 无在途图,删除安全),
        之后更新不再承担该遍历开销。
        """
        if (
            self.world_size <= 1
            or self.model_ddp is None
            or critic_bootstrap
            or not bool(getattr(self.model_ddp, "find_unused_parameters", False))
        ):
            return
        old_wrapper = self.model_ddp
        self.model_ddp = DistributedDataParallel(
            self.model,
            device_ids=[self.device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        del old_wrapper

    @contextmanager
    def _gpu_stage(self, name: str):
        self._sync_cuda()
        with self.profiler.stage(name):
            try:
                yield
            finally:
                self._sync_cuda()

    def _state_dict_source(self) -> nn.Module:
        """权重读写目标:compile 包装时返回底层 ``_orig_mod``(同一组参数,
        state_dict 键不带包装前缀,checkpoint 契约与 eager 完全一致)。"""
        module = self.model
        return getattr(module, "_orig_mod", module)

    def weights(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self._state_dict_source().state_dict().items()
        }

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
        shared_capacity: int | None = None,
        critic_total_capacity: int | None = None,
        kind_row_plan: dict[int, Any] | None = None,
        critic_kind_row_plan: dict[int, Any] | None = None,
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
            validate_structure=self.update_validate_structure,
            shared_capacity=shared_capacity,
            critic_total_capacity=critic_total_capacity,
            kind_row_plan=kind_row_plan,
            critic_kind_row_plan=critic_kind_row_plan,
        )

    def _precompute_reference_logits(self, transitions: RolloutBuffer) -> torch.Tensor:
        """对整个 rank 分片一次性预计算冻结 SFT reference 的 policy_logits。

        大批量(``update_reference_precompute_batch_size``,默认 8192 行)在
        ``no_grad`` + bf16 autocast 下前向,与 minibatch 内的 reference
        前向同构(同一 forward 路径,含 legal mask 的 -inf masking);冻结模型
        行值跨 epoch 不变,故每 update 一次取代每 minibatch 一次(约占 update
        forward 的 40%)。结果 [N, NUM_ACTIONS] fp32 驻留 GPU 至 update 结束,
        由 minibatch 按 indices gather(常量,无梯度)。

        chunk 的 host collate 交给一个后台线程与 GPU 前向重叠(stage 内原本
        collate+H2D 串行):线程只做纯 CPU numpy 工作,不触碰 CUDA;产出的
        host 批经有界队列交给主线程(队列容量 2,反压防内存膨胀)。数值与
        串行版本逐位一致(同一 collate 输入与同一前向顺序)。
        """
        model = self.reference_model
        assert model is not None
        total = len(transitions)
        chunk = self.reference_precompute_batch_size
        chunk_starts = list(range(0, total, chunk))
        if not chunk_starts:
            raise RuntimeError("reference precompute requires a non-empty shard")
        producer_queue: _queue.Queue[Any] = _queue.Queue(maxsize=2)
        producer_error: list[BaseException] = []

        def produce() -> None:
            try:
                for start in chunk_starts:
                    host_batch = transitions.collate(np.arange(start, min(start + chunk, total)))
                    while True:
                        try:
                            producer_queue.put(host_batch, timeout=0.2)
                            break
                        except _queue.Full:
                            continue
            except BaseException as exc:  # noqa: BLE001 - 交由主线程统一抛出
                producer_error.append(exc)
            finally:
                producer_queue.put(None)

        producer = threading.Thread(
            target=produce, name="riichi-ppo-reference-collate", daemon=True,
        )
        producer.start()
        chunks: list[torch.Tensor] = []
        # 用 no_grad 而非 inference_mode:inference_mode 张量在 torch 2.7 的
        # inductor 下无法 lower token_embedding 行表路径的
        # ``torch.empty(pin_memory=True)``,开启 torch_compile_reference 时
        # 编译直接报错;no_grad 数值与张量语义完全一致(冻结模型 + 常量输入
        # 本就无梯度),仅保留版本计数开销,预计算是 GPU-bound,可忽略。
        with torch.no_grad():
            for _start in chunk_starts:
                host_batch = producer_queue.get()
                if host_batch is None:
                    if producer_error:
                        raise producer_error[0] from None
                    raise RuntimeError(
                        "reference collate producer ended before all chunks"
                    ) from None
                shared_capacity = host_batch.pop("shared_capacity", None)
                host_batch.pop("critic_total_capacity", None)
                kind_row_plan = host_batch.pop("kind_row_plan", None)
                host_batch.pop("critic_kind_row_plan", None)
                # 行表上传为 CUDA 张量再进 forward:让 token_embedding 的合并
                # 走纯 GPU cat 分支。torch 2.7 inductor 无法 lower pinned 分配,
                # 冻结模型(无梯度)的编译图里不能出现 numpy→pinned 路径;
                # 训练路径(参数带梯度)保持 numpy 行表不变。
                if kind_row_plan is not None:
                    kind_row_plan = {
                        key: torch.from_numpy(array).to(
                            self.device, non_blocking=True,
                        )
                        for key, array in kind_row_plan.items()
                    }
                batch = transfer_batch_to_device(host_batch, self.device, None)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    # 统一走 forward_actor(policy-only 唯一消费入口):编译开关
                    # 挂在该方法上,开启时此处自动走编译图;critic 侧容量/行表
                    # 在 policy-only 路径不参与,仅从 host 批中移除。
                    logits = model.forward_actor(
                        actor_factors=batch["actor_factors"],
                        actor_numeric=batch["actor_numeric"],
                        actor_lengths=batch["actor_lengths"],
                        query_action_ids=batch["query_action_ids"],
                        query_pair_counts=batch["query_pair_counts"],
                        legal_mask=batch["legal_mask"],
                        shared_capacity=shared_capacity,
                        kind_row_plan=kind_row_plan,
                        validate_structure=self.update_validate_structure,
                    )["policy_logits"]
                chunks.append(logits)
        # 排空生产者可能残留的哨兵/批次(主线程提前异常时避免挂起)。
        while producer_queue.qsize():
            producer_queue.get_nowait()
        producer.join(timeout=5.0)
        if producer.is_alive():
            raise RuntimeError("reference collate producer did not stop within 5s")
        # cat 出普通张量,供 minibatch 内的 KL 计算安全使用(no_grad 产物
        # 本就是普通张量;单 chunk 时 clone 避免别名)。
        return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0].clone()

    def _start_collate_prefetch(
        self,
        transitions: RolloutBuffer,
        epoch_plans: list[tuple[np.ndarray, ...]],
    ) -> _PrefetchState:
        """启动本 update 的 collate 预取线程(每个 update 新建,不跨 update 复用)。"""
        flat_size = sum(len(plan) for plan in epoch_plans)
        state = _PrefetchState(
            queue=_queue.Queue(maxsize=2),
            stop_event=threading.Event(),
            finished_event=threading.Event(),
            errors=[],
            times=[0.0] * flat_size,
        )
        thread = threading.Thread(
            target=_prefetch_collate_worker,
            args=(transitions, epoch_plans, state),
            name="riichi-ppo-collate-prefetch",
            daemon=True,
        )
        state.thread = thread
        thread.start()
        return state

    def _prefetch_get(
        self,
        state: _PrefetchState,
    ) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
        """从预取队列取下一批;early-stop 中断或线程异常时安全退出/抛出。"""
        while True:
            if state.stop_event.is_set():
                raise _PrefetchAborted()
            try:
                return state.queue.get(timeout=0.2)
            except _queue.Empty:
                if state.finished_event.is_set():
                    if state.errors:
                        raise state.errors[0] from None
                    raise RuntimeError(
                        "collate prefetch producer ended before all minibatches"
                    ) from None
                continue

    def _stop_collate_prefetch(self, state: _PrefetchState) -> None:
        """停止预取线程并合并计时;early-stop/异常路径均安全。"""
        state.stop_event.set()
        # 排空队列,解除 producer 可能阻塞的 put。
        try:
            while True:
                state.queue.get_nowait()
        except _queue.Empty:
            pass
        assert state.thread is not None
        state.thread.join(timeout=5.0)
        if state.thread.is_alive():
            raise RuntimeError("collate prefetch thread did not stop within 5s")
        for duration in state.times:
            if duration > 0.0:
                self.profiler.add("update/collate_soa_gather", duration)
        # 预取线程被队列反压阻塞的累计时长(collate 饥饿证据链的一半,
        # 另一半是主线程侧的 update/collate_wait)。
        self.profiler.add("update/collate_put_block", state.blocked_s)

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
        # B1 插计:rank 侧整个 update 调用的总墙钟(经 profiler 上报为
        # timing/update/learner_wall);driver 的 _recv 等待 ≈ shard 传输 +
        # 本墙钟 + 结果回传,用于把未插计段分解到具体位置。
        update_wall_started = time.perf_counter()
        length_metrics = rollout_target_metrics(transitions)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if advantages is None or returns is None:
            if (advantages is None) != (returns is None):
                raise ValueError("advantages and returns must be provided together")
            with self.profiler.stage("update/advantage_normalize"):
                advantages, returns, return_mean, return_std = rollout_update_targets(
                    transitions,
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
            sft_kl_coef = scheduled_piecewise_coefficient(
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
        # B2:SFT reference logits 每 update 预计算一次(冻结模型,行值跨 epoch
        # 不变),取代每 minibatch 一次的 reference 前向;bootstrap 期
        # (sft_kl_coef=0)跳过。
        reference_logits: torch.Tensor | None = None
        if sft_kl_coef > 0.0:
            with self.profiler.stage("update/reference_precompute"):
                reference_logits = self._precompute_reference_logits(transitions)
        value_coef = float(self.hp.get("value_coef", 0.5))

        # bootstrap 结束后关闭 DDP unused 检测,消除每次 backward 的额外图遍历。
        self._rebuild_ddp_without_unused_when_ready(critic_bootstrap)
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
        # collate 预取:先把全部 epoch 的 minibatch 计划在主线程预计算(与串行
        # 路径同一 RNG 序列,保证分桶可复现),再交由后台线程按序 collate,与
        # GPU forward/backward 重叠;``update_collate_prefetch`` 默认开启。
        epoch_plans: list[tuple[np.ndarray, ...]] | None = None
        prefetch_state: _PrefetchState | None = None
        if self.collate_prefetch:
            epoch_plans = []
            for _ in range(configured_epochs):
                with self.profiler.stage("update/length_bucket"):
                    epoch_plans.append(
                        transitions.bucketed_minibatches(
                            minibatch_size,
                            rng=rng,
                            bucket_window_multiplier=int(
                                self.hp.get("bucket_window_multiplier", 1)
                            ),
                        )
                    )
            prefetch_state = self._start_collate_prefetch(transitions, epoch_plans)
        try:
            for epoch_index in range(configured_epochs):
                epochs_started += 1
                if prefetch_state is None:
                    with self.profiler.stage("update/length_bucket"):
                        minibatches = transitions.bucketed_minibatches(
                            minibatch_size,
                            rng=rng,
                            bucket_window_multiplier=int(
                                self.hp.get("bucket_window_multiplier", 1)
                            ),
                        )
                else:
                    assert epoch_plans is not None
                    minibatches = epoch_plans[epoch_index]
                for planned_indices in minibatches:
                    if prefetch_state is not None:
                        # B1 插计:主线程在预取队列上的等待时长(「collate
                        # 饥饿」假设的直接证据);与线程内 collate_soa_gather
                        # (计算)和 collate_put_block(反压阻塞)互补。
                        with self.profiler.stage("update/collate_wait"):
                            indices, host_batch = self._prefetch_get(prefetch_state)
                    else:
                        indices = planned_indices
                        with self.profiler.stage("update/collate_soa_gather"):
                            host_batch = transitions.collate(indices)
                    # host 侧容量标量与类别行表为非张量,不能进 H2D;取出后
                    # 透传 forward,消除 max().item() 与 argsort/tolist 同步。
                    shared_capacity = host_batch.pop("shared_capacity", None)
                    critic_total_capacity = host_batch.pop("critic_total_capacity", None)
                    kind_row_plan = host_batch.pop("kind_row_plan", None)
                    critic_kind_row_plan = host_batch.pop("critic_kind_row_plan", None)
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
                    # 组边界与 should_step 先于 forward 计算:它们只依赖
                    # minibatch 计数与配置,而 planned_minibatches 经
                    # learner_shard_indices 的分片补齐后跨 rank 严格一致,
                    # should_step 因此天然同步,collective 次数在两 rank 对齐。
                    accumulation_steps = self.gradient_accumulation_steps
                    planned_minibatches = (
                        configured_epochs * planned_minibatches_per_epoch
                    )
                    actual_group_size = accumulation_group_size(
                        planned_minibatches,
                        accumulation_steps,
                        updates,
                    )
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
                    # 非组末 minibatch 用 no_sync 跳过梯度 allreduce:梯度仅
                    # 本地累加,到组末 backward(同步批)才一次性整体平均。
                    # DDP 的同步开关在 wrapper.forward 时读取,因此 no_sync
                    # 必须同时覆盖 forward 与 backward。结果与逐批 allreduce
                    # + 累积数学等价(sum_b mean_r g == mean_r sum_b g),仅
                    # 浮点结合顺序不同;单卡/无 DDP 路径不受影响。
                    sync_grads = self.model_ddp is None or should_step
                    sync_context = (
                        nullcontext() if sync_grads else self.model_ddp.no_sync()
                    )
                    with sync_context:
                        with self._gpu_stage("update/model_forward"):
                            with torch.autocast(
                                device_type=self.device.type,
                                dtype=torch.bfloat16,
                                enabled=self.use_bf16,
                            ):
                                output = self._model_forward(
                                    batch, critic_bootstrap=critic_bootstrap,
                                    shared_capacity=shared_capacity,
                                    critic_total_capacity=critic_total_capacity,
                                    kind_row_plan=kind_row_plan,
                                    critic_kind_row_plan=critic_kind_row_plan,
                                )
                                # B2:reference logits 每 update 预计算一次,
                                # 这里按 minibatch indices gather(冻结常量)。
                                reference_logits_batch = (
                                    reference_logits[indices]
                                    if reference_logits is not None
                                    else None
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
                            if reference_logits_batch is None:
                                sft_reference_kl_values = torch.zeros_like(policy_loss_values)
                            else:
                                sft_reference_kl_values = categorical_kl_values(
                                    output["policy_logits"],
                                    reference_logits_batch,
                                )
                            if critic_bootstrap:
                                # 预热期只训 critic:policy/entropy/KL 不进入损失,
                                # actor/shared 学习率同时为 0。实测把 policy 项以
                                # 0 系数接入损失图会让 bootstrap backward 多走整条
                                # policy 反传(104→246ms/step,负优化),故不采用。
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

                        def loss_detail() -> str:
                            # 诊断字符串仅在判定失败时构造:每 minibatch 无条件
                            # float() 取均值会引入 4 次 GPU→CPU 同步。
                            return (
                                f"policy={float(policy_loss_values.mean())} "
                                f"value={float(value_loss_values_.mean())} "
                                f"entropy={float(entropy_values.mean())} "
                                f"sft_kl={float(sft_reference_kl_values.mean())}"
                            )

                        if self.world_size == 1 and not loss_is_finite:
                            raise RuntimeError("non-finite PPO loss: " + loss_detail())
                        with self._gpu_stage("update/backward"):
                            # Gradient accumulation:配置 >1 时,loss 除以累积
                            # 步数后反向,只在最后一累积步执行梯度裁剪 +
                            # optimizer.step,保持 global effective batch ≈
                            # minibatch_size × 累积步数。组边界计算见本 minibatch
                            # 开头(forward 之前,no_sync 需要提前知道 should_step)。
                            scaled_loss = evaluated_loss / float(max(actual_group_size, 1))
                            scaled_loss.backward()
                    if self.world_size > 1 and should_step:
                        # 有限性检查从每 minibatch 降到每累积组一次(与
                        # optimizer step 同节拍,同步次数 868→87/update)。
                        # 组边界由 learner_shard_indices 补齐后的
                        # planned_minibatches 跨 rank 严格对齐,should_step
                        # 同步为真,故该 collective 的调用次数在两 rank 天然
                        # 一致。仍必须在所有 rank backward 完成后做全局判定,
                        # 避免单 rank 提前异常导致 NCCL collective 失配。
                        finite = torch.tensor(
                            float(loss_is_finite), device=self.device,
                        )
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        if finite.item() == 0.0:
                            raise RuntimeError(
                                "non-finite PPO loss on one of the DDP ranks: "
                                + loss_detail()
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
        finally:
            if prefetch_state is not None:
                self._stop_collate_prefetch(prefetch_state)
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
        self.profiler.add(
            "update/learner_wall", time.perf_counter() - update_wall_started,
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
        self._state_dict_source().load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload["iteration"])
        reference_state = payload.get("sft_reference_model")
        if reference_state is not None:
            self._build_reference_model(reference_state)
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
        self._state_dict_source().load_state_dict(payload["model"], strict=True)
        self._build_reference_model(self._state_dict_source().state_dict())
        self.iteration = 0
