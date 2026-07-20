"""PPO optimisation and variable-length batch collation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from ..model import KyokuTransformerActorCritic, ModelConfig
from .profiling import StageProfiler
from .trajectory import Transition
from .metrics import ppo_buffer_metrics


BATCH_MODE_IDS = {"streaming": 0.0, "prefetch": 1.0, "gpu_cache": 2.0}


@dataclass(frozen=True)
class MinibatchPlan:
    indices: np.ndarray
    selected_indices: np.ndarray
    sample_weights_np: np.ndarray
    loss_scale: float
    real_local_samples: int
    global_batch_size: int
    minibatch_tokens: int
    padded_input_tokens: int


@dataclass
class PreparedBatch:
    batch: dict[str, torch.Tensor]
    sample_weights: torch.Tensor
    host_tensors: dict[str, torch.Tensor] | None = None
    stream: torch.cuda.Stream | None = None


def transition_length_metrics(transitions: list[Transition], prefix: str = "update/buffer") -> dict[str, float]:
    """Describe semantic-token lengths and the global-padding baseline."""
    lengths = np.asarray([transition.token_length for transition in transitions], dtype=np.int64)
    if np.any(lengths < 0):
        raise ValueError("token length cannot be negative")
    input_lengths = lengths + 1  # learned query token appended by the model
    global_padded_input_tokens = int(input_lengths.max()) * len(transitions)
    effective_input_tokens = int(input_lengths.sum())
    return {
        f"{prefix}_transition_tokens_mean": float(lengths.mean()),
        f"{prefix}_transition_input_tokens_mean": float(input_lengths.mean()),
        f"{prefix}_transition_input_tokens_max": float(input_lengths.max()),
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
    """Return random-order minibatches whose members have similar lengths.

    Samples are shuffled first (which randomizes equal-length ties), sorted by
    length only to form compact batches, and the completed batches are shuffled
    again. This avoids the stage-correlated optimizer order of one permanent
    global sort while minimizing right-padding within every forward.
    """
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    lengths = np.asarray([transition.token_length for transition in transitions], dtype=np.int64)
    if not len(lengths):
        raise ValueError("cannot bucket an empty rollout")
    if np.any(lengths < 0):
        raise ValueError("token length cannot be negative")
    permutation = rng.permutation if rng is not None else np.random.permutation
    shuffled = permutation(len(transitions))
    sorted_indices = shuffled[np.argsort(lengths[shuffled], kind="stable")]
    batches = tuple(
        sorted_indices[start : start + minibatch_size]
        for start in range(0, len(sorted_indices), minibatch_size)
    )
    batch_order = permutation(len(batches))
    return tuple(batches[index] for index in batch_order)


def scheduled_learning_rate(base: float, update: int, total_updates: int, warmup_fraction: float) -> float:
    """Exp-style warmup followed by linear decay, using 1-based updates."""
    total = max(1, int(total_updates))
    step = max(0, int(update))
    warmup = int(total * float(warmup_fraction))
    if warmup > 0 and step <= warmup:
        return float(base) * float(step) / float(warmup)
    decay_updates = max(1, total - warmup)
    return float(base) * max(0.0, float(total - step + 1) / float(decay_updates))


def scheduled_entropy_coefficient(start: float, end: float, update: int, total_updates: int) -> float:
    """Linearly anneal entropy coefficient from start to end."""
    total = max(1, int(total_updates))
    progress = min(max(float(update) / float(total), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * progress


def value_loss_values(predicted: torch.Tensor, returns: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Return per-sample value loss for the configured PPO value objective."""
    normalized = str(loss_name).lower()
    if normalized == "huber":
        return nn.functional.huber_loss(predicted, returns, reduction="none")
    if normalized == "mse":
        return nn.functional.mse_loss(predicted, returns, reduction="none")
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
    """Put critic predictions and targets into the configured loss space.

    Rollout values, GAE and inference deliberately stay in the original reward
    space.  This only conditions the critic objective against changing return
    scales during self-play.
    """
    normalized_mode = str(mode).lower()
    if normalized_mode == "none":
        return predicted, returns
    if normalized_mode == "batch_std":
        scale = max(float(std), float(std_floor))
        return (predicted - float(mean)) / scale, (returns - float(mean)) / scale
    raise ValueError("value_target_normalization must be one of 'none' or 'batch_std'")


def branch_grad_norms(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return pre-clipping L2 gradient norms for disjoint actor/critic/shared groups."""
    parameter_device = next(model.parameters()).device
    squared_sums = {
        "actor": torch.zeros((), device=parameter_device),
        "critic": torch.zeros((), device=parameter_device),
        "shared": torch.zeros((), device=parameter_device),
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        root = name.removeprefix("module.").split(".", 1)[0]
        if root in {"actor_backbone", "policy_head", "query"}:
            branch = "actor"
        elif root in {"critic_embedding", "critic_backbone", "value_head", "value_query"}:
            branch = "critic"
        elif root in {"token_embedding", "public_backbone"}:
            branch = "shared"
        else:  # Guard against silently misclassifying a future model parameter.
            raise ValueError(f"unclassified model parameter for gradient metrics: {name}")
        squared_sums[branch].add_(parameter.grad.detach().float().square().sum())
    return {name: value.sqrt() for name, value in squared_sums.items()}


def approximate_kl_values(new_logprob: torch.Tensor, old_logprob: torch.Tensor) -> torch.Tensor:
    """Return exp/training's per-sample PPO approximate KL estimate."""
    log_ratio = new_logprob - old_logprob
    ratio = log_ratio.exp()
    return (ratio - 1.0) - log_ratio


def _empty_host_tensor(shape: tuple[int, ...], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, pin_memory=pin_memory)


def materialize_host_batch(
    transitions: list[Transition], profiler: StageProfiler | None = None,
    *, advantages: np.ndarray | None = None, pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    """Pad a length-bucketed minibatch into CPU tensors."""
    if not transitions:
        raise ValueError("cannot collate an empty rollout")
    profile = profiler or StageProfiler(enabled=False)
    with profile.stage("update/collate_shape_and_allocate"):
        max_length = max(t.token_length for t in transitions)
        max_critic_length = max(int(t.critic_length) for t in transitions)
        batch = len(transitions)
        factors = _empty_host_tensor((batch, max_length, 10), torch.uint8, pin_memory).zero_()
        numeric = _empty_host_tensor((batch, max_length, 8), torch.float32, pin_memory).zero_()
        critic_factors = _empty_host_tensor((batch, max_critic_length, 10), torch.uint8, pin_memory).zero_()
        legal = _empty_host_tensor((batch, 241), torch.bool, pin_memory)
        token_lengths = _empty_host_tensor((batch,), torch.long, pin_memory)
        critic_lengths = _empty_host_tensor((batch,), torch.long, pin_memory)
        actions = _empty_host_tensor((batch,), torch.long, pin_memory)
        old_logprobs = _empty_host_tensor((batch,), torch.float32, pin_memory)
        returns = _empty_host_tensor((batch,), torch.float32, pin_memory)
        advantage_values = _empty_host_tensor((batch,), torch.float32, pin_memory)
    with profile.stage("update/collate_host_padding_copy"):
        factors_np = factors.numpy()
        numeric_np = numeric.numpy()
        critic_factors_np = critic_factors.numpy()
        legal_np = legal.numpy()
        token_lengths_np = token_lengths.numpy()
        critic_lengths_np = critic_lengths.numpy()
        actions_np = actions.numpy()
        old_logprobs_np = old_logprobs.numpy()
        returns_np = returns.numpy()
        advantage_values_np = advantage_values.numpy()
        source_advantages = (
            np.asarray([t.advantage for t in transitions], dtype=np.float32)
            if advantages is None else np.asarray(advantages, dtype=np.float32)
        )
        if source_advantages.shape != (batch,):
            raise ValueError("advantages must have one value per transition")
        for row, transition in enumerate(transitions):
            factors_np[row, : transition.token_length] = transition.token_factors
            numeric_np[row, : transition.token_length] = transition.token_numeric
            critic_length = int(transition.critic_length)
            if critic_length:
                if transition.critic_factors is None:
                    raise ValueError("critic transition length requires critic arrays")
                critic_factors_np[row, :critic_length] = transition.critic_factors[:critic_length]
            legal_np[row] = transition.legal_mask
            token_lengths_np[row] = int(transition.token_length)
            critic_lengths_np[row] = critic_length
            actions_np[row] = int(transition.action)
            old_logprobs_np[row] = float(transition.logprob)
            returns_np[row] = float(transition.return_)
            advantage_values_np[row] = source_advantages[row]
    return {
        "token_factors": factors,
        "token_numeric": numeric,
        "token_lengths": token_lengths,
        "critic_factors": critic_factors,
        "critic_lengths": critic_lengths,
        "legal_mask": legal,
        "actions": actions,
        "old_logprobs": old_logprobs,
        "advantages": advantage_values,
        "returns": returns,
    }


def transfer_batch_to_device(
    host_batch: dict[str, torch.Tensor], device: torch.device, profiler: StageProfiler | None = None,
    *, non_blocking: bool = False, stream: torch.cuda.Stream | None = None,
) -> dict[str, torch.Tensor]:
    """Transfer a materialized CPU minibatch to the learner device."""
    profile = profiler or StageProfiler(enabled=False)
    stream_context = torch.cuda.stream(stream) if stream is not None and device.type == "cuda" else nullcontext()
    with profile.stage("update/collate_h2d"):
        with stream_context:
            return {
                name: value.to(device=device, non_blocking=non_blocking)
                for name, value in host_batch.items()
            }


def collate(
    transitions: list[Transition], device: torch.device, profiler: StageProfiler | None = None,
    *, advantages: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Pad a length-bucketed minibatch and transfer it to the learner device."""
    host_batch = materialize_host_batch(transitions, profiler, advantages=advantages, pin_memory=False)
    return transfer_batch_to_device(host_batch, device, profiler)


class PPOLearner:
    def __init__(self, model_size: str, device: str, **hyperparameters: Any) -> None:
        self.device = torch.device(device)
        self.config = replace(ModelConfig.preset(model_size), context_tokens=int(hyperparameters.get("context_tokens", 4096)))
        self.model = KyokuTransformerActorCritic(self.config).to(self.device)
        self.hp = hyperparameters
        # Keep FP32 parameters and optimizer state, but use BF16 tensor cores for
        # CUDA forwards when available.  This matches exp/training's production
        # policy: unsupported hardware is deliberately FP32 rather than FP16.
        requested_bf16 = str(hyperparameters.get("inference_dtype", "bf16")).lower() == "bf16"
        self.use_bf16 = bool(
            requested_bf16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(hyperparameters["learning_rate"]),
            betas=(
                float(hyperparameters.get("adam_beta1", 0.9)),
                float(hyperparameters.get("adam_beta2", 0.999)),
            ),
            eps=float(hyperparameters.get("adam_epsilon", 1e-5)),
            weight_decay=float(hyperparameters.get("weight_decay", 0.01)),
            fused=self.use_bf16,
        )
        self.iteration = 0
        self.profiler = StageProfiler(enabled=bool(hyperparameters.get("profile_enabled", True)))
        self.profile_cuda_sync = bool(hyperparameters.get("profile_cuda_sync", False))
        self.update_batch_mode = str(hyperparameters.get("update_batch_mode", "auto")).lower()
        if self.update_batch_mode not in {"streaming", "prefetch", "gpu_cache", "auto"}:
            raise ValueError("update_batch_mode must be one of streaming, prefetch, gpu_cache or auto")
        self.value_target_normalization = str(hyperparameters.get("value_target_normalization", "batch_std")).lower()
        if self.value_target_normalization not in {"none", "batch_std"}:
            raise ValueError("value_target_normalization must be one of 'none' or 'batch_std'")
        self.value_target_std_floor = float(hyperparameters.get("value_target_std_floor", 1e-2))
        if self.value_target_std_floor <= 0:
            raise ValueError("value_target_std_floor must be positive")
        self.distributed = False

    def enable_distributed(self) -> None:
        if self.distributed:
            return
        if self.device.type != "cuda":
            raise RuntimeError("distributed PPO currently requires CUDA")
        self.model = DistributedDataParallel(self.model, device_ids=[self.device.index or 0])
        self.distributed = True

    def _sync_cuda(self) -> None:
        if self.profile_cuda_sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def _gpu_stage(self, name: str):
        self._sync_cuda()
        # Keep the completion fence *inside* the timed region.  Otherwise
        # asynchronous CUDA work from this stage is charged to the pre-fence
        # of the following stage, which makes the per-stage profile useless.
        with self.profiler.stage(name):
            try:
                yield
            finally:
                self._sync_cuda()

    def weights(self) -> dict[str, torch.Tensor]:
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        return {key: value.detach().cpu() for key, value in model.state_dict().items()}

    def _minibatch_plan(
        self,
        transitions: list[Transition],
        advantages: np.ndarray,
        indices: np.ndarray,
        distributed: bool,
        distributed_rank: int,
        distributed_world_size: int,
    ) -> MinibatchPlan:
        global_batch_size = int(len(indices))
        if distributed:
            local_indices = np.array_split(indices, int(distributed_world_size))[int(distributed_rank)]
            if len(local_indices):
                selected_indices = local_indices
                sample_weights_np = np.ones(len(selected_indices), dtype=np.float32)
            else:
                selected_indices = np.asarray([int(indices[0])], dtype=np.int64)
                sample_weights_np = np.zeros(1, dtype=np.float32)
            loss_scale = float(distributed_world_size) / float(global_batch_size)
            real_local_samples = int(len(local_indices))
        else:
            selected_indices = indices
            sample_weights_np = np.ones(len(selected_indices), dtype=np.float32)
            loss_scale = 1.0 / float(len(selected_indices))
            real_local_samples = int(len(selected_indices))
        minibatch_tokens = sum(transitions[int(index)].token_length for index in indices)
        padded_input_tokens = global_batch_size * (
            max(transitions[int(index)].token_length for index in indices) + 1
        )
        return MinibatchPlan(
            indices=np.asarray(indices, dtype=np.int64),
            selected_indices=np.asarray(selected_indices, dtype=np.int64),
            sample_weights_np=sample_weights_np,
            loss_scale=loss_scale,
            real_local_samples=real_local_samples,
            global_batch_size=global_batch_size,
            minibatch_tokens=minibatch_tokens,
            padded_input_tokens=padded_input_tokens,
        )

    def _estimate_cached_batch_bytes(self, transitions: list[Transition], plans: list[MinibatchPlan]) -> int:
        total = 0
        for plan in plans:
            selected = [transitions[int(index)] for index in plan.selected_indices]
            if not selected:
                continue
            batch = len(selected)
            max_length = max(t.token_length for t in selected)
            max_critic_length = max(int(t.critic_length) for t in selected)
            total += batch * max_length * (10 + 8 * 4)
            total += batch * max_critic_length * (10 + 8 * 4 + 34 * 4)
            total += batch * (241 + 2 * 8 + 4 * 4 + 4)  # masks, lengths, scalar PPO fields and weights
        return int(total)

    def _resolve_batch_mode(self, requested: str, estimated_cache_bytes: int) -> tuple[str, float, float]:
        if self.device.type != "cuda":
            return "streaming", 0.0, 0.0
        if requested != "auto":
            return requested, 0.0, 0.0
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        except RuntimeError:
            return "streaming", 0.0, 0.0
        threshold_bytes = int(free_bytes * 0.25)
        # Benchmarking showed that the large unsynchronized ``collate_h2d``
        # number is mostly CUDA queue attribution rather than copy time.  Keep
        # auto on the fastest measured path and leave cache/prefetch available
        # as explicit diagnostic modes.
        return "streaming", float(free_bytes / 2**20), float(threshold_bytes / 2**20)

    def _prepare_batch(
        self,
        transitions: list[Transition],
        advantages: np.ndarray,
        plan: MinibatchPlan,
        *,
        pin_memory: bool,
        non_blocking: bool,
        stream: torch.cuda.Stream | None = None,
    ) -> PreparedBatch:
        with self.profiler.stage("update/minibatch_collate_total"):
            selected = [transitions[int(index)] for index in plan.selected_indices]
            host = materialize_host_batch(
                selected, self.profiler, advantages=advantages[plan.selected_indices], pin_memory=pin_memory,
            )
            sample_weights_host = torch.as_tensor(plan.sample_weights_np, dtype=torch.float32)
            if pin_memory:
                sample_weights_host = sample_weights_host.pin_memory()
            batch = transfer_batch_to_device(
                host, self.device, self.profiler, non_blocking=non_blocking, stream=stream,
            )
            with self.profiler.stage("update/sample_weights_h2d"):
                if stream is not None and self.device.type == "cuda":
                    with torch.cuda.stream(stream):
                        sample_weights = sample_weights_host.to(
                            device=self.device, non_blocking=non_blocking,
                        )
                else:
                    sample_weights = sample_weights_host.to(device=self.device, non_blocking=non_blocking)
            return PreparedBatch(batch=batch, sample_weights=sample_weights, host_tensors=host, stream=stream)

    def _prepare_batch_in_thread(
        self,
        transitions: list[Transition],
        advantages: np.ndarray,
        plan: MinibatchPlan,
        stream: torch.cuda.Stream,
    ) -> PreparedBatch:
        torch.cuda.set_device(self.device)
        return self._prepare_batch(
            transitions, advantages, plan, pin_memory=True, non_blocking=True, stream=stream,
        )

    def update(
        self,
        transitions: list[Transition],
        *,
        distributed_rank: int = 0,
        distributed_world_size: int = 1,
        shuffle_seed: int | None = None,
    ) -> dict[str, float]:
        distributed = int(distributed_world_size) > 1
        if distributed and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("distributed update requires an initialized torch.distributed process group")
        if not transitions:
            raise ValueError("cannot update from an empty rollout")
        self.profiler.reset()
        length_metrics = transition_length_metrics(transitions)
        length_metrics.update(ppo_buffer_metrics(transitions))
        raw_returns = np.asarray([transition.return_ for transition in transitions], dtype=np.float64)
        raw_values = np.asarray([transition.value for transition in transitions], dtype=np.float64)
        value_target_mean = float(raw_returns.mean())
        value_target_std = float(raw_returns.std())
        length_metrics.update({
            # Keep these raw-space diagnostics independent from the selected
            # loss normalization mode so experiments remain comparable.
            "value_target_mean": value_target_mean,
            "value_target_std": value_target_std,
            "value_prediction_std": float(raw_values.std()),
            "value_explained_variance": float(length_metrics["explained_variance"]),
        })
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with self.profiler.stage("update/advantage_normalize"):
            advantages = np.asarray([transition.advantage for transition in transitions], dtype=np.float32)
            advantages = (
                (advantages - advantages.mean(dtype=np.float64))
                / (advantages.std(dtype=np.float64) + 1e-8)
            ).astype(np.float32)
        # Keep metric reductions on the device.  Reading a scalar with
        # ``float(tensor)`` inside every minibatch synchronizes CUDA and
        # serializes the update stream; transfer the final averages once.
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
        learning_rate = scheduled_learning_rate(
            float(self.hp["learning_rate"]),
            update_number,
            total_updates,
            float(self.hp.get("warmup_fraction", 0.0)),
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        if "entropy_start" in self.hp or "entropy_end" in self.hp:
            entropy_coef = scheduled_entropy_coefficient(
                float(self.hp.get("entropy_start", self.hp.get("entropy_coef", 0.0))),
                float(self.hp.get("entropy_end", self.hp.get("entropy_coef", 0.0))),
                update_number,
                total_updates,
            )
        else:
            entropy_coef = float(self.hp["entropy_coef"])
        epochs_started = 0
        epochs_completed = 0
        executed_samples = 0
        executed_tokens = 0
        executed_padded_input_tokens = 0
        batch_mode_id_sum = 0.0
        batch_mode_epochs = 0
        batch_cache_estimated_bytes = 0
        batch_cache_free_mb = 0.0
        batch_cache_threshold_mb = 0.0
        batch_cache_fallbacks = 0
        self.model.train()
        stop_early = False
        rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
        for _ in range(configured_epochs):
            epochs_started += 1
            epoch_kl_sum: torch.Tensor | None = None
            epoch_kl_count = 0
            with self.profiler.stage("update/length_bucket"):
                minibatches = length_bucketed_minibatches(transitions, minibatch_size, rng=rng)
            plans = [
                self._minibatch_plan(
                    transitions, advantages, indices, distributed, distributed_rank, distributed_world_size,
                )
                for indices in minibatches
            ]
            estimated_cache_bytes = self._estimate_cached_batch_bytes(transitions, plans)
            batch_cache_estimated_bytes = max(batch_cache_estimated_bytes, estimated_cache_bytes)
            batch_mode, free_mb, threshold_mb = self._resolve_batch_mode(self.update_batch_mode, estimated_cache_bytes)
            batch_cache_free_mb = max(batch_cache_free_mb, free_mb)
            batch_cache_threshold_mb = max(batch_cache_threshold_mb, threshold_mb)

            def prepared_batches() -> Any:
                nonlocal batch_mode, batch_cache_fallbacks
                if batch_mode == "gpu_cache":
                    try:
                        cached = [
                            self._prepare_batch(
                                transitions, advantages, plan,
                                pin_memory=False, non_blocking=False,
                            )
                            for plan in plans
                        ]
                    except RuntimeError as exc:
                        if self.device.type == "cuda" and "out of memory" in str(exc).lower():
                            cached = []
                            torch.cuda.empty_cache()
                            batch_cache_fallbacks += 1
                            batch_mode = "prefetch"
                        else:
                            raise
                    else:
                        for plan, prepared in zip(plans, cached):
                            yield plan, prepared
                        return
                if batch_mode == "prefetch" and self.device.type == "cuda" and plans:
                    stream = torch.cuda.Stream(self.device)
                    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppo-prefetch") as executor:
                        future: Future[PreparedBatch] = executor.submit(
                            self._prepare_batch_in_thread, transitions, advantages, plans[0], stream,
                        )
                        for index, plan in enumerate(plans):
                            prepared = future.result()
                            if index + 1 < len(plans):
                                future = executor.submit(
                                    self._prepare_batch_in_thread,
                                    transitions,
                                    advantages,
                                    plans[index + 1],
                                    stream,
                                )
                            current_stream = torch.cuda.current_stream(self.device)
                            current_stream.wait_stream(stream)
                            for value in tuple(prepared.batch.values()) + (prepared.sample_weights,):
                                value.record_stream(current_stream)
                            yield plan, prepared
                    return
                for plan in plans:
                    yield plan, self._prepare_batch(
                        transitions, advantages, plan,
                        pin_memory=False, non_blocking=False,
                    )

            for plan, prepared in prepared_batches():
                batch = prepared.batch
                sample_weights = prepared.sample_weights
                token_factors, token_numeric = batch["token_factors"], batch["token_numeric"]
                critic_factors, critic_lengths = batch["critic_factors"], batch["critic_lengths"]
                legal_mask, token_lengths = batch["legal_mask"], batch["token_lengths"]
                actions, old_logprobs = batch["actions"], batch["old_logprobs"]
                adv, returns = batch["advantages"], batch["returns"]
                executed_samples += plan.global_batch_size
                executed_tokens += plan.minibatch_tokens
                executed_padded_input_tokens += plan.padded_input_tokens
                with self._gpu_stage("update/model_forward"):
                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16,
                    ):
                        output = self.model(
                            token_factors,
                            token_numeric,
                            legal_mask,
                            token_lengths,
                            critic_factors=critic_factors,
                            critic_lengths=critic_lengths,
                        )
                with self._gpu_stage("update/distribution_and_loss"):
                    # The model promotes its policy/value outputs to FP32.  Keep
                    # the PPO ratio, loss and their gradients in FP32 as well.
                    logprobabilities = F.log_softmax(output["policy_logits"].float(), dim=-1)
                    logprob = logprobabilities.gather(1, actions[:, None]).squeeze(1)
                    old_logprobs, adv, returns = old_logprobs.float(), adv.float(), returns.float()
                    ratio = (logprob - old_logprobs).exp()
                    clipped = ratio.clamp(1 - float(self.hp["ppo_clip"]), 1 + float(self.hp["ppo_clip"])) * adv
                    policy_loss_values = -torch.minimum(ratio * adv, clipped)
                    predicted_value = output["value"].float()
                    normalized_value, normalized_returns = normalize_value_targets(
                        predicted_value,
                        returns,
                        mode=self.value_target_normalization,
                        mean=value_target_mean,
                        std=value_target_std,
                        std_floor=self.value_target_std_floor,
                    )
                    value_loss = value_loss_values(
                        normalized_value, normalized_returns, str(self.hp.get("value_loss", "huber")),
                    )
                    raw_value_loss = value_loss_values(
                        predicted_value, returns, str(self.hp.get("value_loss", "huber")),
                    )
                    valid_logprobabilities = torch.isfinite(logprobabilities)
                    probabilities = torch.where(
                        valid_logprobabilities, logprobabilities.exp(), torch.zeros_like(logprobabilities),
                    )
                    safe_logprobabilities = torch.where(
                        valid_logprobabilities, logprobabilities, torch.zeros_like(logprobabilities),
                    )
                    entropy_values = -(probabilities * safe_logprobabilities).sum(-1)
                    legal_action_counts = legal_mask.sum(-1).float().clamp_min(2.0)
                    normalized_entropy_values = entropy_values / legal_action_counts.log()
                    loss_values = (
                        policy_loss_values
                        + float(self.hp["value_coef"]) * value_loss
                        - entropy_coef * entropy_values
                    )
                    weighted_count = sample_weights.sum().clamp_min(1.0)
                    policy_loss = (policy_loss_values * sample_weights).sum() / weighted_count
                    value_loss_scalar = (value_loss * sample_weights).sum() / weighted_count
                    entropy = (entropy_values * sample_weights).sum() / weighted_count
                    loss = (loss_values * sample_weights).sum() * plan.loss_scale
                with self._gpu_stage("update/zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
                with self._gpu_stage("update/backward"):
                    loss.backward()
                with self._gpu_stage("update/gradient_clip"):
                    branch_norms = branch_grad_norms(self.model)
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), float(self.hp["max_grad_norm"]))
                    grad_norm_post_clip = grad_norm.clamp(max=float(self.hp["max_grad_norm"]))
                with self._gpu_stage("update/optimizer_step"):
                    self.optimizer.step()
                with self._gpu_stage("update/diagnostic_metrics"):
                    with torch.no_grad():
                        kl_values = approximate_kl_values(logprob, old_logprobs)
                        clipfrac_values = ((ratio - 1).abs() > float(self.hp["ppo_clip"])).float()
                        if float(self.hp["target_kl"]) > 0:
                            kl_sum = (kl_values * sample_weights).sum()
                            if epoch_kl_sum is None:
                                epoch_kl_sum = kl_sum.detach().clone()
                            else:
                                epoch_kl_sum.add_(kl_sum.detach())
                            epoch_kl_count += plan.real_local_samples
                for name, values in (
                    ("loss", loss_values),
                    ("policy_loss", policy_loss_values),
                    ("value_loss", value_loss),
                    ("value_loss_raw", raw_value_loss),
                    ("entropy", entropy_values),
                    ("entropy_normalized", normalized_entropy_values),
                    ("approx_kl", kl_values),
                    ("clipfrac", clipfrac_values),
                    ("ratio", ratio),
                ):
                    detached = (values.detach() * sample_weights).sum()
                    if name in metric_sample_sums:
                        metric_sample_sums[name].add_(detached)
                    else:
                        metric_sample_sums[name] = detached.clone()
                metric_sample_count += plan.real_local_samples
                # Retaining scalar ratios on-device is inexpensive (one FP32
                # value per executed sample) and avoids synchronizing every
                # minibatch merely to obtain a useful tail diagnostic.
                if plan.real_local_samples:
                    ratio_samples.append(ratio.detach()[sample_weights.bool()])
                detached_grad_norm = grad_norm.detach()
                if "grad_norm" in step_metric_totals:
                    step_metric_totals["grad_norm"].add_(detached_grad_norm)
                else:
                    step_metric_totals["grad_norm"] = detached_grad_norm.clone()
                detached_post_clip_grad_norm = grad_norm_post_clip.detach()
                if "grad_norm_post_clip" in step_metric_totals:
                    step_metric_totals["grad_norm_post_clip"].add_(detached_post_clip_grad_norm)
                else:
                    step_metric_totals["grad_norm_post_clip"] = detached_post_clip_grad_norm.clone()
                for branch, value in branch_norms.items():
                    name = f"grad_norm_{branch}"
                    detached_value = value.detach()
                    if name in step_metric_totals:
                        step_metric_totals[name].add_(detached_value)
                    else:
                        step_metric_totals[name] = detached_value.clone()
                updates += 1
            batch_mode_id_sum += BATCH_MODE_IDS[batch_mode]
            batch_mode_epochs += 1
            epochs_completed += 1
            if float(self.hp["target_kl"]) > 0 and epoch_kl_sum is not None:
                epoch_kl_count_tensor = torch.tensor(float(epoch_kl_count), device=self.device)
                if distributed:
                    dist.all_reduce(epoch_kl_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(epoch_kl_count_tensor, op=dist.ReduceOp.SUM)
                epoch_kl = epoch_kl_sum / epoch_kl_count_tensor.clamp_min(1.0)
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
            "update/executed_transition_input_tokens_mean": float(executed_tokens / executed_samples + 1),
            "update/executed_padded_input_tokens": float(executed_padded_input_tokens),
            "update/executed_padding_input_tokens": float(
                executed_padded_input_tokens - executed_tokens - executed_samples
            ),
            "update/executed_padding_fraction_of_padded_input_tokens": float(
                (executed_padded_input_tokens - executed_tokens - executed_samples)
                / max(executed_padded_input_tokens, 1)
            ),
            "update/batch_mode_id": float(batch_mode_id_sum / max(batch_mode_epochs, 1)),
            "update/batch_cache_estimated_mb": float(batch_cache_estimated_bytes / 2**20),
            "update/batch_cache_free_mb": float(batch_cache_free_mb),
            "update/batch_cache_threshold_mb": float(batch_cache_threshold_mb),
            "update/batch_cache_fallbacks": float(batch_cache_fallbacks),
            "system/learning_rate": float(learning_rate),
            "system/entropy_coef": float(entropy_coef),
        })
        sample_count_tensor = torch.tensor(float(metric_sample_count), device=self.device)
        if distributed:
            for value in metric_sample_sums.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count_tensor, op=dist.ReduceOp.SUM)
        sample_names = tuple(metric_sample_sums)
        sample_values = torch.stack([metric_sample_sums[name] for name in sample_names]).div(
            sample_count_tensor.clamp_min(1.0)
        ).tolist()
        step_names = tuple(step_metric_totals)
        step_values = torch.stack([step_metric_totals[name] for name in step_names]).div(max(updates, 1)).tolist()
        result = (
            dict(zip(sample_names, sample_values))
            | dict(zip(step_names, step_values))
            | {"transitions": float(count)}
            | length_metrics
        )
        if ratio_samples:
            result["ratio_p95"] = float(torch.quantile(torch.cat(ratio_samples), 0.95).item())
        result.update(self.profiler.delta({}, prefix="timing"))
        if self.device.type == "cuda":
            result.update({
                "gpu/torch_memory_allocated_mb": float(torch.cuda.memory_allocated(self.device) / 2**20),
                "gpu/torch_memory_reserved_mb": float(torch.cuda.memory_reserved(self.device) / 2**20),
                "gpu/torch_memory_peak_allocated_mb": float(torch.cuda.max_memory_allocated(self.device) / 2**20),
                "gpu/torch_memory_peak_reserved_mb": float(torch.cuda.max_memory_reserved(self.device) / 2**20),
            })
        return result

    def save(self, path: str | Path, train_config: dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.weights(),
            "optimizer": self.optimizer.state_dict(),
            "model_config": asdict(self.config),
            "train_config": train_config,
            "iteration": self.iteration,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload.get("iteration", 0))
        if "torch_rng" in payload:
            # Checkpoints are loaded onto the learner device so model/optimizer
            # tensors restore without an extra copy.  PyTorch's default RNG
            # state is CPU-only, however; DDP CUDA resume must move it back.
            torch.set_rng_state(payload["torch_rng"].cpu())
            random.setstate(payload["python_rng"])
            np.random.set_state(payload["numpy_rng"])
