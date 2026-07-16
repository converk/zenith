"""PPO optimisation and variable-length batch collation."""

from __future__ import annotations

from dataclasses import asdict, replace
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import KyokuTransformerActorCritic, ModelConfig, MODEL_SCHEMA_VERSION, TOKEN_SCHEMA_VERSION
from .profiling import StageProfiler
from .trajectory import Transition


def transition_length_metrics(transitions: list[Transition], prefix: str = "update/buffer") -> dict[str, float]:
    """Describe V5 lengths and the old global-padding baseline."""
    lengths = np.asarray([transition.token_length for transition in transitions], dtype=np.int64)
    if np.any(lengths < 0):
        raise ValueError("V5 token length cannot be negative")
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
    transitions: list[Transition], minibatch_size: int,
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
        raise ValueError("V5 token length cannot be negative")
    shuffled = np.random.permutation(len(transitions))
    sorted_indices = shuffled[np.argsort(lengths[shuffled], kind="stable")]
    batches = tuple(
        sorted_indices[start : start + minibatch_size]
        for start in range(0, len(sorted_indices), minibatch_size)
    )
    batch_order = np.random.permutation(len(batches))
    return tuple(batches[index] for index in batch_order)


def collate(
    transitions: list[Transition], device: torch.device, profiler: StageProfiler | None = None,
    *, advantages: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Pad a length-bucketed minibatch and transfer it to the learner device."""
    if not transitions:
        raise ValueError("cannot collate an empty rollout")
    profile = profiler or StageProfiler(enabled=False)
    with profile.stage("update/collate_shape_and_allocate"):
        max_length = max(t.token_length for t in transitions)
        batch = len(transitions)
        factors = np.zeros((batch, max_length, 10), dtype=np.uint8)
        numeric = np.zeros((batch, max_length, 8), dtype=np.float32)
    with profile.stage("update/collate_host_padding_copy"):
        for row, transition in enumerate(transitions):
            factors[row, : transition.token_length] = transition.token_factors
            numeric[row, : transition.token_length] = transition.token_numeric
        advantage_values = (
            np.asarray([t.advantage for t in transitions], dtype=np.float32)
            if advantages is None else np.asarray(advantages, dtype=np.float32)
        )
        if advantage_values.shape != (batch,):
            raise ValueError("advantages must have one value per transition")
        legal = np.stack([t.legal_mask for t in transitions])
        lengths = [t.token_length for t in transitions]
        actions = [t.action for t in transitions]
        old_logprobs = [t.logprob for t in transitions]
        returns = [t.return_ for t in transitions]
    with profile.stage("update/collate_h2d"):
        return {
            "token_factors": torch.as_tensor(factors, device=device),
            "token_numeric": torch.as_tensor(numeric, device=device),
            "token_lengths": torch.tensor(lengths, device=device),
            "legal_mask": torch.as_tensor(legal, device=device),
            "actions": torch.tensor(actions, device=device, dtype=torch.long),
            "old_logprobs": torch.tensor(old_logprobs, device=device, dtype=torch.float32),
            "advantages": torch.as_tensor(advantage_values, device=device),
            "returns": torch.tensor(returns, device=device, dtype=torch.float32),
        }


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
            self.model.parameters(), lr=float(hyperparameters["learning_rate"]), fused=self.use_bf16,
        )
        self.iteration = 0
        self.profiler = StageProfiler(enabled=bool(hyperparameters.get("profile_enabled", True)))
        self.profile_cuda_sync = bool(hyperparameters.get("profile_cuda_sync", False))

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
        return {key: value.detach().cpu() for key, value in self.model.state_dict().items()}

    def update(self, transitions: list[Transition]) -> dict[str, float]:
        self.profiler.reset()
        length_metrics = transition_length_metrics(transitions)
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
        metric_totals: dict[str, torch.Tensor] = {}
        updates = 0
        count = len(transitions)
        configured_epochs = int(self.hp["update_epochs"])
        minibatch_size = int(self.hp["minibatch_size"])
        planned_minibatches_per_epoch = (count + minibatch_size - 1) // minibatch_size
        epochs_started = 0
        epochs_completed = 0
        executed_samples = 0
        executed_tokens = 0
        executed_padded_input_tokens = 0
        self.model.train()
        stop_early = False
        for _ in range(configured_epochs):
            epochs_started += 1
            with self.profiler.stage("update/length_bucket"):
                minibatches = length_bucketed_minibatches(transitions, minibatch_size)

            for indices in minibatches:
                with self.profiler.stage("update/minibatch_collate_total"):
                    selected = [transitions[int(index)] for index in indices]
                    batch = collate(selected, self.device, self.profiler, advantages=advantages[indices])
                token_factors, token_numeric = batch["token_factors"], batch["token_numeric"]
                legal_mask, token_lengths = batch["legal_mask"], batch["token_lengths"]
                actions, old_logprobs = batch["actions"], batch["old_logprobs"]
                adv, returns = batch["advantages"], batch["returns"]
                minibatch_tokens = sum(transition.token_length for transition in selected)
                executed_samples += len(selected)
                executed_tokens += minibatch_tokens
                executed_padded_input_tokens += len(selected) * (max(transition.token_length for transition in selected) + 1)
                with self._gpu_stage("update/model_forward"):
                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16,
                    ):
                        output = self.model(token_factors, token_numeric, legal_mask, token_lengths)
                with self._gpu_stage("update/distribution_and_loss"):
                    # The model promotes its policy/value outputs to FP32.  Keep
                    # the PPO ratio, loss and their gradients in FP32 as well.
                    logprobabilities = F.log_softmax(output["policy_logits"].float(), dim=-1)
                    logprob = logprobabilities.gather(1, actions[:, None]).squeeze(1)
                    old_logprobs, adv, returns = old_logprobs.float(), adv.float(), returns.float()
                    ratio = (logprob - old_logprobs).exp()
                    clipped = ratio.clamp(1 - float(self.hp["ppo_clip"]), 1 + float(self.hp["ppo_clip"])) * adv
                    policy_loss = -torch.minimum(ratio * adv, clipped).mean()
                    value_loss = nn.functional.mse_loss(output["value"].float(), returns)
                    valid_logprobabilities = torch.isfinite(logprobabilities)
                    probabilities = torch.where(
                        valid_logprobabilities, logprobabilities.exp(), torch.zeros_like(logprobabilities),
                    )
                    safe_logprobabilities = torch.where(
                        valid_logprobabilities, logprobabilities, torch.zeros_like(logprobabilities),
                    )
                    entropy = -(probabilities * safe_logprobabilities).sum(-1).mean()
                    loss = policy_loss + float(self.hp["value_coef"]) * value_loss - float(self.hp["entropy_coef"]) * entropy
                with self._gpu_stage("update/zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
                with self._gpu_stage("update/backward"):
                    loss.backward()
                with self._gpu_stage("update/gradient_clip"):
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), float(self.hp["max_grad_norm"]))
                with self._gpu_stage("update/optimizer_step"):
                    self.optimizer.step()
                with self._gpu_stage("update/diagnostic_metrics"):
                    with torch.no_grad():
                        kl = (old_logprobs - logprob).mean()
                        clipfrac = ((ratio - 1).abs() > float(self.hp["ppo_clip"])).float().mean()
                for name, value in (("loss", loss), ("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", entropy), ("approx_kl", kl), ("clipfrac", clipfrac), ("grad_norm", grad_norm)):
                    detached = value.detach()
                    if name in metric_totals:
                        metric_totals[name].add_(detached)
                    else:
                        metric_totals[name] = detached.clone()
                updates += 1
                if float(self.hp["target_kl"]) > 0 and float(kl) > float(self.hp["target_kl"]):
                    stop_early = True
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
            "update/executed_transition_input_tokens_mean": float(executed_tokens / executed_samples + 1),
            "update/executed_padded_input_tokens": float(executed_padded_input_tokens),
            "update/executed_padding_input_tokens": float(
                executed_padded_input_tokens - executed_tokens - executed_samples
            ),
            "update/executed_padding_fraction_of_padded_input_tokens": float(
                (executed_padded_input_tokens - executed_tokens - executed_samples)
                / max(executed_padded_input_tokens, 1)
            ),
        })
        names = tuple(metric_totals)
        values = torch.stack([metric_totals[name] for name in names]).div(max(updates, 1)).tolist()
        result = dict(zip(names, values)) | {"transitions": float(count)} | length_metrics
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
        torch.save({"model_schema": MODEL_SCHEMA_VERSION, "token_schema": TOKEN_SCHEMA_VERSION, "model": self.weights(), "optimizer": self.optimizer.state_dict(), "model_config": asdict(self.config), "train_config": train_config, "iteration": self.iteration, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("model_schema") != MODEL_SCHEMA_VERSION or payload.get("token_schema") != TOKEN_SCHEMA_VERSION:
            raise ValueError("checkpoint is not compatible with semantic Token V5")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload.get("iteration", 0))
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"])
            random.setstate(payload["python_rng"])
            np.random.set_state(payload["numpy_rng"])
