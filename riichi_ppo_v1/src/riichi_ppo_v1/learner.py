"""PPO optimisation and variable-length batch collation."""

from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .model import KyokuTransformerActorCritic, ModelConfig, NUM_ACTIONS
from .profiling import StageProfiler
from .trajectory import Transition


def transition_length_metrics(transitions: list[Transition], prefix: str = "update/buffer") -> dict[str, float]:
    """Describe the unpadded transition buffer supplied to a PPO update."""
    lengths = np.asarray([transition.length for transition in transitions], dtype=np.int64)
    history_lengths = np.asarray([transition.history_length for transition in transitions], dtype=np.int64)
    if np.any(lengths <= 0) or np.any(history_lengths < 0) or np.any(history_lengths >= lengths):
        raise ValueError("each transition must have a non-empty snapshot after its history prefix")
    snapshot_lengths = lengths - history_lengths
    return {
        f"{prefix}_transition_sequence_tokens_mean": float(lengths.mean()),
        f"{prefix}_transition_history_tokens_mean": float(history_lengths.mean()),
        f"{prefix}_transition_snapshot_tokens_mean": float(snapshot_lengths.mean()),
    }


def collate(transitions: list[Transition], device: torch.device, profiler: StageProfiler | None = None) -> dict[str, torch.Tensor]:
    if not transitions:
        raise ValueError("cannot collate an empty rollout")
    profile = profiler or StageProfiler(enabled=False)
    with profile.stage("update/collate_shape_and_allocate"):
        max_length = max(t.length for t in transitions)
        batch = len(transitions)
        ids = np.zeros((batch, max_length, 8), dtype=np.int64)
        attention = np.zeros((batch, max_length), dtype=np.bool_)
    with profile.stage("update/collate_host_padding_copy"):
        for row, transition in enumerate(transitions):
            ids[row, : transition.length] = transition.input_ids
            attention[row, : transition.length] = transition.attention_mask
        legal = np.stack([t.legal_mask for t in transitions])
    with profile.stage("update/collate_h2d"):
        return {
            "input_ids": torch.as_tensor(ids, device=device),
            "attention_mask": torch.as_tensor(attention, device=device),
            "sequence_lengths": torch.tensor([t.length for t in transitions], device=device),
            "history_lengths": torch.tensor([t.history_length for t in transitions], device=device),
            "legal_mask": torch.as_tensor(legal, device=device),
            "actions": torch.tensor([t.action for t in transitions], device=device, dtype=torch.long),
            "old_logprobs": torch.tensor([t.logprob for t in transitions], device=device),
            "advantages": torch.tensor([t.advantage for t in transitions], device=device),
            "returns": torch.tensor([t.return_ for t in transitions], device=device),
        }


class PPOLearner:
    def __init__(self, model_size: str, device: str, **hyperparameters: Any) -> None:
        self.device = torch.device(device)
        self.config = ModelConfig.preset(model_size)
        self.model = KyokuTransformerActorCritic(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(hyperparameters["learning_rate"]))
        self.hp = hyperparameters
        self.iteration = 0
        self.profiler = StageProfiler(enabled=bool(hyperparameters.get("profile_enabled", True)))
        self.profile_cuda_sync = bool(hyperparameters.get("profile_cuda_sync", True))

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
        with self._gpu_stage("update/collate_total"):
            batch = collate(transitions, self.device, self.profiler)
        with self._gpu_stage("update/advantage_normalize"):
            advantages = batch["advantages"]
            batch["advantages"] = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        total = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        updates = 0
        count = len(transitions)
        executed_samples = 0
        executed_sequence_tokens = torch.zeros((), device=self.device, dtype=torch.int64)
        executed_snapshot_tokens = torch.zeros((), device=self.device, dtype=torch.int64)
        self.model.train()
        stop_early = False
        for _ in range(int(self.hp["update_epochs"])):
            with self._gpu_stage("update/shuffle_indices"):
                order = torch.randperm(count, device=self.device)
            for start in range(0, count, int(self.hp["minibatch_size"])):
                with self._gpu_stage("update/minibatch_index_select"):
                    index = order[start : start + int(self.hp["minibatch_size"])]
                    input_ids, legal_mask = batch["input_ids"][index], batch["legal_mask"][index]
                    attention_mask, sequence_lengths = batch["attention_mask"][index], batch["sequence_lengths"][index]
                    history_lengths = batch["history_lengths"][index]
                    actions, old_logprobs = batch["actions"][index], batch["old_logprobs"][index]
                    adv, returns = batch["advantages"][index], batch["returns"][index]
                    executed_samples += int(index.numel())
                    executed_sequence_tokens += sequence_lengths.sum()
                    executed_snapshot_tokens += (sequence_lengths - history_lengths).sum()
                with self._gpu_stage("update/model_forward"):
                    output = self.model(input_ids, legal_mask, attention_mask, sequence_lengths)
                with self._gpu_stage("update/distribution_and_loss"):
                    distribution = Categorical(logits=output["policy_logits"])
                    logprob = distribution.log_prob(actions)
                    ratio = (logprob - old_logprobs).exp()
                    clipped = ratio.clamp(1 - float(self.hp["ppo_clip"]), 1 + float(self.hp["ppo_clip"])) * adv
                    policy_loss = -torch.minimum(ratio * adv, clipped).mean()
                    value_loss = nn.functional.mse_loss(output["value"], returns)
                    entropy = distribution.entropy().mean()
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
                    total.setdefault(name, 0.0)
                    total[name] += float(value.detach())
                updates += 1
                if float(self.hp["target_kl"]) > 0 and float(kl) > float(self.hp["target_kl"]):
                    stop_early = True
                    break
            if stop_early:
                break
        self.iteration += 1
        if not executed_samples:
            raise RuntimeError("PPO update completed without a minibatch")
        length_metrics.update({
            "update/executed_transition_samples": float(executed_samples),
            "update/executed_transition_sequence_tokens_mean": float(executed_sequence_tokens.item() / executed_samples),
            "update/executed_transition_snapshot_tokens_mean": float(executed_snapshot_tokens.item() / executed_samples),
        })
        result = {name: value / max(updates, 1) for name, value in total.items()} | {"transitions": float(count)} | length_metrics
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
        torch.save({"model": self.weights(), "optimizer": self.optimizer.state_dict(), "model_config": asdict(self.config), "train_config": train_config, "iteration": self.iteration, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload.get("iteration", 0))
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"])
            random.setstate(payload["python_rng"])
            np.random.set_state(payload["numpy_rng"])
