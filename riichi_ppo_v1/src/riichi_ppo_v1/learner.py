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
    lengths = np.asarray([transition.block_length for transition in transitions], dtype=np.int64)
    if np.any(lengths < 0):
        raise ValueError("V4 block length cannot be negative")
    return {
        f"{prefix}_transition_event_blocks_mean": float(lengths.mean()),
        f"{prefix}_transition_input_tokens_mean": float(lengths.mean() + 12),
        f"{prefix}_transition_board_tokens_mean": 12.0,
    }


def collate(transitions: list[Transition], device: torch.device, profiler: StageProfiler | None = None) -> dict[str, torch.Tensor]:
    if not transitions:
        raise ValueError("cannot collate an empty rollout")
    profile = profiler or StageProfiler(enabled=False)
    with profile.stage("update/collate_shape_and_allocate"):
        max_length = max(t.block_length for t in transitions)
        batch = len(transitions)
        kinds = np.zeros((batch, max_length), dtype=np.uint8)
        turn = np.zeros((batch, max_length, 4, 4), dtype=np.uint8)
        meld = np.zeros((batch, max_length, 8), dtype=np.uint8)
        board = np.stack([t.board_state for t in transitions])
    with profile.stage("update/collate_host_padding_copy"):
        for row, transition in enumerate(transitions):
            kinds[row, : transition.block_length] = transition.event_kinds
            turn[row, : transition.block_length] = transition.turn_fields
            meld[row, : transition.block_length] = transition.meld_fields
        legal = np.stack([t.legal_mask for t in transitions])
    with profile.stage("update/collate_h2d"):
        return {
            "block_kinds": torch.as_tensor(kinds, device=device),
            "turn_fields": torch.as_tensor(turn, device=device),
            "meld_fields": torch.as_tensor(meld, device=device),
            "board_state": torch.as_tensor(board, device=device),
            "block_lengths": torch.tensor([t.block_length for t in transitions], device=device),
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
        with self._gpu_stage("update/collate_total"):
            batch = collate(transitions, self.device, self.profiler)
        with self._gpu_stage("update/advantage_normalize"):
            advantages = batch["advantages"]
            batch["advantages"] = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        total = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        updates = 0
        count = len(transitions)
        configured_epochs = int(self.hp["update_epochs"])
        minibatch_size = int(self.hp["minibatch_size"])
        planned_minibatches_per_epoch = (count + minibatch_size - 1) // minibatch_size
        epochs_started = 0
        epochs_completed = 0
        executed_samples = 0
        executed_event_blocks = torch.zeros((), device=self.device, dtype=torch.int64)
        self.model.train()
        stop_early = False
        for _ in range(configured_epochs):
            epochs_started += 1
            with self._gpu_stage("update/shuffle_indices"):
                order = torch.randperm(count, device=self.device)
            for start in range(0, count, minibatch_size):
                with self._gpu_stage("update/minibatch_index_select"):
                    index = order[start : start + minibatch_size]
                    block_kinds, legal_mask = batch["block_kinds"][index], batch["legal_mask"][index]
                    turn_fields, meld_fields = batch["turn_fields"][index], batch["meld_fields"][index]
                    board_state, block_lengths = batch["board_state"][index], batch["block_lengths"][index]
                    actions, old_logprobs = batch["actions"][index], batch["old_logprobs"][index]
                    adv, returns = batch["advantages"][index], batch["returns"][index]
                    executed_samples += int(index.numel())
                    executed_event_blocks += block_lengths.sum()
                with self._gpu_stage("update/model_forward"):
                    output = self.model(block_kinds, turn_fields, meld_fields, board_state, legal_mask, block_lengths)
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
            "update/executed_transition_event_blocks_mean": float(executed_event_blocks.item() / executed_samples),
            "update/executed_transition_input_tokens_mean": float(executed_event_blocks.item() / executed_samples + 12),
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
