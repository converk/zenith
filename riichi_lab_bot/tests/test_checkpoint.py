from __future__ import annotations

import os
import random

import pytest
import torch

from conftest import default_checkpoint, v16_sft_checkpoint, v17_ppo_checkpoint
from riichi_lab_bot.bridge import OnlineStateBridge
from riichi_lab_bot.policy import PolicyEngine


@pytest.mark.parametrize(
    ("checkpoint", "checkpoint_format"),
    [
        (v16_sft_checkpoint, "sft_v16"),
        (v17_ppo_checkpoint, "ppo_v3"),
    ],
)
def test_real_v16_v17_checkpoint_loads_strictly_and_warms_up(
    checkpoint, checkpoint_format: str,
) -> None:
    engine = PolicyEngine(
        checkpoint(), device="cpu", dtype="fp32"
    )
    assert engine.config.context_tokens == 4096
    assert engine.metadata["checkpoint_format"] == checkpoint_format
    assert engine.metadata["token_schema_version"] == 16
    assert engine.metadata["policy_head_type"] == "symmetric_action_query"
    assert engine.warmup() >= 0.0


def test_missing_model_config_fails_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    path = tmp_path / "bad.pt"
    path.touch()
    monkeypatch.setattr("torch.load", lambda *args, **kwargs: {"model": {}})
    with pytest.raises(ValueError, match="model_config"):
        PolicyEngine(path, device="cpu", dtype="fp32")


def test_non_symmetric_policy_head_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    path = tmp_path / "bad_head.pt"
    path.touch()
    monkeypatch.setattr(
        "torch.load",
        lambda *args, **kwargs: {
            "model_config": {"policy_head_type": "unsupported_query"},
            "model": {},
        },
    )
    with pytest.raises(RuntimeError, match="symmetric_action_query"):
        PolicyEngine(path, device="cpu", dtype="fp32")


def _first_legal_prepared(seed: int = 20260730):
    from riichienv import RiichiEnv
    from riichi_lab_bot.local_play import observation_with_events

    env = RiichiEnv(game_mode="4p-red-half", seed=seed)
    observations = env.reset()
    pending = {seat: [] for seat in range(4)}
    rng = random.Random(seed)
    for _step in range(4000):
        for seat, observation in observations.items():
            pending[int(seat)].extend(observation.new_events())
        for seat, observation in observations.items():
            if not observation.legal_actions():
                continue
            server_observation = observation_with_events(
                observation, pending[int(seat)]
            )
            return OnlineStateBridge(int(seat)).prepare(server_observation)
        actions = {
            seat: rng.choice(observation.legal_actions())
            for seat, observation in observations.items()
            if observation.legal_actions()
        }
        if not actions:
            raise RuntimeError("local environment stalled before a decision")
        observations = env.step(actions)
    raise RuntimeError("no legal decision found in 4000 steps")


def test_fp32_and_bf16_inference_agree_on_l20() -> None:
    visible = os.environ.get("CUDA_DEVICE", "")
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or not any(device in visible.split(",") for device in ("2", "3"))
    ):
        pytest.skip("requires CUDA_DEVICE=2,3 on an L20")
    prepared = _first_legal_prepared()
    fp32 = PolicyEngine(default_checkpoint(), device="cpu", dtype="fp32")
    bf16 = PolicyEngine(default_checkpoint(), device="cuda:0", dtype="bf16")
    fp_result = fp32.infer(prepared)
    bf_result = bf16.infer(prepared)
    assert fp_result.action_id == bf_result.action_id
