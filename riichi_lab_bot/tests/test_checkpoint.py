from __future__ import annotations

import json
import os
import random

import pytest
import torch
from conftest import default_checkpoint

from riichi_lab_bot.bridge import OnlineStateBridge
from riichi_lab_bot.policy import PolicyEngine


@pytest.mark.parametrize(
    "policy_head_type", ["unsupported_query", "query_snapshot", "legacy_snapshot"]
)
def test_legacy_checkpoint_is_rejected(tmp_path, policy_head_type) -> None:
    path = tmp_path / "legacy.pt"
    torch.save(
        {"model_config": {"policy_head_type": policy_head_type}, "model": {}},
        path,
    )
    with pytest.raises(RuntimeError, match="current_state_snapshot"):
        PolicyEngine(path, device="cpu", dtype="fp32")


def test_missing_model_config_fails_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    path = tmp_path / "bad.pt"
    path.touch()
    monkeypatch.setattr("torch.load", lambda *args, **kwargs: {"model": {}})
    with pytest.raises(ValueError, match="model_config"):
        PolicyEngine(path, device="cpu", dtype="fp32")


def test_non_isolated_policy_head_is_rejected(
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
    with pytest.raises(RuntimeError, match="current_state_snapshot"):
        PolicyEngine(path, device="cpu", dtype="fp32")


def _first_legal_prepared(seed: int = 20260730, seat: int | None = None):
    from riichienv import RiichiEnv

    from riichi_lab_bot.local_play import observation_with_events

    env = RiichiEnv(game_mode="4p-red-half", seed=seed)
    observations = env.reset()
    pending = {seat: [] for seat in range(4)}
    rng = random.Random(seed)
    for _step in range(4000):
        for view_seat, observation in observations.items():
            pending[int(view_seat)].extend(observation.new_events())
        for view_seat, observation in observations.items():
            if not observation.legal_actions():
                continue
            if seat is not None and int(view_seat) != seat:
                continue
            server_observation = observation_with_events(
                observation, pending[int(view_seat)]
            )
            bridge = OnlineStateBridge(int(view_seat))
            return bridge, bridge.prepare(server_observation)
        actions = {
            view_seat: rng.choice(observation.legal_actions())
            for view_seat, observation in observations.items()
            if observation.legal_actions()
        }
        if not actions:
            raise RuntimeError("local environment stalled before a decision")
        observations = env.step(actions)
    raise RuntimeError("no legal decision found in 4000 steps")


def test_v19_checkpoint_end_to_end_decision() -> None:
    """加载→warmup→编码→forward→解码→安全校验的完整 V19 链路冒烟。"""
    from riichi_lab_bot.safety import choose_safe_response

    policy = PolicyEngine(default_checkpoint(), device="cpu", dtype="fp32")
    assert policy.metadata["policy_head_type"] == "current_state_snapshot"
    # V19 拓扑:共享 3 + actor 2 + critic 1;context 320。
    assert policy.config.layers == 5
    assert policy.config.shared_layers == 3
    assert policy.config.critic_layers == 1
    assert policy.config.context_tokens == 320
    assert policy.warmup() >= 0.0
    bridge, prepared = _first_legal_prepared()
    result = policy.infer(prepared)
    action = bridge.decode(prepared, result.action_id)
    possible = [json.loads(value) for value in prepared.legal_jsons]
    safe = choose_safe_response(prepared, action, possible, 1)
    assert safe.source == "model"
    assert safe.payload is not None


def test_v19_nonzero_seat_prepare_and_decode() -> None:
    """非 0 座位的桥接必须用正确 batch_index 登记合法动作并解码。"""
    for seat in (1, 2, 3):
        bridge, prepared = _first_legal_prepared(seed=20260730 + seat, seat=seat)
        action_id = int(
            next(
                index
                for index, legal in enumerate(prepared.legal_mask)
                if bool(legal)
            )
        )
        action = bridge.decode(prepared, action_id)
        assert action is not None


def test_fp32_and_bf16_inference_agree_on_l20() -> None:
    visible = os.environ.get("CUDA_DEVICE", "")
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or not any(device in visible.split(",") for device in ("2", "3"))
    ):
        pytest.skip("requires CUDA_DEVICE=2,3 on an L20")
    _bridge, prepared = _first_legal_prepared()
    fp32 = PolicyEngine(default_checkpoint(), device="cpu", dtype="fp32")
    bf16 = PolicyEngine(default_checkpoint(), device="cuda:0", dtype="bf16")
    fp_result = fp32.infer(prepared)
    bf_result = bf16.infer(prepared)
    assert fp_result.action_id == bf_result.action_id
