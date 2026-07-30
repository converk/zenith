from __future__ import annotations

import pytest

from conftest import default_checkpoint
from riichi_lab_bot.model import TOKEN_SCHEMA_VERSION
from riichi_lab_bot.policy import PolicyEngine


def test_real_checkpoint_loads_strictly_and_warms_up() -> None:
    engine = PolicyEngine(
        default_checkpoint(), device="cpu", dtype="fp32"
    )
    assert engine.config.context_tokens == 4096
    assert engine.warmup() >= 0.0


def test_incompatible_schema_fails_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "bad.pt"
    path.touch()
    monkeypatch.setattr(
        "torch.load",
        lambda *args, **kwargs: {
            "token_schema_version": TOKEN_SCHEMA_VERSION - 1
        },
    )
    with pytest.raises(ValueError, match="incompatible token schema"):
        PolicyEngine(path, device="cpu", dtype="fp32")

