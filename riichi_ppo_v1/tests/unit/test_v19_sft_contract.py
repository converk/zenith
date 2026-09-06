"""V19 SFT 契约哈希的动态推导与 fail-closed 校验。"""

from __future__ import annotations

from pathlib import Path

from riichi_ppo_v1.sft.contract import (
    ACTOR_INPUT_CONTRACT_SHA256,
    BELIEF_LABEL_SHAPES,
    STATE_PROTOCOL_VERSION,
    validate_manifest,
)
from riichi_ppo_v1.sft.trainer import load_config


def _base_manifest() -> dict[str, object]:
    return {
        "format": "riichi-sft-encoded-v19",
        "encoding_protocol_version": 19,
        "encoding_contract_sha256": ACTOR_INPUT_CONTRACT_SHA256,
        "state_protocol": STATE_PROTOCOL_VERSION,
        "token_row_width": 32,
        "token_numeric_width": 8,
        "context_tokens": 320,
        "numeric_dtype": "float32",
        "legal_encoding": "packbits-little-241",
        "actor_only": True,
        "belief_labels": True,
        "belief_shape": BELIEF_LABEL_SHAPES,
        "subset_denominator": 5,
        "subset_remainders": [0, 1, 2],
        "game_sample_denominator": 1,
        "game_sample_remainder": 0,
        "source_manifest_sha256": "abc",
        "counts": {
            "train_kyokus": 1, "validation_kyokus": 1,
            "train_decisions": 2, "validation_decisions": 2,
        },
    }


def test_contract_hash_is_stable_sha256() -> None:
    assert len(ACTOR_INPUT_CONTRACT_SHA256) == 64
    assert all(character in "0123456789abcdef" for character in ACTOR_INPUT_CONTRACT_SHA256)


def test_belief_shape_contract() -> None:
    assert BELIEF_LABEL_SHAPES == {
        "hand": [102],
        "shanten": [3],
        "wait": [105],
        "danger": [102],
        "loss": [102],
    }


def test_standard_sft_config_does_not_copy_cadence_constants() -> None:
    """V19 标准 SFT 配置不能复制 3000 步/96 半庄等机制常量（sft/contract.py 单点）。"""
    path = Path(__file__).resolve().parents[2] / "configs" / "v19_sft.yaml"
    config = load_config(path)
    assert "validation_interval_steps" not in config
    assert "checkpoint_interval_steps" not in config
    assert "validation_interval" not in config
    assert "checkpoint_interval" not in config


def test_manifest_fail_closed() -> None:
    validate_manifest(_base_manifest())
    for mutate in (
        {"format": "riichi-sft-encoded-v18"},
        {"encoding_protocol_version": 18},
        {"encoding_contract_sha256": "0" * 64},
        {"state_protocol": "riichi-current-state-v18-1"},
        {"token_row_width": 30},
        {"token_numeric_width": 4},
        {"context_tokens": 128},
        {"numeric_dtype": "float64"},
        {"legal_encoding": "packbits-little-240"},
        {"actor_only": False},
        {"belief_labels": False},
        {"belief_labels": "yes"},
        {"belief_shape": {"hand": [102], "shanten": [3], "wait": [105],
                          "danger": [102], "loss": [101]}},
        {"belief_shape": None},
        {"subset_denominator": 0},
        {"subset_remainders": [7]},
        {"game_sample_denominator": 0},
        {"game_sample_remainder": 2},
        {"source_manifest_sha256": ""},
        {"counts": {"train_kyokus": 0, "validation_kyokus": 1,
                    "train_decisions": 1, "validation_decisions": 1}},
    ):
        manifest = _base_manifest()
        manifest.update(mutate)
        try:
            validate_manifest(manifest)
        except RuntimeError:
            continue
        raise AssertionError(f"manifest accepted invalid mutation: {mutate}")
