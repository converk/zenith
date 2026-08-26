"""V18 SFT 契约哈希的动态推导与 fail-closed 校验。"""

from __future__ import annotations

from riichi_ppo_v1.sft.contract import (
    ACTOR_INPUT_CONTRACT_SHA256,
    STATE_PROTOCOL_VERSION,
    validate_manifest,
)
from riichi_ppo_v1.sft.precompute import precompute  # 保证模块可导入


def _base_manifest() -> dict[str, object]:
    return {
        "format": "riichi-sft-encoded-v18",
        "encoding_protocol_version": 18,
        "encoding_contract_sha256": ACTOR_INPUT_CONTRACT_SHA256,
        "state_protocol": STATE_PROTOCOL_VERSION,
        "source_manifest_sha256": "abc",
        "counts": {
            "train_kyokus": 1, "validation_kyokus": 1,
            "train_decisions": 2, "validation_decisions": 2,
        },
    }


def test_contract_hash_is_stable_sha256() -> None:
    assert len(ACTOR_INPUT_CONTRACT_SHA256) == 64
    assert all(character in "0123456789abcdef" for character in ACTOR_INPUT_CONTRACT_SHA256)


def test_manifest_fail_closed() -> None:
    validate_manifest(_base_manifest())
    for mutate in (
        {"format": "riichi-sft-encoded-v19"},
        {"encoding_protocol_version": 17},
        {"encoding_contract_sha256": "0" * 64},
        {"state_protocol": "old"},
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
