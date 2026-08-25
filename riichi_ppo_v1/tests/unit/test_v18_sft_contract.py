"""V18 SFT 契约哈希的动态推导与 fail-closed 校验。"""

import hashlib
import json

import pytest

from riichi_ppo_v1.sft.contract import (
    ACTOR_INPUT_CONTRACT_SHA256,
    _ACTOR_INPUT_CONTRACT_PAYLOAD,
    validate_manifest,
)


def test_contract_hash_is_deterministic_and_covers_full_schema() -> None:
    assert len(ACTOR_INPUT_CONTRACT_SHA256) == 64
    assert all(char in "0123456789abcdef" for char in ACTOR_INPUT_CONTRACT_SHA256)
    payload = _ACTOR_INPUT_CONTRACT_PAYLOAD
    assert payload["protocol_version"] == 18
    assert payload["snapshot_field_count"] == 54
    assert payload["num_actions"] == 241
    assert len(payload["snapshot_schema"]) == 54
    assert [field[0] for field in payload["snapshot_schema"]] == list(range(1, 55))
    assert ACTOR_INPUT_CONTRACT_SHA256 == hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def test_manifest_fail_closed_rejects_unknown_contract_hash() -> None:
    manifest = {
        "format": "riichi-sft-encoded-v18",
        "encoding_protocol_version": 18,
        "encoding_contract_sha256": "0" * 64,
        "source_manifest_sha256": "sample",
        "counts": {
            "train_kyokus": 1, "validation_kyokus": 1,
            "train_decisions": 1, "validation_decisions": 1,
        },
    }
    with pytest.raises(RuntimeError, match="contract hash"):
        validate_manifest(manifest)
    manifest["encoding_contract_sha256"] = ACTOR_INPUT_CONTRACT_SHA256
    validate_manifest(manifest)
