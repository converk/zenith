"""Single fail-closed contract boundary for the supported v13 SFT path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..model.feature_schema import ENCODED_FORMAT

SFT_CONTRACT_VERSION = "riichi-sft-v13-1"
RUNTIME_CONTRACT_ID = "riichi-runtime-v13-1"
DATA_PLAN_VERSION = 1
DATA_CURSOR_VERSION = 1
TRAINING_MODES = frozenset({"actor_only", "actor_public_value", "joint_actor_critic"})
_FORMAL_V13_MANIFEST_CONTRACT = (
    13,
    "ad8dc752f116d6d6430930e16c6a17322b3da980549d3350a5ddc461ee123036",
    4,
    16,
)


def dataset_manifest_hash(dataset: Path) -> str:
    return hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()


def load_manifest(dataset: Path) -> dict[str, Any]:
    path = dataset / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read SFT dataset manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"SFT dataset manifest must be an object: {path}")
    return value


def validate_v13_manifest(manifest: Mapping[str, Any]) -> None:
    """Accept the current formal cache or the compact replacement contract.

    The formal v13 cache is immutable and predates ``sft_contract_version``.
    Its exact four-field tuple is recognized explicitly, not inferred.  Newly
    generated caches write only the unified contract identifier.
    """
    if manifest.get("format") != ENCODED_FORMAT:
        raise RuntimeError("only the v13 encoded SFT format is supported")
    contract = manifest.get("sft_contract_version")
    if contract is not None:
        if contract != SFT_CONTRACT_VERSION:
            raise RuntimeError(f"unsupported SFT contract: {contract!r}")
        return
    legacy_tuple = (
        manifest.get("token_schema_version"),
        manifest.get("feature_schema_sha256"),
        manifest.get("rust_analysis_version"),
        manifest.get("decision_analysis_version"),
    )
    if legacy_tuple != _FORMAL_V13_MANIFEST_CONTRACT:
        raise RuntimeError(
            "encoded dataset lacks the supported v13 SFT contract; re-encode it"
        )


def training_mode(config: Mapping[str, Any]) -> str:
    if bool(config["train_critic"]):
        return "joint_actor_critic"
    if bool(config.get("train_public_value", False)):
        return "actor_public_value"
    return "actor_only"


def assert_runtime_contract() -> None:
    """Check the two native boundaries represented by one runtime ID."""
    import riichi
    import riichienv

    if getattr(riichi, "ANALYSIS_VERSION", None) != 4:
        raise RuntimeError(f"installed riichi extension violates {RUNTIME_CONTRACT_ID}")
    if getattr(riichienv, "REPLAY_SEMANTICS_VERSION", None) != 1:
        raise RuntimeError(f"installed RiichiEnv extension violates {RUNTIME_CONTRACT_ID}")
