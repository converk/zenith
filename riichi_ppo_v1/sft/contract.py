"""受支持 v13 SFT 路径的唯一 fail-closed 契约边界。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..model.encoding_protocol import (
    ENCODED_FORMAT as V16_ENCODED_FORMAT,
    ENCODING_PROTOCOL_VERSION,
)
from ..model.feature_schema import ENCODED_FORMAT
from ..model.schema import NUM_ACTIONS

SFT_CONTRACT_VERSION = "riichi-sft-v13-1"
RUNTIME_CONTRACT_ID = "riichi-runtime-v13-1"
DATA_PLAN_VERSION = 1
DATA_CURSOR_VERSION = 1
TRAINING_MODES = frozenset({"actor_only", "actor_public_value", "joint_actor_critic"})
# 固定 SFT 节奏(宪法原则 IV):验证、启发式评测与 checkpoint 保存每 3000 steps
# 一次,最终评估保持 96 半庄。参数只在代码中定义一处,禁止在实验配置里复制。
SFT_CADENCE_STEPS = 3000
SFT_FINAL_EVAL_HANCHAN_COUNT = 96
_FORMAL_V13_MANIFEST_CONTRACT = (
    13,
    "ad8dc752f116d6d6430930e16c6a17322b3da980549d3350a5ddc461ee123036",
    4,
    16,
)
# v16 协议契约的冻结内容哈希(specs/003-v16-model-rework/contracts/
# actor-input-v16.md),数据集 manifest 与校验器共用此单一来源。
V16_ACTOR_INPUT_CONTRACT_SHA256 = (
    "56874dfb4738af3a506221c1001083ac67fe5188aa2042ae1576f5a852a7ed3b"
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


def validate_v16_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on the v16 单协议版本 manifest 契约。

    只接受单一 ``format=riichi-sft-encoded-v16``、单一
    ``encoding_protocol_version=16`` 与冻结的协议契约 sha256;多版本字段一律
    视为未知格式拒绝。
    """
    if manifest.get("format") != V16_ENCODED_FORMAT:
        raise RuntimeError("only the v16 encoded SFT format is supported")
    if manifest.get("encoding_protocol_version") != ENCODING_PROTOCOL_VERSION:
        raise RuntimeError(
            f"v16 SFT manifest requires encoding_protocol_version={ENCODING_PROTOCOL_VERSION}"
        )
    if manifest.get("encoding_contract_sha256") != V16_ACTOR_INPUT_CONTRACT_SHA256:
        raise RuntimeError(
            "encoded dataset carries an unknown v16 protocol contract hash"
        )
    if not isinstance(manifest.get("source_manifest_sha256"), str) or not manifest["source_manifest_sha256"]:
        raise RuntimeError("v16 SFT manifest lacks source_manifest_sha256")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(name, 0)) <= 0
        for name in ("train_kyokus", "validation_kyokus", "train_decisions", "validation_decisions")
    ):
        raise RuntimeError("v16 SFT requires positive train/validation kyoku and decision counts")


def validate_v15_reused_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on the immutable V13 cache fields required by V15 SFT."""
    validate_v13_manifest(manifest)
    counts = manifest.get("counts")
    statistics = manifest.get("field_statistics")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(name, 0)) <= 0
        for name in ("train_kyokus", "validation_kyokus", "train_decisions", "validation_decisions")
    ):
        raise RuntimeError("V15 SFT requires positive train/validation kyoku and decision counts")
    if not isinstance(statistics, Mapping):
        raise RuntimeError("V15 SFT requires audited field_statistics")
    for name in ("legal_action_id_counts", "expert_action_id_counts"):
        coverage = statistics.get(name)
        if (
            not isinstance(coverage, list)
            or len(coverage) != NUM_ACTIONS
            or any(int(value) <= 0 for value in coverage)
        ):
            raise RuntimeError(f"V15 SFT requires complete {NUM_ACTIONS}-action {name}")
    if manifest.get("complete_action_coverage_required") is not True:
        raise RuntimeError("V15 SFT manifest does not require complete action coverage")
    if manifest.get("ordered_public_history_verified") is not True:
        raise RuntimeError("V15 SFT manifest lacks ordered public-history verification")
    if manifest.get("audit_skipped") is not False or not manifest.get("audit_reports"):
        raise RuntimeError("V15 SFT requires completed cache audit reports")


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
