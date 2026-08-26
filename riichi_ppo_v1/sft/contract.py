"""V18 当前局面 SFT 路径的唯一 fail-closed 契约边界。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..model.encoding_protocol import (
    ACTION_TYPE_CARDINALITY,
    CATEGORY_SCHEMAS,
    CONTEXT_TOKENS,
    DEFENSE_SLOT_ORDER,
    ENCODED_FORMAT,
    ENCODING_PROTOCOL_VERSION,
    KIND_BOS,
    NUM_SEPARATORS,
    OFFENSE_SLOT_ORDER,
    QUERY_ROW_WIDTH,
    QUERY_SLOT_COUNT,
    SEPARATOR_IDS,
    SLOT_CARDINALITIES,
    STATE_PROTOCOL_VERSION,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
)
from ..model.schema import NUM_ACTIONS, TILE_KINDS, TID_COUNT

SFT_CONTRACT_VERSION = "riichi-sft-v18-2"
RUNTIME_CONTRACT_ID = "riichi-runtime-v18-1"
DATA_PLAN_VERSION = 2
DATA_CURSOR_VERSION = 1
TRAINING_MODES = frozenset({"actor_only", "actor_public_value", "joint_actor_critic"})

# 固定 SFT 节奏(宪法原则 IV):验证、checkpoint 保存每 3000 steps 一次,
# 最终评估保持 96 半庄。参数只在代码中定义一处,禁止在实验配置里复制。
SFT_CADENCE_STEPS = 3000
SFT_FINAL_EVAL_HANCHAN_COUNT = 96

# V18 当前局面输入契约的规范化载荷：协议版本、格式、token 行宽/数值宽、
# 类别 schema 全表（kind/name/segment/离散字段基数/数值字段）、分隔符、动作空间。
# 任何一项变化都会使哈希变化，旧数据集 manifest 会 fail closed。
_ACTOR_INPUT_CONTRACT_PAYLOAD = {
    "protocol_version": ENCODING_PROTOCOL_VERSION,
    "encoded_format": ENCODED_FORMAT,
    "state_protocol": STATE_PROTOCOL_VERSION,
    "token_row_width": TOKEN_ROW_WIDTH,
    "token_numeric_width": TOKEN_NUMERIC_WIDTH,
    "context_tokens": CONTEXT_TOKENS,
    "num_separators": NUM_SEPARATORS,
    "separator_ids": SEPARATOR_IDS,
    "categories": [
        (
            schema.kind, schema.name, schema.segment, schema.cls,
            tuple((field.name, field.cardinality) for field in schema.discrete),
            tuple((field.name, field.scale) for field in schema.numeric),
            schema.slot_count,
        )
        for schema in sorted(CATEGORY_SCHEMAS.values(), key=lambda value: value.kind)
    ],
    "query_row_width": QUERY_ROW_WIDTH,
    "query_slot_count": QUERY_SLOT_COUNT,
    "action_type_cardinality": ACTION_TYPE_CARDINALITY,
    "num_actions": NUM_ACTIONS,
    "tile_kinds": TILE_KINDS,
    "tid_count": TID_COUNT,
    "offense_slot_order": OFFENSE_SLOT_ORDER,
    "defense_slot_order": DEFENSE_SLOT_ORDER,
    "slot_cardinalities": SLOT_CARDINALITIES,
}
ACTOR_INPUT_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(_ACTOR_INPUT_CONTRACT_PAYLOAD, sort_keys=True, ensure_ascii=False).encode("utf-8")
).hexdigest()


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


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """对 V18 当前局面单版本 manifest 执行 fail-closed 校验。"""
    if manifest.get("format") != ENCODED_FORMAT:
        raise RuntimeError("only the V18 encoded SFT format is supported")
    if manifest.get("encoding_protocol_version") != ENCODING_PROTOCOL_VERSION:
        raise RuntimeError(
            f"V18 SFT manifest requires encoding_protocol_version={ENCODING_PROTOCOL_VERSION}"
        )
    if manifest.get("encoding_contract_sha256") != ACTOR_INPUT_CONTRACT_SHA256:
        raise RuntimeError(
            "encoded dataset carries an unknown V18 protocol contract hash"
        )
    if manifest.get("state_protocol") != STATE_PROTOCOL_VERSION:
        raise RuntimeError("encoded dataset carries an unknown state protocol version")
    if not isinstance(manifest.get("source_manifest_sha256"), str) or not manifest["source_manifest_sha256"]:
        raise RuntimeError("V18 SFT manifest lacks source_manifest_sha256")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(name, 0)) <= 0
        for name in (
            "train_kyokus", "validation_kyokus",
            "train_decisions", "validation_decisions",
        )
    ):
        raise RuntimeError(
            "V18 SFT requires positive train/validation kyoku and decision counts"
        )


def training_mode(config: Mapping[str, Any]) -> str:
    if bool(config["train_critic"]):
        return "joint_actor_critic"
    if bool(config.get("train_public_value", False)):
        return "actor_public_value"
    return "actor_only"


def assert_runtime_contract() -> None:
    """检查 V18 输入边界依赖的两个原生扩展版本。"""
    import riichi
    import riichienv

    if getattr(riichi, "ANALYSIS_VERSION", None) != 4:
        raise RuntimeError(f"installed riichi extension violates {RUNTIME_CONTRACT_ID}")
    if getattr(riichienv, "REPLAY_SEMANTICS_VERSION", None) != 1:
        raise RuntimeError(f"installed RiichiEnv extension violates {RUNTIME_CONTRACT_ID}")
    if getattr(riichi, "ENCODING_PROTOCOL_VERSION", None) != ENCODING_PROTOCOL_VERSION:
        raise RuntimeError(f"installed riichi extension violates {RUNTIME_CONTRACT_ID}")
