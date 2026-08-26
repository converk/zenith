"""V18 当前局面协议单点来源测试。"""

from __future__ import annotations

from riichi_ppo_v1.model.encoding_protocol import (
    ACTION_TYPE_CARDINALITY,
    CATEGORY_SCHEMAS,
    CONTEXT_TOKENS,
    DEFENSE_SLOT_ORDER,
    ENCODED_FORMAT,
    ENCODING_PROTOCOL_VERSION,
    KIND_BOS,
    KIND_CRITIC_FUTURE,
    KIND_CRITIC_HAND,
    KIND_SEP_ACTIONS,
    KIND_SEP_CRITIC,
    KIND_SEP_OPPONENT_ANALYSIS,
    KIND_TABLE,
    KIND_TILE_STATE,
    NUM_SEPARATORS,
    OFFENSE_SLOT_ORDER,
    OPEN_MELD_YAKUHAI_HAN_OVERFLOW_BUCKET,
    QUERY_ROW_WIDTH,
    QUERY_SLOT_COUNT,
    SEPARATOR_IDS,
    SEPARATOR_KINDS,
    SEPARATOR_SEGMENTS,
    SEGMENT_ACTIONS,
    SEGMENT_ANALYSIS,
    SEGMENT_CRITIC_FUTURE,
    SEGMENT_CRITIC_PRIVATE,
    SEGMENT_SHARED,
    SLOT_CARDINALITIES,
    STATE_PROTOCOL_VERSION,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
    VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET,
)


def test_protocol_version_and_widths() -> None:
    assert ENCODING_PROTOCOL_VERSION == 18
    assert ENCODED_FORMAT == "riichi-sft-encoded-v18"
    assert STATE_PROTOCOL_VERSION.startswith("riichi-current-state-v18")
    assert TOKEN_ROW_WIDTH == 32
    assert TOKEN_NUMERIC_WIDTH == 8
    assert CONTEXT_TOKENS == 256
    assert QUERY_ROW_WIDTH == 15
    assert QUERY_SLOT_COUNT == 10


def test_separators_single_source() -> None:
    assert len(SEPARATOR_IDS) == 11
    assert NUM_SEPARATORS == 11
    for name, separator_id in SEPARATOR_IDS.items():
        assert SEPARATOR_KINDS[name] == 100 + separator_id
    assert SEPARATOR_SEGMENTS[KIND_SEP_OPPONENT_ANALYSIS] == SEGMENT_ANALYSIS
    assert SEPARATOR_SEGMENTS[KIND_SEP_ACTIONS] == SEGMENT_ACTIONS
    assert SEPARATOR_SEGMENTS[KIND_SEP_CRITIC] == SEGMENT_CRITIC_PRIVATE


def test_category_schema_domains() -> None:
    assert set(CATEGORY_SCHEMAS) == {
        KIND_BOS, KIND_TABLE, 3, 4, 5, 6, 7, 8, KIND_TILE_STATE, 10, 11, 12,
        KIND_CRITIC_HAND, KIND_CRITIC_FUTURE,
    }
    for kind, schema in CATEGORY_SCHEMAS.items():
        assert schema.kind == kind
        assert schema.segment in (1, 2, 3, 4, 5)
        assert schema.cls in ("SIMPLE", "DENSE", "SEPARATOR")
        assert all(field.cardinality > 0 for field in schema.discrete)
    assert CATEGORY_SCHEMAS[KIND_TABLE].segment == SEGMENT_SHARED
    assert CATEGORY_SCHEMAS[KIND_TILE_STATE].segment == SEGMENT_SHARED


def test_query_slot_orders_and_cardinalities() -> None:
    assert OFFENSE_SLOT_ORDER == tuple(f"O{i}" for i in range(10))
    assert DEFENSE_SLOT_ORDER == tuple(f"D{i}" for i in range(10))
    assert ACTION_TYPE_CARDINALITY == 12
    for slot in OFFENSE_SLOT_ORDER + DEFENSE_SLOT_ORDER:
        assert SLOT_CARDINALITIES[slot] > 0


def test_overflow_bucket_constants_mirror_rust() -> None:
    """Rust 单源溢出桶常量与 Python 镜像一致,且与类别 schema 的「+1 基数」约定相符。"""
    assert OPEN_MELD_YAKUHAI_HAN_OVERFLOW_BUCKET == 6
    assert VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET == 8
    buckets = {
        "open_meld_yakuhai_han": OPEN_MELD_YAKUHAI_HAN_OVERFLOW_BUCKET,
        "visible_meld_dora_aka_han": VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET,
    }
    for kind, schema in CATEGORY_SCHEMAS.items():
        for field in schema.discrete:
            if field.name in buckets:
                assert field.cardinality == buckets[field.name] + 1, (kind, field.name)


def test_no_legacy_snapshot_constants() -> None:
    import riichi_ppo_v1.model.encoding_protocol as module

    for legacy in (
        "SNAPSHOT_FIELD_COUNT", "SNAPSHOT_FACTOR_WIDTH", "SNAPSHOT_NUMERIC_WIDTH",
        "SNAPSHOT_FIELDS", "SNAPSHOT_FACTOR_CARDINALITIES",
    ):
        assert not hasattr(module, legacy), legacy
