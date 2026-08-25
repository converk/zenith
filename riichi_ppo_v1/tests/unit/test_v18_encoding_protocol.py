"""V18 schema 顺序与领域契约。"""

from riichi_ppo_v1.model.encoding_protocol import (
    ENCODED_FORMAT,
    ENCODING_PROTOCOL_VERSION,
    SNAPSHOT_FACTOR_CARDINALITIES,
    SNAPSHOT_FIELD_COUNT,
    SNAPSHOT_FIELDS,
)


def test_v18_protocol_and_exact_field_order() -> None:
    assert ENCODING_PROTOCOL_VERSION == 18
    assert ENCODED_FORMAT == "riichi-sft-encoded-v18"
    assert SNAPSHOT_FIELD_COUNT == 29
    assert tuple(field.field_id for field in SNAPSHOT_FIELDS) == tuple(range(1, 30))
    assert len({field.name for field in SNAPSHOT_FIELDS}) == 29


def test_domains_derive_from_rust_schema() -> None:
    assert SNAPSHOT_FACTOR_CARDINALITIES[0] == 30
    assert all(0 <= field.relative_seat <= 3 for field in SNAPSHOT_FIELDS)
    assert all(field.categorical_max >= 0 and field.tile_max >= 0 for field in SNAPSHOT_FIELDS)
    assert sum(field.numeric for field in SNAPSHOT_FIELDS) == 3
