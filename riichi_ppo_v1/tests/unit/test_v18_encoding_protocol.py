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
    assert SNAPSHOT_FIELD_COUNT == 54
    assert tuple(field.field_id for field in SNAPSHOT_FIELDS) == tuple(range(1, 55))
    assert len({field.name for field in SNAPSHOT_FIELDS}) == 54
    assert tuple(field.name for field in SNAPSHOT_FIELDS[4:17]) == (
        "opponent_1_riichi_status", "opponent_1_riichi_turn", "opponent_1_open_meld_count",
        "opponent_1_tedashi_count", "opponent_1_tsumogiri_count",
        "opponent_1_post_riichi_tsumogiri_count",
        "opponent_1_first_six_man_count", "opponent_1_first_six_pin_count",
        "opponent_1_first_six_sou_count", "opponent_1_first_six_terminal_honor_count",
        "opponent_1_open_meld_yakuhai_han", "opponent_1_visible_meld_dora_aka_han",
        "opponent_1_riichi_declaration_tile",
    )
    assert tuple(field.name for field in SNAPSHOT_FIELDS[47:51]) == (
        "fully_visible_tile_kind_count", "unknown_distinct_dora_copy_count",
        "self_improve_tile_count", "self_win_tile_count",
    )


def test_domains_derive_from_rust_schema() -> None:
    assert SNAPSHOT_FACTOR_CARDINALITIES[0] == 55
    assert all(0 <= field.relative_seat <= 3 for field in SNAPSHOT_FIELDS)
    assert all(field.categorical_max >= 0 and field.tile_max >= 0 for field in SNAPSHOT_FIELDS)
    assert sum(field.numeric for field in SNAPSHOT_FIELDS) == 3
    assert all(field.categorical_max == 6 for field in SNAPSHOT_FIELDS if "first_six" in field.name)
    assert all(field.categorical_max == 8 for field in SNAPSHOT_FIELDS if "dora_aka" in field.name)
    assert all(
        field.categorical_max == 16
        for field in SNAPSHOT_FIELDS if "post_riichi_tsumogiri" in field.name
    )
    assert all(
        field.categorical_max == 40
        for field in SNAPSHOT_FIELDS if field.name in ("self_improve_tile_count", "self_win_tile_count")
    )
