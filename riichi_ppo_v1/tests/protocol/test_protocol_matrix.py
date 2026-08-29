"""V18 当前局面协议矩阵：规范类别顺序与关键字段一致性的轻量矩阵测试。"""

from __future__ import annotations

import numpy as np

from riichi_ppo_v1.model.encoding_protocol import (
    KIND_ACTION_DEFENSE_QUERY,
    KIND_ACTION_OFFENSE_QUERY,
    KIND_BOS,
    KIND_OPPONENT_ANALYSIS,
    KIND_PLAYER,
    KIND_RIVER_DISCARD,
    KIND_RIVER_SUMMARY,
    KIND_SELF_HAND,
    KIND_SELF_STATE_ANALYSIS,
    KIND_SEP_ACTIONS,
    KIND_SEP_SELF_HAND,
    KIND_SEP_TILE_STATE,
    KIND_TABLE,
    KIND_TILE_STATE,
    SEGMENT_ANALYSIS,
    SEGMENT_SHARED,
    is_separator_kind,
)
from riichi_ppo_v1.sft.data import encode_kyoku
from riichi_ppo_v1.tests.v18_fixtures import first_kyoku_record


def _summary_of_kinds(sample) -> list[tuple[str, int]]:
    """返回 (类别名, 出现次数) 的稳定摘要。"""
    kinds = sample.actor_factors[:, 1].astype(int)
    order: list[tuple[str, int]] = []
    for kind in kinds:
        if kind == KIND_BOS:
            name = "BOS"
        elif kind == KIND_TABLE:
            name = "TABLE"
        elif kind == KIND_SEP_SELF_HAND:
            name = "SEP_SELF_HAND"
        elif kind == KIND_SELF_HAND:
            name = "SELF_HAND"
        elif kind == KIND_SELF_STATE_ANALYSIS:
            name = "SELF_STATE"
        elif kind == KIND_PLAYER:
            name = "PLAYER"
        elif kind == KIND_RIVER_SUMMARY:
            name = "RIVER_SUMMARY"
        elif kind == KIND_RIVER_DISCARD:
            name = "RIVER_DISCARD"
        elif kind == KIND_TILE_STATE:
            name = "TILE_STATE"
        elif kind == KIND_OPPONENT_ANALYSIS:
            name = "OPPONENT_ANALYSIS"
        elif kind == KIND_SEP_TILE_STATE:
            name = "SEP_TILE_STATE"
        elif kind == KIND_SEP_ACTIONS:
            name = "SEP_ACTIONS"
        elif kind in (KIND_ACTION_OFFENSE_QUERY, KIND_ACTION_DEFENSE_QUERY):
            name = "ACTION_QUERY"
        elif is_separator_kind(int(kind)):
            name = "SEPARATOR"
        else:  # pragma: no cover
            name = f"KIND_{kind}"
        if order and order[-1][0] == name:
            order[-1] = (name, order[-1][1] + 1)
        else:
            order.append((name, 1))
    return order


def test_canonical_category_order_matrix() -> None:
    record, _game_id = first_kyoku_record()
    samples = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    matrix = _summary_of_kinds(samples[0])
    names = [name for name, _count in matrix]
    assert names[0] == "BOS"
    assert names[1] == "TABLE"
    assert names[2] == "SEP_SELF_HAND"
    assert "SELF_STATE" in names
    assert "PLAYER" in names
    assert sum(count for name, count in matrix if name == "RIVER_SUMMARY") == 6
    assert "TILE_STATE" in names
    assert "OPPONENT_ANALYSIS" in names and names[-3] == "OPPONENT_ANALYSIS"
    assert names[-2] == "SEP_ACTIONS"
    assert names[-1] == "ACTION_QUERY"


def test_segments_and_action_block() -> None:
    record, _game_id = first_kyoku_record()
    samples = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    sample = samples[0]
    segments = sample.actor_factors[:, 0].astype(int)
    kinds = sample.actor_factors[:, 1].astype(int)
    assert np.any(segments == SEGMENT_SHARED)
    assert np.any(segments == SEGMENT_ANALYSIS)
    action_start = int(np.flatnonzero(kinds == KIND_SEP_ACTIONS)[0]) + 1
    action_block = kinds[action_start:]
    assert action_block[0::2].tolist() == [KIND_ACTION_OFFENSE_QUERY] * (len(action_block) // 2)
    assert action_block[1::2].tolist() == [KIND_ACTION_DEFENSE_QUERY] * (len(action_block) // 2)
