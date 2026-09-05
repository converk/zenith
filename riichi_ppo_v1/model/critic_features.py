"""V19 centralized-value Critic 的严格私有事实。

Critic 私有输入契约:SEP_CRITIC + 三家真实闭手(固定相对座次顺序)。
V18 的未来五张牌山行已删除(D1:纯随机信息,不再进入任何输入)。
行布局与 Actor 相同的 32 宽:[segment, kind, fields...];segment/kind 与
``encoding_protocol.py`` 单点定义一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .encoding_protocol import (
    KIND_CRITIC_HAND,
    KIND_SEP_CRITIC,
    SEGMENT_CRITIC_PRIVATE,
    TOKEN_ROW_WIDTH,
)
from .schema import NUM_PLAYERS, TID_COUNT


@dataclass(frozen=True)
class TableState:
    hands: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CriticFeatures:
    factors: np.ndarray  # [T, 32]
    length: int


def tile_id_to_type(tile: Any) -> int | None:
    """返回 34 类牌型，故意把红五并入普通五。"""
    if tile is None:
        return None
    value = int(tile)
    if not 0 <= value < TID_COUNT:
        return None
    return value // 4


def _tile_type_red(tile: Any) -> tuple[int, int]:
    value = int(tile)
    tile_type = tile_id_to_type(value)
    if tile_type is None:
        raise ValueError(f"invalid tile id {value}")
    return tile_type, int(bool(_is_red(value)))


def _seat_tiles(values: Any, seat: int) -> tuple[int, ...]:
    """读取每座位实体牌行，缺行/非法牌忽略。"""
    try:
        row = values[seat]
    except (IndexError, KeyError, TypeError):
        return ()
    return tuple(int(tile) for tile in row if tile_id_to_type(tile) is not None)


def collect_visible_table_state(observations: dict[int, Any]) -> TableState:
    """收集四家闭手实体牌(Critic 私有行的唯一事实来源)。

    V19 起,``GameState.get_observation`` 携带 ``privileged_hands``(全状态真手),
    训练环境与 replay 路径走同一数据源;缺省回退到每观测自身的 masked hands
    (仅测试/人工场景)。
    """
    if set(observations) != set(range(NUM_PLAYERS)):
        raise RuntimeError("critic features require all four player observations")
    privileged: tuple[tuple[int, ...], ...] | None = None
    for seat in range(NUM_PLAYERS):
        raw = getattr(observations[seat], "privileged_hands", None)
        if raw is not None:
            privileged = tuple(
                tuple(int(tile) for tile in row if tile_id_to_type(tile) is not None)
                for row in raw
            )
            break
    if privileged is not None:
        return TableState(privileged)
    hands: list[tuple[int, ...]] = []
    for seat in range(NUM_PLAYERS):
        hand_rows = getattr(observations[seat], "hands", None)
        if hand_rows is None:
            raise RuntimeError("Observation must expose hands for critic features")
        hands.append(_seat_tiles(hand_rows, seat))
    return TableState(tuple(hands))


def _is_red(tile: int) -> bool:
    return int(tile) in {16, 52, 88}


def _critic_sep_row() -> np.ndarray:
    row = np.zeros(TOKEN_ROW_WIDTH, dtype=np.uint8)
    row[0] = SEGMENT_CRITIC_PRIVATE
    row[1] = KIND_SEP_CRITIC
    return row


def encode_opponent_hand_tokens(table_state: TableState, observer: int) -> list[np.ndarray]:
    """三家真实闭手：相对座次 1,2,3，同一座次按牌型升序。"""
    rows: list[np.ndarray] = []
    for relative in (1, 2, 3):
        seat = (int(observer) + relative) % NUM_PLAYERS
        counts: dict[int, tuple[int, int]] = {}
        for tile in table_state.hands[seat]:
            tile_type, red = _tile_type_red(tile)
            key = (tile_type, red)
            current_count, current_red = counts.get(key, (0, 0))
            counts[key] = (current_count + 1, current_red or red)
        for (tile_type, red), (count, _any_red) in sorted(counts.items()):
            row = np.zeros(TOKEN_ROW_WIDTH, dtype=np.uint8)
            row[0] = SEGMENT_CRITIC_PRIVATE
            row[1] = KIND_CRITIC_HAND
            row[2] = relative
            row[3] = tile_type + 1
            row[4] = red
            row[5] = count
            rows.append(row)
    return rows


def encode_critic_features(
    table_state: TableState,
    observer: int,
) -> CriticFeatures:
    rows: list[np.ndarray] = [_critic_sep_row()]
    rows.extend(encode_opponent_hand_tokens(table_state, observer))
    factors = np.asarray(rows, dtype=np.uint8).reshape(-1, TOKEN_ROW_WIDTH)
    return CriticFeatures(factors, len(rows))


def empty_critic_features() -> CriticFeatures:
    return CriticFeatures(np.zeros((0, TOKEN_ROW_WIDTH), dtype=np.uint8), 0)


def pad_critic_feature_rows(features: list[CriticFeatures]) -> tuple[np.ndarray, np.ndarray]:
    if not features:
        raise ValueError("cannot pad an empty critic feature batch")
    batch = len(features)
    lengths = np.asarray([feature.length for feature in features], dtype=np.int64)
    maximum = int(lengths.max(initial=0))
    factors = np.zeros((batch, maximum, TOKEN_ROW_WIDTH), dtype=np.uint8)
    for row, feature in enumerate(features):
        length = int(feature.length)
        if length:
            factors[row, :length] = feature.factors[:length]
    return factors, lengths
