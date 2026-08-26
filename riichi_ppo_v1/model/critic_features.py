"""V18 centralized-value Critic 的严格私有事实（新行宽）。

Critic 私有输入保持现行契约：三家真实闭手（固定相对座次顺序）+ 未来五张牌（固定摸牌顺序）。
行布局改为与 Actor 相同的 32 宽：[segment, kind, fields...]；segment/kind 与
``encoding_protocol.py`` 单点定义一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .encoding_protocol import (
    KIND_CRITIC_FUTURE,
    KIND_CRITIC_HAND,
    KIND_SEP_CRITIC,
    SEGMENT_CRITIC_FUTURE,
    SEGMENT_CRITIC_PRIVATE,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
)

NUM_PLAYERS = 4
FUTURE_WALL_TILE_COUNT = 5


@dataclass(frozen=True)
class TableState:
    hands: tuple[tuple[int, ...], ...]
    discards: tuple[tuple[int, ...], ...] = ((), (), (), ())
    melds: tuple[tuple["MeldState", ...], ...] = ((), (), (), ())


@dataclass(frozen=True)
class MeldState:
    """One validated public portion of a RiichiEnv meld."""

    field: int
    tiles: tuple[int, ...]
    called_tile: int | None
    source: int | None


@dataclass(frozen=True)
class CriticFeatures:
    factors: np.ndarray  # [T, 32]
    length: int


from .schema import TID_COUNT


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


def _meld_state(value: Any) -> MeldState | None:
    """把 RiichiEnv 的 Meld（或测试替身）映射为稳定结构。"""
    name = str(getattr(value, "meld_type", getattr(value, "type", ""))).lower()
    field = {"chi": 1, "pon": 2, "daiminkan": 3, "ankan": 4, "kakan": 5}.get(name)
    if field is None:
        return None
    tiles = tuple(tile for tile in getattr(value, "tiles", ()) if tile_id_to_type(tile) is not None)
    if not tiles:
        return None
    called = getattr(value, "called_tile", None)
    called_tile = int(called) if tile_id_to_type(called) is not None else None
    source = getattr(value, "from_who", None)
    source_seat = int(source) if isinstance(source, (int, np.integer)) and 0 <= int(source) < NUM_PLAYERS else None
    return MeldState(field, tiles, called_tile, source_seat)


def _public_meld_rows(values: Any, seat: int) -> tuple[MeldState, ...]:
    try:
        row = values[seat]
    except (IndexError, KeyError, TypeError):
        return ()
    return tuple(meld for value in row if (meld := _meld_state(value)) is not None)


def collect_visible_table_state(observations: dict[int, Any], *, include_public_state: bool = True) -> TableState:
    """收集四家闭手与可选公开牌河/副露投影（仅用于 Critic 测试与装配）。"""
    if set(observations) != set(range(NUM_PLAYERS)):
        raise RuntimeError("critic features require all four player observations")
    hands: list[tuple[int, ...]] = []
    for seat in range(NUM_PLAYERS):
        hand_rows = getattr(observations[seat], "hands", None)
        if hand_rows is None:
            raise RuntimeError("Observation must expose hands for critic features")
        hands.append(_seat_tiles(hand_rows, seat))
    if not include_public_state:
        return TableState(tuple(hands))
    public_observation = observations[0]
    discards = getattr(public_observation, "discards", ())
    melds = getattr(public_observation, "melds", ())
    return TableState(
        tuple(hands),
        tuple(_seat_tiles(discards, seat) for seat in range(NUM_PLAYERS)),
        tuple(_public_meld_rows(melds, seat) for seat in range(NUM_PLAYERS)),
    )


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


def encode_future_wall_tokens(wall: Iterable[int]) -> list[np.ndarray]:
    """未来五张活牌，按摸牌顺序 position 1..5。"""
    tiles = tuple(wall)
    if len(tiles) != FUTURE_WALL_TILE_COUNT:
        raise ValueError("future wall must contain exactly five ordered tiles")
    rows: list[np.ndarray] = []
    for position, tile in enumerate(tiles, start=1):
        if tile is None:
            raise ValueError("future wall contains a missing tile id")
        tile_type, red = _tile_type_red(tile)
        row = np.zeros(TOKEN_ROW_WIDTH, dtype=np.uint8)
        row[0] = SEGMENT_CRITIC_FUTURE
        row[1] = KIND_CRITIC_FUTURE
        row[2] = position
        row[3] = tile_type + 1
        row[4] = red
        rows.append(row)
    return rows


def encode_critic_features(
    table_state: TableState,
    observer: int,
    *,
    include_public_state: bool = False,
    future_wall_tiles: Iterable[int] = (),
) -> CriticFeatures:
    rows: list[np.ndarray] = [_critic_sep_row()]
    rows.extend(encode_opponent_hand_tokens(table_state, observer))
    if future_wall_tiles:
        rows.extend(encode_future_wall_tokens(future_wall_tiles))
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
