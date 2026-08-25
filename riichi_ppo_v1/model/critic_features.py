"""V18 centralized-value Critic 的严格私有事实。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .architecture import TOKEN_WIDTH

NUM_PLAYERS = 4

SEGMENT_CRITIC_PRIVATE = 4
SEGMENT_CRITIC_FUTURE_WALL = 5
TOKEN_KIND_TILE_COUNT = 4
TOKEN_KIND_FUTURE_WALL = 6
FIELD_OPPONENT_HAND = 2
FIELD_FUTURE_WALL = 3
FUTURE_WALL_TILE_COUNT = 5

_MELD_FIELDS = {
    "chi": 1,
    "pon": 2,
    "daiminkan": 3,
    "ankan": 4,
    "kakan": 5,
}


@dataclass(frozen=True)
class TableState:
    hands: tuple[tuple[int, ...], ...]
    discards: tuple[tuple[int, ...], ...] = ((), (), (), ())
    melds: tuple[tuple["MeldState", ...], ...] = ((), (), (), ())


@dataclass(frozen=True)
class MeldState:
    """Validated public portion of one RiichiEnv meld."""

    field: int
    tiles: tuple[int, ...]
    called_tile: int | None
    source: int | None


@dataclass(frozen=True)
class CriticFeatures:
    factors: np.ndarray
    length: int


from .schema import TID_COUNT


def tile_id_to_type(tile: Any) -> int | None:
    """返回 34 类牌型,故意把红五并入普通五。"""
    if tile is None:
        return None
    value = int(tile)
    if not 0 <= value < TID_COUNT:
        return None
    return value // 4


def _seat_tiles(values: Any, seat: int) -> tuple[int, ...]:
    """Read a public per-seat tile row, treating absent/malformed data as empty."""
    try:
        row = values[seat]
    except (IndexError, KeyError, TypeError):
        return ()
    return tuple(int(tile) for tile in row if tile_id_to_type(tile) is not None)


def _meld_field(value: Any) -> int | None:
    """Map RiichiEnv's enum (or a test double) to a stable compact code."""
    name = str(getattr(value, "name", value)).lower()
    for kind, field in _MELD_FIELDS.items():
        if kind in name:
            return field
    return None


def _meld_state(value: Any) -> MeldState | None:
    field = _meld_field(getattr(value, "meld_type", None))
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
    """Collect concealed hands and optional public river/meld projection."""
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

    # Rivers and melds are public and identical from every observer's view.
    # Selecting one canonical observation avoids multiplying extraction work.
    public_observation = observations[0]
    discards = getattr(public_observation, "discards", ())
    melds = getattr(public_observation, "melds", ())
    return TableState(
        tuple(hands),
        tuple(_seat_tiles(discards, seat) for seat in range(NUM_PLAYERS)),
        tuple(_public_meld_rows(melds, seat) for seat in range(NUM_PLAYERS)),
    )


def _tile_factors(tile_type: int) -> tuple[int, int]:
    tile_type = int(tile_type)
    suit = tile_type // 9 + 1 if tile_type < 27 else 4
    rank = tile_type % 9 + 1 if tile_type < 27 else tile_type - 26
    return suit, rank


def _is_red(tile: int) -> bool:
    return int(tile) in {16, 52, 88}


def _tile_count_rows(field: int, relative: int, tiles: Iterable[int], *, flag: int = 0) -> list[tuple[int, ...]]:
    counts: dict[tuple[int, bool], int] = {}
    for tile in tiles:
        tile_type = tile_id_to_type(tile)
        if tile_type is None:
            continue
        key = (tile_type, _is_red(tile))
        counts[key] = counts.get(key, 0) + 1
    rows: list[tuple[int, ...]] = []
    for (tile_type, red), count in sorted(counts.items()):
        suit, rank = _tile_factors(tile_type)
        rows.append((
            SEGMENT_CRITIC_PRIVATE,
            TOKEN_KIND_TILE_COUNT,
            int(field),
            int(relative),
            suit,
            rank,
            int(red),
            int(count),
            int(flag),
            1,
        ))
    return rows


def encode_opponent_hand_tokens(table_state: TableState, observer: int) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    for relative in (2, 3, 4):
        seat = (int(observer) + relative - 1) % NUM_PLAYERS
        # Aka dora are semantically distinct tiles.  The value branch is
        # allowed to see concealed opponent hands, so folding their red fives
        # here would discard information that is retained for rivers/melds.
        rows.extend(_tile_count_rows(FIELD_OPPONENT_HAND, relative, table_state.hands[seat]))
    return rows


def encode_future_wall_tokens(wall: Iterable[int]) -> list[tuple[int, ...]]:
    """Encode the next five live-wall tiles as ordered critic-only tokens.

    ``wall`` 必须已截取为后五张活牌。牌按摸牌顺序编码为 position 1..5;
    缺失、数量不符或非法牌号均为硬错误。
    """
    tiles = tuple(wall)
    if len(tiles) != FUTURE_WALL_TILE_COUNT:
        raise ValueError("future wall must contain exactly five ordered tiles")
    rows: list[tuple[int, ...]] = []
    for position, tile in enumerate(tiles, start=1):
        if tile is None:
            raise ValueError("future wall contains a missing tile id")
        value = int(tile)
        if not 0 <= value < TID_COUNT:
            raise ValueError(f"invalid RiichiEnv tile id {value} in future wall")
        tile_type = tile_id_to_type(value)
        if tile_type is None:
            raise ValueError(f"invalid RiichiEnv tile id {value} in future wall")
        suit, rank = _tile_factors(tile_type)
        rows.append((
            SEGMENT_CRITIC_FUTURE_WALL,
            TOKEN_KIND_FUTURE_WALL,
            FIELD_FUTURE_WALL,
            position,
            suit,
            rank,
            int(_is_red(value)),
            1,
            0,
            1,
        ))
    return rows


def encode_critic_features(
    table_state: TableState,
    observer: int,
    *,
    include_public_state: bool = False,
    future_wall_tiles: Iterable[int] = (),
) -> CriticFeatures:
    rows = encode_opponent_hand_tokens(table_state, observer)
    if future_wall_tiles:
        rows.extend(encode_future_wall_tokens(future_wall_tiles))
    if not rows:
        return empty_critic_features()
    factors = np.asarray(rows, dtype=np.uint8).reshape(-1, TOKEN_WIDTH)
    return CriticFeatures(factors, len(rows))


def empty_critic_features() -> CriticFeatures:
    return CriticFeatures(
        np.zeros((0, TOKEN_WIDTH), dtype=np.uint8),
        0,
    )


def pad_critic_feature_rows(features: list[CriticFeatures]) -> tuple[np.ndarray, np.ndarray]:
    if not features:
        raise ValueError("cannot pad an empty critic feature batch")
    batch = len(features)
    lengths = np.asarray([feature.length for feature in features], dtype=np.int64)
    maximum = int(lengths.max(initial=0))
    factors = np.zeros((batch, maximum, TOKEN_WIDTH), dtype=np.uint8)
    for row, feature in enumerate(features):
        length = int(feature.length)
        if length:
            factors[row, :length] = feature.factors[:length]
    return factors, lengths
