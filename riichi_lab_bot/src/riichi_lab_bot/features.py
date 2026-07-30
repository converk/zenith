"""Actor-visible public river and meld summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .model import TOKEN_WIDTH

NUM_PLAYERS = 4
SEGMENT_CRITIC_PRIVATE = 4
SEGMENT_PUBLIC_SUMMARY = 3
TOKEN_KIND_TILE_COUNT = 4
TOKEN_KIND_MELD = 5
FIELD_PUBLIC_RIVER = 6
FIELD_PUBLIC_MELD_TILE = 7

_MELD_FIELDS = {
    "chi": 1,
    "pon": 2,
    "daiminkan": 3,
    "ankan": 4,
    "kakan": 5,
}


@dataclass(frozen=True)
class MeldState:
    field: int
    tiles: tuple[int, ...]
    called_tile: int | None
    source: int | None


@dataclass(frozen=True)
class PublicState:
    discards: tuple[tuple[int, ...], ...]
    melds: tuple[tuple[MeldState, ...], ...]


def tile_id_to_type(tile: Any) -> int | None:
    if tile is None:
        return None
    value = int(tile)
    return value // 4 if 0 <= value < 136 else None


def _public_seat_rows(values: Any, seat: int) -> tuple[int, ...]:
    try:
        row = values[seat]
    except (IndexError, KeyError, TypeError):
        return ()
    return tuple(
        int(tile) for tile in row if tile_id_to_type(tile) is not None
    )


def _meld_field(value: Any) -> int | None:
    name = str(getattr(value, "name", value)).lower()
    return next(
        (field for kind, field in _MELD_FIELDS.items() if kind in name), None
    )


def _meld_state(value: Any) -> MeldState | None:
    field = _meld_field(getattr(value, "meld_type", None))
    if field is None:
        return None
    tiles = tuple(
        int(tile)
        for tile in getattr(value, "tiles", ())
        if tile_id_to_type(tile) is not None
    )
    if not tiles:
        return None
    called = getattr(value, "called_tile", None)
    called_tile = (
        int(called) if tile_id_to_type(called) is not None else None
    )
    source = getattr(value, "from_who", None)
    source_seat = (
        int(source)
        if isinstance(source, (int, np.integer))
        and 0 <= int(source) < NUM_PLAYERS
        else None
    )
    return MeldState(field, tiles, called_tile, source_seat)


def _public_meld_rows(values: Any, seat: int) -> tuple[MeldState, ...]:
    try:
        row = values[seat]
    except (IndexError, KeyError, TypeError):
        return ()
    return tuple(
        meld for value in row if (meld := _meld_state(value)) is not None
    )


def collect_public_state(observation: Any) -> PublicState:
    discards = getattr(observation, "discards", ())
    melds = getattr(observation, "melds", ())
    return PublicState(
        tuple(_public_seat_rows(discards, seat) for seat in range(NUM_PLAYERS)),
        tuple(_public_meld_rows(melds, seat) for seat in range(NUM_PLAYERS)),
    )


def _tile_factors(tile_type: int) -> tuple[int, int]:
    suit = tile_type // 9 + 1 if tile_type < 27 else 4
    rank = tile_type % 9 + 1 if tile_type < 27 else tile_type - 26
    return suit, rank


def _is_red(tile: int) -> bool:
    return int(tile) in {16, 52, 88}


def _tile_count_rows(
    field: int, relative: int, tiles: Iterable[int], *, flag: int = 0
) -> list[tuple[int, ...]]:
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
        rows.append(
            (
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
            )
        )
    return rows


def _relative_seat(observer: int, seat: int) -> int:
    return (int(seat) - int(observer)) % NUM_PLAYERS + 1


def encode_public_summary(
    observation: Any, observer: int
) -> np.ndarray:
    state = collect_public_state(observation)
    rows: list[tuple[int, ...]] = []
    for relative in (1, 2, 3, 4):
        seat = (int(observer) + relative - 1) % NUM_PLAYERS
        river_rows = _tile_count_rows(
            FIELD_PUBLIC_RIVER, relative, state.discards[seat]
        )
        rows.extend((SEGMENT_PUBLIC_SUMMARY, *row[1:]) for row in river_rows)
        for meld_index, meld in enumerate(state.melds[seat], start=1):
            if meld.called_tile is None:
                suit = rank = red = 0
            else:
                tile_type = tile_id_to_type(meld.called_tile)
                assert tile_type is not None
                suit, rank = _tile_factors(tile_type)
                red = int(_is_red(meld.called_tile))
            source = (
                0
                if meld.source is None
                else _relative_seat(observer, meld.source)
            )
            rows.append(
                (
                    SEGMENT_PUBLIC_SUMMARY,
                    TOKEN_KIND_MELD,
                    meld.field,
                    int(relative),
                    suit,
                    rank,
                    red,
                    source,
                    meld_index,
                    1,
                )
            )
            meld_rows = _tile_count_rows(
                FIELD_PUBLIC_MELD_TILE,
                relative,
                meld.tiles,
                flag=meld_index,
            )
            rows.extend(
                (SEGMENT_PUBLIC_SUMMARY, *row[1:]) for row in meld_rows
            )
    if not rows:
        return np.zeros((0, TOKEN_WIDTH), dtype=np.uint8)
    return np.asarray(rows, dtype=np.uint8).reshape(-1, TOKEN_WIDTH)
