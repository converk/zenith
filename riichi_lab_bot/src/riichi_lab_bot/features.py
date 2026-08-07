"""Actor-visible public river and meld summary.

The encoding itself is delegated to the training-side
``riichi_ppo_v1.model.critic_features`` implementation so the bot and the
training path can never drift.  Only the single-observation adapter lives
here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from riichi_ppo_v1.model.critic_features import (
    MeldState,
    TableState,
    encode_public_summary as _encode_training_public_summary,
    tile_id_to_type,
)

NUM_PLAYERS = 4


def _seat_tiles(values: Any, seat: int) -> tuple[int, ...]:
    try:
        row = values[seat]
    except (IndexError, KeyError, TypeError):
        return ()
    return tuple(
        int(tile) for tile in row if tile_id_to_type(tile) is not None
    )


def _meld_field(value: Any) -> int | None:
    name = str(getattr(value, "meld_type", value)).lower()
    for kind, field in {
        "chi": 1,
        "pon": 2,
        "daiminkan": 3,
        "ankan": 4,
        "kakan": 5,
    }.items():
        if kind in name:
            return field
    return None


def _meld_state(value: Any) -> MeldState | None:
    field = _meld_field(value)
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


def encode_public_summary(
    observation: Any, observer: int
) -> np.ndarray:
    discards = getattr(observation, "discards", ())
    melds = getattr(observation, "melds", ())
    table = TableState(
        hands=((), (), (), ()),
        discards=tuple(
            _seat_tiles(discards, seat) for seat in range(NUM_PLAYERS)
        ),
        melds=tuple(
            _public_meld_rows(melds, seat) for seat in range(NUM_PLAYERS)
        ),
    )
    return _encode_training_public_summary(table, observer).factors
