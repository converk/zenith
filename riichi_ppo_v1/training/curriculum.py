"""Deterministic current/history self-play lineup helpers."""

from __future__ import annotations

import random
from typing import Sequence

from ..model.bridge import NUM_PLAYERS


def rollout_lineups(
    count: int,
    *,
    update: int,
    worker_id: int,
    history_ids: Sequence[str],
    pool_start_update: int = 1501,
) -> list[tuple[str, str, str, str]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if int(update) < int(pool_start_update) or len(history_ids) < 2:
        return [("current",) * NUM_PLAYERS for _ in range(count)]
    rng = random.Random((int(update) << 24) ^ (int(worker_id) << 12) ^ 0xB8)
    choices = tuple(history_ids)
    result: list[tuple[str, str, str, str]] = []
    for _ in range(count):
        first, second = rng.sample(choices, 2)
        seats = ["current", "current", first, second]
        rng.shuffle(seats)
        result.append(tuple(seats))
    return result
