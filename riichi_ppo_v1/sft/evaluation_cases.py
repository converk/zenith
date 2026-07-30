"""Deterministic, seat-balanced schedules for heuristic policy evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


EFFICIENCY = "heuristic_efficiency"
DEFENSE = "heuristic_defense"


def evaluation_cases(
    seed_base: int, hanchan_count: int, *, cycle: int = 0,
) -> list[tuple[int, int, tuple[str, str, str]]]:
    """Return fixed candidate seats and three rotating heuristic opponents."""
    recipes = (
        (EFFICIENCY, DEFENSE, EFFICIENCY),
        (DEFENSE, EFFICIENCY, DEFENSE),
    )
    return [
        (
            int(seed_base) + int(cycle) * 1_000_003 + index,
            index % 4,
            recipes[(index // 4) % len(recipes)],
        )
        for index in range(int(hanchan_count))
    ]


def merge_evaluation_summaries(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    """Merge seat-balanced evaluation shards using their metric counts."""
    rows = list(rows)
    if not rows:
        return {}
    result: dict[str, float] = {}
    for name in {key for row in rows for key in row}:
        values = [float(row[name]) for row in rows if name in row and math.isfinite(float(row[name]))]
        if not values:
            continue
        if name.endswith("/count") or name.endswith("_count"):
            result[name] = float(sum(values))
            continue
        prefix = name.rsplit("/", 1)[0]
        count_name = f"{prefix}/count"
        weighted = [(float(row[name]), float(row.get(count_name, 0.0))) for row in rows if name in row]
        total = sum(weight for _value, weight in weighted)
        result[name] = float(sum(value * weight for value, weight in weighted) / total) if total else float(np.mean(values))
        if name.endswith("/point_delta_mean") and len(values) > 1:
            result[f"{name}_stderr"] = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return result
