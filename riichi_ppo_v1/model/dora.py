"""Shared public dora-indicator semantics for actor feature encoding."""

from __future__ import annotations

from collections.abc import Iterable


def dora_type(indicator: int) -> int:
    """Return the 34-tile type selected by one physical indicator."""
    tile = int(indicator) // 4
    if tile < 27:
        base = tile // 9 * 9
        return base + (tile - base + 1) % 9
    if tile <= 30:
        return 27 + (tile - 27 + 1) % 4
    return 31 + (tile - 31 + 1) % 3


def dora_type_multiplicities(indicators: Iterable[int]) -> dict[int, int]:
    """Count indicator multiplicity instead of collapsing equal dora types."""
    result: dict[int, int] = {}
    for indicator in indicators:
        tile = dora_type(int(indicator))
        result[tile] = result.get(tile, 0) + 1
    return result
