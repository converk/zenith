"""演员特征编码共用的公开宝牌指示牌语义。"""

from __future__ import annotations

from collections.abc import Iterable


def dora_type(indicator: int) -> int:
    """返回一枚物理指示牌翻开后对应的 34 类牌型。"""
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
