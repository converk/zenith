"""SFT 训练与评测共用的规范化动作分组。"""

from __future__ import annotations

import json

from .schema import NUM_ACTIONS
from .validation import TILE37, _chi_pairs, _pon_pairs


def action_group(action_id: int) -> str:
    """返回 241 维动作空间中指定动作 id 的语义分组。"""
    boundaries = (
        (1, "pass"),
        (75, "discard"),
        (76, "reach"),
        (133, "chi"),
        (170, "pon"),
        (239, "kan"),
        (240, "hora"),
        (NUM_ACTIONS, "ryukyoku"),
    )
    return next(name for end, name in boundaries if int(action_id) < end)


def _action_kind_and_row(action: object) -> tuple[str, dict[str, object]]:
    """解析原生动作,同时保留 MJAI 行用于固定 id 映射。"""
    try:
        row = json.loads(action.to_mjai())
        if isinstance(row, dict):
            return str(row.get("type", "")).lower(), row
    except (AttributeError, TypeError, ValueError):
        pass
    value = getattr(action, "action_type", getattr(action, "type", ""))
    return str(getattr(value, "name", value)).lower().rsplit(".", 1)[-1], {}


def action_kind(action: object) -> str:
    """返回规范 MJAI 动作类型。"""
    return _action_kind_and_row(action)[0]


def consumed_tiles(action: object) -> tuple[int, ...]:
    """返回吃/碰/杠从手牌消耗的实体牌,兼容离线重放多带叫牌的表示。"""
    values = getattr(action, "consume_tiles", getattr(action, "consumed", ())) or ()
    result = [int(value) for value in values]
    kind = action_kind(action)
    expected = {"chi": 2, "pon": 2, "daiminkan": 3}.get(kind)
    called = getattr(action, "tile", None)
    if expected is not None and len(result) == expected + 1 and called is not None:
        try:
            result.remove(int(called))
        except ValueError:
            pass
    return tuple(result)


def _action_id_from_normalized(
    action: object,
    observation: object,
    kind: str,
    row: dict[str, object],
) -> int | None:
    """把规范化动作映射到固定 241 维动作 id。"""
    expected_consumed = {"chi": 2, "pon": 2, "daiminkan": 3}.get(kind)
    consumed_row = row.get("consumed")
    if (
        expected_consumed is not None
        and isinstance(consumed_row, list)
        and len(consumed_row) == expected_consumed + 1
    ):
        consumed_row = list(consumed_row)
        try:
            consumed_row.remove(row.get("pai"))
        except ValueError:
            return None
        row["consumed"] = consumed_row
    if kind in {"none", "pass"}:
        return 0
    if kind == "dahai":
        pai = str(row.get("pai", ""))
        drawn = getattr(observation, "drawn_tile", None)
        tile = getattr(action, "tile", None)
        mode = int(drawn is not None and tile is not None and int(drawn) == int(tile))
        try:
            return 1 + 2 * TILE37.index(pai) + mode
        except ValueError:
            return None
    if kind == "reach":
        return 75
    if kind == "chi":
        consumed = tuple(sorted(str(value) for value in row.get("consumed", ())))
        for index, pair in enumerate(_chi_pairs()):
            if consumed == tuple(sorted(pair)):
                return 76 + index
        return None
    if kind == "pon":
        consumed = tuple(sorted(str(value) for value in row.get("consumed", ())))
        for index, pair in enumerate(_pon_pairs()):
            if consumed == tuple(sorted(pair)):
                return 133 + index
        return None
    if kind == "daiminkan":
        return 170
    if kind in {"ankan", "kakan"}:
        values = consumed_tiles(action)
        tile = getattr(action, "tile", None)
        tile_type = (values[0] if values else int(tile)) // 4
        return (171 if kind == "ankan" else 205) + tile_type
    if kind in {"hora", "ron", "tsumo"}:
        return 239
    if kind in {"ryukyoku", "kyushukyuhai", "kyushu_kyuhai"}:
        return 240
    return None


def action_id(action: object, observation: object) -> int | None:
    """把 RiichiEnv 动作映射到固定 241 维策略 id。"""
    kind, row = _action_kind_and_row(action)
    return _action_id_from_normalized(action, observation, kind, row)
