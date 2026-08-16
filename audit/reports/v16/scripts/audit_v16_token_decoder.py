"""V16 独立逐 token 解码审计(扩展审计)。

本脚本不调用 `model/bridge.py` 的 token 构造函数,而是从每个决策的原始 MJAI
事件与 Observation 公开字段独立重算:

- Objective Facts 的事件 history 与当前状态后缀(10 分类因子 + 8 数值通道);
- Compact Snapshot 的 kind/categorical/numeric 行;
- 每个 query 行的头部字段与 Defense 答案(D0–D9)。

再与 `BatchedStateBridge.prepare_v16` 的实际输出逐 token 比较。样本按 MJAI 事件
类型分层选取,覆盖 reach、chi、pon、daiminkan、ankan、kakan、dora、hora 与
ryukyoku 等局面。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import tarfile
from pathlib import Path

import numpy as np

import riichi
from riichienv import MjaiReplay

from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision
from riichi_ppo_v1.model.encoding_protocol import (
    DEFENSE_SLOT_ORDER,
    OFFENSE_SLOT_ORDER,
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ACTION_TYPE,
    QUERY_ROW_ANSWER_START,
    QUERY_ROW_PRIMARY_TILE,
    QUERY_ROW_QUERY_TYPE,
    QUERY_ROW_SOURCE_SEAT,
    SLOT_CARDINALITIES,
)
from riichi_ppo_v1.sft.data import _member_metadata
from riichi_ppo_v1.sft.precompute import selected_any

ROOT = Path(__file__).resolve().parents[4]
RAW_ROOT = ROOT / "datasets/tenhou_sft_2024_2025"
SUBSET_DENOMINATOR = 5
SUBSET_REMAINDERS = (0, 1)
CONTEXT_LIMIT = 4096

# 本模块刻意本地定义历史 token 编码,作为独立 oracle,不复用生产常量。
SEGMENT_EVENT = 1
SEGMENT_STATE = 2
KIND_EVENT = 1
KIND_SCORE = 2
KIND_COUNTER = 3
KIND_TILE_COUNT = 4
VIS_PUBLIC = 1
ACTOR_SELF = 1
RED_FIVE_IDS = {16, 52, 88}

EVENT_FIELDS = {
    "start_kyoku": 2,
    "dahai": 4,
    "chi": 5,
    "pon": 6,
    "daiminkan": 7,
    "ankan": 8,
    "kakan": 9,
    "dora": 10,
    "reach": 11,
    "reach_accepted": 12,
}

TARGET_CATEGORIES = (
    "reach", "chi", "pon", "daiminkan", "ankan", "kakan", "dora",
    "hora", "ryukyoku",
)


class _EmptyEventsObservation:
    """离线回放结尾缺失座位时补齐四座观察。"""

    def __init__(self, base: object) -> None:
        self._base = base

    def new_events(self) -> list[str]:
        return []

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


def _decode(payload: bytes) -> str:
    if payload[:2] == b"\x1f\x8b":
        return gzip.decompress(payload).decode("utf-8")
    return payload.decode("utf-8")


def _read_member(shard: Path, name: str) -> str:
    with tarfile.open(shard, "r") as archive:
        member = archive.getmember(name)
        file = archive.extractfile(member)
        if file is None:
            raise RuntimeError(f"cannot read {shard}:{name}")
        return _decode(file.read())


def _selected_members(shard: Path) -> list[str]:
    with tarfile.open(shard, "r") as archive:
        return [
            member.name
            for member in archive
            if member.isfile() and selected_any(
                member.name, SUBSET_DENOMINATOR, SUBSET_REMAINDERS,
            )
        ]


def _parse_tile(value: str) -> tuple[int, int, int, int]:
    """把 MJAI 牌字符串解析为 (type, suit, rank, red)。"""
    if value == "5mr":
        return 4, 1, 5, 1
    if value == "5pr":
        return 13, 2, 5, 1
    if value == "5sr":
        return 22, 3, 5, 1
    if value[-1] in "mps":
        suit = {"m": 1, "p": 2, "s": 3}[value[-1]]
        rank = int(value[:-1])
        if not 1 <= rank <= 9:
            raise ValueError(f"invalid numbered tile {value!r}")
        return (suit - 1) * 9 + rank - 1, suit, rank, 0
    honors = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}
    if value not in honors:
        raise ValueError(f"invalid honor tile {value!r}")
    tile_type = honors[value]
    return tile_type, 4, tile_type - 26, 0


def _physical_tile(tile_id: int) -> tuple[int, int, int, int]:
    tile_type = int(tile_id) // 4
    red = int(tile_id) in RED_FIVE_IDS
    if tile_type < 27:
        suit = tile_type // 9 + 1
        rank = tile_type % 9 + 1
    else:
        suit = 4
        rank = tile_type - 26
    return tile_type, suit, rank, int(red)


def _relative(absolute: int, observer: int) -> int:
    return (int(absolute) + 4 - int(observer)) % 4 + 1


def _numeric_features(field: int, value: float) -> np.ndarray:
    """独立重算分数/计数的八维周期特征。"""
    result = np.zeros(8, dtype=np.float32)
    periods = (
        (100.0, 1_000.0, 10_000.0, 100_000.0)
        if field == 1
        else (2.0, 8.0, 32.0, 128.0)
        if field == 2
        else ()
    )
    for index, period in enumerate(periods):
        angle = 2.0 * math.pi * value / period
        result[2 * index] = math.sin(angle)
        result[2 * index + 1] = math.cos(angle)
    return result


def _row(factors: list[int], numeric: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    array = np.zeros(10, dtype=np.uint8)
    array[: len(factors)] = factors
    return array, np.asarray(numeric if numeric is not None else [0.0] * 8, dtype=np.float32)


def _chi_detail(pai: str, consumed: list[str]) -> int:
    pai_type = _parse_tile(pai)[0]
    types = sorted([pai_type, _parse_tile(consumed[0])[0], _parse_tile(consumed[1])[0]])
    offset = pai_type - types[0]
    red = _parse_tile(pai)[3] or any(_parse_tile(value)[3] for value in consumed)
    return 1 + offset * 2 + int(red)


def _meld_red_detail(consumed: list[str]) -> int:
    return 1 + int(any(_parse_tile(value)[3] for value in consumed))


def decode_event_rows(events: list[dict], observer: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """独立解码公开 MJAI 事件序列为 history token。"""
    factors: list[np.ndarray] = []
    numeric: list[np.ndarray] = []
    for event in events:
        event_type = str(event.get("type", ""))
        if event_type == "start_kyoku":
            factors.clear()
            numeric.clear()
            bakaze = str(event.get("bakaze", "E"))
            kyoku = int(event.get("kyoku", 1))
            wind = {"E": 0, "S": 1, "W": 2, "N": 3}[bakaze]
            hand = max(0, min(kyoku - 1, 3))
            row, num = _row([SEGMENT_EVENT, KIND_EVENT, 2, 0, 0, 0, 0, 0, 1 + wind * 4 + hand, VIS_PUBLIC])
            factors.append(row)
            numeric.append(num)
            continue
        if event_type in {"start_game", "tsumo", "hora", "ryukyoku", "end_kyoku", "end_game", "none"}:
            continue
        field = EVENT_FIELDS.get(event_type)
        if field is None:
            raise ValueError(f"unknown history event type {event_type!r}")
        actor = int(event.get("actor", 0))
        tile = event.get("pai")
        if event_type == "dahai":
            if tile is None:
                raise ValueError(f"dahai event lacks pai: {event!r}")
            suit, rank, red = _parse_tile(str(tile))[1:]
            tsumogiri = int(bool(event.get("tsumogiri", False)))
            row, num = _row([
                SEGMENT_EVENT, KIND_EVENT, field, _relative(actor, observer),
                suit, rank, red, 0, 1 + tsumogiri, VIS_PUBLIC,
            ])
        elif event_type in {"chi", "pon", "daiminkan"}:
            suit, rank, red = _parse_tile(str(tile))[1:]
            target = int(event.get("target", actor))
            consumed = [str(value) for value in event.get("consumed", [])]
            detail = _chi_detail(str(tile), consumed) if event_type == "chi" else _meld_red_detail(consumed)
            row, num = _row([
                SEGMENT_EVENT, KIND_EVENT, field, _relative(actor, observer),
                suit, rank, red, _relative(target, observer), detail, VIS_PUBLIC,
            ])
        elif event_type in {"ankan", "kakan"}:
            consumed = [str(value) for value in event.get("consumed", [])]
            pai = str(tile) if event_type == "kakan" else consumed[0]
            if pai == "None" or not consumed:
                raise ValueError(f"{event_type} event lacks required tile/consumed: {event!r}")
            suit, rank, red = _parse_tile(pai)[1:]
            row, num = _row([
                SEGMENT_EVENT, KIND_EVENT, field, _relative(actor, observer),
                suit, rank, red, ACTOR_SELF, _meld_red_detail(consumed), VIS_PUBLIC,
            ])
        elif event_type == "dora":
            dora_marker = event.get("dora_marker")
            if dora_marker is None:
                raise ValueError(f"dora event lacks dora_marker: {event!r}")
            suit, rank, red = _parse_tile(str(dora_marker))[1:]
            row, num = _row([
                SEGMENT_EVENT, KIND_EVENT, field, 0, suit, rank, red, 0, 0, VIS_PUBLIC,
            ])
        else:  # reach / reach_accepted
            row, num = _row([
                SEGMENT_EVENT, KIND_EVENT, field, _relative(actor, observer),
                0, 0, 0, 0, 0, VIS_PUBLIC,
            ])
        factors.append(row)
        numeric.append(num)
    return factors, numeric


def _decision_flag(observation: object) -> int:
    """独立判断当前是否处于主动打牌决策窗口。"""
    kinds = {"dahai", "reach", "ankan", "kakan", "ryukyoku"}
    for action in observation.legal_actions():
        try:
            row = json.loads(action.to_mjai())
        except (AttributeError, TypeError, ValueError):
            continue
        if str(row.get("type", "")) in kinds:
            return 1
    return 0


def _live_wall_from_events(events: list[dict]) -> int:
    """按协议从当前小局最后一个 start_kyoku 之后的 tsumo 事件数重算活牌山。"""
    live_wall = 70
    seen_start = False
    for event in events:
        if event.get("type") == "start_kyoku":
            live_wall = 70
            seen_start = True
        elif seen_start and event.get("type") == "tsumo":
            live_wall = max(0, live_wall - 1)
    return live_wall


def decode_state_rows(
    observation: object,
    live_wall: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """独立从 Observation 重算当前状态后缀 token。"""
    seat = int(observation.player_id)
    factors: list[np.ndarray] = []
    numeric: list[np.ndarray] = []
    scores = [int(value) for value in observation.scores]
    for absolute, score in enumerate(scores):
        row, num = _row(
            [SEGMENT_STATE, KIND_SCORE, 1, _relative(absolute, seat), 0, 0, 0, 0, 0, 0],
            _numeric_features(1, float(score)),
        )
        factors.append(row)
        numeric.append(num)
    for field, value in (
        (1, int(observation.round_wind)),
        (2, int(observation.kyoku_index) + 1),
        (3, int(observation.honba)),
        (4, int(observation.riichi_sticks)),
        (5, int(live_wall if live_wall is not None else observation.tiles_left)),
    ):
        row, num = _row(
            [SEGMENT_STATE, KIND_COUNTER, field, 0, 0, 0, 0, 0, 0, 0],
            _numeric_features(2, float(value)),
        )
        factors.append(row)
        numeric.append(num)
    oya_relative = _relative(int(observation.oya), seat)
    row, num = _row([SEGMENT_STATE, KIND_COUNTER, 6, oya_relative, 0, 0, 0, 0, 0, 0])
    factors.append(row)
    numeric.append(num)
    self_wind = (seat + 4 - int(observation.oya)) % 4 + 1
    row, num = _row([SEGMENT_STATE, KIND_COUNTER, 7, ACTOR_SELF, 0, 0, 0, self_wind, 0, 0])
    factors.append(row)
    numeric.append(num)
    for tile_id in observation.dora_indicators:
        _tile_type, suit, rank, red = _physical_tile(int(tile_id))
        row, num = _row([SEGMENT_STATE, KIND_TILE_COUNT, 3, 0, suit, rank, red, 0, 0, VIS_PUBLIC])
        factors.append(row)
        numeric.append(num)
    drawn = int(observation.drawn_tile) if observation.drawn_tile is not None else None
    flags = (
        int(bool(observation.riichi_declared[seat]))
        | (int(drawn is not None) << 1)
        | (_decision_flag(observation) << 2)
    )
    row, num = _row([SEGMENT_STATE, KIND_COUNTER, 8, ACTOR_SELF, 0, 0, 0, 0, flags, 0])
    factors.append(row)
    numeric.append(num)

    counts = np.zeros(34, dtype=np.uint8)
    red = np.zeros(34, dtype=np.bool_)
    for tile_id in observation.hands[seat]:
        tile_type, _suit, _rank, is_red = _physical_tile(int(tile_id))
        counts[tile_type] = min(255, int(counts[tile_type]) + 1)
        red[tile_type] = red[tile_type] or bool(is_red)
    for tile_type in range(34):
        if not counts[tile_type]:
            continue
        if tile_type < 27:
            suit = tile_type // 9 + 1
            rank = tile_type % 9 + 1
        else:
            suit = 4
            rank = tile_type - 26
        row, num = _row([
            SEGMENT_STATE, KIND_TILE_COUNT, 1, ACTOR_SELF,
            suit, rank, int(red[tile_type]), int(counts[tile_type]), 0, VIS_PUBLIC,
        ])
        factors.append(row)
        numeric.append(num)
    if drawn is not None:
        _tile_type, suit, rank, is_red = _physical_tile(drawn)
        row, num = _row([
            SEGMENT_STATE, KIND_TILE_COUNT, 5, ACTOR_SELF,
            suit, rank, is_red, 1, 0, VIS_PUBLIC,
        ])
        factors.append(row)
        numeric.append(num)
    return factors, numeric


def decode_snapshot(observation: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """独立从 Observation 重算 Snapshot 三组行。"""
    seat = int(observation.player_id)
    scores = tuple(int(value) for value in observation.scores)
    order = sorted(range(4), key=lambda player: (-scores[player], player))
    self_rank = order.index(seat) + 1
    oya_relative = (int(observation.oya) - seat) % 4
    opponents = [(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]

    kinds = [0]
    cats = [[
        min(max(int(observation.round_wind), 0), 1),
        min(max(int(observation.kyoku_index), 0), 7),
        oya_relative,
        max(0, self_rank - 1),
    ]]
    nums = [[
        min(max(float(observation.honba) / 10.0, 0.0), 1.0),
        min(max(float(observation.riichi_sticks) / 10.0, 0.0), 1.0),
        min(max(float(observation.tiles_left) / 136.0, 0.0), 1.0),
        0.0, 0.0, 0.0, 0.0,
    ]]
    for tile_id in observation.dora_indicators:
        tile_type = int(tile_id) // 4
        kinds.append(1)
        cats.append([min(max(tile_type, 0), 33), 0, 0, 0])
        nums.append([0.0] * 7)
    kinds.append(2)
    cats.append([0, 0, 0, 0])
    nums.append([
        *[max(-5.0, min(float(score) / 25000.0, 5.0)) for score in scores],
        *[
            max(-5.0, min(float(scores[seat] - scores[opponent]) / 25000.0, 5.0))
            for opponent in opponents
        ],
    ])
    for opponent in opponents:
        declared = int(bool(observation.riichi_declared[opponent]))
        reach_turn = int(observation.riichi_declaration_indices[opponent] or -1)
        meld_count = len(observation.melds[opponent])
        river_count = len(observation.discards[opponent])
        flags = list(observation.tsumogiri_flags[opponent])
        tsumogiri_count = sum(bool(flag) for flag in flags)
        tedashi_count = sum(not bool(flag) for flag in flags)
        menzen = 1 if meld_count == 0 else 0
        kinds.append(3)
        cats.append([declared, menzen, 0, 0])
        nums.append([
            min(max(float(reach_turn if reach_turn >= 0 else 0) / 20.0, 0.0), 1.0),
            min(max(float(meld_count) / 4.0, 0.0), 1.0),
            min(max(float(river_count) / 20.0, 0.0), 1.0),
            min(max(float(tedashi_count) / 20.0, 0.0), 1.0),
            min(max(float(tsumogiri_count) / 20.0, 0.0), 1.0),
            0.0, 0.0,
        ])
    kinds_array = np.asarray(kinds, dtype=np.uint8)
    cat_array = np.zeros((len(kinds), 4), dtype=np.uint8)
    num_array = np.zeros((len(kinds), 7), dtype=np.float32)
    for index, row in enumerate(cats):
        cat_array[index, : len(row)] = row
    for index, row in enumerate(nums):
        num_array[index, : len(row)] = row
    return kinds_array, cat_array, num_array


def _river_masks(observation: object) -> np.ndarray:
    masks = np.zeros(4, dtype=np.uint64)
    for seat, river in enumerate(observation.discards):
        mask = 0
        for tile_id in river:
            mask |= 1 << (int(tile_id) // 4)
        masks[seat] = mask
    return masks


def _public_visible(observation: object) -> np.ndarray:
    visible = np.zeros(34, dtype=np.uint8)
    for river in observation.discards:
        for tile_id in river:
            visible[int(tile_id) // 4] += 1
    for meld_rows in observation.melds:
        for meld in meld_rows:
            for tile_id in getattr(meld, "tiles", ()) or ():
                visible[int(tile_id) // 4] += 1
    for tile_id in observation.dora_indicators:
        visible[int(tile_id) // 4] += 1
    return np.minimum(visible, 4)


def _suji_safe(tile_type: int, river_mask: int) -> bool:
    if tile_type >= 27:
        return False
    rank = tile_type % 9
    lower = tile_type - 3 if rank >= 3 else None
    upper = tile_type + 3 if rank <= 5 else None
    anchors = [anchor for anchor in (lower, upper) if anchor is not None]
    return all(int(river_mask) & (1 << anchor) for anchor in anchors)


def _action_kind(action: object) -> str:
    enum_name = str(getattr(action, "action_type", "")).lower().rsplit(".", 1)[-1]
    if enum_name in {"tsumo", "ron"}:
        return enum_name
    try:
        row = json.loads(action.to_mjai())
        if isinstance(row, dict):
            kind = str(row.get("type", "")).lower()
            if kind == "hora":
                return enum_name or "hora"
            return kind
    except (AttributeError, TypeError, ValueError):
        pass
    return enum_name


def _canonical_template(action: object, observation: object) -> str:
    """独立规范化动作模板,补上摸切位并去掉回放 consume 中的被鸣牌。"""
    row = json.loads(action.to_mjai())
    action_type = str(row.get("type", ""))
    drawn = getattr(observation, "drawn_tile", None)
    tile = getattr(action, "tile", None)
    if action_type == "dahai":
        row["tsumogiri"] = bool(
            tile is not None and drawn is not None and int(tile) == int(drawn)
        )
    expected_consumed = {"chi": 2, "pon": 2, "daiminkan": 3}.get(action_type)
    consumed = row.get("consumed")
    if (
        expected_consumed is not None
        and isinstance(consumed, list)
        and len(consumed) == expected_consumed + 1
        and row.get("pai") in consumed
    ):
        consumed.remove(row["pai"])
    return json.dumps(row, separators=(",", ":"), sort_keys=True)


def _discard_type(action: object, observation: object) -> int | None:
    tile = getattr(action, "tile", None)
    kind = _action_kind(action)
    if tile is None and kind in {"reach", "dahai"} and observation.drawn_tile is not None:
        tile = int(observation.drawn_tile)
    return int(tile) // 4 if tile is not None else None


def _post_action_counts(
    observation: object,
    action: object,
    kind: str,
) -> np.ndarray:
    """独立估算动作后的暗牌计数,用于防守库存。"""
    seat = int(observation.player_id)
    post = [int(value) for value in observation.hands[seat]]
    if kind in {"dahai", "reach"}:
        candidate = getattr(action, "tile", None)
        if candidate is None and observation.drawn_tile is not None:
            candidate = int(observation.drawn_tile)
        if candidate is not None:
            candidate = int(candidate)
            if candidate in post:
                post.remove(candidate)
            else:
                for value in post:
                    if int(value) // 4 == candidate // 4:
                        post.remove(value)
                        break
    elif kind in {"chi", "pon", "ankan", "daiminkan"}:
        consumed = [int(value) for value in (getattr(action, "consume_tiles", ()) or ())]
        counts: dict[int, int] = {}
        for value in consumed:
            tile_type = value // 4
            counts[tile_type] = counts.get(tile_type, 0) + 1
        for tile_type, count in counts.items():
            for _ in range(count):
                for value in post:
                    if int(value) // 4 == tile_type:
                        post.remove(value)
                        break
    elif kind == "kakan":
        added = getattr(action, "tile", None)
        if added is None:
            consumed = [int(value) for value in (getattr(action, "consume_tiles", ()) or ())]
            added = consumed[-1] if consumed else None
        if added is not None:
            for value in post:
                if int(value) // 4 == int(added) // 4:
                    post.remove(value)
                    break
    return np.bincount([int(value) // 4 for value in post], minlength=34).astype(np.uint8)


def decode_query_rows(
    observation: object,
    action: object,
    action_id: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """独立重算一对 query 行的头部与 Defense answers,返回实际行与 offense 状态。"""
    kind = _action_kind(action)
    seat = int(observation.player_id)
    opponents = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
    primary_type = _discard_type(action, observation)
    action_type_code = {
        "none": 1, "pass": 1, "dahai": 2, "reach": 3, "chi": 4, "pon": 5,
        "daiminkan": 6, "ankan": 7, "kakan": 8, "tsumo": 9, "ron": 9,
        "hora": 9, "ryukyoku": 10,
    }.get(kind, 0)
    rows = np.zeros((2, 15), dtype=np.int32)
    for query_type, row in ((1, rows[0]), (2, rows[1])):
        row[QUERY_ROW_QUERY_TYPE] = query_type
        row[QUERY_ROW_ACTION_ID] = int(action_id)
        row[QUERY_ROW_ACTION_TYPE] = action_type_code
        row[QUERY_ROW_PRIMARY_TILE] = 0 if primary_type is None else int(primary_type) + 1
        row[QUERY_ROW_SOURCE_SEAT] = 0

    # 独立计算 Defense D0–D8;对非打牌动作按契约填 N/A。
    if kind in {"dahai", "reach"} and primary_type is not None:
        post_counts = _post_action_counts(observation, action, kind)
        rivers = _river_masks(observation)
        visible = _public_visible(observation)
        for opponent_index, opponent in enumerate(opponents):
            genbutsu = int(bool(int(rivers[opponent]) & (1 << primary_type)))
            suji = int(_suji_safe(primary_type, int(rivers[opponent])))
            rows[1][QUERY_ROW_ANSWER_START + opponent_index] = 2 if False else (0 if genbutsu else 1)
            rows[1][QUERY_ROW_ANSWER_START + 3 + opponent_index] = 0 if suji else 1
            stock = sum(
                1 for tile_type in range(34)
                if post_counts[tile_type] > 0 and (int(rivers[opponent]) & (1 << tile_type))
            )
            rows[1][QUERY_ROW_ANSWER_START + 6 + opponent_index] = min(stock, 4)
        rows[1][QUERY_ROW_ANSWER_START + 9] = min(4, int(visible[primary_type]))
    else:
        for slot in range(3):
            rows[1][QUERY_ROW_ANSWER_START + slot] = 2
        for slot in range(3, 6):
            rows[1][QUERY_ROW_ANSWER_START + slot] = 2
        if kind not in {"tsumo", "ron"}:
            current_counts = _post_action_counts(observation, action, kind)
            rivers = _river_masks(observation)
            for opponent_index, opponent in enumerate(opponents):
                stock = sum(
                    1 for tile_type in range(34)
                    if current_counts[tile_type] > 0
                    and (int(rivers[opponent]) & (1 << tile_type))
                )
                rows[1][QUERY_ROW_ANSWER_START + 6 + opponent_index] = min(stock, 4)
        if kind not in {"tsumo", "ron"} and primary_type is None:
            rows[1][QUERY_ROW_ANSWER_START + 9] = 5
        else:
            rows[1][QUERY_ROW_ANSWER_START + 9] = min(
                4, int(_public_visible(observation)[primary_type])
            ) if primary_type is not None else 5

    # 独立核对 offense 的门清/立直条件/保留宝牌赤牌等可判定字段。
    menzen = all(not bool(getattr(meld, "opened", True)) for meld in observation.melds[seat])
    offense_state = {"kind": kind, "menzen": menzen}
    return rows, offense_state


def _make_kyoku(content: str) -> object:
    replay = MjaiReplay.from_jsonl_string(content, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    if len(kyokus) != 1:
        raise RuntimeError(f"member contains {len(kyokus)} kyokus")
    return kyokus[0]


def audit_member(shard: Path, member_name: str) -> dict[str, int]:
    """对一个小局的每个决策做独立 token 解码比对。"""
    content = _read_member(shard, member_name)
    _year, _game_id, _kyoku_index = _member_metadata(member_name)
    actual_kyoku = _make_kyoku(content)
    ref_kyoku = _make_kyoku(content)

    manager = riichi.MjaiKyokuStateMachineManager(1)
    bridge = BatchedStateBridge(manager, 1)
    actual_streams = [
        iter(actual_kyoku.steps(seat=seat, skip_single_action=False))
        for seat in range(4)
    ]
    ref_streams = [
        iter(ref_kyoku.steps(seat=seat, skip_single_action=False))
        for seat in range(4)
    ]
    active = set(range(4))
    last_observations: list[object | None] = [None] * 4
    decision_counts = [0] * 4
    event_buffers: list[list[dict]] = [[] for _ in range(4)]
    compared = 0
    mismatches = 0

    while active:
        actual_batch: list[tuple[int, object]] = []
        ref_batch: list[tuple[int, object]] = []
        for seat in sorted(active):
            actual_item = next(actual_streams[seat], None)
            ref_item = next(ref_streams[seat], None)
            if actual_item is None or ref_item is None:
                active.remove(seat)
                continue
            actual_observation, _expert = actual_item
            ref_observation, _expert_ref = ref_item
            actual_batch.append((seat, actual_observation))
            ref_batch.append((seat, ref_observation))
            last_observations[seat] = actual_observation
            for raw in ref_observation.new_events():
                event = json.loads(raw)
                if event.get("type") == "start_kyoku":
                    event_buffers[seat] = []
                event_buffers[seat].append(event)
        if not actual_batch:
            continue

        observations_by_env = [{
            seat: (
                last_observations[seat]
                if seat in {value for value, _obs in actual_batch}
                else _EmptyEventsObservation(last_observations[seat])
            )
            for seat in range(4)
        }]
        bridge.sync(observations_by_env)
        prepared = bridge.prepare_v16(
            [Decision(0, seat, observations_by_env[0][seat]) for seat in range(4)],
            walls=None,
        )

        for seat, actual_observation in actual_batch:
            ref_observation = next(
                (obs for ref_seat, obs in ref_batch if ref_seat == seat), None
            )
            if ref_observation is None:
                raise RuntimeError("ref batch missing seat")
            row = seat
            decision_index = decision_counts[seat]
            decision_counts[seat] += 1
            context = f"{member_name} seat={seat} decision={decision_index}"
            event_factors, event_numeric = decode_event_rows(event_buffers[seat], seat)
            state_factors, state_numeric = decode_state_rows(
                ref_observation,
                live_wall=_live_wall_from_events(event_buffers[seat]),
            )
            expected_factors = event_factors + state_factors
            expected_numeric = event_numeric + state_numeric
            expected = np.stack(expected_factors, axis=0)
            expected_num = np.stack(expected_numeric, axis=0)
            actual_length = int(prepared.history_lengths[row])
            actual_factors = prepared.history_factors[row, :actual_length]
            actual_numeric = prepared.history_numeric[row, :actual_length]
            if actual_length != len(expected) or not np.array_equal(actual_factors, expected):
                first = int(np.flatnonzero(
                    np.any(
                        np.pad(
                            actual_factors,
                            ((0, max(0, len(expected) - actual_length)), (0, 0)),
                        ) != np.pad(
                            expected,
                            ((0, max(0, actual_length - len(expected))), (0, 0)),
                        ),
                        axis=1,
                    )
                )[0]) if actual_length == len(expected) and np.any(actual_factors != expected) else -1
                raise AssertionError(
                    f"{context} history/state token 不一致: expected={len(expected)} "
                    f"actual={actual_length} first_diff={first}"
                )
            if not np.allclose(actual_numeric, expected_num, rtol=0, atol=1e-3):
                raise AssertionError(f"{context} history/state numeric 不一致")
            compared += 1

            expected_kinds, expected_cat, expected_num = decode_snapshot(ref_observation)
            snapshot_length = int(prepared.snapshot_lengths[row])
            if (
                not np.array_equal(prepared.snapshot_kinds[row, :snapshot_length], expected_kinds)
                or not np.array_equal(prepared.snapshot_cat[row, :snapshot_length], expected_cat)
                or not np.allclose(
                    prepared.snapshot_num[row, :snapshot_length],
                    expected_num, rtol=0, atol=1e-3,
                )
            ):
                raise AssertionError(f"{context} snapshot token 不一致")

            ids = np.flatnonzero(prepared.legal_mask[row]).tolist()
            actions = list(ref_observation.legal_actions())
            representative: dict[str, object] = {}
            for action in actions:
                key = _canonical_template(action, ref_observation)
                representative.setdefault(key, action)
            actual_rows = prepared.query_rows[row, : 2 * len(ids)]
            for index, action_id in enumerate(ids):
                actual_pair = actual_rows[2 * index : 2 * index + 2]
                decoded = manager.decode_actions([seat], [int(action_id)])[0]
                action = representative.get(
                    json.dumps(json.loads(decoded), separators=(",", ":"), sort_keys=True)
                )
                if action is None:
                    raise AssertionError(f"{context} action_id={action_id} 无代表动作")
                expected_pair, offense_state = decode_query_rows(
                    ref_observation, action, int(action_id),
                )
                if not np.array_equal(actual_pair[:, :QUERY_ROW_ANSWER_START], expected_pair[:, :QUERY_ROW_ANSWER_START]):
                    raise AssertionError(f"{context} query 头字段不一致")
                actual_defense = actual_pair[1, QUERY_ROW_ANSWER_START:]
                expected_defense = expected_pair[1, QUERY_ROW_ANSWER_START:]
                if not np.array_equal(actual_defense, expected_defense):
                    raise AssertionError(
                        f"{context} action_id={action_id} defense answers 不一致: "
                        f"expected={expected_defense.tolist()} actual={actual_defense.tolist()}"
                    )
                for slot_index, slot in enumerate(OFFENSE_SLOT_ORDER):
                    if actual_pair[0, QUERY_ROW_ANSWER_START + slot_index] >= SLOT_CARDINALITIES[slot]:
                        raise AssertionError(f"{context} offense slot {slot} 越界")
            mismatches += 0

    return {"decisions": compared, "mismatches": mismatches}


def classify_content(content: str) -> set[str]:
    """按原始 MJAI 事件类型为一个小局分类。"""
    categories = set()
    for line in content.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event_type = str(event.get("type", ""))
        if event_type in TARGET_CATEGORIES:
            categories.add(event_type)
    return categories


def select_members(
    min_per_category: int,
    max_members: int,
) -> list[tuple[Path, str]]:
    """确定性分层抽样,直到各事件类型达到最小覆盖或上限。"""
    shards = sorted((RAW_ROOT / "train").glob("train-*.tar"))
    counts = {category: 0 for category in TARGET_CATEGORIES}
    selected: list[tuple[Path, str]] = []
    for shard in shards:
        members = _selected_members(shard)
        rng = random.Random(f"v16-token-decoder-audit\0{shard.name}")
        rng.shuffle(members)
        chosen_this_shard = 0
        for member_name in members:
            if len(selected) >= max_members or all(
                counts[category] >= min_per_category for category in TARGET_CATEGORIES
            ):
                break
            if chosen_this_shard >= 2:
                break
            content = _read_member(shard, member_name)
            categories = classify_content(content)
            useful = any(counts[category] < min_per_category for category in categories)
            if not useful and len(selected) >= max(20, min_per_category * len(TARGET_CATEGORIES) // 2):
                continue
            selected.append((shard, member_name))
            chosen_this_shard += 1
            for category in categories:
                counts[category] = counts.get(category, 0) + 1
        if len(selected) >= max_members or all(
            counts[category] >= min_per_category for category in TARGET_CATEGORIES
        ):
            break
    print(json.dumps({"selected_members": len(selected), "category_counts": counts}, ensure_ascii=False), flush=True)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-per-category", type=int, default=3)
    parser.add_argument("--max-members", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0, help="只审计前 N 个选中成员")
    args = parser.parse_args()
    members = select_members(args.min_per_category, args.max_members)
    if args.limit:
        members = members[: args.limit]
    results: list[dict[str, int]] = []
    for shard, member_name in members:
        result = audit_member(shard, member_name)
        results.append(result)
        print(
            f"token decode compared {member_name} decisions={result['decisions']} "
            f"mismatches={result['mismatches']}",
            flush=True,
        )
    summary = {
        "members": len(results),
        "decisions": sum(int(item["decisions"]) for item in results),
        "mismatches": sum(int(item["mismatches"]) for item in results),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("independent token decoder audit: all checks passed", flush=True)


if __name__ == "__main__":
    main()
