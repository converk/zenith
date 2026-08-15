"""V16 统一 Action Query 取值层。

为每个合法动作生成一对 Offense/Defense Query,每个 Query 固定 10 个 answer slot;
slot 语义、基数与终局约定见 `encoding_protocol.py` 与
`specs/003-v16-model-rework/contracts/actor-input-v16.md`。

事实来源:手牌结构评价用 core(`riichienv.analyze_offense_v16`),公开状态防守事实
用 state-machine(`riichi.analyze_defense_v16`),宝牌/赤牌聚合在模型输入转换侧复用
`model/dora.py`。终局/边缘动作按契约约定在本层直接填值。

约定补充(Assumptions A5 之外由本层拍板):
- 吃/碰/杠等副露动作的 Offense 以「动作后含新副露的形状」计算:吃/碰为 14 张
  形状(取 14 张向听),杠以 13 张物理牌计算向听(暗杠含 4 张杠牌的 4-copy 近似),
  有效牌/等待/役/振听等听牌相关 slot 一律 N/A;O8 对吃/碰取 NO(非门清),对杠取
  N/A(杠本身不是可立直打牌)。
- none/pass/ryukyoku 等不改变手牌的动作按当前手牌现状计算;14 张形状只给向听。
- 终局动作(自摸/荣和)按 A5 终局约定填值。
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

import riichi
import riichienv

from .dora import dora_type_multiplicities
from .encoding_protocol import (
    ACTION_TYPE_CODES,
    QUERY_DEFENSE,
    QUERY_OFFENSE,
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ACTION_TYPE,
    QUERY_ROW_ANSWER_START,
    QUERY_ROW_PRIMARY_TILE,
    QUERY_ROW_QUERY_TYPE,
    QUERY_ROW_SOURCE_SEAT,
    QUERY_ROW_WIDTH,
    bucket_d6,
    bucket_d9,
    bucket_o1,
    bucket_o2,
    bucket_o3,
    bucket_o5,
    bucket_o9,
)
from .schema import TILE_KINDS

# O8 标签为 YES=0/NO=1/N/A=2;内核输出 0=N/A、1=YES、2=NO,需映射。
_O8_CODE = {0: 2, 1: 0, 2: 1}
_O7_YES = 0
_O7_NO = 1
_KAN_KINDS = frozenset({"ankan", "daiminkan", "kakan"})


@dataclass(frozen=True)
class ActionQuery:
    query_type: int
    action_id: int
    action_type: str
    primary_tile: int | None
    source_seat: int | None
    answers: tuple[int, ...]


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


def _own_hand(observation: object) -> list[int]:
    seat = int(observation.player_id)
    return [int(tile) for tile in observation.hands[seat]]


def _own_melds(observation: object) -> list[object]:
    seat = int(observation.player_id)
    return list(observation.melds[seat])


def _remaining_counts(observation: object) -> np.ndarray:
    """公开剩余张数(物理牌去重)。"""
    own_tiles = _own_hand(observation)
    visible: set[int] = set()
    for river in observation.discards:
        visible.update(int(tile) for tile in river)
    for meld_rows in observation.melds:
        for meld in meld_rows:
            visible.update(int(tile) for tile in getattr(meld, "tiles", ()) or ())
    visible.update(int(tile) for tile in observation.dora_indicators)
    result = np.full(TILE_KINDS, 4, dtype=np.int16)
    for tile in own_tiles:
        result[int(tile) // 4] -= 1
    for tile in visible.difference(set(own_tiles)):
        result[int(tile) // 4] -= 1
    return np.maximum(result, 0)


def _river_masks(observation: object) -> np.ndarray:
    masks = np.zeros(4, dtype=np.uint64)
    for seat, river in enumerate(observation.discards):
        mask = 0
        for tile in river:
            mask |= 1 << (int(tile) // 4)
        masks[seat] = mask
    return masks


def _count_dora_aka(tiles: list[int], dora_mult: dict[int, int]) -> int:
    """统计一组物理牌中保留的宝牌 + 赤牌张数。"""
    dora = sum(dora_mult.get(int(tile) // 4, 0) for tile in tiles)
    aka = sum(int(tile) in {16, 52, 88} for tile in tiles)
    return int(dora + aka)


def _public_visible_counts(observation: object) -> np.ndarray:
    """各牌型已公开出现的张数(河 + 副露 + 宝牌指示,不含自身手牌)。"""
    result = np.zeros(TILE_KINDS, dtype=np.uint8)
    for river in observation.discards:
        for tile in river:
            result[int(tile) // 4] += 1
    for meld_rows in observation.melds:
        for meld in meld_rows:
            for tile in getattr(meld, "tiles", ()) or ():
                result[int(tile) // 4] += 1
    for tile in observation.dora_indicators:
        result[int(tile) // 4] += 1
    return np.minimum(result, 4)


def _visible_code(public_visible: np.ndarray, tile_type: int | None) -> int:
    """D9 编码:公开出现数 0..4,N/A=5。"""
    if tile_type is None:
        return bucket_d9(None)
    return bucket_d9(min(4, int(public_visible[tile_type])))


def _physical_tiles(hand: list[int], melds: list[object]) -> list[int]:
    tiles = list(hand)
    for meld in melds:
        tiles.extend(int(tile) for tile in (getattr(meld, "tiles", ()) or ()))
    return tiles


def _menzen(melds: list[object]) -> bool:
    return all(not bool(getattr(meld, "opened", True)) for meld in melds)


def _analyze_offense_row(
    shape_counts: np.ndarray,
    concealed_tiles: list[int],
    melds: list[object],
    remaining: np.ndarray,
    own_river: int,
    missed_doujun: bool,
    missed_riichi: bool,
    riichi_declared: bool,
    score: int,
    dora_indicators: list[int],
    player_wind: int,
    round_wind: int,
    honba: int,
    riichi_sticks: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    """调用 state-machine 与 core 内核,返回 (O0..O6, O8)(不含 O7/O9)。"""
    result = riichi.analyze_offense_v16(
        np.ascontiguousarray(shape_counts[None], dtype=np.uint8),
        np.asarray([sum(1 for meld in melds if bool(getattr(meld, "opened", True)))], dtype=np.uint8),
        np.ascontiguousarray(remaining[None], dtype=np.uint8),
        np.asarray([own_river], dtype=np.uint64),
        np.asarray([bool(missed_doujun)], dtype=bool),
        np.asarray([bool(missed_riichi)], dtype=bool),
        np.asarray([bool(riichi_declared)], dtype=bool),
        np.asarray([int(score)], dtype=np.int32),
    )
    shanten = int(np.asarray(result.shanten)[0])
    kinds = int(np.asarray(result.effective_kinds)[0])
    effective_remaining = int(np.asarray(result.effective_remaining)[0])
    wait_kinds = int(np.asarray(result.wait_kinds)[0])
    wait_mask = int(np.asarray(result.wait_mask)[0])
    furiten = int(np.asarray(result.furiten)[0])
    can_riichi = _O8_CODE[int(np.asarray(result.can_riichi)[0])]
    yaku_class = 0
    base_han = 0
    if wait_kinds > 0:
        yaku = riichienv.analyze_offense_v16(
            [concealed_tiles],
            [[meld for meld in melds]],
            np.asarray([wait_mask], dtype=np.uint64),
            [dora_indicators],
            np.asarray([int(player_wind)], dtype=np.uint8),
            np.asarray([int(round_wind)], dtype=np.uint8),
            np.asarray([int(honba)], dtype=np.uint8),
            np.asarray([int(riichi_sticks)], dtype=np.uint8),
        )
        yaku_class = int(np.asarray(yaku.yaku_class)[0])
        base_han = int(np.asarray(yaku.base_han)[0])
    return (
        0 if shanten < 0 else min(shanten, 5) + 1,
        bucket_o1(kinds),
        bucket_o2(effective_remaining),
        bucket_o3(wait_kinds if wait_kinds > 0 else None),
        yaku_class,
        bucket_o5(base_han if base_han > 0 else None),
        furiten,
        can_riichi,
    )


def _analyze_defense_row(
    discard_type: int | None,
    post_hand_counts: np.ndarray,
    remaining: np.ndarray,
    river_masks: np.ndarray,
    opponents: tuple[int, int, int],
) -> tuple[list[int], list[int], list[int], int]:
    """调用 state-machine 内核,返回 D0–D2、D3–D5、D6–D8 与 D9 编码。"""
    result = riichi.analyze_defense_v16(
        np.asarray([-1 if discard_type is None else int(discard_type)], dtype=np.int16),
        np.ascontiguousarray(
            np.asarray([river_masks[seat] for seat in opponents], dtype=np.uint64)[None]
        ),
        np.ascontiguousarray(post_hand_counts[None], dtype=np.uint8),
        np.ascontiguousarray(remaining[None], dtype=np.uint8),
    )
    # 内核用 1=是/0=否;契约标签为 GENBUTSU=0/NOT=1、SUJI=0/NOT=1,需取反(N/A=2
    # 不变)。
    genbutsu = [
        2 if int(value) == 2 else 1 - int(value)
        for value in np.asarray(result.genbutsu)[0]
    ]
    suji = [
        2 if int(value) == 2 else 1 - int(value)
        for value in np.asarray(result.suji)[0]
    ]
    stock = [bucket_d6(int(value)) for value in np.asarray(result.stock)[0]]
    visible = int(np.asarray(result.visible)[0])
    return genbutsu, suji, stock, visible


def _source_seat(observation: object, kind: str) -> int | None:
    if kind in {"chi", "pon", "daiminkan", "ron"}:
        last = getattr(observation, "last_discard", None)
        if last is not None:
            try:
                return int(last[0])
            except (TypeError, ValueError):
                return None
    return None


def _shanten_14(concealed_types: list[int], open_melds: int) -> int:
    """14 张形状的向听数 = 暗牌去掉一张后的最小向听(吃/碰后的强制打牌形状)。"""
    counts = np.bincount(concealed_types, minlength=TILE_KINDS)
    best = 127
    for tile in range(TILE_KINDS):
        if counts[tile] > 0:
            reduced = counts.copy()
            reduced[tile] -= 1
            result = riichi.analyze_hands(
                np.ascontiguousarray(reduced, dtype=np.uint8)[None],
                np.asarray([int(open_melds)], dtype=np.uint8),
            )
            best = min(best, int(np.asarray(result.shanten)[0, 0]))
    return best


def _shanten_counts(counts: np.ndarray, open_melds: int = 0) -> int:
    """34 计数形状的向听(state-machine shanten,生产路径)。"""
    result = riichi.analyze_hands(
        np.ascontiguousarray(counts, dtype=np.uint8)[None],
        np.asarray([int(open_melds)], dtype=np.uint8),
    )
    return int(np.asarray(result.shanten)[0, 0])


def analyze_action_queries(
    observation: object,
    action: object,
    action_id: int,
) -> tuple[ActionQuery, ActionQuery]:
    """生成一个动作的 Offense/Defense 两个 Query。"""
    kind = _action_kind(action)
    seat = int(observation.player_id)
    opponents = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
    hand = _own_hand(observation)
    melds = _own_melds(observation)
    remaining = _remaining_counts(observation)
    rivers = _river_masks(observation)
    own_river = int(rivers[seat])
    dora_mult = dora_type_multiplicities(observation.dora_indicators)
    public_visible = _public_visible_counts(observation)
    dora_indicators = [int(tile) for tile in observation.dora_indicators]
    score = int(observation.scores[seat])
    player_wind = (seat - int(observation.oya)) % 4
    round_wind = int(observation.round_wind)
    honba = int(observation.honba)
    sticks = int(observation.riichi_sticks)
    declared = bool(observation.riichi_declared[seat])
    tile = getattr(action, "tile", None)
    # 离线回放里 reach/dahai 可能不在 Action 上携带物理牌,改为取当前摸牌。
    if tile is None and kind in {"reach", "dahai"}:
        drawn = getattr(observation, "drawn_tile", None)
        if drawn is not None:
            tile = int(drawn)
    primary_type = int(tile) // 4 if tile is not None else None
    menzen = _menzen(melds)

    offense: tuple[int, ...]
    defense_genbutsu: list[int]
    defense_suji: list[int]
    defense_stock: list[int]
    defense_visible: int

    if kind in {"tsumo", "ron"}:
        # A5 终局约定:和了。
        win_type = primary_type
        full = _physical_tiles(hand, melds)
        if tile is not None:
            full.append(int(tile))
        offense = (
            0, 0, 0, 0, 0, 0, 0,
            _O7_YES if menzen else _O7_NO,
            2,
            bucket_o9(_count_dora_aka(full, dora_mult)),
        )
        defense_genbutsu = [2, 2, 2]
        defense_suji = [2, 2, 2]
        defense_stock = [0, 0, 0]
        defense_visible = _visible_code(public_visible, win_type)
    elif kind == "reach":
        post = list(hand)
        if tile is not None:
            try:
                post.remove(int(tile))
            except ValueError:
                pass
        shape = np.bincount([t // 4 for t in post], minlength=TILE_KINDS).astype(np.uint8)
        head = _analyze_offense_row(
            shape, post, melds, remaining, own_river,
            bool(observation.missed_agari_doujun),
            bool(observation.missed_agari_riichi),
            True, score, dora_indicators, player_wind, round_wind, honba, sticks,
        )
        offense = (
            *head[:7],
            _O7_YES if menzen else _O7_NO,
            2,  # 立直宣告动作本身:O8=N/A
            bucket_o9(_count_dora_aka(_physical_tiles(post, melds), dora_mult)),
        )
        defense_genbutsu, defense_suji, defense_stock, _kernel_visible = _analyze_defense_row(
            primary_type, shape, remaining, rivers, opponents,
        )
        defense_visible = _visible_code(public_visible, primary_type)
    elif kind == "dahai":
        post = list(hand)
        if tile is not None:
            try:
                post.remove(int(tile))
            except ValueError:
                pass
        shape = np.bincount([t // 4 for t in post], minlength=TILE_KINDS).astype(np.uint8)
        head = _analyze_offense_row(
            shape, post, melds, remaining, own_river,
            bool(observation.missed_agari_doujun),
            bool(observation.missed_agari_riichi),
            declared, score, dora_indicators, player_wind, round_wind, honba, sticks,
        )
        offense = (
            *head[:7],
            _O7_YES if menzen else _O7_NO,
            head[7],
            bucket_o9(_count_dora_aka(_physical_tiles(post, melds), dora_mult)),
        )
        defense_genbutsu, defense_suji, defense_stock, _kernel_visible = _analyze_defense_row(
            primary_type, shape, remaining, rivers, opponents,
        )
        defense_visible = _visible_code(public_visible, primary_type)
    elif kind in {"chi", "pon", "ankan", "daiminkan", "kakan"}:
        consumed = [int(value) for value in (getattr(action, "consume_tiles", ()) or ())]
        post = list(hand)
        for value in consumed:
            try:
                post.remove(value)
            except ValueError:
                continue
        called = int(tile) if tile is not None else None
        if kind == "ankan":
            # RiichiEnv 暗杠后手牌仍保留 4 张杠牌;取 13 张物理牌计算向听。
            full = list(post)
        elif kind == "daiminkan":
            full = list(post) + ([called] * 3 if called is not None else consumed)
        elif kind == "kakan":
            full = _physical_tiles(post, melds) + ([called] if called is not None else [])
        else:  # chi / pon
            new_meld = [*consumed]
            if called is not None and called not in new_meld:
                new_meld.append(called)
            full = _physical_tiles(post, melds) + new_meld
        opened_count = sum(
            1 for meld in melds if bool(getattr(meld, "opened", True))
        )
        post_menzen = menzen and kind == "ankan"
        if kind in {"chi", "pon"}:
            shanten = _shanten_14([t // 4 for t in post], opened_count + 1)
        else:
            shanten = _shanten_counts(
                np.bincount([t // 4 for t in post], minlength=TILE_KINDS),
                opened_count + (0 if kind == "ankan" else 1),
            )
        offense = (
            0 if shanten < 0 else min(shanten, 5) + 1,
            0, 0, 0, 0, 0, 0,
            _O7_YES if post_menzen else _O7_NO,
            2 if kind in _KAN_KINDS else 1,
            bucket_o9(_count_dora_aka(full, dora_mult)),
        )
        post_counts = np.bincount(
            [t // 4 for t in post], minlength=TILE_KINDS
        ).astype(np.uint8)
        defense_genbutsu, defense_suji, defense_stock, _kernel_visible = _analyze_defense_row(
            None, post_counts, remaining, rivers, opponents,
        )
        defense_visible = _visible_code(public_visible, primary_type)
    else:
        # none / pass / ryukyoku:当前手牌不变。
        full = _physical_tiles(hand, melds)
        if len(full) == 13:
            shape = np.bincount([t // 4 for t in full], minlength=TILE_KINDS).astype(np.uint8)
            head = _analyze_offense_row(
                shape, hand, melds, remaining, own_river,
                bool(observation.missed_agari_doujun),
                bool(observation.missed_agari_riichi),
                declared, score, dora_indicators, player_wind, round_wind, honba, sticks,
            )
            offense = (
                *head[:7],
                _O7_YES if menzen else _O7_NO,
                2,  # 非打牌动作:O8=N/A
                bucket_o9(_count_dora_aka(full, dora_mult)),
            )
        else:
            opened_count = sum(
                1 for meld in melds if bool(getattr(meld, "opened", True))
            )
            shanten = _shanten_14([t // 4 for t in hand], opened_count)
            offense = (
                0 if shanten < 0 else min(shanten, 5) + 1,
                0, 0, 0, 0, 0, 0,
                _O7_YES if menzen else _O7_NO,
                2,
                bucket_o9(_count_dora_aka(full, dora_mult)),
            )
        shape_full = np.bincount([t // 4 for t in full], minlength=TILE_KINDS).astype(np.uint8)
        defense_genbutsu, defense_suji, defense_stock, _kernel_visible = _analyze_defense_row(
            None, shape_full, remaining, rivers, opponents,
        )
        defense_visible = bucket_d9(None)

    offense_query = ActionQuery(
        QUERY_OFFENSE,
        int(action_id),
        kind,
        primary_type,
        _source_seat(observation, kind),
        tuple(int(value) for value in offense),
    )
    defense_query = ActionQuery(
        QUERY_DEFENSE,
        int(action_id),
        kind,
        primary_type,
        _source_seat(observation, kind),
        tuple(
            int(value) for value in (
                *defense_genbutsu, *defense_suji, *defense_stock, defense_visible
            )
        ),
    )
    return offense_query, defense_query


def encode_query_row(query: ActionQuery) -> np.ndarray:
    """把 ActionQuery 编码为固定宽度存储行(见 QUERY_ROW_WIDTH)。

    行布局:[query_type, action_id, action_type_code, primary_tile_code,
    source_seat_code, answer_0..answer_9];primary_tile/source_seat 的 0 表示
    N/A,answer 直接使用各 slot 的 categorical 编码。
    """
    row = np.zeros(QUERY_ROW_WIDTH, dtype=np.int32)
    row[QUERY_ROW_QUERY_TYPE] = int(query.query_type)
    row[QUERY_ROW_ACTION_ID] = int(query.action_id)
    row[QUERY_ROW_ACTION_TYPE] = int(ACTION_TYPE_CODES.get(query.action_type, 0))
    row[QUERY_ROW_PRIMARY_TILE] = (
        0 if query.primary_tile is None else int(query.primary_tile) + 1
    )
    row[QUERY_ROW_SOURCE_SEAT] = (
        0 if query.source_seat is None else int(query.source_seat) + 1
    )
    if len(query.answers) != 10:
        raise ValueError(f"action query must have 10 answers, got {len(query.answers)}")
    row[QUERY_ROW_ANSWER_START:] = [int(value) for value in query.answers]
    return row
