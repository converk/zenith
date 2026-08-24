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
    three_melds, kan_types = _decompose_melds(melds)
    shape_counts, three_melds = _kernel_shape(
        concealed_tiles, three_melds, kan_types, target=13,
    )
    result = riichi.analyze_offense_v16(
        np.ascontiguousarray(shape_counts[None], dtype=np.uint8),
        np.asarray([three_melds], dtype=np.uint8),
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


def _shanten_counts(counts: np.ndarray, open_melds: int = 0) -> int:
    """34 计数形状的向听(state-machine shanten,生产路径)。"""
    result = riichi.analyze_hands(
        np.ascontiguousarray(counts, dtype=np.uint8)[None],
        np.asarray([int(open_melds)], dtype=np.uint8),
    )
    return int(np.asarray(result.shanten)[0, 0])


def _decompose_melds(melds: list[object]) -> tuple[int, list[int]]:
    """把副露拆成「3 张副露数 + 4 张杠牌型列表」。

    杠的向听等价于一组刻子;其第四张并入暗牌计数(4-copy),不进入 3×副露数。
    """
    three_melds = 0
    kan_types: list[int] = []
    for meld in melds:
        tiles = [int(tile) // 4 for tile in (getattr(meld, "tiles", ()) or ())]
        if len(tiles) == 4 and tiles:
            kan_types.append(tiles[0])
        elif tiles:
            three_melds += 1
    return three_melds, kan_types


def _kernel_shape(
    concealed_tiles: list[int],
    three_melds: int = 0,
    kan_types: list[int] | None = None,
    target: int = 13,
) -> tuple[np.ndarray, int]:
    """内核形状:暗牌计数 + 杠 4-copy,并把偶发多余暗牌剔除到目标张数。"""
    counts = np.bincount(
        [tile // 4 for tile in concealed_tiles], minlength=TILE_KINDS,
    ).astype(np.uint8)
    for tile_type in kan_types or ():
        counts[int(tile_type)] += 4
    excess = int(counts.sum()) + 3 * int(three_melds) - int(target)
    for _ in range(max(0, excess)):
        for tile_type in range(TILE_KINDS):
            if counts[tile_type] > 0:
                counts[tile_type] -= 1
                break
    return counts, int(three_melds)


def _remove_first_by_type(hand: list[int], tile_type: int, count: int) -> None:
    """按牌型从手牌移除指定张数(回放可能把同类牌归一为同一物理 id)。"""
    for _ in range(int(count)):
        for index, physical in enumerate(hand):
            if int(physical) // 4 == int(tile_type):
                hand.pop(index)
                break


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
            post, melds, remaining, own_river,
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
            post, melds, remaining, own_river,
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
        called = int(tile) if tile is not None else None
        # 离线回放把同类牌归一化为同一物理 id,必须按牌型取走指定张数。
        if kind == "kakan":
            added = called if called is not None else (consumed[-1] if consumed else None)
            if added is not None:
                _remove_first_by_type(post, int(added) // 4, 1)
        else:
            counts: dict[int, int] = {}
            for value in consumed:
                tile_type = value // 4
                counts[tile_type] = counts.get(tile_type, 0) + 1
            for tile_type, count in sorted(counts.items()):
                _remove_first_by_type(post, tile_type, count)
        three_melds, kan_types = _decompose_melds(melds)
        kan_type = (
            (called if called is not None else consumed[0]) // 4
            if (called is not None or consumed)
            else None
        )
        post_menzen = menzen and kind == "ankan"
        if kind in {"chi", "pon"}:
            # 吃/碰后还有强制打牌:按 14 张形状取向听。
            counts, three_melds = _kernel_shape(post, three_melds + 1, kan_types, target=14)
            shanten = _shanten_counts(counts, three_melds)
            new_meld = [*consumed]
            if called is not None and called not in new_meld:
                new_meld.append(called)
            full = _physical_tiles(post, melds) + new_meld
        elif kind == "daiminkan":
            counts, three_melds = _kernel_shape(
                post, three_melds, [*kan_types, kan_type] if kan_type is not None else kan_types,
                target=13,
            )
            shanten = _shanten_counts(counts, three_melds)
            full = list(post) + ([called] * 3 if called is not None else consumed)
        elif kind == "ankan":
            # 暗杠 4 张仍属于自身手牌;向听按 13 张物理牌(4-copy 近似)计算。
            counts, three_melds = _kernel_shape(
                post, three_melds, [*kan_types, kan_type] if kan_type is not None else kan_types,
                target=13,
            )
            shanten = _shanten_counts(counts, three_melds)
            full = list(post) + consumed
        else:
            # 加杠:从手牌再移除升级的那一张,组数不变。
            added = called if called is not None else (consumed[-1] if consumed else None)
            if added is not None:
                try:
                    post.remove(added)
                except ValueError:
                    pass
            # 加杠把既有碰升级为 4 张杠:3 张组数减一,杠 4-copy 并入暗牌。
            counts, three_melds = _kernel_shape(
                post, max(0, three_melds - 1),
                [*kan_types, kan_type] if kan_type is not None else kan_types,
                target=13,
            )
            shanten = _shanten_counts(counts, three_melds)
            full = _physical_tiles(post, melds) + ([added] if added is not None else [])
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
            # 内核按「暗牌计数 + 3×副露数」还原总张数;副露牌已固定在场上,
            # 手牌形状只统计暗牌。
            head = _analyze_offense_row(
                hand, melds, remaining, own_river,
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
            three_melds, kan_types = _decompose_melds(melds)
            if not melds:
                # 门清 14 张:取去掉一张后的最小向听。
                shanten = min(
                    _shanten_counts(
                        _kernel_shape(
                            [tile for index, tile in enumerate(hand) if index != drop],
                            0, [], target=13,
                        )[0],
                        0,
                    )
                    for drop in range(len(hand))
                )
            else:
                counts, three_melds = _kernel_shape(hand, three_melds, kan_types, target=13)
                shanten = _shanten_counts(counts, three_melds)
            offense = (
                0 if shanten < 0 else min(shanten, 5) + 1,
                0, 0, 0, 0, 0, 0,
                _O7_YES if menzen else _O7_NO,
                2,
                bucket_o9(_count_dora_aka(full, dora_mult)),
            )
        # 防守库存只统计仍可打出的暗牌,不含已固定的副露牌。
        shape_concealed = np.bincount(
            [t // 4 for t in hand], minlength=TILE_KINDS,
        ).astype(np.uint8)
        defense_genbutsu, defense_suji, defense_stock, _kernel_visible = _analyze_defense_row(
            None, shape_concealed, remaining, rivers, opponents,
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


def _observation_facts(observation: object) -> dict[str, object]:
    """提取一个观察者的全部不变量事实(同一次决策的多个合法动作共享)。

    这些事实只依赖 observation,不依赖具体 action;对同一决策批的多个动作
    只计算一次,避免每个动作重复 ``_remaining_counts`` / ``_river_masks`` /
    ``_public_visible_counts`` 等 O(牌数) 扫描。
    """
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
    return {
        "seat": seat,
        "opponents": opponents,
        "hand": hand,
        "melds": melds,
        "remaining": remaining,
        "rivers": rivers,
        "own_river": own_river,
        "dora_mult": dora_mult,
        "public_visible": public_visible,
        "dora_indicators": dora_indicators,
        "score": score,
        "player_wind": player_wind,
        "round_wind": round_wind,
        "honba": honba,
        "sticks": sticks,
        "declared": declared,
        "menzen": _menzen(melds),
        "missed_doujun": bool(observation.missed_agari_doujun),
        "missed_riichi": bool(observation.missed_agari_riichi),
    }


def _post_shape(hand: list[int], tile: int | None, *, kind: str) -> np.ndarray:
    """打牌/立直/副露后的暗牌 34 计数形状(不含已经固定的副露牌)。"""
    post = list(hand)
    if tile is not None:
        try:
            post.remove(int(tile))
        except ValueError:
            pass
    return np.bincount([t // 4 for t in post], minlength=TILE_KINDS).astype(np.uint8)


def analyze_action_queries_batch(
    rows: list[tuple[object, object, int]],
) -> list[tuple[ActionQuery, ActionQuery]]:
    """批量生成一批 (observation, action, action_id) 的 Offense/Defense Query。

    语义与 ``analyze_action_queries`` 完全一致,但:
    - 同一 observation 的不变量事实只计算一次(``_observation_facts``);
    - 所有动作的 offense / defense / shanten 内核调用汇聚为每内核 1 次 batch
      调用,消除逐动作的 numpy 组装与结果提取 Python 开销(尤其
      ``_analyze_defense_row`` 的 ~31μs/动作)。
    """
    if not rows:
        return []
    facts_cache: dict[int, dict[str, object]] = {}

    kind_of: list[str] = []
    row_primary: list[int | None] = []
    row_source: list[int | None] = []
    o7_of: list[int] = []
    o9_of: list[int | None] = []
    off_slot: list[int | None] = []
    def_slot: list[int | None] = []
    hand_slot: list[int | None] = []
    fixed_offense: list[tuple[int, ...] | None] = []
    fixed_defense: list[tuple[list[int], list[int], list[int], int] | None] = []
    off_reqs: list[tuple[int, dict[str, object]]] = []
    def_reqs: list[tuple[int, dict[str, object]]] = []
    hand_reqs: list[tuple[int, dict[str, object]]] = []

    n = len(rows)
    for row, (observation, action, action_id) in enumerate(rows):
        kind = _action_kind(action)
        facts = facts_cache.setdefault(id(observation), _observation_facts(observation))
        seat = int(facts["seat"])
        hand = list(facts["hand"])
        melds = list(facts["melds"])
        remaining = np.asarray(facts["remaining"])
        rivers = np.asarray(facts["rivers"])
        own_river = int(facts["own_river"])
        dora_mult = facts["dora_mult"]
        public_visible = np.asarray(facts["public_visible"])
        dora_indicators = list(facts["dora_indicators"])
        score = int(facts["score"])
        player_wind = int(facts["player_wind"])
        round_wind = int(facts["round_wind"])
        honba = int(facts["honba"])
        sticks = int(facts["sticks"])
        declared = bool(facts["declared"])
        menzen = bool(facts["menzen"])
        opponents = facts["opponents"]

        tile = getattr(action, "tile", None)
        if tile is None and kind in {"reach", "dahai"}:
            drawn = getattr(observation, "drawn_tile", None)
            if drawn is not None:
                tile = int(drawn)
        primary_type = int(tile) // 4 if tile is not None else None
        source_seat = _source_seat(observation, kind)

        kind_of.append(kind)
        row_primary.append(primary_type)
        row_source.append(source_seat)
        o7_of.append(_O7_YES if menzen else _O7_NO)
        o9_of.append(None)
        off_slot.append(None)
        def_slot.append(None)
        hand_slot.append(None)
        fixed_offense.append(None)
        fixed_defense.append(None)

        if kind in {"tsumo", "ron"}:
            win_type = primary_type
            full = _physical_tiles(hand, melds)
            if tile is not None:
                full.append(int(tile))
            fixed_offense[row] = (
                0, 0, 0, 0, 0, 0, 0,
                o7_of[row], 2,
                bucket_o9(_count_dora_aka(full, dora_mult)),
            )
            fixed_defense[row] = (
                [2, 2, 2], [2, 2, 2], [0, 0, 0],
                _visible_code(public_visible, win_type),
            )
        elif kind in {"reach", "dahai"}:
            post = list(hand)
            if tile is not None:
                try:
                    post.remove(int(tile))
                except ValueError:
                    pass
            shape = np.bincount([t // 4 for t in post], minlength=TILE_KINDS).astype(np.uint8)
            o9_of[row] = bucket_o9(_count_dora_aka(_physical_tiles(post, melds), dora_mult))
            off_reqs.append((row, {
                "post": post, "melds": melds, "remaining": remaining,
                "own_river": own_river, "score": score,
                "dora_indicators": dora_indicators, "player_wind": player_wind,
                "round_wind": round_wind, "honba": honba, "sticks": sticks,
                "missed_doujun": facts["missed_doujun"],
                "missed_riichi": facts["missed_riichi"],
                "declared": True if kind == "reach" else declared,
            }))
            off_slot[row] = len(off_reqs) - 1
            def_reqs.append((row, {
                "discard_type": primary_type, "shape": shape,
                "remaining": remaining, "rivers": rivers, "opponents": opponents,
            }))
            def_slot[row] = len(def_reqs) - 1
        elif kind in {"chi", "pon", "ankan", "daiminkan", "kakan"}:
            consumed = [int(value) for value in (getattr(action, "consume_tiles", ()) or ())]
            post = list(hand)
            called = int(tile) if tile is not None else None
            if kind == "kakan":
                added = called if called is not None else (consumed[-1] if consumed else None)
                if added is not None:
                    _remove_first_by_type(post, int(added) // 4, 1)
            else:
                counts_temp: dict[int, int] = {}
                for value in consumed:
                    tile_type = value // 4
                    counts_temp[tile_type] = counts_temp.get(tile_type, 0) + 1
                for tile_type, count in sorted(counts_temp.items()):
                    _remove_first_by_type(post, tile_type, count)
            three_melds, kan_types = _decompose_melds(melds)
            kan_type = (
                (called if called is not None else consumed[0]) // 4
                if (called is not None or consumed)
                else None
            )
            post_menzen = menzen and kind == "ankan"
            o7_of[row] = _O7_YES if post_menzen else _O7_NO
            if kind in {"chi", "pon"}:
                counts, three_melds = _kernel_shape(post, three_melds + 1, kan_types, target=14)
                new_meld = [*consumed]
                if called is not None and called not in new_meld:
                    new_meld.append(called)
                full = _physical_tiles(post, melds) + new_meld
            elif kind == "daiminkan":
                counts, three_melds = _kernel_shape(
                    post, three_melds, [*kan_types, kan_type] if kan_type is not None else kan_types,
                    target=13,
                )
                full = list(post) + ([called] * 3 if called is not None else consumed)
            elif kind == "ankan":
                counts, three_melds = _kernel_shape(
                    post, three_melds, [*kan_types, kan_type] if kan_type is not None else kan_types,
                    target=13,
                )
                full = list(post) + consumed
            else:
                added = called if called is not None else (consumed[-1] if consumed else None)
                if added is not None:
                    try:
                        post.remove(added)
                    except ValueError:
                        pass
                counts, three_melds = _kernel_shape(
                    post, max(0, three_melds - 1),
                    [*kan_types, kan_type] if kan_type is not None else kan_types,
                    target=13,
                )
                full = _physical_tiles(post, melds) + ([added] if added is not None else [])
            hand_reqs.append((row, {"counts": counts, "melds": three_melds}))
            hand_slot[row] = len(hand_reqs) - 1
            # O9 在组装时与 hand_results 中的 shanten 结合;此处不设 fixed_offense。
            o9_of[row] = bucket_o9(_count_dora_aka(full, dora_mult))
            post_counts = np.bincount(
                [t // 4 for t in post], minlength=TILE_KINDS,
            ).astype(np.uint8)
            def_reqs.append((row, {
                "discard_type": None, "shape": post_counts,
                "remaining": remaining, "rivers": rivers, "opponents": opponents,
            }))
            def_slot[row] = len(def_reqs) - 1
        else:
            # none / pass / ryukyoku:当前手牌不变。
            full = _physical_tiles(hand, melds)
            if len(full) == 13:
                off_reqs.append((row, {
                    "post": hand, "melds": melds, "remaining": remaining,
                    "own_river": own_river, "score": score,
                    "dora_indicators": dora_indicators, "player_wind": player_wind,
                    "round_wind": round_wind, "honba": honba, "sticks": sticks,
                    "missed_doujun": facts["missed_doujun"],
                    "missed_riichi": facts["missed_riichi"],
                    "declared": declared,
                }))
                off_slot[row] = len(off_reqs) - 1
                o9_of[row] = bucket_o9(_count_dora_aka(full, dora_mult))
            else:
                three_melds, kan_types = _decompose_melds(melds)
                if not melds:
                    shanten = min(
                        _shanten_counts(
                            _kernel_shape(
                                [tile for index, tile in enumerate(hand) if index != drop],
                                0, [], target=13,
                            )[0],
                            0,
                        )
                        for drop in range(len(hand))
                    )
                else:
                    counts, three_melds = _kernel_shape(hand, three_melds, kan_types, target=13)
                    shanten = _shanten_counts(counts, three_melds)
                fixed_offense[row] = (
                    0 if shanten < 0 else min(shanten, 5) + 1,
                    0, 0, 0, 0, 0, 0,
                    o7_of[row], 2,
                    bucket_o9(_count_dora_aka(full, dora_mult)),
                )
            shape_concealed = np.bincount(
                [t // 4 for t in hand], minlength=TILE_KINDS,
            ).astype(np.uint8)
            def_reqs.append((row, {
                "discard_type": None, "shape": shape_concealed,
                "remaining": remaining, "rivers": rivers, "opponents": opponents,
            }))
            def_slot[row] = len(def_reqs) - 1

    # ---- 第 2 遍:批量调用 offense / defense / shanten 内核 ----
    off_results: dict[int, tuple[int, ...]] = {}
    if off_reqs:
        off_inputs = []
        for row, req in off_reqs:
            post = req["post"]
            three_melds, kan_types = _decompose_melds(req["melds"])
            counts, three_melds = _kernel_shape(post, three_melds, kan_types, target=13)
            off_inputs.append((row, counts, three_melds, req, post))
        shape_arr = np.ascontiguousarray(np.stack([x[1] for x in off_inputs]), dtype=np.uint8)
        meld_arr = np.asarray([int(x[2]) for x in off_inputs], dtype=np.uint8)
        remain_arr = np.ascontiguousarray(np.stack([x[3]["remaining"] for x in off_inputs]), dtype=np.uint8)
        river_arr = np.asarray([int(x[3]["own_river"]) for x in off_inputs], dtype=np.uint64)
        doujun_arr = np.asarray([bool(x[3]["missed_doujun"]) for x in off_inputs], dtype=bool)
        rmiss_arr = np.asarray([bool(x[3]["missed_riichi"]) for x in off_inputs], dtype=bool)
        decl_arr = np.asarray([bool(x[3]["declared"]) for x in off_inputs], dtype=bool)
        score_arr = np.asarray([int(x[3]["score"]) for x in off_inputs], dtype=np.int32)
        offense_result = riichi.analyze_offense_v16(
            shape_arr, meld_arr, remain_arr, river_arr,
            doujun_arr, rmiss_arr, decl_arr, score_arr,
        )
        shanten_v = np.asarray(offense_result.shanten)
        kinds_v = np.asarray(offense_result.effective_kinds)
        effrem_v = np.asarray(offense_result.effective_remaining)
        waits_v = np.asarray(offense_result.wait_kinds)
        waitmask_v = np.asarray(offense_result.wait_mask)
        furiten_v = np.asarray(offense_result.furiten)
        canriichi_v = np.asarray(offense_result.can_riichi)
        for slot, (row, _counts, _tm, req, post) in enumerate(off_inputs):
            shanten = int(shanten_v[slot])
            kinds = int(kinds_v[slot])
            effrem = int(effrem_v[slot])
            wait_kinds = int(waits_v[slot])
            wait_mask = int(waitmask_v[slot])
            furiten = int(furiten_v[slot])
            can_riichi = _O8_CODE[int(canriichi_v[slot])]
            yaku_class = 0
            base_han = 0
            if wait_kinds > 0:
                melds_list = req["melds"]
                yaku = riichienv.analyze_offense_v16(
                    [post], [melds_list], np.asarray([wait_mask], dtype=np.uint64),
                    [req["dora_indicators"]], np.asarray([req["player_wind"]], dtype=np.uint8),
                    np.asarray([req["round_wind"]], dtype=np.uint8),
                    np.asarray([req["honba"]], dtype=np.uint8),
                    np.asarray([req["sticks"]], dtype=np.uint8),
                )
                yaku_class = int(np.asarray(yaku.yaku_class)[0])
                base_han = int(np.asarray(yaku.base_han)[0])
            off_results[row] = (
                0 if shanten < 0 else min(shanten, 5) + 1,
                bucket_o1(kinds),
                bucket_o2(effrem),
                bucket_o3(wait_kinds if wait_kinds > 0 else None),
                yaku_class,
                bucket_o5(base_han if base_han > 0 else None),
                furiten,
                can_riichi,
            )

    def_results: dict[int, tuple[list[int], list[int], list[int], int]] = {}
    if def_reqs:
        def_discard = np.array(
            [-1 if req["discard_type"] is None else int(req["discard_type"]) for _r, req in def_reqs],
            dtype=np.int16,
        )
        def_rivers = np.stack([
            np.asarray([req["rivers"][opp] for opp in req["opponents"]], dtype=np.uint64)
            for _r, req in def_reqs
        ])
        def_hand = np.ascontiguousarray(np.stack([req["shape"] for _r, req in def_reqs]), dtype=np.uint8)
        def_remain = np.ascontiguousarray(np.stack([req["remaining"] for _r, req in def_reqs]), dtype=np.uint8)
        defense_result = riichi.analyze_defense_v16(
            def_discard, np.ascontiguousarray(def_rivers), def_hand, def_remain,
        )
        genbutsu_v = np.asarray(defense_result.genbutsu)
        suji_v = np.asarray(defense_result.suji)
        stock_v = np.asarray(defense_result.stock)
        visible_v = np.asarray(defense_result.visible)
        for slot, (_r, _req) in enumerate(def_reqs):
            genbutsu = [
                2 if int(value) == 2 else 1 - int(value)
                for value in genbutsu_v[slot]
            ]
            suji = [
                2 if int(value) == 2 else 1 - int(value)
                for value in suji_v[slot]
            ]
            stock = [bucket_d6(int(value)) for value in stock_v[slot]]
            visible = int(visible_v[slot])
            def_results[slot] = (genbutsu, suji, stock, visible)

    hand_results: dict[int, int] = {}
    if hand_reqs:
        hand_counts = np.ascontiguousarray(np.stack([req["counts"] for _r, req in hand_reqs]), dtype=np.uint8)
        hand_melds = np.asarray([int(req["melds"]) for _r, req in hand_reqs], dtype=np.uint8)
        hand_result = riichi.analyze_hands(hand_counts, hand_melds)
        shanten_v = np.asarray(hand_result.shanten)
        for slot, (_r, _req) in enumerate(hand_reqs):
            hand_results[slot] = int(shanten_v[slot, 0])

    # ---- 第 3 遍:组装 ----
    out: list[tuple[ActionQuery, ActionQuery]] = []
    for row in range(n):
        observation, action, action_id = rows[row]
        kind = kind_of[row]
        primary_type = row_primary[row]
        source_seat = row_source[row]
        facts = facts_cache[id(observation)]
        if fixed_offense[row] is not None:
            offense = fixed_offense[row]
        elif off_slot[row] is not None:
            head = off_results[row]
            # O8:立直宣告=2;打牌=can_riichi(head[7]);非打牌(none/pass/ryukyoku)=2。
            o8 = 2 if kind != "dahai" else head[7]
            offense = (
                *head[:7],
                o7_of[row],
                o8,
                o9_of[row],
            )
        else:
            # chi/pon/ankan/daiminkan/kakan:shanten 已在 hand_results 中。
            slot = hand_slot[row]
            shanten = hand_results[slot]
            offense = (
                0 if shanten < 0 else min(shanten, 5) + 1,
                0, 0, 0, 0, 0, 0,
                o7_of[row],
                2 if kind in _KAN_KINDS else 1,
                o9_of[row] if o9_of[row] is not None else 0,
            )
        if fixed_defense[row] is not None:
            d_gen, d_suji, d_stock, d_visible = fixed_defense[row]
        else:
            slot = def_slot[row]
            d_gen, d_suji, d_stock, d_visible = def_results[slot]
        if kind in {"reach", "dahai"} or kind in {"chi", "pon", "ankan", "daiminkan", "kakan"}:
            d_visible = _visible_code(facts["public_visible"], primary_type)
        offense_query = ActionQuery(
            QUERY_OFFENSE, int(action_id), kind, primary_type, source_seat,
            tuple(int(value) for value in offense),
        )
        defense_query = ActionQuery(
            QUERY_DEFENSE, int(action_id), kind, primary_type, source_seat,
            tuple(int(value) for value in (*d_gen, *d_suji, *d_stock, d_visible)),
        )
        out.append((offense_query, defense_query))
    return out


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
