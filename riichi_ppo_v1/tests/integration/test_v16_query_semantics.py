"""V16 action query 语义正确性硬门槛。

独立 oracle:每个局面用「手算期望值」逐 slot 比对,不调用编码器的任何计算路径;
任何 slot 不一致即失败。物理牌号映射:1m=0..3,4p=48..51,6p=56..59,
5s=88..91,6s=92..95,7s=96..99,8s=100..103,9s=104..107,9p=68..71。
"""

from __future__ import annotations

import pytest

from riichienv import Meld, MeldType, RiichiEnv
from riichienv.action import Action, ActionType

from riichi_ppo_v1.model.action_query import analyze_action_queries
from riichi_ppo_v1.model.encoding_protocol import (
    DEFENSE_SLOT_LABELS,
    OFFENSE_SLOT_LABELS,
)


def _make_observation(
    seat: int,
    hands: list[list[int]],
    discards: list[list[int]],
    melds: list[list[Meld]],
    dora: list[int] | None = None,
    *,
    oya: int = 0,
) -> object:
    env = RiichiEnv(seed=0, game_mode="4p-red-half")
    env.reset()
    env.hands = [list(row) for row in hands]
    env.discards = [list(row) for row in discards]
    env.melds = [list(row) for row in melds]
    env.dora_indicators = list(dora or [])
    env.oya = oya
    env.round_wind = 0
    env.honba = 0
    env.riichi_sticks = 0
    return env.get_observation(seat)


def _labels(query_type: int, slot: str) -> tuple[str, ...]:
    table = OFFENSE_SLOT_LABELS if query_type == 1 else DEFENSE_SLOT_LABELS
    return table[slot]


def _assert_slots(offense: object, defense: object, expected: dict[str, str]) -> None:
    """按标签名断言全部 20 个 slot(允许只给子集)。"""
    for slot, label in expected.items():
        if slot.startswith("D"):
            query = defense
        else:
            query = offense
        index = int(slot[1:])
        got = _labels(query.query_type, slot)[query.answers[index]]
        assert got == label, (
            f"{slot}: expected {label!r}, got {got!r} "
            f"(answers={query.answers})"
        )


def test_closed_pinfu_discard_all_slots() -> None:
    """门清听牌打牌:123m 456p 789s 55m 67s + 打 9p,听 5s/8s(平和)。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 68],
        [68],   # 下家河:9p(现物)
        [56],   # 对家河:6p(9p 的筋)
        [69],   # 上家河:9p(现物)
    ]
    observation = _make_observation(0, hands, [[], [68], [56], [69]], [[], [], [], []])
    action = Action(ActionType.DISCARD, tile=68)
    offense, defense = analyze_action_queries(observation, action, 17)

    _assert_slots(offense, defense, {
        "O0": "0", "O1": "0", "O2": "0", "O3": "2", "O4": "ALL_YAKU",
        "O5": "1", "O6": "NO_FURITEN", "O7": "YES", "O8": "YES", "O9": "0",
        "D0": "GENBUTSU", "D1": "NOT_GENBUTSU", "D2": "GENBUTSU",
        "D3": "NOT_SUJI", "D4": "SUJI", "D5": "NOT_SUJI",
        "D6": "0", "D7": "1", "D8": "0", "D9": "2",
    })


def test_open_hand_discard_waits_and_furiten() -> None:
    """副露手打牌:碰 5s 后 123m 456p 78s 11m,打 9m,听 6s/9s(无役)。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 0, 1, 32],
        [],
        [92],
        [],
    ]
    melds = [
        [Meld(MeldType.Pon, [89, 90, 91], True, 2, None)],
        [],
        [],
        [],
    ]
    observation = _make_observation(0, hands, [[92], [], [92], []], melds)
    action = Action(ActionType.DISCARD, tile=32)
    offense, defense = analyze_action_queries(observation, action, 32)

    _assert_slots(offense, defense, {
        "O0": "0", "O1": "0", "O2": "0", "O3": "2", "O4": "NO_YAKU",
        "O5": "N/A", "O6": "PERMANENT_FURITEN", "O7": "NO", "O8": "NO",
        "O9": "0",
        "D0": "NOT_GENBUTSU", "D1": "NOT_GENBUTSU", "D2": "NOT_GENBUTSU",
        "D3": "NOT_SUJI", "D4": "NOT_SUJI", "D5": "NOT_SUJI",
        "D6": "0", "D7": "0", "D8": "0", "D9": "0",
    })
    # 上家河含 6s(等待之一)→ 永久振听。
    assert offense.answers[6] == 2  # PERMANENT_FURITEN


def test_terminal_tsumo_and_ron_conventions() -> None:
    """终局动作按 A5 约定填值。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 89],
        [],
        [],
        [],
    ]
    observation = _make_observation(0, hands, [[], [], [], []], [[], [], [], []])
    tsumo = Action(ActionType.TSUMO, tile=89)
    offense, defense = analyze_action_queries(observation, tsumo, 170)
    _assert_slots(offense, defense, {
        "O0": "AGARI", "O1": "0", "O2": "0", "O3": "N/A", "O4": "N/A",
        "O5": "N/A", "O6": "N/A", "O7": "YES", "O8": "N/A", "O9": "0",
        "D0": "N/A", "D3": "N/A", "D6": "0", "D9": "0",
    })

    ron = Action(ActionType.RON, tile=89)
    offense, defense = analyze_action_queries(observation, ron, 169)
    _assert_slots(offense, defense, {
        "O0": "AGARI", "O7": "YES", "O8": "N/A", "D6": "0", "D9": "0",
    })


def test_pass_and_chi_conventions() -> None:
    """pass/none 与吃牌动作的 N/A 规则。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 16, 17, 92, 96],
        [],
        [],
        [],
    ]
    observation = _make_observation(0, hands, [[], [], [], []], [[], [], [], []])
    none = Action(ActionType.PASS)
    offense, defense = analyze_action_queries(observation, none, 240)
    _assert_slots(offense, defense, {
        "O8": "N/A", "D0": "N/A", "D3": "N/A", "D6": "0", "D9": "N/A",
    })
    assert offense.answers[0] == 1  # 当前 13 张形状向听 0

    # 吃 4m5m(吃 6m):动作后 14 张形状。
    chi_hand = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 16, 17, 20, 24],
        [],
        [],
        [],
    ]
    chi_obs = _make_observation(0, chi_hand, [[], [], [16], []], [[], [], [], []])
    chi = Action(ActionType.CHI, tile=16, consume_tiles=[20, 24])
    offense, defense = analyze_action_queries(chi_obs, chi, 76)
    _assert_slots(offense, defense, {
        "O1": "0", "O2": "0", "O3": "N/A", "O4": "N/A", "O5": "N/A",
        "O6": "N/A", "O7": "NO", "O8": "NO",
        "D0": "N/A", "D3": "N/A", "D9": "1",
    })


@pytest.mark.parametrize("scenario", ["pinfu", "open", "tsumo", "chi"])
def test_codes_stay_within_declared_cardinality(scenario: str) -> None:
    """全部编码值必须在 contracts 声明的基数内。"""
    if scenario == "pinfu":
        hands = [[0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 68], [], [], []]
        observation = _make_observation(0, hands, [[], [], [], []], [[], [], [], []])
        queries = analyze_action_queries(observation, Action(ActionType.DISCARD, tile=68), 17)
    elif scenario == "open":
        hands = [[0, 4, 8, 48, 53, 56, 96, 100, 0, 1, 32], [], [], []]
        melds = [[Meld(MeldType.Pon, [89, 90, 91], True, 2, None)], [], [], []]
        observation = _make_observation(0, hands, [[], [], [], []], melds)
        queries = analyze_action_queries(observation, Action(ActionType.DISCARD, tile=32), 32)
    elif scenario == "tsumo":
        hands = [[0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 89], [], [], []]
        observation = _make_observation(0, hands, [[], [], [], []], [[], [], [], []])
        queries = analyze_action_queries(observation, Action(ActionType.TSUMO, tile=89), 170)
    else:
        hands = [[0, 4, 8, 48, 53, 56, 96, 100, 104, 16, 17, 20, 24], [], [], []]
        observation = _make_observation(0, hands, [[], [], [], []], [[], [], [], []])
        queries = analyze_action_queries(
            observation, Action(ActionType.CHI, tile=16, consume_tiles=[20, 24]), 76,
        )
    offense, defense = queries
    for query in (offense, defense):
        assert len(query.answers) == 10
    for slot in ("O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9"):
        assert 0 <= offense.answers[int(slot[1:])] < len(OFFENSE_SLOT_LABELS[slot])
    for slot in ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"):
        assert 0 <= defense.answers[int(slot[1:])] < len(DEFENSE_SLOT_LABELS[slot])
