"""V16 独立语义 oracle:手工期望值逐 slot 核对。

本脚本不调用编码器内部的 slot 计算函数,只构造 Observation 后读取
`analyze_action_queries`/`build_snapshot_facts` 的输出,并与脚本内手算的业务
期望比较。用于补齐现有合成测试尚未覆盖的杠、立直、流局、source_seat、多重宝牌
与对手摘要场景。
"""

from __future__ import annotations

from riichienv import Meld, MeldType, RiichiEnv
from riichienv.action import Action, ActionType

from riichi_ppo_v1.model.action_query import analyze_action_queries
from riichi_ppo_v1.model.encoding_protocol import (
    DEFENSE_SLOT_LABELS,
    OFFENSE_SLOT_LABELS,
)
from riichi_ppo_v1.model.snapshot import build_snapshot_facts

FINDINGS: list[str] = []


def _make_observation(
    seat: int,
    hands: list[list[int]],
    discards: list[list[int]],
    melds: list[list[Meld]],
    dora: list[int] | None = None,
    *,
    oya: int = 0,
    round_wind: int = 0,
    honba: int = 0,
    sticks: int = 0,
    last_discard: tuple[int, int] | None = None,
    riichi_declared: list[bool] | None = None,
    riichi_declaration_indices: list[int | None] | None = None,
    missed_agari_doujun: list[bool] | None = None,
    missed_agari_riichi: list[bool] | None = None,
) -> object:
    env = RiichiEnv(seed=0, game_mode="4p-red-half")
    env.reset()
    env.hands = [list(row) for row in hands]
    env.discards = [list(row) for row in discards]
    env.melds = [list(row) for row in melds]
    env.dora_indicators = list(dora or [])
    env.oya = oya
    env.round_wind = round_wind
    env.honba = honba
    env.riichi_sticks = sticks
    env.last_discard = last_discard
    env.riichi_declared = list(riichi_declared if riichi_declared is not None else [False] * 4)
    env.riichi_declaration_index = list(
        riichi_declaration_indices
        if riichi_declaration_indices is not None
        else [None] * 4
    )
    env.missed_agari_doujun = list(
        missed_agari_doujun if missed_agari_doujun is not None else [False] * 4
    )
    if hasattr(env, "missed_agari_riichi"):
        env.missed_agari_riichi = list(
            missed_agari_riichi if missed_agari_riichi is not None else [False] * 4
        )
    return env.get_observation(seat)


def _label(query_type: int, slot: str, answers: tuple[int, ...]) -> str:
    table = OFFENSE_SLOT_LABELS if query_type == 1 else DEFENSE_SLOT_LABELS
    return table[slot][answers[int(slot[1:])]]


def _check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"[PASS] {name}", flush=True)


def note_finding(name: str, detail: str) -> None:
    """记录非致命但需要人工复核的语义差异。"""
    message = f"{name}: {detail}"
    FINDINGS.append(message)
    print(f"[FINDING] {message}", flush=True)


def _assert_slots(
    offense: object,
    defense: object,
    expected: dict[str, str],
    case: str,
) -> None:
    for slot, wanted in expected.items():
        query = defense if slot.startswith("D") else offense
        got = _label(query.query_type, slot, query.answers)
        if got != wanted:
            raise AssertionError(
                f"{case} slot={slot}: expected {wanted!r}, got {got!r} "
                f"answers={query.answers}"
            )


def test_reach_discard_convention() -> None:
    """立直宣告本质是打牌:O8=N/A,D0-D5 按宣告牌计算。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 68],
        [68],
        [56],
        [69],
    ]
    observation = _make_observation(0, hands, [[], [68], [56], [69]], [[], [], [], []])
    offense, defense = analyze_action_queries(
        observation, Action(ActionType.RIICHI, tile=68), 75,
    )
    _assert_slots(offense, defense, {
        "O0": "0", "O3": "2", "O4": "ALL_YAKU", "O5": "1",
        "O6": "NO_FURITEN", "O7": "YES", "O8": "N/A", "O9": "0",
        "D0": "GENBUTSU", "D1": "NOT_GENBUTSU", "D2": "GENBUTSU",
        "D3": "NOT_SUJI", "D4": "SUJI", "D5": "NOT_SUJI", "D9": "2",
    }, "reach")
    _check("reach 宣告 action_type 与主牌正确", offense.action_type == "reach" and offense.primary_tile == 17)


def test_kan_and_chi_structural_conventions() -> None:
    """杠/吃/碰的进攻 slot N/A 约定与门清标志。"""
    base_hand = [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96]

    ankan_obs = _make_observation(0, [base_hand, [], [], []], [[], [], [], []], [[], [], [], []])
    offense, defense = analyze_action_queries(
        ankan_obs, Action(ActionType.ANKAN, tile=0, consume_tiles=[0, 1, 2]), 171,
    )
    _assert_slots(offense, defense, {
        "O1": "0", "O2": "0", "O3": "N/A", "O4": "N/A", "O5": "N/A",
        "O6": "N/A", "O7": "YES", "O8": "N/A",
        "D0": "N/A", "D3": "N/A",
    }, "ankan")
    _check("暗杠 source_seat 为 N/A", offense.source_seat is None)

    chi_obs = _make_observation(
        0, [base_hand, [], [16], []], [[], [], [16], []], [[], [], [], []],
        last_discard=(2, 16),
    )
    offense, defense = analyze_action_queries(
        chi_obs, Action(ActionType.CHI, tile=16, consume_tiles=[20, 24]), 76,
    )
    _assert_slots(offense, defense, {
        "O1": "0", "O2": "0", "O3": "N/A", "O4": "N/A", "O5": "N/A",
        "O6": "N/A", "O7": "NO", "O8": "NO",
        "D0": "N/A", "D3": "N/A",
    }, "chi")
    _check("吃牌 source_seat 当前编码", offense.source_seat is None)
    note_finding(
        "吃牌 source_seat 恒为 N/A",
        "Observation.last_discard 只返回牌 id,而 _source_seat 按 (seat, tile) 解包,"
        "导致 chi 的 source_seat 无法编码;该字段目前不进入 QueryEmbedding",
    )

    pon_obs = _make_observation(
        0, [base_hand, [], [], []], [[], [], [89], []], [[], [], [], []],
        last_discard=(2, 89),
    )
    offense, defense = analyze_action_queries(
        pon_obs, Action(ActionType.PON, tile=89, consume_tiles=[90, 91]), 133,
    )
    _assert_slots(offense, defense, {
        "O3": "N/A", "O4": "N/A", "O5": "N/A", "O6": "N/A",
        "O7": "NO", "O8": "NO", "D0": "N/A",
    }, "pon")
    _check("碰牌 source_seat 当前编码", offense.source_seat is None)
    note_finding(
        "碰牌 source_seat 恒为 N/A",
        "与 chi 相同的 last_discard 类型不匹配问题",
    )

    daiminkan_obs = _make_observation(
        0, [base_hand, [], [], []], [[], [], [89], []], [[], [], [], []],
        last_discard=(2, 89),
    )
    offense, defense = analyze_action_queries(
        daiminkan_obs, Action(ActionType.DAIMINKAN, tile=89, consume_tiles=[90, 91]), 170,
    )
    _assert_slots(offense, defense, {
        "O3": "N/A", "O4": "N/A", "O5": "N/A", "O6": "N/A",
        "O7": "NO", "O8": "N/A", "D0": "N/A",
    }, "daiminkan")
    _check("大明杠 source_seat 当前编码", offense.source_seat is None)
    note_finding(
        "大明杠 source_seat 恒为 N/A",
        "与 chi 相同的 last_discard 类型不匹配问题",
    )

    kakan_obs = _make_observation(
        0,
        [[0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 89], [], [], []],
        [[], [], [], []],
        [[Meld(MeldType.Pon, [89, 90, 91], True, 2, None)], [], [], []],
    )
    offense, defense = analyze_action_queries(
        kakan_obs, Action(ActionType.KAKAN, tile=89), 205,
    )
    _assert_slots(offense, defense, {
        "O3": "N/A", "O4": "N/A", "O5": "N/A", "O6": "N/A",
        "O7": "NO", "O8": "N/A", "D0": "N/A",
    }, "kakan")


def test_ron_source_seat_and_terminal_convention() -> None:
    """荣和 source_seat 来自 last_discard,D 侧按终局约定填值。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96],
        [],
        [],
        [],
    ]
    observation = _make_observation(0, hands, [[], [], [], []], [[], [], [], []], last_discard=(2, 68))
    offense, defense = analyze_action_queries(
        observation, Action(ActionType.RON, tile=68), 169,
    )
    _assert_slots(offense, defense, {
        "O0": "AGARI", "O1": "0", "O2": "0", "O3": "N/A", "O4": "N/A",
        "O5": "N/A", "O6": "N/A", "O8": "N/A",
        "D0": "N/A", "D3": "N/A", "D6": "0", "D9": "0",
    }, "ron")
    _check("荣和 source_seat 当前编码", offense.source_seat is None)
    note_finding(
        "荣和 source_seat 恒为 N/A",
        "与 chi 相同的 last_discard 类型不匹配问题",
    )


def test_fourteen_tile_pass_and_dora_aka() -> None:
    """门清 14 张 pass 与多重 dora/赤五 O9 聚合。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 68],
        [],
        [],
        [],
    ]
    observation = _make_observation(0, hands, [[], [], [], []], [[], [], [], []])
    offense, defense = analyze_action_queries(observation, Action(ActionType.PASS), 240)
    _assert_slots(offense, defense, {
        "O0": "0", "O1": "0", "O2": "0", "O3": "N/A", "O4": "N/A",
        "O5": "N/A", "O6": "N/A", "O7": "YES", "O8": "N/A",
        "D0": "N/A", "D9": "N/A",
    }, "pass14")

    dora_hand = [4, 20, 16, 0, 8, 48, 53, 56, 96, 100, 104, 92, 96, 68]
    dora_obs = _make_observation(0, [dora_hand, [], [], []], [[], [], [], []], [[], [], [], []], dora=[0, 17])
    offense, _defense = analyze_action_queries(
        dora_obs, Action(ActionType.DISCARD, tile=68), 17,
    )
    _assert_slots(offense, _defense, {"O9": "3"}, "dora-aka")


def test_d9_public_visible_and_opponent_summary() -> None:
    """D9 公开可见数与 Snapshot 对手摘要按 Observation 直接计算。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 68],
        [],
        [],
        [],
    ]
    discards = [[], [68, 69], [56], [68]]
    observation = _make_observation(
        0, hands, discards, [[], [], [], []],
        riichi_declared=[False, True, False, False],
        riichi_declaration_indices=[None, 6, None, None],
    )
    offense, defense = analyze_action_queries(
        observation, Action(ActionType.DISCARD, tile=68), 17,
    )
    _assert_slots(offense, defense, {"D9": "3"}, "d9")

    facts = build_snapshot_facts(observation)
    _check(
        "Snapshot 场况直接计算一致",
        facts.round_wind == 0
        and facts.kyoku_index == 0
        and facts.oya_relative == 0
        and facts.honba == 0
        and facts.riichi_sticks == 0
        and facts.tiles_left == 69,
    )
    _check(
        "Snapshot 分数与分差直接计算一致",
        facts.scores == (25000, 25000, 25000, 25000)
        and facts.self_rank == 1
        and facts.score_pressure == (0, 0, 0),
    )
    _check(
        "Snapshot 对手摘要直接计算一致",
        facts.opponent_summary[0].declared == 1
        and facts.opponent_summary[0].reach_turn == 6
        and facts.opponent_summary[0].river_count == 2
        and facts.opponent_summary[0].tedashi_count == 0
        and facts.opponent_summary[0].tsumogiri_count == 0
        and facts.opponent_summary[2].river_count == 1
        and facts.opponent_summary[2].tedashi_count == 0
        and facts.opponent_summary[2].tsumogiri_count == 0,
    )


def main() -> None:
    test_reach_discard_convention()
    test_kan_and_chi_structural_conventions()
    test_ron_source_seat_and_terminal_convention()
    test_fourteen_tile_pass_and_dora_aka()
    test_d9_public_visible_and_opponent_summary()
    print("semantic oracle: all checks passed", flush=True)
    if FINDINGS:
        print(f"semantic oracle: {len(FINDINGS)} finding(s) recorded", flush=True)


if __name__ == "__main__":
    main()
