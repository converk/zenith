"""V16 回放/桥接一致性:Actor 输入不含隐藏信息,Snapshot 事实与局面一致。"""

from __future__ import annotations

from riichienv import RiichiEnv
from riichienv.action import Action, ActionType

from riichi_ppo_v1.model.action_query import analyze_action_queries
from riichi_ppo_v1.model.snapshot import build_snapshot_facts


def _observation(seat: int, hands: list[list[int]], discards: list[list[int]]) -> object:
    env = RiichiEnv(seed=0, game_mode="4p-red-half")
    env.reset()
    env.hands = [list(row) for row in hands]
    env.discards = [list(row) for row in discards]
    env.melds = [[], [], [], []]
    env.dora_indicators = [36]
    env.oya = 0
    env.round_wind = 0
    env.honba = 0
    env.riichi_sticks = 0
    return env.get_observation(seat)


def test_actor_facts_ignore_opponent_hidden_hands() -> None:
    """篡改三家对手手牌不影响 Actor 侧 Snapshot 与 Query(无隐藏信息)。"""
    base_hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 68],
        [68],
        [56],
        [69],
    ]
    observation_a = _observation(0, base_hands, [[], [68], [56], [69]])
    # 对手手牌换成哨兵值,只有 Critic 特权才可见。
    tampered = [base_hands[0], [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    observation_b = _observation(0, tampered, [[], [68], [56], [69]])

    action = Action(ActionType.DISCARD, tile=68)
    facts_a = build_snapshot_facts(observation_a)
    facts_b = build_snapshot_facts(observation_b)
    assert facts_a == facts_b
    queries_a = analyze_action_queries(observation_a, action, 17)
    queries_b = analyze_action_queries(observation_b, action, 17)
    assert queries_a == queries_b


def test_snapshot_facts_match_observation() -> None:
    """Snapshot 事实与公开局面逐项一致(秩、分差、对手摘要)。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 68],
        [68],
        [56, 69],
        [69],
    ]
    observation = _observation(0, hands, [[], [68], [56, 69], [69]])
    facts = build_snapshot_facts(observation)
    assert facts.round_wind == 0
    assert facts.kyoku_index == 0
    assert facts.oya_relative == 0
    assert facts.honba == 0
    assert facts.riichi_sticks == 0
    assert facts.scores == (25000, 25000, 25000, 25000)
    assert facts.self_rank == 1
    assert facts.score_pressure == (0, 0, 0)
    assert facts.dora_indicator_types == (9,)  # 1p 指示 → 2p 宝牌
    # 下家:河 1 张、全摸切;对家:河 2 张。
    assert facts.opponent_summary[0].river_count == 1
    assert facts.opponent_summary[1].river_count == 2
    assert facts.opponent_summary[0].menzen == 1


def test_action_query_primary_tile_and_id_roundtrip() -> None:
    """Query 头字段与动作身份一致,answer 槽数固定为 10。"""
    hands = [
        [0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96, 68],
        [],
        [],
        [],
    ]
    observation = _observation(0, hands, [[], [], [], []])
    offense, defense = analyze_action_queries(observation, Action(ActionType.DISCARD, tile=68), 17)
    assert offense.action_id == 17 and defense.action_id == 17
    assert offense.action_type == "dahai" and defense.action_type == "dahai"
    assert offense.primary_tile == 17 and defense.primary_tile == 17
    assert len(offense.answers) == 10 and len(defense.answers) == 10
    assert offense.query_type == 1 and defense.query_type == 2

