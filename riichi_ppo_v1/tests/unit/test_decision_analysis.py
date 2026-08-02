from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from riichienv import Action, ActionType, Conditions, HandEvaluator, Meld, MeldType

from riichi_ppo_v1.model.bridge import Decision
from riichi_ppo_v1.model.feature_schema import BREAK_MELD
from riichi_ppo_v1.training.opponents.heuristic import HeuristicPolicy
from riichi_ppo_v1.training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
    public_remaining,
)
from riichi_ppo_v1.training.rewards.decision import (
    Candidate,
    DecisionAnalysis,
    action_key,
)


def observation(
    hand: list[int],
    actions: list[object],
    *,
    melds: list[object] | None = None,
    river: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        player_id=0,
        hands=[hand, [], [], []],
        melds=[melds or [], [], [], []],
        discards=[river or [], [], [], []],
        dora_indicators=[],
        riichi_declared=[False] * 4,
        oya=0,
        round_wind=1,
        honba=0,
        riichi_sticks=0,
        drawn_tile=hand[-1],
        legal_actions=lambda: actions,
    )


def test_decision_253_east_is_strictly_worse_than_best_discards() -> None:
    # 1m 2m 4m 5m 66m 2p 1s 3s 4s 7s 9s EE
    hand = [0, 4, 12, 16, 20, 21, 40, 72, 80, 84, 96, 104, 108, 109]
    actions = [Action(ActionType.DISCARD, tile) for tile in hand]
    obs = observation(hand, actions)
    decision = Decision(0, 0, obs)
    row = DecisionAnalysisBatch.build(
        [decision], analyzer=EfficiencyAnalyzer(),
    ).for_decision(decision)
    by_type = {int(item.action.tile) // 4: item for item in row.candidates}
    assert by_type[27].structural_shanten == 3  # East
    assert by_type[5].structural_shanten == 2   # 6m
    assert by_type[10].structural_shanten == 2  # 2p
    assert by_type[18].structural_shanten == 2  # 1s
    assert by_type[27].rank > min(by_type[tile].rank for tile in (5, 10, 18))


def test_decision_198_marks_open_no_yaku_and_discard_furiten() -> None:
    # Standing tiles and three open melds from the V7 trace.
    hand = [44, 53, 104, 105, 56]  # 3p 5p 99s 6p
    melds = [
        Meld(MeldType.Chi, [20, 24, 28], True),   # 678m
        Meld(MeldType.Pon, [72, 73, 74], True),  # 111s
        Meld(MeldType.Chi, [40, 45, 48], True),  # 234p
    ]
    actions = [Action(ActionType.DISCARD, tile) for tile in hand]
    obs = observation(hand, actions, melds=melds, river=[60, 61])  # prior 7p
    decision = Decision(0, 0, obs)
    row = DecisionAnalysisBatch.build(
        [decision], analyzer=EfficiencyAnalyzer(),
    ).for_decision(decision)
    discard_3p = next(item for item in row.candidates if int(item.action.tile) == 44)
    assert discard_3p.structural_shanten == 0
    assert discard_3p.effective_shanten == 1
    assert discard_3p.open_no_yaku
    assert discard_3p.furiten
    assert discard_3p.live_ron == 0
    assert discard_3p.live_tsumo == 0


def test_public_remaining_deduplicates_called_physical_tile() -> None:
    called = 0
    own_meld = Meld(MeldType.Pon, [called, 1, 2], True)
    obs = observation([40] * 4, [], melds=[own_meld], river=[called])
    remaining = public_remaining(obs)
    assert remaining[0] == 1


def test_red_physical_ids_and_exhausted_waits_share_one_tile_count() -> None:
    # Red 5m and three normal copies are the same tile type.  Repeating the
    # called physical id in a river must still leave the type exhausted, not -1.
    own_meld = Meld(MeldType.Pon, [16, 17, 18], True)
    obs = observation([40], [], melds=[own_meld], river=[16, 19])
    assert public_remaining(obs)[4] == 0


def test_closed_no_ron_yaku_keeps_riichi_and_tsumo_routes_separate() -> None:
    # 123m 456m 789p 234s 5s, plus an unrelated drawn East to discard.
    hand = [0, 4, 8, 12, 16, 20, 60, 64, 68, 76, 80, 84, 88, 108]
    east = Action(ActionType.DISCARD, 108)
    obs = observation(hand, [east])
    decision = Decision(0, 0, obs)
    candidate = DecisionAnalysisBatch.build(
        [decision], analyzer=EfficiencyAnalyzer(),
    ).for_decision(decision).candidates[0]
    assert candidate.structural_shanten == 0
    assert candidate.riichi_route
    assert not candidate.has_yaku
    assert candidate.can_tsumo
    assert candidate.has_legal_route
    assert candidate.live_ron == 0
    assert candidate.live_tsumo > 0
    assert candidate.ron_value_sum == 0
    assert candidate.tsumo_value_sum > 0
    assert candidate.riichi_ron_points > 0
    assert candidate.riichi_ron_value_sum > 0


def test_declared_riichi_does_not_turn_riichi_itself_into_damaten_yaku() -> None:
    # The same no-intrinsic-ron-yaku hand remains legal after declaring
    # riichi, but the riichi yaku must not populate the damaten-yaku field.
    hand = [0, 4, 8, 12, 16, 20, 60, 64, 68, 76, 80, 84, 88, 108]
    east = Action(ActionType.DISCARD, 108)
    obs = observation(hand, [east])
    obs.riichi_declared = [True, False, False, False]
    decision = Decision(0, 0, obs)
    candidate = DecisionAnalysisBatch.build(
        [decision], analyzer=EfficiencyAnalyzer(),
    ).for_decision(decision).candidates[0]
    assert not candidate.has_yaku
    assert not candidate.riichi_route
    assert candidate.has_legal_route
    assert candidate.live_ron > 0
    assert candidate.live_tsumo > 0


def test_close_one_shanten_discards_receive_second_order_ukeire_scores() -> None:
    # 123m 456m 78p 23s 55s EE: several equal-ukeire one-shanten cuts
    # differ in the quality of their next shapes.
    hand = [0, 4, 8, 12, 16, 20, 60, 64, 76, 80, 88, 89, 108, 109]
    actions = [Action(ActionType.DISCARD, tile) for tile in hand]
    obs = observation(hand, actions)
    decision = Decision(0, 0, obs)
    analyzer = EfficiencyAnalyzer()
    row = DecisionAnalysisBatch.build(
        [decision], analyzer=analyzer,
    ).for_decision(decision)
    policy = HeuristicPolicy(
        analyzer, PublicStateTracker(1), defensive=False,
    )

    bonuses = policy._second_order_bonuses(list(row.candidates), row)

    assert len(bonuses) >= 2
    assert min(bonuses.values()) > 0
    assert len(set(bonuses.values())) >= 2


def test_double_wind_open_pon_is_recognized_as_yaku() -> None:
    hand = [44, 53, 104, 105, 56]
    melds = [
        Meld(MeldType.Chi, [20, 24, 28], True),
        Meld(MeldType.Pon, [108, 109, 110], True),  # East: seat + round wind
        Meld(MeldType.Chi, [40, 45, 48], True),
    ]
    actions = [Action(ActionType.DISCARD, tile) for tile in hand]
    obs = observation(hand, actions, melds=melds)
    obs.round_wind = 0
    decision = Decision(0, 0, obs)
    discard_3p = next(
        item for item in DecisionAnalysisBatch.build(
            [decision], analyzer=EfficiencyAnalyzer(),
        ).for_decision(decision).candidates
        if int(item.action.tile) == 44
    )
    assert discard_3p.has_yaku
    assert not discard_3p.open_no_yaku
    assert discard_3p.effective_shanten == 0


def test_decision_225_complete_open_shape_has_no_legal_win_value() -> None:
    hand = [0, 1, 8, 9, 2]  # 111m 33m
    melds = [
        Meld(MeldType.Chi, [24, 28, 32], True),
        Meld(MeldType.Pon, [4, 5, 6], True),
        Meld(MeldType.Pon, [104, 105, 106], True),
    ]
    result = HandEvaluator(hand, melds).calc(
        2, [], Conditions(tsumo=True, player_wind=0, round_wind=1),
    )
    assert result.has_win_shape
    assert not result.is_win


class FakeAction:
    def __init__(self, raw: str, tile: int | None = None, consumed: tuple[int, ...] = ()) -> None:
        self.raw, self.tile, self.consume_tiles = raw, tile, list(consumed)

    def to_mjai(self) -> str:
        return self.raw


def test_call_action_key_ignores_native_object_identity_and_pass_can_regret() -> None:
    first = FakeAction('{"type":"pon","pai":"E","consumed":["E","E"]}', 108, (109, 110))
    decoded = FakeAction('{"consumed":["E","E"],"pai":"E","type":"pon"}', 111, (108, 109))
    passed = FakeAction('{"type":"none"}')
    assert action_key(first) == action_key(decoded)
    best = Candidate(first, 133, 0, 0, 4)
    pass_candidate = Candidate(passed, 0, 1, 1, 12)
    teacher = np.zeros(241, dtype=bool)
    teacher[133] = True
    row = DecisionAnalysis(
        object(), (first, passed), (best, pass_candidate), best.rank, teacher, "call",
    )
    assert row.candidate_for(decoded) is best
    assert row.selected_regrets(decoded) == (0.0, 0.0)
    discard, call = row.selected_regrets(passed)
    assert discard == 0.0
    assert -1.0 <= call < 0.0


def test_parallel_best_teacher_and_candidate_tokens_align_to_legal_mask() -> None:
    hand = [0, 4, 12, 16, 20, 21, 40, 72, 80, 84, 96, 104, 108, 109]
    actions = [Action(ActionType.DISCARD, tile) for tile in hand]
    obs = observation(hand, actions)
    decision = Decision(0, 0, obs)
    batch = DecisionAnalysisBatch.build([decision], analyzer=EfficiencyAnalyzer())
    row = batch.for_decision(decision)
    assert row.teacher_mask.sum() >= 2
    legal = np.zeros((1, 241), dtype=bool)
    for candidate in row.candidates:
        legal[0, candidate.action_id] = True
    factors, numeric = batch.candidate_tokens([decision], legal)
    assert factors[0].shape == (2 * int(legal.sum()), 10)
    assert numeric[0].shape == (2 * int(legal.sum()), 8)
    assert np.all(factors[0][0::2, 2] == np.flatnonzero(legal) + 1)
    assert np.array_equal(factors[0][0::2, 2], factors[0][1::2, 2])
    assert np.all(factors[0][0::2, 9] == 1)
    assert np.all(factors[0][1::2, 9] == 2)


def test_ankan_and_kakan_simulations_keep_all_four_kan_tiles() -> None:
    # 9m indicator makes 1m dora. Both kan variants must preserve four copies
    # in the simulated meld rather than dropping the added/closed fourth tile.
    ankan_hand = [0, 1, 2, 3, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76]
    ankan = Action(ActionType.ANKAN, 0, [0, 1, 2, 3])
    ankan_obs = observation(ankan_hand, [ankan])
    ankan_obs.dora_indicators = [32]
    ankan_decision = Decision(0, 0, ankan_obs)
    ankan_row = DecisionAnalysisBatch.build(
        [ankan_decision], analyzer=EfficiencyAnalyzer(),
    ).for_decision(ankan_decision)
    assert ankan_row.candidates[0].preserve_dora == 4

    pon = Meld(MeldType.Pon, [0, 1, 2], True)
    kakan_hand = [3, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76]
    kakan = Action(ActionType.KAKAN, 3, [0, 1, 2, 3])
    kakan_obs = observation(kakan_hand, [kakan], melds=[pon])
    kakan_obs.dora_indicators = [32]
    kakan_decision = Decision(0, 0, kakan_obs)
    kakan_row = DecisionAnalysisBatch.build(
        [kakan_decision], analyzer=EfficiencyAnalyzer(),
    ).for_decision(kakan_decision)
    assert kakan_row.candidates[0].preserve_dora == 4


def test_equal_dora_indicators_multiply_preserved_candidate_dora() -> None:
    hand = [0, 1, 2, 3, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76]
    ankan = Action(ActionType.ANKAN, 0, [0, 1, 2, 3])
    obs = observation(hand, [ankan])
    obs.dora_indicators = [32, 32]  # both 9m indicators select 1m
    decision = Decision(0, 0, obs)
    batch = DecisionAnalysisBatch.build([decision], analyzer=EfficiencyAnalyzer())
    candidate = batch.for_decision(decision).candidates[0]
    assert candidate.preserve_dora == 8
    legal = np.zeros((1, 241), dtype=bool)
    legal[0, candidate.action_id] = True
    factors, _numeric = batch.candidate_tokens([decision], legal)
    assert int(factors[0][0, 3]) == 7  # clipped at six, then shifted by one


def test_native_ankan_with_only_representative_tile_is_normalized() -> None:
    hand = [108, 109, 110, 111, 0, 4, 8, 12, 16, 20, 36, 40, 44, 72]
    ankan = Action(ActionType.ANKAN, 108)
    obs = observation(hand, [ankan])
    decision = Decision(0, 0, obs)
    batch = DecisionAnalysisBatch.build([decision], analyzer=EfficiencyAnalyzer())
    candidate = batch.for_decision(decision).candidates[0]

    assert candidate.action_id == 198
    assert candidate.open_meld_count == 1
    assert sum(candidate.concealed_counts) + 3 * candidate.open_meld_count == 13


def test_actor_analysis_ignores_opponent_concealed_hands() -> None:
    hand = [0, 4, 12, 16, 20, 21, 40, 72, 80, 84, 96, 104, 108, 109]
    actions = [Action(ActionType.DISCARD, tile) for tile in hand]
    first = observation(hand, actions)
    second = observation(hand, actions)
    second.hands[1:] = [[135] * 13, [67] * 13, [31] * 13]
    rows = []
    for obs in (first, second):
        decision = Decision(0, 0, obs)
        analysis = DecisionAnalysisBatch.build([decision], analyzer=EfficiencyAnalyzer())
        rows.append([(item.action_id, item.rank) for item in analysis.for_decision(decision).candidates])
    assert rows[0] == rows[1]


def test_pass_candidate_explicitly_records_a_foregone_win() -> None:
    hand = [0, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76, 80, 84]
    passed = FakeAction('{"type":"none"}')
    hora = FakeAction('{"type":"hora"}')
    obs = observation(hand, [passed, hora])
    obs.drawn_tile = None
    decision = Decision(0, 0, obs)
    batch = DecisionAnalysisBatch.build([decision], analyzer=EfficiencyAnalyzer())
    row = batch.for_decision(decision)
    assert len(row.candidates) == 1
    assert row.candidates[0].action_id == 0
    assert row.candidates[0].foregone_win
    legal = np.zeros((1, 241), dtype=bool)
    legal[0, [0, 239]] = True
    factors, numeric = batch.candidate_tokens([decision], legal)
    assert int(factors[0][1, 4]) == 1  # pass has an applicable risk summary
    assert float(numeric[0][1, 7]) == 1.0
    assert not factors[0][2, 3:9].any()  # terminal hora offense is explicit N/A
    assert not factors[0][3, 3:9].any()  # terminal hora defense is explicit N/A


def test_discarding_one_of_four_copies_does_not_claim_to_break_a_triplet() -> None:
    hand = [0, 1, 2, 3, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76]
    discard = Action(ActionType.DISCARD, 0)
    obs = observation(hand, [discard])
    decision = Decision(0, 0, obs)
    candidate = DecisionAnalysisBatch.build(
        [decision], analyzer=EfficiencyAnalyzer(),
    ).for_decision(decision).candidates[0]
    assert candidate.structure_known_mask & BREAK_MELD
    assert not candidate.structure_break_mask & BREAK_MELD


def test_temporary_and_riichi_missed_win_states_are_furiten() -> None:
    hand = [0, 4, 8, 12, 16, 20, 60, 64, 68, 76, 80, 84, 88, 108]
    discard = Action(ActionType.DISCARD, 108)
    for field in ("missed_agari_doujun", "missed_agari_riichi"):
        obs = observation(hand, [discard])
        setattr(obs, field, True)
        decision = Decision(0, 0, obs)
        candidate = DecisionAnalysisBatch.build(
            [decision], analyzer=EfficiencyAnalyzer(),
        ).for_decision(decision).candidates[0]
        assert candidate.furiten
        assert candidate.live_ron == 0


def test_schema_11_keeps_kan_candidate_features_frozen_as_na() -> None:
    hand = [0, 1, 2, 3, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76]
    ankan = Action(ActionType.ANKAN, 0, [0, 1, 2, 3])
    obs = observation(hand, [ankan])
    decision = Decision(0, 0, obs)
    row = DecisionAnalysisBatch.build_legacy_v11(
        [decision], analyzer=EfficiencyAnalyzer(),
    ).for_decision(decision)
    assert row.candidates == ()


def test_red_and_normal_five_discards_keep_distinct_protocol_candidates() -> None:
    hand = [16, 17, 0, 4, 8, 12, 20, 36, 40, 44, 72, 76, 80, 108]
    actions = [Action(ActionType.DISCARD, 16), Action(ActionType.DISCARD, 17)]
    obs = observation(hand, actions)
    obs.drawn_tile = 108
    decision = Decision(0, 0, obs)
    row = DecisionAnalysisBatch.build(
        [decision], analyzer=EfficiencyAnalyzer(),
    ).for_decision(decision)
    assert len(row.candidates) == 2
    assert len({candidate.action_id for candidate in row.candidates}) == 2
