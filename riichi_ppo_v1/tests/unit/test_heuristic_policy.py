from dataclasses import dataclass

import numpy as np

from riichi_ppo_v1.training.opponents.heuristic import HeuristicPolicy
from riichi_ppo_v1.training.rewards.decision import (
    Candidate,
    DecisionAnalysis,
    DecisionAnalysisBatch,
)
from riichi_ppo_v1.training.rewards.efficiency import (
    DiscardAnalysisBatch,
    DiscardCandidate,
    DiscardDecisionAnalysis,
    HandAnalysis,
)
from riichi_ppo_v1.training.rewards.public_state import PublicStateTracker


@dataclass(frozen=True)
class _Action:
    action_type: str
    tile: int | None = None


@dataclass
class _Observation:
    actions: list[_Action]
    dora_indicators: list[int] | None = None
    hands: tuple[tuple[int, ...], ...] = ((), (), (), ())
    melds: tuple[tuple[object, ...], ...] = ((), (), (), ())
    player_id: int = 0
    oya: int = 1
    round_wind: int = 0
    riichi_declared: tuple[bool, ...] = (False, False, False, False)

    def legal_actions(self) -> list[_Action]:
        return self.actions


@dataclass
class _Decision:
    observation: _Observation
    env_index: int = 0
    seat_id: int = 0


def _row(decision: _Decision, candidates: list[DiscardCandidate]) -> DiscardAnalysisBatch:
    return DiscardAnalysisBatch({
        id(decision): DiscardDecisionAnalysis(
            decision=decision,
            actions=tuple(decision.observation.actions),
            counts=np.zeros(34, dtype=np.uint8),
            opened=0,
            candidates=tuple(candidates),
            best_shanten=min((candidate.shanten for candidate in candidates), default=None),
            best_ukeire=max((candidate.ukeire for candidate in candidates), default=0),
        ),
    })


def _candidate(action: _Action, *, shanten: int = 0, ukeire: int = 4) -> DiscardCandidate:
    return DiscardCandidate(action, HandAnalysis(shanten, 0), shanten, ukeire)


def test_heuristic_always_accepts_a_legal_win() -> None:
    discard = _Action("discard", 0)
    tsumo = _Action("tsumo")
    decision = _Decision(_Observation([discard, tsumo]))
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)

    assert policy.select_batch([decision], _row(decision, [_candidate(discard)])) == [tsumo]


def test_efficiency_heuristic_explicitly_declares_riichi() -> None:
    discard = _Action("discard", 0)
    riichi = _Action("riichi")
    decision = _Decision(_Observation([discard, riichi]))
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)

    assert policy.select_batch([decision], _row(decision, [_candidate(discard)])) == [riichi]


def test_defensive_heuristic_declines_unsafe_riichi_under_threat() -> None:
    unsafe_tenpai = _Action("discard", 0)
    safe_fallback = _Action("discard", 108)
    riichi = _Action("riichi")
    decision = _Decision(_Observation([unsafe_tenpai, safe_fallback, riichi]))
    public = PublicStateTracker(1)
    public.riichi[0, 1] = True
    public.discard_masks[0, 1] = 1 << 27
    policy = HeuristicPolicy(object(), public, defensive=True)

    result = policy.select_batch(
        [decision],
        _row(decision, [_candidate(unsafe_tenpai, shanten=0), _candidate(safe_fallback, shanten=1)]),
    )

    assert result == [safe_fallback]


def test_defensive_heuristic_riichis_with_a_safe_tenpai_discard() -> None:
    safe_tenpai = _Action("discard", 108)
    riichi = _Action("riichi")
    decision = _Decision(_Observation([safe_tenpai, riichi]))
    public = PublicStateTracker(1)
    public.riichi[0, 1] = True
    public.discard_masks[0, 1] = 1 << 27
    policy = HeuristicPolicy(object(), public, defensive=True)

    assert policy.select_batch([decision], _row(decision, [_candidate(safe_tenpai)])) == [riichi]


def test_efficiency_preserves_dora_and_red_five_on_equal_shape() -> None:
    red_dora = _Action("discard", 16)
    plain = _Action("discard", 20)
    decision = _Decision(_Observation([red_dora, plain], dora_indicators=[12]))  # 4m indicator => 5m dora
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)

    assert policy.select_batch([decision], _row(decision, [_candidate(red_dora), _candidate(plain)])) == [plain]


def _rule_row(
    decision: _Decision, candidates: list[Candidate],
) -> DecisionAnalysisBatch:
    return DecisionAnalysisBatch({
        id(decision): DecisionAnalysis(
            decision=decision,
            actions=tuple(decision.observation.actions),
            candidates=tuple(candidates),
            best_rank=min(candidate.rank for candidate in candidates),
            teacher_mask=np.zeros(241, dtype=np.bool_),
            selected_kind="call",
        ),
    })


def test_efficiency_rejects_open_no_yaku_tenpai_call() -> None:
    passed = _Action("pass")
    pon = _Action("pon", 108)
    decision = _Decision(_Observation([passed, pon]))
    pass_candidate = Candidate(passed, 0, 1, 1, 8)
    no_yaku_call = Candidate(
        pon, 133, 0, 1, 12, open_no_yaku=True, closed=False,
    )
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)

    assert policy.select_batch(
        [decision], _rule_row(decision, [pass_candidate, no_yaku_call]),
    ) == [passed]


def test_defensive_policy_passes_calls_under_riichi_threat() -> None:
    passed = _Action("pass")
    chi = _Action("chi", 0)
    decision = _Decision(_Observation([passed, chi]))
    public = PublicStateTracker(1)
    public.riichi[0, 1] = True
    policy = HeuristicPolicy(object(), public, defensive=True)

    assert policy.select_batch(
        [decision],
        _rule_row(
            decision,
            [Candidate(passed, 0, 2, 2, 5), Candidate(chi, 76, 1, 1, 12)],
        ),
    ) == [passed]


def test_tenpai_offence_uses_opportunity_weighted_value_before_wait_count() -> None:
    cheap = _Action("discard", 0)
    expensive = _Action("discard", 4)
    decision = _Decision(_Observation([cheap, expensive]))
    candidates = [
        Candidate(
            cheap, 1, 0, 0, 6,
            live_ron=6, ron_points=1_000, ron_value_sum=6_000,
        ),
        Candidate(
            expensive, 3, 0, 0, 4,
            live_ron=4, ron_points=8_000, ron_value_sum=32_000,
        ),
    ]
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)

    assert policy.select_batch(
        [decision], _rule_row(decision, candidates),
    ) == [expensive]


def test_closed_no_yaku_tenpai_uses_riichi_value_for_discard_choice() -> None:
    cheap = _Action("discard", 0)
    expensive = _Action("discard", 4)
    decision = _Decision(_Observation([cheap, expensive]))
    candidates = [
        Candidate(
            cheap, 1, 0, 0, 6,
            live_tsumo=6, riichi_route=True,
            riichi_ron_points=1_300, riichi_ron_value_sum=7_800,
        ),
        Candidate(
            expensive, 3, 0, 0, 4,
            live_tsumo=4, riichi_route=True,
            riichi_ron_points=8_000, riichi_ron_value_sum=32_000,
        ),
    ]
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)

    assert policy.select_batch(
        [decision], _rule_row(decision, candidates),
    ) == [expensive]


def test_non_tenpai_value_tiles_trade_against_only_a_small_ukeire_gap() -> None:
    keep_dora = _Action("discard", 0)
    gain_two = _Action("discard", 4)
    gain_three = _Action("discard", 8)
    decision = _Decision(_Observation([keep_dora, gain_two, gain_three]))
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)
    base = Candidate(
        keep_dora, 1, 1, 1, 8, preserve_dora=1,
    )

    assert policy.select_batch(
        [decision],
        _rule_row(
            decision,
            [base, Candidate(gain_two, 3, 1, 1, 10)],
        ),
    ) == [keep_dora]
    assert policy.select_batch(
        [decision],
        _rule_row(
            decision,
            [base, Candidate(gain_three, 5, 1, 1, 11)],
        ),
    ) == [gain_three]


def test_calls_require_a_near_tenpai_open_hand_or_value_honour() -> None:
    passed = _Action("pass")
    ordinary = _Action("pon", 0)
    value_honour = _Action("pon", 108)  # East, round wind for the fixture.
    decision = _Decision(_Observation([passed, ordinary, value_honour]))
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)
    fallback = Candidate(passed, 0, 3, 3, 12)

    assert policy.select_batch(
        [decision],
        _rule_row(
            decision,
            [fallback, Candidate(ordinary, 133, 2, 2, 16, closed=False)],
        ),
    ) == [passed]
    assert policy.select_batch(
        [decision],
        _rule_row(
            decision,
            [
                fallback,
                Candidate(value_honour, 133, 2, 2, 16, closed=False),
            ],
        ),
    ) == [value_honour]


def test_daiminkan_requires_a_scoring_tenpai() -> None:
    passed = _Action("pass")
    kan = _Action("daiminkan", 108)
    decision = _Decision(_Observation([passed, kan]))
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)
    fallback = Candidate(passed, 0, 1, 1, 8)

    assert policy.select_batch(
        [decision],
        _rule_row(
            decision,
            [fallback, Candidate(kan, 170, 0, 0, 4, has_yaku=False, closed=False)],
        ),
    ) == [passed]
    assert policy.select_batch(
        [decision],
        _rule_row(
            decision,
            [fallback, Candidate(kan, 170, 0, 0, 4, has_yaku=True, closed=False)],
        ),
    ) == [kan]


def test_high_value_and_late_narrow_tenpai_stay_dama() -> None:
    discard = _Action("discard", 0)
    riichi = _Action("riichi")
    decision = _Decision(_Observation([discard, riichi]))
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)
    high_value = Candidate(
        discard, 1, 0, 0, 6,
        live_ron=6, ron_points=7_700, has_yaku=True,
    )

    assert policy.select_batch(
        [decision], _rule_row(decision, [high_value]),
    ) == [discard]

    policy.public.discard_counts[0] = 56
    late_narrow = Candidate(
        discard, 1, 0, 0, 4,
        live_ron=4, ron_points=2_000, has_yaku=True,
    )
    assert policy.select_batch(
        [decision], _rule_row(decision, [late_narrow]),
    ) == [discard]


def test_kan_without_shape_analysis_is_declined_and_riichi_ankan_is_used() -> None:
    discard = _Action("discard", 0)
    ankan = _Action("ankan", 0)
    kakan = _Action("kakan", 0)
    policy = HeuristicPolicy(object(), PublicStateTracker(1), defensive=False)
    candidate = Candidate(discard, 1, 1, 1, 8)

    before = _Decision(_Observation([discard, ankan, kakan]))
    assert policy.select_batch(
        [before], _rule_row(before, [candidate]),
    ) == [discard]

    after = _Decision(_Observation(
        [discard, ankan, kakan],
        riichi_declared=(True, False, False, False),
    ))
    assert policy.select_batch(
        [after], _rule_row(after, [candidate]),
    ) == [ankan]


def test_pre_riichi_kan_requires_a_non_worsening_valuable_shape() -> None:
    class _KanAnalyzer:
        def __init__(self, shanten: int) -> None:
            self.shanten = shanten

        def analyze(self, hands, opened):
            assert len(hands) == len(opened) == 1
            return [HandAnalysis(self.shanten, 0)]

    ankan = _Action("ankan", 0)
    observation = _Observation(
        [ankan],
        hands=((0, 1, 2, 3, 4, 8, 12, 36, 40, 44, 72, 76, 80, 108), (), (), ()),
    )
    best = Candidate(
        _Action("discard", 108), 55, 1, 1, 10, preserve_dora=2,
    )

    assert HeuristicPolicy(
        _KanAnalyzer(1), PublicStateTracker(1), defensive=False,
    )._kan_is_worthwhile(ankan, best, observation)
    assert not HeuristicPolicy(
        _KanAnalyzer(2), PublicStateTracker(1), defensive=False,
    )._kan_is_worthwhile(ankan, best, observation)


def test_defensive_policy_pushes_good_tenpai_but_folds_weak_tenpai() -> None:
    unsafe = _Action("discard", 0)
    safe = _Action("discard", 108)
    decision = _Decision(_Observation([unsafe, safe]))
    public = PublicStateTracker(1)
    public.riichi[0, 1] = True
    public.discard_masks[0, 1] = 1 << 27
    policy = HeuristicPolicy(object(), public, defensive=True)
    fallback = Candidate(safe, 55, 1, 1, 5)

    strong = Candidate(unsafe, 1, 0, 0, 5, live_ron=5, ron_points=1_000)
    assert policy.select_batch(
        [decision], _rule_row(decision, [strong, fallback]),
    ) == [unsafe]

    weak = Candidate(unsafe, 1, 0, 0, 2, live_ron=2, ron_points=1_000)
    assert policy.select_batch(
        [decision], _rule_row(decision, [weak, fallback]),
    ) == [safe]


def test_danger_score_accounts_for_walls_and_multiple_riichi_players() -> None:
    discard = _Action("discard", 16)  # 5m
    decision = _Decision(_Observation([discard]))
    candidate = Candidate(discard, 9, 2, 2, 4)
    row = _rule_row(decision, [candidate]).for_decision(decision)

    one = PublicStateTracker(1)
    one.riichi[0, 1] = True
    policy = HeuristicPolicy(object(), one, defensive=True)
    open_score = policy._danger_score(candidate, row, 0, 0)

    one.visible[0, 3] = 3  # one-chance 4m
    one.visible[0, 5] = 3  # one-chance 6m
    one_chance_score = policy._danger_score(candidate, row, 0, 0)
    assert 0 < one_chance_score < open_score

    one.visible[0, 3] = 4  # 4m wall
    one.visible[0, 5] = 4  # 6m wall
    wall_score = policy._danger_score(candidate, row, 0, 0)
    assert 0 < wall_score < one_chance_score

    one.riichi[0, 2] = True
    assert policy._danger_score(candidate, row, 0, 0) == 2 * wall_score


def test_defensive_policy_raises_push_threshold_against_two_riichi_players() -> None:
    unsafe = _Action("discard", 0)
    safe = _Action("discard", 108)
    decision = _Decision(_Observation([unsafe, safe]))
    public = PublicStateTracker(1)
    public.riichi[0, 1:3] = True
    public.discard_masks[0, 1] = 1 << 27
    public.discard_masks[0, 2] = 1 << 27
    policy = HeuristicPolicy(object(), public, defensive=True)

    assert policy.select_batch(
        [decision],
        _rule_row(
            decision,
            [
                Candidate(
                    unsafe, 1, 0, 0, 5,
                    live_ron=5, ron_points=1_000,
                ),
                Candidate(safe, 55, 1, 1, 5),
            ],
        ),
    ) == [safe]
