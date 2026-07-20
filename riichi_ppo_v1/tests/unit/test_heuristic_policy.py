from dataclasses import dataclass

import numpy as np

from riichi_ppo_v1.training.opponents.heuristic import HeuristicPolicy
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
