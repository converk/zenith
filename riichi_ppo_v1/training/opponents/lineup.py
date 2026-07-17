"""Deterministic stage-aware seat ownership for mixed-policy self-play."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from ..curriculum import CurriculumSnapshot

CURRENT = "current"
EFFICIENCY = "heuristic_efficiency"
DEFENSE = "heuristic_defense"


@dataclass(frozen=True, slots=True)
class Lineup:
    policies: tuple[str, str, str, str]
    learner_mask: int

    def is_learner(self, seat: int) -> bool:
        return bool(self.learner_mask & (1 << int(seat)))


class LineupSampler:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def sample(self, snapshot: CurriculumSnapshot, history: Sequence[str], env_index: int, generation: int) -> Lineup:
        stage = snapshot.stage.lineup_recipe
        if stage == "self_play":
            return Lineup((CURRENT,) * 4, 0b1111)
        learner = self._learner_seats(env_index, generation)
        opponents = self._opponents(stage, history)
        policies = [CURRENT if seat in learner else opponents.pop(0) for seat in range(4)]
        return Lineup(tuple(policies), sum(1 << seat for seat in learner))

    @staticmethod
    def _learner_seats(env_index: int, generation: int) -> tuple[int, int]:
        offset = (int(env_index) + int(generation)) % 4
        return (offset, (offset + 2) % 4)

    def _opponents(self, stage: str, history: Sequence[str]) -> list[str]:
        available = list(dict.fromkeys(history))
        def one_history() -> str | None:
            return self.rng.choice(available) if available else None
        def two_history() -> list[str] | None:
            return self.rng.sample(available, 2) if len(available) >= 2 else None
        if stage == "heuristic_pair":
            return [EFFICIENCY, DEFENSE]
        draw = self.rng.random()
        if stage == "history_intro":
            if draw < .50:
                return [EFFICIENCY, DEFENSE]
            selected = one_history()
            return ([EFFICIENCY if draw < .75 else DEFENSE, selected] if selected else [EFFICIENCY, DEFENSE])
        if stage == "history_focus":
            pair = two_history()
            if draw < .50 and pair:
                return pair
            selected = one_history()
            return ([EFFICIENCY if draw < .75 else DEFENSE, selected] if selected else [EFFICIENCY, DEFENSE])
        if stage == "rank":
            pair = two_history()
            if draw < .80 and pair:
                return pair
            selected = one_history()
            return ([EFFICIENCY if draw < .90 else DEFENSE, selected] if selected else [EFFICIENCY, DEFENSE])
        raise ValueError(f"unknown lineup recipe {stage!r}")
