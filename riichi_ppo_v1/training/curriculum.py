"""Fixed five-stage curriculum for rollout rewards and opponent lineups."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    start_fraction: float
    end_fraction: float
    weights: tuple[float, float, float]
    lineup_recipe: str
    train_all_seats: bool = False


STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage("bootstrap", 0.00, 0.20, (0.85, 0.15, 0.00), "self_play", True),
    CurriculumStage("heuristic", 0.20, 0.50, (0.25, 0.75, 0.00), "heuristic_pair"),
    CurriculumStage("history_intro", 0.50, 0.68, (0.20, 0.60, 0.20), "history_intro"),
    CurriculumStage("history_focus", 0.68, 0.84, (0.15, 0.45, 0.40), "history_focus"),
    CurriculumStage("rank", 0.84, 1.01, (0.05, 0.35, 0.60), "rank"),
)


@dataclass(frozen=True, slots=True)
class CurriculumSnapshot:
    update: int
    progress: float
    stage: CurriculumStage

    @property
    def weights(self) -> tuple[float, float, float]:
        return self.stage.weights


class Curriculum:
    """Maps an update number to a deterministic stage without mastery gates."""

    def __init__(self, total_updates: int) -> None:
        if int(total_updates) <= 0:
            raise ValueError("total_updates must be positive")
        self.total_updates = int(total_updates)

    def snapshot(self, update: int) -> CurriculumSnapshot:
        progress = min(max(float(update) / float(self.total_updates), 0.0), 1.0)
        stage = next(item for item in STAGES if progress < item.end_fraction)
        return CurriculumSnapshot(int(update), progress, stage)


def checkpoint_cadence(total_updates: int) -> int:
    """Return the requested 2.5%-of-training checkpoint cadence."""
    return max(1, round(int(total_updates) * 0.025))
