import random
import unittest

from riichi_ppo_v1.bridge import NUM_PLAYERS
from riichi_ppo_v1.worker import active_decisions, sample_training_seats, should_record_decision


class _Observation:
    def __init__(self, active: bool) -> None:
        self.active = active

    def legal_actions(self) -> list[int]:
        return [0] if self.active else []


class WorkerSamplingTest(unittest.TestCase):
    def test_sample_training_seats_is_distinct_and_seeded(self) -> None:
        first = random.Random(91)
        second = random.Random(91)
        draws = [sample_training_seats(first) for _ in range(32)]

        self.assertEqual(draws, [sample_training_seats(second) for _ in range(32)])
        self.assertTrue(all(len(seats) == 2 and seats[0] < seats[1] for seats in draws))
        self.assertTrue(all(set(seats).issubset(range(NUM_PLAYERS)) for seats in draws))

    def test_all_active_seats_are_model_decisions_but_only_samples_are_recorded(self) -> None:
        decisions = active_decisions([{seat: _Observation(True) for seat in range(NUM_PLAYERS)}])

        self.assertEqual([decision.seat_id for decision in decisions], list(range(NUM_PLAYERS)))
        self.assertEqual(
            [decision.seat_id for decision in decisions if should_record_decision(decision, [(0, 2)])],
            [0, 2],
        )
