from collections import Counter

import pytest

from riichi_ppo_v1.sft.head_to_head import balanced_team_a_seats, select_winner


def test_320_hanchan_schedule_is_exactly_seat_balanced() -> None:
    schedule = balanced_team_a_seats(320)
    assert len(schedule) == 320
    assert all(len(set(seats)) == 2 for seats in schedule)
    assert Counter(seat for seats in schedule for seat in seats) == {
        0: 160,
        1: 160,
        2: 160,
        3: 160,
    }
    assert all(
        set(schedule[index]).isdisjoint(schedule[index + 1])
        for index in range(0, len(schedule), 2)
    )


def test_schedule_requires_a_positive_even_count() -> None:
    with pytest.raises(ValueError):
        balanced_team_a_seats(0)
    with pytest.raises(ValueError):
        balanced_team_a_seats(319)


def test_winner_selection_uses_win_rate_before_tiebreakers() -> None:
    selected, reason = select_winner(
        model_a="a.pt",
        model_b="b.pt",
        wins_a=161,
        wins_b=159,
        ties=0,
        team_point_diff_sum=-100_000,
        first_places_a=100,
        first_places_b=220,
    )
    assert selected == "a.pt"
    assert reason == "team_win_rate"


def test_winner_selection_has_deterministic_tiebreakers() -> None:
    selected, reason = select_winner(
        model_a="a.pt",
        model_b="b.pt",
        wins_a=160,
        wins_b=160,
        ties=0,
        team_point_diff_sum=-1,
        first_places_a=200,
        first_places_b=120,
    )
    assert selected == "b.pt"
    assert reason == "team_point_diff"
