"""Opponent-mix rollout helpers: history pool, fractions, labels, recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from riichi_ppo_v1.training.worker import (
    build_rollout_lineups,
    eligible_history_updates,
    history_namespace,
    normalize_opponent_fractions,
    parse_checkpoint_updates,
    resolve_opponent_fractions,
    rollout_lineup,
    should_record_transition,
)


def test_empty_history_pool_renormalizes_to_70_30_current_sft() -> None:
    current, sft, historical, random = resolve_opponent_fractions(
        current_frac=0.7,
        sft_frac=0.2,
        historical_frac=0.1,
        random_frac=0.0,
        history_pool=[],
    )
    assert np.isclose(current, 0.7)
    assert np.isclose(sft, 0.3)
    assert historical == 0.0
    assert random == 0.0


def test_nonempty_history_pool_keeps_configured_fractions() -> None:
    current, sft, historical, random = resolve_opponent_fractions(
        current_frac=0.7,
        sft_frac=0.2,
        historical_frac=0.1,
        random_frac=0.0,
        history_pool=[30],
    )
    assert np.isclose(current, 0.7)
    assert np.isclose(sft, 0.2)
    assert np.isclose(historical, 0.1)
    assert random == 0.0


def test_fractions_always_normalize_to_one() -> None:
    values = normalize_opponent_fractions(0.8, 0.1, 0.05, 0.05)
    assert np.isclose(sum(values), 1.0)
    with pytest.raises(ValueError, match="positive"):
        normalize_opponent_fractions(0.0, 0.0, 0.0, 0.0)


def test_history_lag_and_min_update_filter() -> None:
    updates = [30, 60, 90, 120]
    assert eligible_history_updates(
        updates,
        current_update=120,
        min_update=60,
        lag_updates=60,
    ) == [60]
    assert eligible_history_updates(
        updates,
        current_update=180,
        min_update=60,
        lag_updates=60,
    ) == [60, 90, 120]
    assert eligible_history_updates(
        updates,
        current_update=59,
        min_update=60,
        lag_updates=60,
    ) == []


def test_history_namespace_label_format() -> None:
    assert history_namespace(60) == "history:u060"
    assert history_namespace(780) == "history:u780"


def test_checkpoint_pool_scans_complete_five_digit_files_only(tmp_path: Path) -> None:
    for name in (
        "checkpoint_00030.pt",
        "checkpoint_00180.pt",
        "checkpoint_00060.pt.tmp",
        "latest.pt",
        "junk.txt",
    ):
        (tmp_path / name).touch()
    assert parse_checkpoint_updates(tmp_path) == [30, 180]
    assert parse_checkpoint_updates(tmp_path / "missing") == []


def test_lineup_rolls_whole_tables_with_one_opponent_seat() -> None:
    rng = np.random.default_rng(12345)
    lineups = build_rollout_lineups(
        num_envs=200,
        rng=rng,
        current_frac=0.7,
        sft_frac=0.2,
        historical_frac=0.1,
        random_frac=0.0,
        history_pool=[60],
    )
    assert len(lineups) == 200
    for lineup in lineups:
        assert set(lineup) <= {"current", "sft", "history:u060"}
        if set(lineup) == {"current"}:
            continue
        assert sum(policy == "current" for policy in lineup) == 3
        assert sum(policy != "current" for policy in lineup) == 1


def test_history_lineup_uses_uniform_pool_member() -> None:
    rng = np.random.default_rng(7)
    picked = {
        rollout_lineup(
            rng,
            current_frac=0.0,
            sft_frac=0.0,
            historical_frac=1.0,
            random_frac=0.0,
            history_pool=[60, 90, 120],
        )
        for _ in range(50)
    }
    opponents = {
        next(policy for policy in lineup if policy != "current")
        for lineup in picked
    }
    assert opponents == {"history:u060", "history:u090", "history:u120"}


def test_only_current_seats_record_transitions() -> None:
    assert should_record_transition("current") is True
    for policy in ("sft", "random", "history:u060"):
        assert should_record_transition(policy) is False
