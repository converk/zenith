"""Terminal kyoku reward scaling shared by rollout workers and tests."""

from __future__ import annotations


POINTS_PER_REWARD_UNIT = 1_000.0


def terminal_kyoku_reward(point_delta: int | float, clip_points: int | float) -> float:
    """Scale one seat's terminal point delta after symmetric clipping."""
    limit = float(clip_points)
    if limit <= 0.0:
        raise ValueError("kyoku_reward_clip_points must be positive")
    clipped = min(max(float(point_delta), -limit), limit)
    return clipped / POINTS_PER_REWARD_UNIT
