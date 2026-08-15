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


def terminal_hanchan_rank_rewards(scores: list[int] | tuple[int, ...]) -> tuple[float, ...]:
    """Return zero-sum placement rewards, breaking score ties by physical seat."""
    if len(scores) != 4:
        raise ValueError("hanchan rank reward requires four final scores")
    ranking = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    by_rank = (16.0, 8.0, -8.0, -16.0)
    rewards = [0.0] * 4
    for rank, seat in enumerate(ranking):
        rewards[seat] = by_rank[rank]
    return tuple(rewards)
