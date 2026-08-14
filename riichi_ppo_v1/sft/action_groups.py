"""Canonical action-id grouping shared by SFT training and evaluation."""

from __future__ import annotations


def action_group(action_id: int) -> str:
    """Return the semantic action group for a 241-way action id."""
    boundaries = (
        (1, "pass"),
        (75, "discard"),
        (76, "reach"),
        (133, "chi"),
        (170, "pon"),
        (239, "kan"),
        (240, "hora"),
        (241, "ryukyoku"),
    )
    return next(name for end, name in boundaries if int(action_id) < end)
