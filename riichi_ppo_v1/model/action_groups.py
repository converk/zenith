"""SFT 训练与评测共用的规范化动作分组。"""

from __future__ import annotations

from .schema import NUM_ACTIONS


def action_group(action_id: int) -> str:
    """返回 241 维动作空间中指定动作 id 的语义分组。"""
    boundaries = (
        (1, "pass"),
        (75, "discard"),
        (76, "reach"),
        (133, "chi"),
        (170, "pon"),
        (239, "kan"),
        (240, "hora"),
        (NUM_ACTIONS, "ryukyoku"),
    )
    return next(name for end, name in boundaries if int(action_id) < end)
