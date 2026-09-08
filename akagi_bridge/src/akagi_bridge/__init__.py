"""Akagi v3 云端推理适配层。

对外提供 Akagi `bot.api` 需要的 HTTP 端点,对内把 Akagi 的 mjai 事件流
重建成 RiichiEnv 的当前局面 Observation,复用 `riichi_lab_bot` 的
`OnlineStateBridge` + `PolicyEngine` 得到 V18 动作。
"""

from .engine import (
    AkagiCandidate,
    AkagiDecision,
    AkagiDecisionEngine,
    AkagiProtocolError,
    build_observation,
)

__all__ = [
    "AkagiCandidate",
    "AkagiDecision",
    "AkagiDecisionEngine",
    "AkagiProtocolError",
    "build_observation",
]
