"""Akagi v3 云端推理协议适配:mjai 事件流 → V18 决策。

Akagi 的 `bot.api`(云端推理)在每次决策时 POST 一个**从 `start_game` 起的完整
mjai 事件流**(对手手牌以 `"?"` 隐藏),期望拿回一个 mjai 动作。本模块把这条
流重建成 RiichiEnv 的当前局面 `Observation`,再复用 `riichi_lab_bot` 的
`OnlineStateBridge` + `PolicyEngine` 得到 V18 动作并解码回 mjai JSON。

与 `riichi_lab_bot.client` 的 RiichiLab 路径的关系:两者共用同一套 bridge 与
策略,差别只在 Observation 的来源——RiichiLab 由服务端直接下发,
Akagi 侧必须自己从事件流重建。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 重依赖只在真正决策时导入,HTTP 层可脱离 torch/原生扩展导入
    from riichi_lab_bot.bridge import OnlineStateBridge
    from riichi_lab_bot.policy import PolicyEngine

# 单次请求的事件条数上限。Akagi 按局发送(每局几百条),上限只是防止畸形请求
# 把服务端拖死。
MAX_EVENTS = 4096

# Akagi 的 `react` 默认超时是 3s、上限 10s(`bot.api.react_timeout_ms`),
# 这里给出一个自检用的软目标:超过它就在日志里提示,不中断服务。
SLOW_DECISION_MS = 1500.0


class AkagiProtocolError(RuntimeError):
    """请求不符合 Akagi 云端推理协议(由 HTTP 层映射为 422)。"""


@dataclass(frozen=True)
class AkagiCandidate:
    """一条粗粒度候选动作,用于 Akagi HUD 的 Bot Show 卡片。"""

    action: str
    prob: float


@dataclass(frozen=True)
class AkagiDecision:
    """一次 `/v3/react` 的完整结果。"""

    reaction: dict[str, Any] | None
    candidates: tuple[AkagiCandidate, ...]
    action_id: int
    elapsed_ms: float


def _coarse_label(mjai: dict[str, Any]) -> str:
    """把 mjai 动作压成 Akagi HUD 使用的粗粒度标签。"""
    kind = str(mjai.get("type", "none"))
    if kind == "dahai":
        pai = mjai.get("pai")
        return f"dahai:{pai}" if isinstance(pai, str) else "dahai"
    return kind


def _parse_event(event: Any, index: int) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    if isinstance(event, str):
        try:
            parsed = json.loads(event)
        except json.JSONDecodeError as exc:
            raise AkagiProtocolError(
                f"events[{index}] is not valid JSON: {exc}"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise AkagiProtocolError(f"events[{index}] must be a JSON object")


def _resolve_rule(game_rule: Any, rule: str) -> Any:
    if rule == "tenhou":
        return game_rule.default_tenhou()
    if rule == "mjsoul":
        return game_rule.default_mjsoul()
    raise AkagiProtocolError(f"unknown rule: {rule!r}")


def build_observation(
    events: list[Any],
    player_id: int,
    *,
    game_mode: str = "4p-red-half",
    rule: str = "tenhou",
) -> Any:
    """把 Akagi 的 mjai 事件流重建成 `player_id` 的当前局面 Observation。

    使用 `RiichiEnv.apply_event` 逐条推进状态,最后一次 `get_observations`
    取到待决策的 Observation——这样 `new_events()` 返回的是本局完整事件
    (而不是增量),正是 `OnlineStateBridge.prepare` 需要的形态。
    """
    try:
        from riichienv import GameRule, RiichiEnv
    except ImportError as exc:  # pragma: no cover - 仅在环境缺失时触发
        raise AkagiProtocolError(
            "the local riichienv extension is not installed"
        ) from exc

    if not events:
        raise AkagiProtocolError("events must not be empty")
    if len(events) > MAX_EVENTS:
        raise AkagiProtocolError(
            f"too many events: {len(events)} > {MAX_EVENTS}"
        )
    if not 0 <= int(player_id) < 4:
        raise AkagiProtocolError(f"player_id must be in [0, 3], got {player_id}")

    env = RiichiEnv(game_mode=game_mode, rule=_resolve_rule(GameRule, rule))
    for index, raw in enumerate(events):
        event = _parse_event(raw, index)
        try:
            env.apply_event(event)
        except Exception as exc:  # noqa: BLE001 -- 协议错误统一转成 422 由 Akagi 兜底
            raise AkagiProtocolError(
                f"events[{index}] rejected by RiichiEnv: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    observations = env.get_observations([int(player_id)])
    observation = observations[int(player_id)]
    if not list(observation.legal_actions()):
        raise AkagiProtocolError(
            "no pending decision for seat "
            f"{player_id}: the event stream ends outside a decision window"
        )
    return observation


class AkagiDecisionEngine:
    """把一条 mjai 事件流变成 Akagi 需要的 mjai 动作。

    每次 `react` 都新建 bridge(有状态),因此同一个引擎可以无状态地服务
    任意多局请求;模型权重与设备只在构造时加载一次。
    """

    def __init__(
        self,
        policy: PolicyEngine,
        *,
        topk: int = 5,
        rule: str = "tenhou",
        game_mode: str = "4p-red-half",
    ) -> None:
        if topk < 0:
            raise ValueError("topk must be >= 0")
        self.policy = policy
        self.topk = int(topk)
        self.rule = rule
        self.game_mode = game_mode

    def react(self, events: list[Any], player_id: int) -> AkagiDecision:
        from riichi_lab_bot.bridge import OnlineStateBridge

        started = time.perf_counter()
        observation = build_observation(
            events,
            player_id,
            game_mode=self.game_mode,
            rule=self.rule,
        )
        bridge = OnlineStateBridge(int(player_id))
        prepared = bridge.prepare(observation)
        inference = self.policy.infer(prepared, topk=self.topk)
        reaction = self._decode(bridge, prepared, inference.action_id)
        candidates: list[AkagiCandidate] = []
        for candidate in inference.candidates:
            payload = self._decode_safely(bridge, prepared, candidate.action_id)
            if payload is None:
                continue
            candidates.append(
                AkagiCandidate(_coarse_label(payload), float(candidate.prob))
            )
        return AkagiDecision(
            reaction=reaction,
            candidates=tuple(candidates),
            action_id=inference.action_id,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _decode(
        bridge: OnlineStateBridge, prepared: Any, action_id: int
    ) -> dict[str, Any]:
        mjai = bridge.decode(prepared, action_id)
        payload = json.loads(mjai.to_mjai())
        if not isinstance(payload, dict):  # pragma: no cover - RiichiEnv 保证是对象
            raise AkagiProtocolError(f"decoded action is not an object: {payload!r}")
        return payload

    @classmethod
    def _decode_safely(
        cls, bridge: OnlineStateBridge, prepared: Any, action_id: int
    ) -> dict[str, Any] | None:
        """解码 top-k 候选;单个候选失败不影响主决策。"""
        try:
            return cls._decode(bridge, prepared, action_id)
        except Exception:  # noqa: BLE001 -- 候选仅用于 HUD,失败即跳过
            return None
