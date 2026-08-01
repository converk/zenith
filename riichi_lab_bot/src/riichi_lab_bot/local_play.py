"""Local RiichiEnv simulation through the online-shaped bridge."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import time
from typing import Any

from .bridge import OnlineStateBridge
from .policy import PolicyEngine
from .safety import choose_safe_response
from .telemetry import SessionMetrics


@dataclass(frozen=True)
class LocalGameResult:
    game: int
    seed: int
    elapsed_seconds: float
    steps: int
    scores: tuple[int, ...]
    ranks: tuple[int, ...]
    metrics: dict[str, Any]


def observation_with_events(
    observation: Any, events: list[str]
) -> Any:
    """Round-trip an observation with server-like per-seat event deltas."""
    from riichienv import Observation

    raw = base64.b64decode(observation.serialize_to_base64())
    value = json.loads(raw)
    value["events"] = list(events)
    encoded = base64.b64encode(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return Observation.deserialize_from_base64(encoded)


def play_local_game(
    policy: PolicyEngine,
    *,
    game: int,
    seed: int,
    max_steps: int = 4000,
    recorder: Any = None,
) -> LocalGameResult:
    from riichienv import RiichiEnv

    env = RiichiEnv(game_mode="4p-red-half", seed=seed)
    observations = env.reset()
    bridges = {seat: OnlineStateBridge(seat) for seat in range(4)}
    pending_events: dict[int, list[str]] = {
        seat: [] for seat in range(4)
    }
    metrics = SessionMetrics()
    request_id = 0
    global_step = 0
    started = time.perf_counter()
    for step in range(max_steps):
        for seat, observation in observations.items():
            pending_events[int(seat)].extend(observation.new_events())
        actions: dict[int, Any] = {}
        for seat, original in observations.items():
            if not original.legal_actions():
                continue
            online_observation = observation_with_events(
                original, pending_events[int(seat)]
            )
            pending_events[int(seat)].clear()
            prepared = bridges[int(seat)].prepare(online_observation)
            possible = [
                json.loads(value) for value in prepared.legal_jsons
            ]
            inference = policy.infer(prepared)
            metrics.requests += 1
            metrics.inference_ms.append(inference.elapsed_ms)
            primary = bridges[int(seat)].decode(
                prepared, inference.action_id
            )
            safe = choose_safe_response(
                prepared, primary, possible, request_id
            )
            request_id += 1
            if safe.payload is None:
                metrics.withheld_actions += 1
                if recorder is not None:
                    recorder.emit(
                        "action_withheld",
                        game=game,
                        seed=seed,
                        step=global_step,
                        seat=int(seat),
                        reason=safe.reason,
                        inference_ms=inference.elapsed_ms,
                        action_id=inference.action_id,
                    )
                raise RuntimeError(
                    f"local action was withheld: {safe.reason}"
                )
            selected = original.select_action_from_mjai(safe.payload)
            if selected is None:
                raise RuntimeError(
                    f"local action failed original observation: {safe.payload}"
                )
            actions[int(seat)] = selected
            metrics.responses += 1
            if safe.source == "model":
                metrics.model_actions += 1
            else:
                metrics.fallback_actions += 1
            if recorder is not None:
                recorder.emit(
                    "action_sent",
                    game=game,
                    seed=seed,
                    step=global_step,
                    seat=int(seat),
                    request_id=request_id - 1,
                    action_id=inference.action_id,
                    source=safe.source,
                    payload=safe.payload,
                    inference_ms=inference.elapsed_ms,
                    legal_action_count=len(possible),
                )
            global_step += 1
        if not actions:
            if env.done():
                break
            raise RuntimeError("local environment stalled without actions")
        observations = env.step(actions)
        if env.done():
            break
    else:
        raise RuntimeError(
            f"local game exceeded max_steps={max_steps}"
        )
    elapsed = time.perf_counter() - started
    return LocalGameResult(
        game=game,
        seed=seed,
        elapsed_seconds=elapsed,
        steps=step + 1,
        scores=tuple(int(value) for value in env.scores()),
        ranks=tuple(int(value) for value in env.ranks()),
        metrics=metrics.summary(),
    )
