"""MJAI response construction and chombo-avoidance checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

from .bridge import EventContext, PreparedDecision


@dataclass(frozen=True)
class SafeResponse:
    payload: dict[str, Any] | None
    source: str
    reason: str | None = None


def _consumed(value: dict[str, Any]) -> tuple[str, ...] | None:
    items = value.get("consumed")
    if items is None:
        return None
    if not isinstance(items, list) or not all(
        isinstance(item, str) for item in items
    ):
        return ()
    return tuple(sorted(items))


def possible_action_matches(
    response: dict[str, Any], possible: dict[str, Any]
) -> bool:
    if response.get("type") != possible.get("type"):
        return False
    action_type = response.get("type")
    if action_type not in {"hora", "reach", "none", "ryukyoku"}:
        if (
            "pai" in possible
            and response.get("pai") != possible.get("pai")
        ):
            return False
    response_consumed = _consumed(response)
    possible_consumed = _consumed(possible)
    if (
        possible_consumed is not None
        and response_consumed != possible_consumed
    ):
        return False
    if (
        action_type == "dahai"
        and "tsumogiri" in possible
        and bool(response.get("tsumogiri", False))
        != bool(possible.get("tsumogiri"))
    ):
        return False
    return True


def matches_possible_actions(
    response: dict[str, Any],
    possible_actions: Iterable[dict[str, Any]],
) -> bool:
    return any(
        possible_action_matches(response, possible)
        for possible in possible_actions
        if isinstance(possible, dict)
    )


def action_to_response(
    action: Any,
    observation: Any,
    context: EventContext,
    request_id: int,
) -> dict[str, Any]:
    value = json.loads(action.to_mjai())
    action_type = value.get("type")
    value["request_id"] = int(request_id)
    if action_type == "dahai":
        drawn = getattr(observation, "drawn_tile", None)
        tile = getattr(action, "tile", None)
        value["tsumogiri"] = (
            drawn is not None
            and tile is not None
            and int(drawn) == int(tile)
        )
    if action_type in {"chi", "pon", "daiminkan"}:
        if context.last_type == "dahai" and context.actor is not None:
            value["target"] = context.actor
    if action_type == "hora":
        seat = int(observation.player_id)
        if (
            context.last_type == "dahai"
            and context.actor is not None
            and context.actor != seat
        ):
            value["target"] = context.actor
            if context.pai is not None:
                value["pai"] = context.pai
    return value


def validate_response(
    response: dict[str, Any],
    observation: Any,
    possible_actions: list[dict[str, Any]],
) -> bool:
    if observation.select_action_from_mjai(response) is None:
        return False
    return matches_possible_actions(response, possible_actions)


def _fallback_priority(action: Any, observation: Any) -> tuple[int, int]:
    value = json.loads(action.to_mjai())
    action_type = value.get("type")
    if action_type == "hora":
        return (0, 0)
    if action_type == "none":
        return (1, 0)
    if action_type == "dahai":
        drawn = getattr(observation, "drawn_tile", None)
        tile = getattr(action, "tile", None)
        if drawn is not None and tile is not None and int(drawn) == int(tile):
            return (2, 0)
        return (3, int(tile) if tile is not None else 999)
    return (4, 0)


def choose_safe_response(
    prepared: PreparedDecision,
    primary_action: Any | None,
    possible_actions: list[dict[str, Any]],
    request_id: int,
    *,
    primary_error: str | None = None,
) -> SafeResponse:
    if primary_action is not None:
        response = action_to_response(
            primary_action,
            prepared.observation,
            prepared.event_context,
            request_id,
        )
        if validate_response(
            response, prepared.observation, possible_actions
        ):
            return SafeResponse(response, "model")

    for action in sorted(
        prepared.observation.legal_actions(),
        key=lambda item: _fallback_priority(item, prepared.observation),
    ):
        response = action_to_response(
            action,
            prepared.observation,
            prepared.event_context,
            request_id,
        )
        if validate_response(
            response, prepared.observation, possible_actions
        ):
            return SafeResponse(
                response,
                "fallback",
                primary_error or "model action failed safety validation",
            )
    return SafeResponse(
        None,
        "withheld",
        primary_error or "no action passed both legal-action checks",
    )
