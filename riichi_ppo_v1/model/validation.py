"""Reusable validation helpers for the 4-player semantic-token integration boundary."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .bridge import BatchedStateBridge, Decision, action_jsons
from .schema import NUM_ACTIONS

EVENT_TYPES = frozenset({
    "start_game", "start_kyoku", "tsumo", "dahai", "chi", "pon", "daiminkan",
    "ankan", "kakan", "dora", "reach", "reach_accepted", "hora", "ryukyoku",
    "end_kyoku", "end_game",
})
MODEL_ACTION_TYPES = frozenset({
    "none", "dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan", "hora", "ryukyoku",
})
TILE34 = tuple(
    [f"{rank}{suit}" for suit in "mps" for rank in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C"]
)
TILE37 = TILE34 + ("5mr", "5pr", "5sr")


def canonical(value: str | dict[str, Any]) -> str:
    return json.dumps(json.loads(value) if isinstance(value, str) else value, sort_keys=True, separators=(",", ":"))


def _chi_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    locals_ = ((1, 2), (2, 3), (3, 4), (4, 5), (4, "5r"), (5, 6), ("5r", 6), (6, 7), (7, 8), (8, 9), (1, 3), (2, 4), (3, 5), (3, "5r"), (4, 6), (5, 7), ("5r", 7), (6, 8), (7, 9))
    for suit in "mps":
        for left, right in locals_:
            def name(value: int | str) -> str:
                return f"{value}{suit}" if isinstance(value, int) else f"5{suffix(suit)}r"
            pairs.append((name(left), name(right)))
    return pairs


def _chi_pai(pair: tuple[str, str]) -> str:
    """Choose one valid called tile for a chi-consumed pair.

    A consumed pair can be valid for either end of two adjacent sequences.  The
    fixed id is intentionally shared because the current discard determines
    which one is legal at a real decision; this helper only needs one valid
    template for static action-space coverage.
    """
    suit = pair[0][1]
    ranks = sorted(int(tile[0]) for tile in pair)
    for start in range(1, 8):
        sequence = [start, start + 1, start + 2]
        if all(rank in sequence for rank in ranks) and len(set(ranks)) == 2:
            missing = next(rank for rank in sequence if rank not in ranks)
            return f"{missing}{suit}"
    raise AssertionError(f"invalid chi pair {pair}")


def suffix(suit: str) -> str:
    """Map m/p/s to the red-five spelling's suit character."""
    return suit


def _pon_pairs() -> list[tuple[str, str]]:
    pairs = [(tile, tile) for tile in TILE34 if tile not in {"5m", "5p", "5s"}]
    pairs.extend([("5m", "5m"), ("5m", "5mr"), ("5p", "5p"), ("5p", "5pr"), ("5s", "5s"), ("5s", "5sr")])
    return pairs


def all_action_templates() -> list[dict[str, Any]]:
    """One MJAI template for every fixed action id, in id order."""
    templates: list[dict[str, Any]] = [{"type": "none"}]
    templates.extend({"type": "dahai", "pai": tile, "tsumogiri": bool(mode)} for tile in TILE37 for mode in range(2))
    templates.append({"type": "reach"})
    templates.extend({"type": "chi", "pai": _chi_pai(pair), "consumed": list(pair)} for pair in _chi_pairs())
    templates.extend({"type": "pon", "pai": pair[0].replace("r", ""), "consumed": list(pair)} for pair in _pon_pairs())
    templates.append({"type": "daiminkan", "pai": "E", "consumed": ["E", "E", "E"]})
    templates.extend({"type": "ankan", "consumed": [tile] * 4} for tile in TILE34)
    templates.extend({"type": "kakan", "pai": tile, "consumed": [tile] * 3} for tile in TILE34)
    templates.extend(({"type": "hora"}, {"type": "ryukyoku"}))
    if len(templates) != NUM_ACTIONS:
        raise AssertionError(f"generated {len(templates)} templates instead of {NUM_ACTIONS}")
    return templates


def assert_full_action_space(manager: Any) -> None:
    templates = all_action_templates()
    mask = np.asarray(
        manager.prepare_decisions(
            [0], [[json.dumps(template, separators=(",", ":")) for template in templates]]
        ),
        dtype=bool,
    )
    if mask.shape != (1, NUM_ACTIONS) or not mask[0].all():
        raise AssertionError(f"expected all {NUM_ACTIONS} slots, got shape={mask.shape} count={mask.sum(axis=1)}")
    for action_id, expected in enumerate(templates):
        returned = manager.decode_actions([0], [action_id])[0]
        if canonical(returned) != canonical(expected):
            raise AssertionError(f"action_id={action_id} expected={expected} returned={returned}")


def _action_signature(observation: Any, action: Any) -> tuple[int, str, str | None, tuple[str, ...], bool | None]:
    """Return only the Action fields that can affect the environment transition.

    RiichiEnv intentionally does not preserve physical identity for otherwise
    identical non-red tiles.  MJAI does preserve red tiles, called/consumed
    tiles and the discard mode, which are the distinctions the 241-space must
    carry.  ``actor`` is deliberately excluded: the current observation is
    already the action's actor and ``select_action_from_mjai`` treats it as an
    optional compatibility field.
    """
    value = json.loads(action.to_mjai())
    action_type = str(value["type"])
    tsumogiri: bool | None = None
    if action_type == "dahai":
        drawn = getattr(observation, "drawn_tile", None)
        tile = getattr(action, "tile", None)
        tsumogiri = drawn is not None and tile is not None and int(tile) == int(drawn)
    return (
        int(action.action_type),
        action_type,
        value.get("pai"),
        tuple(sorted(value.get("consumed", []))),
        tsumogiri,
    )


def assert_observation_roundtrip(bridge: BatchedStateBridge, env_index: int, seat_id: int, observation: Any) -> set[int]:
    """Prove that the current legal window is lossless through the 241-space.

    Besides exact MJAI-template equality, this validates the selected
    RiichiEnv Action's semantic transition fields.  It catches aliases that
    happen to be accepted by the environment but lose an aka, consumed tile,
    hand-cut/tsumogiri, or action-kind distinction.
    """
    decision = Decision(env_index, seat_id, observation)
    legal = list(observation.legal_actions())
    templates = action_jsons(observation)
    if len(templates) != len(legal):
        raise AssertionError(f"env={env_index} seat={seat_id} legal/template length mismatch")
    expected_by_template: dict[str, list[Any]] = defaultdict(list)
    for action, template in zip(legal, templates, strict=True):
        expected_by_template[canonical(template)].append(action)
    expected = set(expected_by_template)
    prepared = bridge.prepare([decision])
    mask = prepared.legal_mask[0]
    ids = set(np.flatnonzero(mask).tolist())
    if not ids:
        raise AssertionError(f"env={env_index} seat={seat_id} empty mask legal={list(map(str, legal))}")
    returned: set[str] = set()
    for action_id in ids:
        mjai = bridge.state_machine.decode_actions([decision.batch_index], [action_id])[0]
        template = canonical(mjai)
        source_actions = expected_by_template.get(template, [])
        if not source_actions:
            raise AssertionError(
                f"env={env_index} seat={seat_id} action_id={action_id} decoded unexpected MJAI={mjai} "
                f"expected={sorted(expected)}"
            )
        # Exercise the public bridge path as well as the raw Rust decoder.
        env_action = bridge.decode([decision], [action_id])[0]
        actual = _action_signature(observation, env_action)
        expected_signatures = {_action_signature(observation, action) for action in source_actions}
        if actual not in expected_signatures:
            raise AssertionError(
                f"env={env_index} seat={seat_id} action_id={action_id} mjai={mjai} "
                f"selected={actual} expected={sorted(expected_signatures)}"
            )
        returned.add(template)
    if returned != expected:
        raise AssertionError(
            f"env={env_index} seat={seat_id} expected={sorted(expected)} returned={sorted(returned)} "
            f"mask_ids={sorted(ids)}"
        )
    return ids


def run_random_coverage(games: int = 128, seed: int = 20260713, max_steps: int = 2500) -> dict[str, Any]:
    """Run strict mask/decode verification for every decision in seeded real games."""
    import riichi
    from riichienv import BatchedRiichiEnv

    rng = random.Random(seed)
    event_counts: Counter[str] = Counter()
    offered_action_type_counts: Counter[str] = Counter()
    offered_action_id_counts: Counter[int] = Counter()
    executed_action_type_counts: Counter[str] = Counter()
    executed_action_id_counts: Counter[int] = Counter()
    def record_events(events_by_player: list[list[str]], game: int, step: int) -> None:
        for seat, events in enumerate(events_by_player):
            for raw in events:
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type not in EVENT_TYPES:
                    raise AssertionError(f"unknown 4p event game={game} step={step} seat={seat}: {raw}")
                event_counts[event_type] += 1
    for game in range(games):
        env = BatchedRiichiEnv(1, seed=seed + game, step_threads=1)
        manager = riichi.MjaiKyokuStateMachineManager(1)
        bridge = BatchedStateBridge(manager, 1)
        observations = list(env.reset())
        bridge.sync(observations)
        record_events(bridge.last_events[0], game, -1)
        for step in range(max_steps):
            actions: dict[int, Any] = {}
            for seat, observation in observations[0].items():
                legal = observation.legal_actions()
                if not legal:
                    continue
                ids = assert_observation_roundtrip(bridge, 0, int(seat), observation)
                offered_action_id_counts.update(ids)
                offered_action_type_counts.update(json.loads(value)["type"] for value in action_jsons(observation))
                action_id = rng.choice(sorted(ids))
                action = bridge.decode([Decision(0, int(seat), observation)], [action_id])[0]
                actions[int(seat)] = action
                executed_action_id_counts[action_id] += 1
                executed_action_type_counts[_action_signature(observation, action)[1]] += 1
            if not actions:
                if env.done():
                    break
                raise AssertionError(f"environment stalled in game={game} step={step}")
            observations = list(env.step_batch([actions]))
            bridge.sync(observations)
            record_events(bridge.last_events[0], game, step)
            if env.done()[0]:
                break
        else:
            raise AssertionError(f"game={game} exceeded max_steps={max_steps}")
    return {
        "schema_version": 2,
        "games": games,
        "seed": seed,
        "event_types": dict(sorted(event_counts.items())),
        # Keep the original field names as aliases for existing consumers.
        "action_types": dict(sorted(offered_action_type_counts.items())),
        "action_ids": sorted(offered_action_id_counts),
        "offered_action_types": dict(sorted(offered_action_type_counts.items())),
        "offered_action_ids": sorted(offered_action_id_counts),
        "executed_action_types": dict(sorted(executed_action_type_counts.items())),
        "executed_action_ids": sorted(executed_action_id_counts),
        "missing_naturally_observed_events": sorted(EVENT_TYPES - set(event_counts)),
        "missing_naturally_offered_action_types": sorted(MODEL_ACTION_TYPES - set(offered_action_type_counts)),
        "missing_naturally_executed_action_types": sorted(MODEL_ACTION_TYPES - set(executed_action_type_counts)),
        "missing_naturally_offered_action_ids": sorted(set(range(NUM_ACTIONS)) - set(offered_action_id_counts)),
        "missing_naturally_executed_action_ids": sorted(set(range(NUM_ACTIONS)) - set(executed_action_id_counts)),
        "missing_naturally_observed_action_types": sorted(MODEL_ACTION_TYPES - set(offered_action_type_counts)),
    }


def write_coverage(summary: dict[str, Any], output: str | Path) -> None:
    Path(output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
