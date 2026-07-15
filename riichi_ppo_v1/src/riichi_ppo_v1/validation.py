"""Reusable validation helpers for the 4-player V3/241 integration boundary."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from .bridge import BatchedStateBridge, Decision, NUM_ACTIONS, NUM_PLAYERS, action_jsons

EVENT_TYPES = frozenset({
    "start_game", "start_kyoku", "tsumo", "dahai", "chi", "pon", "daiminkan",
    "ankan", "kakan", "dora", "reach", "reach_accepted", "hora", "ryukyoku",
    "end_kyoku", "end_game",
})
MODEL_EVENT_TYPES = frozenset({
    "start_kyoku", "tsumo", "dahai", "chi", "pon", "daiminkan", "ankan", "kakan",
    "dora", "reach", "reach_accepted", "hora", "ryukyoku",
})
TOKEN_BY_EVENT = {
    "start_kyoku": 26, "tsumo": 27, "dahai": 28, "chi": 29, "pon": 30,
    "daiminkan": 31, "kakan": 32, "ankan": 33, "dora": 35, "reach": 36,
    "reach_accepted": 37, "hora": 38, "ryukyoku": 39,
}
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


def suffix(suit: str) -> str:
    """Map m/p/s to the red-five spelling's suit character."""
    return suit


def _pon_pairs() -> list[tuple[str, str]]:
    pairs = [(tile, tile) for tile in TILE34 if tile not in {"5m", "5p", "5s"}]
    pairs.extend([("5m", "5m"), ("5m", "5mr"), ("5p", "5p"), ("5p", "5pr"), ("5s", "5s"), ("5s", "5sr")])
    return pairs


def all_action_templates() -> list[dict[str, Any]]:
    """One MJAI template for every KyokuActionSpace V2 action id, in id order."""
    templates: list[dict[str, Any]] = [{"type": "none"}]
    templates.extend({"type": "dahai", "pai": tile, "tsumogiri": bool(mode)} for tile in TILE37 for mode in range(2))
    templates.append({"type": "reach"})
    templates.extend({"type": "chi", "pai": "5m", "consumed": list(pair)} for pair in _chi_pairs())
    templates.extend({"type": "pon", "pai": pair[0].replace("r", ""), "consumed": list(pair)} for pair in _pon_pairs())
    templates.append({"type": "daiminkan", "pai": "E", "consumed": ["E", "E", "E"]})
    templates.extend({"type": "ankan", "consumed": [tile] * 4} for tile in TILE34)
    templates.extend({"type": "kakan", "pai": tile, "consumed": [tile] * 3} for tile in TILE34)
    templates.extend(({"type": "hora"}, {"type": "ryukyoku"}))
    if len(templates) != NUM_ACTIONS:
        raise AssertionError(f"generated {len(templates)} templates instead of {NUM_ACTIONS}")
    return templates


def fixture_snapshot() -> str:
    return json.dumps({
        "player_id": 0, "oya": 0, "round_wind": 0, "kyoku_index": 0, "honba": 0,
        "riichi_sticks": 0, "scores": [25000] * 4, "dora_indicators": ["2p"],
        "hand": ["1m"] * 13, "drawn_tile": None, "riichi_declared": [False] * 4,
        "last_discard": None, "last_tedashis": [None] * 4,
    }, separators=(",", ":"))


def assert_full_action_space(manager: Any) -> None:
    templates = all_action_templates()
    _ids, _attention, _lengths, mask, _history_lengths, _history_generations = manager.prepare_decisions(
        [0], [[json.dumps(template, separators=(",", ":")) for template in templates]], [fixture_snapshot()]
    )
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (1, NUM_ACTIONS) or not mask[0].all():
        raise AssertionError(f"expected all 241 slots, got shape={mask.shape} count={mask.sum(axis=1)}")
    for action_id, expected in enumerate(templates):
        returned = manager.decode_actions([0], [action_id])[0]
        if canonical(returned) != canonical(expected):
            raise AssertionError(f"action_id={action_id} expected={expected} returned={returned}")


def assert_observation_roundtrip(bridge: BatchedStateBridge, env_index: int, seat_id: int, observation: Any) -> set[int]:
    decision = Decision(env_index, seat_id, observation)
    expected = {canonical(value) for value in action_jsons(observation)}
    _inputs, _attention, _lengths, masks, _history_lengths, _history_generations = bridge.prepare([decision])
    mask = masks[0]
    ids = set(np.flatnonzero(mask).tolist())
    if not ids:
        raise AssertionError(f"empty mask for seat={seat_id}, legal={list(map(str, observation.legal_actions()))}")
    returned: set[str] = set()
    for action_id in ids:
        mjai = bridge.state_machine.decode_actions([decision.batch_index], [action_id])[0]
        env_action = observation.select_action_from_mjai(mjai)
        if mjai is None or env_action is None:
            raise AssertionError(f"seat={seat_id} action_id={action_id} mask={mask.tolist()} mjai={mjai}")
        returned.add(canonical(mjai))
    if returned != expected:
        raise AssertionError(f"seat={seat_id} expected={sorted(expected)} returned={sorted(returned)} mask_ids={sorted(ids)}")
    return ids


def run_random_coverage(games: int = 128, seed: int = 20260713, max_steps: int = 2500) -> dict[str, Any]:
    """Run strict mask/decode verification for every decision in seeded real games."""
    import riichi
    from riichienv import BatchedRiichiEnv

    rng = random.Random(seed)
    event_counts: Counter[str] = Counter()
    action_type_counts: Counter[str] = Counter()
    action_id_counts: Counter[int] = Counter()
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
                action_id_counts.update(ids)
                action_type_counts.update(json.loads(value)["type"] for value in action_jsons(observation))
                actions[int(seat)] = rng.choice(legal)
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
        "games": games,
        "seed": seed,
        "event_types": dict(sorted(event_counts.items())),
        "action_types": dict(sorted(action_type_counts.items())),
        "action_ids": sorted(action_id_counts),
        "missing_naturally_observed_events": sorted(EVENT_TYPES - set(event_counts)),
        "missing_naturally_observed_action_types": sorted({"none", "dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan", "hora", "ryukyoku"} - set(action_type_counts)),
    }


def write_coverage(summary: dict[str, Any], output: str | Path) -> None:
    Path(output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
