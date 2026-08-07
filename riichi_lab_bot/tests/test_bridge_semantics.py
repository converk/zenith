"""Bot bridge semantic edge matrix for V13 online observations."""

from __future__ import annotations

import base64
import json
import random

import numpy as np
import pytest

from riichi_lab_bot.bridge import OnlineStateBridge
from riichi_lab_bot.local_play import observation_with_events
from riichi_lab_bot.observation import (
    ObservationView,
    missing_observation_fields,
    normalize_observation_base64,
)
from riichi_ppo_v1.model.semantic_validation import assert_actor_token_semantics


def _prepare(env, seat: int) -> tuple[OnlineStateBridge, object]:
    bridge = OnlineStateBridge(seat)
    observation = env.get_observation(seat)
    events = list(observation.new_events())
    prepared = bridge.prepare(observation_with_events(observation, events))
    assert_actor_token_semantics(
        prepared.token_factors[None],
        prepared.token_numeric[None],
        np.asarray([prepared.token_length], dtype=np.int64),
    )
    return bridge, prepared


def _decode_every_legal_id(bridge: OnlineStateBridge, prepared) -> None:
    ids = np.flatnonzero(prepared.legal_mask)
    assert len(ids) > 0
    for action_id in ids:
        action = bridge.decode(prepared, int(action_id))
        assert action is not None
        value = json.loads(action.to_mjai())
        assert isinstance(value.get("type"), str)


def test_kyushu_ryukyoku_window_prepares_and_decodes() -> None:
    from RiichiEnv.tests.env.helper import helper_setup_env

    env = helper_setup_env(
        hands=[
            [0, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132],
            [], [], [],
        ],
        current_player=0,
        active_players=[0],
        drawn_tile=3,
        wall=list(range(136)),
    )
    bridge, prepared = _prepare(env, 0)
    assert bool(prepared.legal_mask[240])
    _decode_every_legal_id(bridge, prepared)


def test_kakan_window_prepares_and_decodes() -> None:
    from riichienv import Meld, MeldType, Phase
    from RiichiEnv.tests.env.helper import helper_setup_env

    env = helper_setup_env(
        hands=[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [], [], []],
        melds=[[Meld(MeldType.Pon, tiles=[0, 1, 2], opened=True)], [], [], []],
        active_players=[0],
        current_player=0,
        phase=Phase.WaitAct,
        needs_tsumo=False,
        drawn_tile=13,
        wall=list(range(136)),
    )
    bridge, prepared = _prepare(env, 0)
    action_types = [
        json.loads(bridge.decode(prepared, int(action_id)).to_mjai())["type"]
        for action_id in np.flatnonzero(prepared.legal_mask)
    ]
    assert "kakan" in action_types
    _decode_every_legal_id(bridge, prepared)


def test_chi_pon_daiminkan_and_red_five_windows_prepares() -> None:
    from riichienv import Action, ActionType
    from RiichiEnv.tests.env.helper import helper_setup_env

    # Player 0 discards 1m; player 1 can pon/daiminkan and must pass.
    env = helper_setup_env(
        hands=[
            [0, 4, 8, 12, 16, 20, 24, 36, 40, 44, 48, 52, 56],
            [1, 2, 3, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42], [], [],
        ],
        current_player=0,
        active_players=[0],
        drawn_tile=108,
        wall=list(range(136)),
    )
    responses = env.step({0: Action(ActionType.DISCARD, tile=0)})
    observation = responses[1]
    bridge = OnlineStateBridge(1)
    events = list(observation.new_events())
    prepared = bridge.prepare(observation_with_events(observation, events))
    action_types = {
        json.loads(bridge.decode(prepared, int(action_id)).to_mjai())["type"]
        for action_id in np.flatnonzero(prepared.legal_mask)
    }
    assert {"none", "pon", "daiminkan"} <= action_types
    _decode_every_legal_id(bridge, prepared)

    # A 5m discard gives the shimocha all three chi shapes; one variant uses
    # a red 5m and must stay distinct.
    env = helper_setup_env(
        hands=[
            [8, 0, 4, 28, 32, 52, 56, 60, 64, 68, 92, 96, 100],
            [12, 16, 17, 20, 24, 36, 40, 44, 48, 72, 76, 80, 84], [], [],
        ],
        current_player=0,
        active_players=[0],
        drawn_tile=8,
        wall=list(range(136)),
    )
    responses = env.step({0: Action(ActionType.DISCARD, tile=8)})
    observation = responses[1]
    bridge = OnlineStateBridge(1)
    events = list(observation.new_events())
    prepared = bridge.prepare(observation_with_events(observation, events))
    action_types = {
        json.loads(bridge.decode(prepared, int(action_id)).to_mjai())["type"]
        for action_id in np.flatnonzero(prepared.legal_mask)
    }
    assert {"none", "chi"} <= action_types
    decoded = [
        json.loads(bridge.decode(prepared, int(action_id)).to_mjai())
        for action_id in np.flatnonzero(prepared.legal_mask)
    ]
    assert any(
        value.get("type") == "chi" and any("5mr" in str(tile) for tile in value.get("consumed", []))
        for value in decoded
    )
    _decode_every_legal_id(bridge, prepared)


def _first_legal_observation(seed: int = 20260730):
    from riichienv import RiichiEnv

    env = RiichiEnv(game_mode="4p-red-half", seed=seed)
    observations = env.reset()
    pending = {seat: [] for seat in range(4)}
    rng = random.Random(seed)
    for _step in range(4000):
        for seat, observation in observations.items():
            pending[int(seat)].extend(observation.new_events())
        for seat, observation in observations.items():
            if observation.legal_actions() and pending[int(seat)]:
                return observation, pending[int(seat)]
        actions = {
            seat: rng.choice(observation.legal_actions())
            for seat, observation in observations.items()
            if observation.legal_actions()
        }
        if not actions:
            raise RuntimeError("local environment stalled before a decision")
        observations = env.step(actions)
    raise RuntimeError("no legal decision with events found")


def test_unknown_events_and_fields_are_ignored_without_token_drift() -> None:
    observation, events = _first_legal_observation()
    baseline_obs = observation_with_events(observation, events)
    bridge = OnlineStateBridge(int(observation.player_id))
    baseline = bridge.prepare(baseline_obs)

    modified_events = list(events)
    if modified_events:
        value = json.loads(modified_events[-1])
        value["future_field"] = 123
        modified_events[-1] = json.dumps(
            value, separators=(",", ":"), sort_keys=True
        )
    modified_events.append(
        json.dumps(
            {"type": "future_event", "actor": 9, "pai": "?"},
            separators=(",", ":"),
        )
    )
    modified_obs = observation_with_events(observation, modified_events)
    modified = OnlineStateBridge(int(observation.player_id)).prepare(modified_obs)
    assert modified.token_length == baseline.token_length
    assert np.array_equal(
        modified.token_factors, baseline.token_factors
    )
    assert np.array_equal(
        modified.token_numeric, baseline.token_numeric
    )
    assert np.array_equal(modified.legal_mask, baseline.legal_mask)


def test_server_observation_with_missing_snapshot_fields_prepares() -> None:
    from riichienv import Observation, RiichiEnv

    env = RiichiEnv(game_mode="4p-red-half", seed=42)
    observation = env.reset()[0]
    full_bridge = OnlineStateBridge(0)
    full = full_bridge.prepare(observation)

    data = json.loads(
        base64.b64decode(observation.serialize_to_base64()).decode("utf-8")
    )
    missing = (
        "riichi_accepted",
        "riichi_declaration_indices",
        "missed_agari_doujun",
        "missed_agari_riichi",
        "tiles_left",
        "tsumogiri_flags",
        "last_tedashis",
        "riichi_sutehais",
        "waits",
        "is_tenpai",
    )
    for field in missing:
        data.pop(field, None)
    encoded = base64.b64encode(
        json.dumps(data, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    server_observation = ObservationView(
        Observation.deserialize_from_base64(
            normalize_observation_base64(encoded)
        ),
        missing_fields=missing_observation_fields(encoded),
    )
    server_bridge = OnlineStateBridge(0)
    prepared = server_bridge.prepare(server_observation)

    assert prepared.token_length == full.token_length
    assert np.array_equal(prepared.token_factors, full.token_factors)
    assert np.array_equal(prepared.token_numeric, full.token_numeric)
    assert np.array_equal(prepared.legal_mask, full.legal_mask)


def test_reach_declared_without_declaration_tile_is_normalized() -> None:
    from riichienv import Observation, RiichiEnv

    env = RiichiEnv(game_mode="4p-red-half", seed=42)
    observation = env.reset()[0]
    data = json.loads(
        base64.b64decode(observation.serialize_to_base64()).decode("utf-8")
    )
    data["riichi_declared"] = [True, False, False, False]
    data["riichi_sutehais"] = [None, None, None, None]
    for field in (
        "riichi_accepted",
        "riichi_declaration_indices",
        "missed_agari_doujun",
        "missed_agari_riichi",
        "tiles_left",
    ):
        data.pop(field, None)
    encoded = base64.b64encode(
        json.dumps(data, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    server_observation = ObservationView(
        Observation.deserialize_from_base64(
            normalize_observation_base64(encoded)
        ),
        missing_fields=missing_observation_fields(encoded),
    )
    bridge = OnlineStateBridge(0)
    prepared = bridge.prepare(server_observation)
    assert_actor_token_semantics(
        prepared.token_factors[None],
        prepared.token_numeric[None],
        np.asarray([prepared.token_length], dtype=np.int64),
    )


def test_server_riichi_snapshot_with_stale_sutehai_and_accepted_missing_prepares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server may provide riichi_declared=True but riichi_sutehais=None."""
    from riichienv import Observation, RiichiEnv

    env = RiichiEnv(game_mode="4p-red-half", seed=42)
    observation = env.reset()[0]
    data = json.loads(
        base64.b64decode(observation.serialize_to_base64()).decode("utf-8")
    )
    data["riichi_declared"] = [True, False, False, False]
    data["riichi_sutehais"] = [None, None, None, None]
    for field in (
        "riichi_accepted",
        "riichi_declaration_indices",
        "missed_agari_doujun",
        "missed_agari_riichi",
        "tiles_left",
    ):
        data.pop(field, None)
    encoded = base64.b64encode(
        json.dumps(data, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    server_observation = ObservationView(
        Observation.deserialize_from_base64(
            normalize_observation_base64(encoded)
        ),
        missing_fields=missing_observation_fields(encoded),
    )
    bridge = OnlineStateBridge(0)
    bridge.threats.riichi_declared = [True, False, False, False]
    bridge.threats.riichi_accepted = [True, False, False, False]
    bridge.threats.riichi_declaration_indices = [12, None, None, None]
    bridge.threats.riichi_sutehais = [84, None, None, None]
    bridge.threats.discard_counts = [12, 0, 0, 0]
    monkeypatch.setattr(
        bridge.threats, "apply_events", lambda _events: None
    )
    prepared = bridge.prepare(server_observation)
    assert prepared.token_length > 0
