"""bot bridge 的 V18 在线 Observation 语义边界矩阵。"""

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
from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_input_semantics,
)
from riichi_ppo_v1.model.encoding_protocol import SNAPSHOT_FIELD_BY_NAME

_FURITEN_TEHAIS_JSON = (
    '["3m","3m","5m","6m","7m","8m","9m","1p","2p","3p","4p","5p","6p"],'
    '["1s","2s","3s","4s","5s","6s","7s","8s","9s","1m","1m","1m","2m"],'
    '["1z","2z","3z","4z","5z","6z","7z","7p","8p","9p","3m","8s","9s"],'
    '["4s","5s","6s","7s","8s","9s","4m","5m","6m","7m","8m","9m","1z"]'
)


def _assert_v18(prepared) -> None:
    assert_actor_input_semantics(
        prepared.history_factors[None],
        prepared.history_numeric[None],
        np.asarray([prepared.history_length], dtype=np.int64),
        prepared.snapshot_factors[None],
        prepared.snapshot_numeric[None],
        np.asarray([prepared.snapshot_length], dtype=np.int64),
        prepared.query_rows[None],
        prepared.query_action_ids[None],
        np.asarray([prepared.query_pair_count], dtype=np.int64),
        prepared.legal_mask[None],
    )


def _assert_same(left, right) -> None:
    assert left.history_length == right.history_length
    assert left.snapshot_length == right.snapshot_length
    assert left.query_pair_count == right.query_pair_count
    assert np.array_equal(left.history_factors, right.history_factors)
    assert np.array_equal(left.history_numeric, right.history_numeric)
    assert np.array_equal(left.snapshot_factors, right.snapshot_factors)
    assert np.array_equal(left.snapshot_numeric, right.snapshot_numeric)
    assert np.array_equal(left.query_rows, right.query_rows)
    assert np.array_equal(left.query_action_ids, right.query_action_ids)
    assert np.array_equal(left.legal_mask, right.legal_mask)


def _prepare(env, seat: int) -> tuple[OnlineStateBridge, object]:
    bridge = OnlineStateBridge(seat)
    observation = env.get_observation(seat)
    events = list(observation.new_events())
    prepared = bridge.prepare(observation_with_events(observation, events))
    _assert_v18(prepared)
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

    # 0 号玩家打出 1m;1 号玩家可以碰/大明杠,也必须能 pass。
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

    # 5m 被下家吃时会出现三种吃法;含赤 5m 的变体必须保持可区分。
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


def _server_observation_missing_fields(
    observation, events: list[str], missing: tuple[str, ...],
):
    from riichienv import Observation

    online = observation_with_events(observation, events)
    data = json.loads(
        base64.b64decode(online.serialize_to_base64()).decode("utf-8")
    )
    for field in missing:
        data.pop(field, None)
    encoded = base64.b64encode(
        json.dumps(data, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return ObservationView(
        Observation.deserialize_from_base64(
            normalize_observation_base64(encoded)
        ),
        missing_fields=missing_observation_fields(encoded),
    )


def _furiten_replay_steps(lines: list[str], tmp_path):
    from riichienv import MjaiReplay

    path = tmp_path / "furiten.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    replay = MjaiReplay.from_jsonl(str(path), rule="tenhou")
    steps = []
    for kyoku in replay.take_kyokus():
        steps.extend(kyoku.steps(seat=None, skip_single_action=False))
    return steps


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
    assert modified.history_length == baseline.history_length
    assert modified.snapshot_length == baseline.snapshot_length
    assert modified.query_pair_count == baseline.query_pair_count
    assert np.array_equal(modified.history_factors, baseline.history_factors)
    assert np.array_equal(modified.history_numeric, baseline.history_numeric)
    assert np.array_equal(modified.snapshot_factors, baseline.snapshot_factors)
    assert np.array_equal(modified.snapshot_numeric, baseline.snapshot_numeric)
    assert np.array_equal(modified.query_rows, baseline.query_rows)
    assert np.array_equal(modified.query_action_ids, baseline.query_action_ids)
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

    assert prepared.history_length == full.history_length
    assert prepared.snapshot_length == full.snapshot_length
    assert prepared.query_pair_count == full.query_pair_count
    assert np.array_equal(prepared.history_factors, full.history_factors)
    assert np.array_equal(prepared.history_numeric, full.history_numeric)
    assert np.array_equal(prepared.snapshot_factors, full.snapshot_factors)
    assert np.array_equal(prepared.snapshot_numeric, full.snapshot_numeric)
    assert np.array_equal(prepared.query_rows, full.query_rows)
    assert np.array_equal(prepared.query_action_ids, full.query_action_ids)
    assert np.array_equal(prepared.legal_mask, full.legal_mask)


def test_missing_server_fields_match_full_observation_across_hanchan() -> None:
    from riichienv import RiichiEnv

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
        "drawn_tile",
    )
    env = RiichiEnv(game_mode="4p-red-half", seed=20260820)
    observations = env.reset()
    full_bridges = {seat: OnlineStateBridge(seat) for seat in range(4)}
    server_bridges = {seat: OnlineStateBridge(seat) for seat in range(4)}
    pending = {seat: [] for seat in range(4)}
    rng = random.Random(20260820)
    compared = 0
    for _step in range(4000):
        for seat, observation in observations.items():
            pending[int(seat)].extend(observation.new_events())
        actions = {}
        for seat, observation in observations.items():
            if not observation.legal_actions():
                continue
            events = pending[int(seat)]
            full_observation = observation_with_events(observation, events)
            server_observation = _server_observation_missing_fields(
                observation, events, missing
            )
            pending[int(seat)].clear()
            full = full_bridges[int(seat)].prepare(full_observation)
            server = server_bridges[int(seat)].prepare(server_observation)
            _assert_v18(server)
            _assert_same(server, full)
            action = rng.choice(observation.legal_actions())
            payload = json.loads(action.to_mjai())
            full_bridges[int(seat)].record_response(full, payload)
            server_bridges[int(seat)].record_response(server, payload)
            actions[int(seat)] = action
            compared += 1
        if not actions:
            if env.done():
                break
            raise RuntimeError("local environment stalled before a decision")
        observations = env.step(actions)
        if env.done():
            break
    else:
        pytest.fail("missing-field hanchan did not finish")
    assert compared > 100


def test_missed_agari_doujun_tracks_pass_and_own_discard(tmp_path) -> None:
    lines = [
        '{"type":"start_game"}',
        '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyoutaku":0,'
        '"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"2p",'
        '"tehais":[' + _FURITEN_TEHAIS_JSON + "]}",
        '{"type":"tsumo","actor":0,"pai":"7p"}',
        '{"type":"dahai","actor":0,"pai":"3m","tsumogiri":false}',
        '{"type":"tsumo","actor":1,"pai":"4z"}',
        '{"type":"dahai","actor":1,"pai":"4z","tsumogiri":true}',
        '{"type":"tsumo","actor":2,"pai":"5z"}',
        '{"type":"dahai","actor":2,"pai":"3m","tsumogiri":false}',
        '{"type":"tsumo","actor":3,"pai":"3z"}',
        '{"type":"dahai","actor":3,"pai":"3z","tsumogiri":true}',
        '{"type":"ryukyoku","reason":"yao9"}',
        '{"type":"end_kyoku"}',
        '{"type":"end_game"}',
    ]
    bridge = OnlineStateBridge(1)
    missing = ("missed_agari_doujun", "missed_agari_riichi")
    p1_steps = [
        (observation, action)
        for pid, observation, action in _furiten_replay_steps(lines, tmp_path)
        if int(pid) == 1
    ]

    first_pass = _server_observation_missing_fields(
        p1_steps[0][0], list(p1_steps[0][0].new_events()), missing
    )
    prepared = bridge.prepare(first_pass)
    assert prepared.observation.missed_agari_doujun is False
    assert any(json.loads(value).get("type") == "hora" for value in prepared.legal_jsons)
    bridge.record_response(prepared, json.loads(p1_steps[0][1].to_mjai()))
    assert bridge.threats.missed_agari_doujun is True
    assert bridge.threats.missed_agari_riichi is False

    own_discard = _server_observation_missing_fields(
        p1_steps[1][0], list(p1_steps[1][0].new_events()), missing
    )
    prepared = bridge.prepare(own_discard)
    assert prepared.observation.missed_agari_doujun is True
    bridge.record_response(prepared, json.loads(p1_steps[1][1].to_mjai()))
    assert bridge.threats.missed_agari_doujun is False

    second_pass = _server_observation_missing_fields(
        p1_steps[2][0], list(p1_steps[2][0].new_events()), missing
    )
    prepared = bridge.prepare(second_pass)
    assert prepared.observation.missed_agari_doujun is False
    assert any(json.loads(value).get("type") == "hora" for value in prepared.legal_jsons)


def test_missed_agari_riichi_persists_and_start_kyoku_resets(tmp_path) -> None:
    lines = [
        '{"type":"start_game"}',
        '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyoutaku":0,'
        '"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"2p",'
        '"tehais":[' + _FURITEN_TEHAIS_JSON + "]}",
        '{"type":"tsumo","actor":0,"pai":"7p"}',
        '{"type":"dahai","actor":0,"pai":"7p","tsumogiri":true}',
        '{"type":"tsumo","actor":1,"pai":"4z"}',
        '{"type":"reach","actor":1}',
        '{"type":"dahai","actor":1,"pai":"4z","tsumogiri":true}',
        '{"type":"reach_accepted","actor":1}',
        '{"type":"tsumo","actor":2,"pai":"5z"}',
        '{"type":"dahai","actor":2,"pai":"5z","tsumogiri":true}',
        '{"type":"tsumo","actor":3,"pai":"2z"}',
        '{"type":"dahai","actor":3,"pai":"2z","tsumogiri":true}',
        '{"type":"tsumo","actor":0,"pai":"1z"}',
        '{"type":"dahai","actor":0,"pai":"3m","tsumogiri":false}',
        '{"type":"tsumo","actor":1,"pai":"6z"}',
        '{"type":"dahai","actor":1,"pai":"6z","tsumogiri":true}',
        '{"type":"tsumo","actor":2,"pai":"7z"}',
        '{"type":"dahai","actor":2,"pai":"3m","tsumogiri":false}',
        '{"type":"tsumo","actor":3,"pai":"3z"}',
        '{"type":"dahai","actor":3,"pai":"3z","tsumogiri":true}',
        '{"type":"ryukyoku","reason":"yao9"}',
        '{"type":"end_kyoku"}',
        '{"type":"end_game"}',
    ]
    bridge = OnlineStateBridge(1)
    missing = ("missed_agari_doujun", "missed_agari_riichi")
    p1_steps = [
        (observation, action)
        for pid, observation, action in _furiten_replay_steps(lines, tmp_path)
        if int(pid) == 1
    ]

    ron_index = None
    for index, (observation, action) in enumerate(p1_steps):
        server_observation = _server_observation_missing_fields(
            observation, list(observation.new_events()), missing
        )
        prepared = bridge.prepare(server_observation)
        if any(json.loads(value).get("type") == "hora" for value in prepared.legal_jsons):
            bridge.record_response(prepared, json.loads(action.to_mjai()))
            ron_index = index
            break
        bridge.record_response(prepared, json.loads(action.to_mjai()))
    assert ron_index is not None
    assert bridge.threats.missed_agari_doujun is True
    assert bridge.threats.missed_agari_riichi is True

    for observation, action in p1_steps[ron_index + 1:]:
        server_observation = _server_observation_missing_fields(
            observation, list(observation.new_events()), missing
        )
        prepared = bridge.prepare(server_observation)
        bridge.record_response(prepared, json.loads(action.to_mjai()))
    assert bridge.threats.missed_agari_doujun is False
    assert bridge.threats.missed_agari_riichi is True

    bridge.threats.apply_events([
        '{"type":"start_kyoku","bakaze":"S","kyoku":2,"honba":0,'
        '"kyoutaku":0,"oya":1,"scores":[25000,25000,25000,25000],'
        '"dora_marker":"1m","tehais":[[],[],[],[]]}'
    ])
    assert bridge.threats.missed_agari_doujun is False
    assert bridge.threats.missed_agari_riichi is False


def test_present_empty_server_tsumogiri_flags_are_overridden() -> None:
    from riichienv import Observation, RiichiEnv

    observation = RiichiEnv(game_mode="4p-red-half", seed=42).reset()[0]
    data = json.loads(
        base64.b64decode(observation.serialize_to_base64()).decode("utf-8")
    )
    data["tsumogiri_flags"] = [[], [], [], []]
    data["discards"][1] = [36, 40, 44]
    data["events"].extend([
        json.dumps({
            "type": "dahai", "actor": 1, "pai": "1p",
            "tsumogiri": False,
        }),
        json.dumps({
            "type": "dahai", "actor": 1, "pai": "2p",
            "tsumogiri": True,
        }),
        json.dumps({
            "type": "dahai", "actor": 1, "pai": "3p",
            "tsumogiri": True,
        }),
    ])
    encoded = base64.b64encode(
        json.dumps(data, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    server_observation = ObservationView(
        Observation.deserialize_from_base64(encoded),
        missing_fields=missing_observation_fields(encoded),
    )

    prepared = OnlineStateBridge(0).prepare(server_observation)

    assert prepared.observation.tsumogiri_flags[1] == [False, True, True]
    # 下家三次弃牌中 1 次手切、2 次摸切,当前连续摸切为 2。
    assert int(prepared.snapshot_factors[7, 2]) == 1
    assert int(prepared.snapshot_factors[8, 2]) == 2
    streak_row = SNAPSHOT_FIELD_BY_NAME["opponent_1_tsumogiri_streak"].field_id - 1
    assert int(prepared.snapshot_factors[streak_row, 2]) == 2


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
    _assert_v18(prepared)


def test_server_riichi_snapshot_with_stale_sutehai_and_accepted_missing_prepares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """线上可能给 riichi_declared=True,但 riichi_sutehais=None。"""
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
    _assert_v18(prepared)
    assert prepared.history_length > 0
