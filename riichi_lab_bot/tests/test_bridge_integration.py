from __future__ import annotations

import json
import random

import numpy as np
import pytest
from conftest import default_checkpoint
from riichi_lab_bot.bridge import OnlineStateBridge
from riichi_lab_bot.local_play import observation_with_events
from riichi_lab_bot.policy import PolicyEngine
from riichi_lab_bot.safety import choose_safe_response

from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_input_semantics,
)


def _assert_prepared_semantics(prepared) -> None:
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


def _assert_matches_training(prepared, batch) -> None:
    assert prepared.history_length == int(batch.history_lengths[0])
    assert prepared.snapshot_length == int(batch.snapshot_lengths[0])
    assert prepared.query_pair_count == int(batch.query_pair_counts[0])
    assert np.array_equal(
        prepared.history_factors,
        batch.history_factors[0, : prepared.history_length],
    )
    assert np.array_equal(
        prepared.history_numeric,
        batch.history_numeric[0, : prepared.history_length],
    )
    assert np.array_equal(
        prepared.snapshot_factors,
        batch.snapshot_factors[0, : prepared.snapshot_length],
    )
    assert np.array_equal(
        prepared.snapshot_numeric,
        batch.snapshot_numeric[0, : prepared.snapshot_length],
    )
    assert np.array_equal(
        prepared.query_rows,
        batch.query_rows[0, : 2 * prepared.query_pair_count],
    )
    assert np.array_equal(
        prepared.query_action_ids,
        batch.query_action_ids[0, : prepared.query_pair_count],
    )
    assert np.array_equal(prepared.legal_mask, batch.legal_mask[0])


def test_serialized_observation_model_action_roundtrip() -> None:
    from riichienv import RiichiEnv

    env = RiichiEnv(game_mode="4p-red-half", seed=20260730)
    observations = env.reset()
    engine = PolicyEngine(
        default_checkpoint(), device="cpu", dtype="fp32"
    )
    bridges = {seat: OnlineStateBridge(seat) for seat in range(4)}
    pending = {seat: [] for seat in range(4)}
    decision_count = 0
    for _step in range(2000):
        for seat, observation in observations.items():
            pending[seat].extend(observation.new_events())
        actions = {}
        for seat, original in observations.items():
            if not original.legal_actions():
                continue
            obs = observation_with_events(original, pending[seat])
            pending[seat].clear()
            prepared = bridges[seat].prepare(obs)
            _assert_prepared_semantics(prepared)
            inference = engine.infer(prepared)
            primary = bridges[seat].decode(
                prepared, inference.action_id
            )
            safe = choose_safe_response(
                prepared,
                primary,
                [json.loads(value) for value in prepared.legal_jsons],
                decision_count,
            )
            assert safe.payload is not None
            bridges[seat].record_response(prepared, safe.payload)
            selected = original.select_action_from_mjai(safe.payload)
            assert selected is not None
            actions[seat] = selected
            decision_count += 1
        observations = env.step(actions)
        if env.done():
            break
    else:
        pytest.fail("fixed-seed hanchan did not finish")
    assert decision_count > 100


def test_bot_reuses_training_bridge_helpers() -> None:
    import riichi_lab_bot.bridge as bot_bridge
    import riichi_ppo_v1.model.bridge as training_bridge

    assert bot_bridge.BatchedStateBridge is training_bridge.BatchedStateBridge
    assert (
        bot_bridge.action_jsons_and_decision_flag
        is training_bridge.action_jsons_and_decision_flag
    )


def test_single_seat_bridge_matches_training_bridge() -> None:
    riichi = pytest.importorskip("riichi")
    training_bridge = pytest.importorskip("riichi_ppo_v1.model.bridge")
    from riichienv import BatchedRiichiEnv

    env = BatchedRiichiEnv(
        1, seed=20260731, step_threads=1, game_mode="4p-red-half"
    )
    rows = list(env.reset())
    reference = training_bridge.BatchedStateBridge(
        riichi.MjaiKyokuStateMachineManager(1), 1
    )
    online = {seat: OnlineStateBridge(seat) for seat in range(4)}
    pending = {seat: [] for seat in range(4)}
    rng = random.Random(20260731)
    compared = [0, 0, 0, 0]
    last_prepared: dict[int, object] = {}
    for _step in range(2500):
        for seat, observation in rows[0].items():
            pending[seat].extend(observation.new_events())
        reference.sync(rows)
        last_prepared.clear()
        for seat, observation in rows[0].items():
            if not observation.legal_actions():
                continue
            decision = training_bridge.Decision(0, int(seat), observation)
            expected = reference.prepare([decision])
            server_observation = observation_with_events(
                observation, pending[seat]
            )
            pending[seat].clear()
            actual = online[seat].prepare(server_observation)
            _assert_prepared_semantics(actual)
            _assert_matches_training(actual, expected)
            compared[seat] += 1
            last_prepared[seat] = actual
        actions = {
            seat: rng.choice(observation.legal_actions())
            for seat, observation in rows[0].items()
            if observation.legal_actions()
        }
        for seat, action in actions.items():
            prepared = last_prepared.get(seat)
            if prepared is not None:
                online[seat].record_response(
                    prepared, json.loads(action.to_mjai())
                )
        rows = list(env.step_batch([actions]))
        if env.done()[0]:
            break
    else:
        pytest.fail("bridge-equivalence hanchan did not finish")
    assert all(count > 100 for count in compared)
