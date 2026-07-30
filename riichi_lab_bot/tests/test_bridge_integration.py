from __future__ import annotations

import json
import random

import numpy as np
import pytest

from riichi_lab_bot.bridge import OnlineStateBridge
from riichi_lab_bot.local_play import observation_with_events
from riichi_lab_bot.policy import PolicyEngine
from riichi_lab_bot.safety import choose_safe_response
from conftest import default_checkpoint


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
    for _step in range(2500):
        for seat, observation in rows[0].items():
            pending[seat].extend(observation.new_events())
        reference.sync(rows)
        for seat, observation in rows[0].items():
            if not observation.legal_actions():
                continue
            expected = reference.prepare(
                [training_bridge.Decision(0, seat, observation)]
            )
            server_observation = observation_with_events(
                observation, pending[seat]
            )
            pending[seat].clear()
            actual = online[seat].prepare(server_observation)
            (
                expected_factors,
                expected_numeric,
                expected_lengths,
                expected_mask,
            ) = expected[:4]
            assert actual.token_length == int(expected_lengths[0])
            assert np.array_equal(
                actual.token_factors,
                expected_factors[0, : actual.token_length],
            )
            assert np.array_equal(
                actual.token_numeric,
                expected_numeric[0, : actual.token_length],
            )
            assert np.array_equal(actual.legal_mask, expected_mask[0])
            compared[seat] += 1
        actions = {
            seat: rng.choice(observation.legal_actions())
            for seat, observation in rows[0].items()
            if observation.legal_actions()
        }
        rows = list(env.step_batch([actions]))
        if env.done()[0]:
            break
    else:
        pytest.fail("bridge-equivalence hanchan did not finish")
    assert all(count > 100 for count in compared)
