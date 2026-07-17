"""Batch-native state and environment contracts used by the PPO worker."""

from __future__ import annotations

import json
import random
import unittest

import numpy as np

try:
    import riichi
    from riichienv import BatchedRiichiEnv, RiichiEnv
except ImportError:  # pragma: no cover
    riichi = None
    BatchedRiichiEnv = None
    RiichiEnv = None

from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision


@unittest.skipUnless(riichi is not None and BatchedRiichiEnv is not None, "local extensions are not installed")
class BatchedPipelineTest(unittest.TestCase):
    def test_legacy_split_state_machine_api_is_not_exported(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(1)
        for name in ("apply_player_events", "apply_env_player_events", "set_legal_actions", "set_legal_actions_batch", "model_inputs", "model_inputs_with_snapshots", "action_mask", "model_action_to_mjai"):
            self.assertFalse(hasattr(manager, name), name)

    def test_two_tables_progress_and_decision_rows_are_isolated(self) -> None:
        envs = BatchedRiichiEnv(2, seed=91, step_threads=2)
        bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(2), 2)
        observations = list(envs.reset())
        end_kyoku, end_game = bridge.sync(observations)
        self.assertEqual(end_kyoku.shape, (2,))
        self.assertFalse(end_kyoku.any())
        self.assertFalse(end_game.any())

        decisions = []
        for env_index, row in enumerate(observations):
            for seat, observation in row.items():
                if observation.legal_actions():
                    decisions.append(Decision(env_index, seat, observation))
        self.assertGreaterEqual(len(decisions), 2)
        (
            factors,
            numeric,
            lengths,
            masks,
            _history_generations,
            critic_factors,
            critic_lengths,
        ) = bridge.prepare(decisions)
        self.assertEqual(factors.shape, (len(decisions), factors.shape[1], 10))
        self.assertEqual(numeric.shape, (*factors.shape[:2], 8))
        self.assertEqual(critic_factors.shape, (len(decisions), critic_factors.shape[1], 10))
        self.assertEqual(critic_lengths.shape, (len(decisions),))
        self.assertEqual(masks.shape, (len(decisions), 241))
        self.assertTrue(np.all(masks.any(axis=1)))

        action_ids = [int(np.flatnonzero(mask)[0]) for mask in masks]
        decoded = bridge.decode(decisions, action_ids)
        self.assertEqual(len(decoded), len(decisions))

        rng = random.Random(91)
        actions_by_env = []
        for row in observations:
            actions_by_env.append({seat: rng.choice(obs.legal_actions()) for seat, obs in row.items() if obs.legal_actions()})
        observations = list(envs.step_batch(actions_by_env))
        end_kyoku, end_game = bridge.sync(observations)
        self.assertEqual(end_kyoku.shape, (2,))
        self.assertEqual(end_game.shape, (2,))

    def test_boundary_flags_are_per_environment(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(2)
        events = [
            [[json.dumps({"type": "end_kyoku"})], [], [], []],
            [[json.dumps({"type": "end_kyoku"}), json.dumps({"type": "end_game"})], [], [], []],
        ]
        end_kyoku, end_game = manager.apply_events_batch([0, 1], events)
        self.assertEqual(np.asarray(end_kyoku, dtype=bool).tolist(), [True, True])
        self.assertEqual(np.asarray(end_game, dtype=bool).tolist(), [False, True])

    def test_single_table_batch_matches_scalar_environment(self) -> None:
        scalar = RiichiEnv(game_mode="4p-red-half", seed=731)
        batch = BatchedRiichiEnv(1, seed=731, step_threads=1)
        scalar_observations = scalar.reset()
        batch_observations = list(batch.reset())[0]
        rng = random.Random(731)
        for _ in range(80):
            scalar_all = scalar.get_observations(players=[0, 1, 2, 3])
            scalar_actions = {}
            batch_actions = {}
            for seat in range(4):
                scalar_legal = scalar_all[seat].legal_actions()
                batch_legal = batch_observations[seat].legal_actions()
                self.assertEqual(len(scalar_legal), len(batch_legal))
                if scalar_legal:
                    index = rng.randrange(len(scalar_legal))
                    scalar_actions[seat] = scalar_legal[index]
                    batch_actions[seat] = batch_legal[index]
            if not scalar_actions:
                break
            scalar_observations = scalar.step(scalar_actions)
            batch_observations = list(batch.step_batch([batch_actions]))[0]
            self.assertEqual(scalar.scores(), batch.scores()[0])
            self.assertEqual(scalar.done(), batch.done()[0])
            if scalar.done():
                break
