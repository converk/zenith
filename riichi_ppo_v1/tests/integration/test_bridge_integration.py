"""Requires locally built ``riichi`` and ``riichienv`` extensions."""

from __future__ import annotations

import random
import unittest

try:
    import riichi
    from riichienv import BatchedRiichiEnv
except ImportError:  # pragma: no cover - expected before local extension setup
    riichi = None
    BatchedRiichiEnv = None

from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision, action_jsons
from riichi_ppo_v1.model.validation import assert_observation_roundtrip, run_random_coverage


@unittest.skipUnless(riichi is not None and BatchedRiichiEnv is not None, "local RiichiEnv extensions are not installed")
class BridgeIntegrationTest(unittest.TestCase):
    def test_random_game_masks_and_mjai_roundtrip(self) -> None:
        env = BatchedRiichiEnv(2, seed=7, step_threads=2)
        state_machine = riichi.MjaiKyokuStateMachineManager(1)
        state_machine = riichi.MjaiKyokuStateMachineManager(2)
        bridge = BatchedStateBridge(state_machine, 2)
        observations = list(env.reset())
        bridge.sync(observations)
        rng = random.Random(7)
        for _ in range(120):
            actions_by_env = []
            for env_index, env_observations in enumerate(observations):
                actions = {}
                for seat, observation in env_observations.items():
                    legal = observation.legal_actions()
                    if not legal:
                        continue
                    decision = Decision(env_index, seat, observation)
                    ids = assert_observation_roundtrip(bridge, env_index, int(seat), observation)
                    self.assertEqual(len(action_jsons(observation)), len(legal))
                    # This is deliberately the model-side action path.  A
                    # legal RiichiEnv action selected directly would not prove
                    # that a decoded 241-space choice can advance the game.
                    action_id = rng.choice(sorted(ids))
                    actions[seat] = bridge.decode([decision], [action_id])[0]
                actions_by_env.append(actions)
            if not any(actions_by_env):
                break
            observations = list(env.step_batch(actions_by_env))
            bridge.sync(observations)
            if all(env.done()):
                break

    def test_coverage_report_distinguishes_offered_and_decoded_actions(self) -> None:
        summary = run_random_coverage(games=1, seed=20260713, max_steps=2500)
        self.assertEqual(summary["schema_version"], 2)
        self.assertTrue(summary["offered_action_ids"])
        self.assertTrue(summary["executed_action_ids"])
        self.assertTrue(set(summary["executed_action_ids"]) <= set(summary["offered_action_ids"]))
        self.assertEqual(summary["action_ids"], summary["offered_action_ids"])
        self.assertEqual(summary["action_types"], summary["offered_action_types"])
        self.assertIn("missing_naturally_observed_events", summary)
