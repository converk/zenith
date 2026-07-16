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

from riichi_ppo_v1.bridge import BatchedStateBridge, Decision, action_jsons


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
                    _factors, _numeric, _lengths, masks, _history_generations = bridge.prepare([decision])
                    mask = masks[0]
                    self.assertEqual(mask.shape, (241,))
                    self.assertTrue(mask.any())
                    self.assertEqual(len(action_jsons(observation)), len(legal))
                    decoded = bridge.decode([decision], [int(mask.nonzero()[0][0])])[0]
                    self.assertIsNotNone(decoded)
                    actions[seat] = rng.choice(legal)
                actions_by_env.append(actions)
            if not any(actions_by_env):
                break
            observations = list(env.step_batch(actions_by_env))
            bridge.sync(observations)
            if all(env.done()):
                break
