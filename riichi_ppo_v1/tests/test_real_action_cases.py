"""Controlled RiichiEnv legal windows must map and decode through the 241-space."""

from __future__ import annotations

import unittest

try:
    import riichi
    from riichienv import Action, ActionType, Meld, MeldType, Phase, RiichiEnv
    from RiichiEnv.tests.env.helper import helper_setup_env
except ImportError:  # pragma: no cover
    riichi = None

from riichi_ppo_v1.bridge import BatchedStateBridge, action_jsons
from riichi_ppo_v1.validation import assert_observation_roundtrip


@unittest.skipUnless(riichi is not None, "local RiichiEnv and riichi extensions are not installed")
class RealActionCasesTest(unittest.TestCase):
    def assert_window(self, observation, required: set[str]) -> None:
        bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
        observed = {__import__("json").loads(value)["type"] for value in action_jsons(observation)}
        self.assertTrue(required <= observed, (required, observed))
        assert_observation_roundtrip(bridge, 0, int(observation.player_id), observation)

    def test_draw_window_reach_ankan_and_kyushu(self) -> None:
        env = helper_setup_env(
            hands=[[0, 1, 2, 4, 8, 12, 16, 20, 36, 40, 44, 108, 112], [], [], []],
            current_player=0, active_players=[0], drawn_tile=3, wall=list(range(136)),
        )
        self.assert_window(env.get_observation(0), {"dahai", "reach", "ankan"})
        env = helper_setup_env(
            hands=[[0, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132], [], [], []],
            current_player=0, active_players=[0], drawn_tile=3, wall=list(range(136)),
        )
        self.assert_window(env.get_observation(0), {"dahai", "ryukyoku"})

    def test_kakan_window(self) -> None:
        env = helper_setup_env(
            hands=[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [], [], []],
            melds=[[Meld(MeldType.Pon, tiles=[0, 1, 2], opened=True)], [], [], []],
            active_players=[0], current_player=0, phase=Phase.WaitAct, needs_tsumo=False, drawn_tile=13,
        )
        self.assert_window(env.get_observation(0), {"dahai", "kakan"})

    def test_chi_pon_daiminkan_and_pass_response_windows(self) -> None:
        # Player 0 discards 1m; player 1 has three 1m, so Pon/Daiminkan/Pass are all legal.
        env = helper_setup_env(
            hands=[
                [0, 4, 8, 12, 16, 20, 24, 36, 40, 44, 48, 52, 56],
                [1, 2, 3, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42], [], [],
            ],
            current_player=0, active_players=[0], drawn_tile=108, wall=list(range(136)),
        )
        responses = env.step({0: Action(ActionType.DISCARD, tile=0)})
        self.assert_window(responses[1], {"none", "pon", "daiminkan"})

        # A 5m discard gives the shimocha all three chi shapes.
        env = helper_setup_env(
            hands=[
                [0, 4, 28, 32, 52, 56, 60, 64, 68, 92, 96, 100, 17],
                [8, 12, 20, 24, 36, 40, 44, 48, 72, 76, 80, 84, 109], [], [],
            ],
            current_player=0, active_players=[0], drawn_tile=108, wall=list(range(136)),
        )
        responses = env.step({0: Action(ActionType.DISCARD, tile=17)})
        self.assert_window(responses[1], {"none", "chi"})

    def test_ron_and_tsumo_hora_windows(self) -> None:
        env = RiichiEnv(seed=42, game_mode="4p-red-half")
        env.reset()
        hands = env.hands
        hands[0] = sorted([124, 125, 0, 4, 8, 5, 9, 12, 16, 20, 24, 14, 15])
        hands[1].append(126)
        env.hands = hands
        env.current_player = 1
        env.phase = Phase.WaitAct
        env.active_players = [1]
        responses = env.step({1: Action(ActionType.DISCARD, tile=126)})
        self.assert_window(responses[0], {"hora"})

        env = RiichiEnv(seed=42, game_mode="4p-red-half")
        env.reset()
        hands = env.hands
        hands[0] = sorted([124, 125, 126, 0, 4, 8, 5, 9, 13, 16, 20, 24, 12])
        env.hands = hands
        env.drawn_tile = 14
        env.current_player = 0
        env.active_players = [0]
        env.is_first_turn = False
        discards = env.discards
        discards[0].append(0)
        env.discards = discards
        self.assert_window(env.get_observation(0), {"hora"})
