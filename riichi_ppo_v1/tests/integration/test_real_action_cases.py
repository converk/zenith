"""Controlled RiichiEnv legal windows must map and decode through the 241-space."""

from __future__ import annotations

import unittest

try:
    import riichi
    from riichienv import Action, ActionType, Meld, MeldType, Phase, RiichiEnv
    from RiichiEnv.tests.env.helper import helper_setup_env
except ImportError:  # pragma: no cover
    riichi = None

from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision, action_jsons
from riichi_ppo_v1.model.validation import assert_observation_roundtrip
from riichi_ppo_v1.training.opponents.heuristic import HeuristicPolicy
from riichi_ppo_v1.training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)


@unittest.skipUnless(riichi is not None, "local RiichiEnv and riichi extensions are not installed")
class RealActionCasesTest(unittest.TestCase):
    def assert_window(self, observation, required: set[str]) -> None:
        bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
        observed = {__import__("json").loads(value)["type"] for value in action_jsons(observation)}
        self.assertTrue(required <= observed, (required, observed))
        assert_observation_roundtrip(bridge, 0, int(observation.player_id), observation)

    @staticmethod
    def heuristic_action(observation, public=None, *, defensive=False):
        public = public or PublicStateTracker(1)
        analyzer = EfficiencyAnalyzer()
        decision = Decision(0, int(observation.player_id), observation)
        analysis = DecisionAnalysisBatch.build(
            [decision], analyzer=analyzer, public=public,
        )
        return HeuristicPolicy(
            analyzer, public, defensive=defensive,
        ).select_batch([decision], analysis)[0]

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

    def test_red_five_choices_and_kuikae_discard_mask(self) -> None:
        # P0 discards 3m. P1 may chi with either a normal or red 5m; both
        # templates must remain distinct 241-space actions and round-trip.
        env = helper_setup_env(
            hands=[
                [8, 0, 4, 28, 32, 52, 56, 60, 64, 68, 92, 96, 100],
                [12, 16, 17, 20, 24, 36, 40, 44, 48, 72, 76, 80, 84],
                [], [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=8,
            wall=list(range(136)),
        )
        responses = env.step({0: Action(ActionType.DISCARD, tile=8)})
        chi_observation = responses[1]
        self.assert_window(chi_observation, {"none", "chi"})
        chi_actions = [action for action in chi_observation.legal_actions() if action.action_type == ActionType.CHI]
        self.assertGreaterEqual(len(chi_actions), 2)
        self.assertTrue(any(16 in action.consume_tiles for action in chi_actions))
        self.assertTrue(any(17 in action.consume_tiles for action in chi_actions))

        # After choosing 4m+5m for 3m, kuikae blocks both 3m and 6m.  The
        # The mask must exactly mirror that restricted post-call observation.
        normal_chi = next(action for action in chi_actions if set(action.consume_tiles) == {12, 17})
        next_observations = env.step({1: normal_chi})
        post_chi = next_observations[1]
        self.assert_window(post_chi, {"dahai"})
        forbidden = {8, 20}
        offered = {action.tile for action in post_chi.legal_actions() if action.action_type == ActionType.DISCARD}
        self.assertTrue(forbidden.isdisjoint(offered), (forbidden, offered))

    def test_same_face_hand_cut_and_tsumogiri_are_distinct_semantics(self) -> None:
        # Three ordinary 5m are present and another ordinary 5m was drawn.
        # The two 241 ids must select different physical roles even though
        # they share the same MJAI pai spelling.
        env = helper_setup_env(
            hands=[[0, 4, 17, 18, 36, 40, 44, 48, 72, 76, 80, 84, 108], [], [], []],
            current_player=0,
            active_players=[0],
            drawn_tile=19,
        )
        observation = env.get_observation(0)
        bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
        decision = Decision(0, 0, observation)
        _factors, _numeric, _lengths, mask, _generation, *_critic = bridge.prepare([decision])
        five_m = [value for value in map(__import__("json").loads, action_jsons(observation))
                  if value.get("type") == "dahai" and value.get("pai") == "5m"]
        self.assertEqual({value["tsumogiri"] for value in five_m}, {False, True})
        # TILE37 index 4 is 5m; mode 0/1 is tedashi/tsumogiri.
        self.assertTrue(mask[0, 9])
        self.assertTrue(mask[0, 10])
        hand_cut, tsumogiri = bridge.decode([decision, decision], [9, 10])
        self.assertNotEqual(hand_cut.tile, observation.drawn_tile)
        self.assertEqual(tsumogiri.tile, observation.drawn_tile)

    def test_riichi_stage_and_furiten_action_masks(self) -> None:
        # During the reach declaration stage, ankan is removed even when the
        # pre-reach hand offered it; only the resulting legal set may be set.
        hand = [8, 9, 10, 12, 20, 40, 44, 48, 60, 61, 80, 84, 88]
        env = helper_setup_env(
            hands=[[0] * 13, [0] * 13, [0] * 13, hand],
            current_player=3,
            active_players=[3],
            phase=Phase.WaitAct,
            needs_tsumo=False,
            drawn_tile=11,
            riichi_declared=[False] * 4,
            points=[25000] * 4,
        )
        before = env.get_observations()[3]
        self.assert_window(before, {"dahai", "reach", "ankan"})
        env.riichi_stage = [False, False, False, True]
        after = env.get_observations()[3]
        self.assert_window(after, {"dahai"})
        after_types = {action.action_type for action in after.legal_actions()}
        self.assertNotIn(ActionType.ANKAN, after_types)
        self.assertNotIn(ActionType.RIICHI, after_types)

        # A permanent furiten hand must not expose a ron id.  This combines an
        # explicit negative rule assertion with exact-mask round-tripping.
        env = RiichiEnv(seed=42, game_mode="4p-red-half")
        env.reset()
        hands = env.hands
        hands[0] = [4, 8, 52, 53, 60, 61, 62, 92, 93, 94, 108, 109, 110]
        hands[1].append(12)
        env.hands = hands
        discards = env.discards
        discards[0].append(0)
        env.discards = discards
        env.current_player = 1
        env.active_players = [1]
        env.phase = Phase.WaitAct
        responses = env.step({1: Action(ActionType.DISCARD, tile=12)})
        self.assertNotIn(0, responses)
        for seat, observation in responses.items():
            self.assert_window(observation, {"none"})

    def test_heuristic_declines_speculative_real_kans(self) -> None:
        # The draw exposes both reach and ankan.  The heuristic may reach or
        # discard, but must no longer kan merely because the action is legal.
        env = helper_setup_env(
            hands=[[0, 1, 2, 4, 8, 12, 16, 20, 36, 40, 44, 108, 112], [], [], []],
            current_player=0,
            active_players=[0],
            drawn_tile=3,
            wall=list(range(136)),
        )
        action = self.heuristic_action(env.get_observation(0))
        self.assertNotEqual(__import__("json").loads(action.to_mjai())["type"], "ankan")

        # This real kakan worsens an inexpensive shape and is declined.
        env = helper_setup_env(
            hands=[[3, 4, 8, 12, 36, 40, 44, 72, 76, 108], [], [], []],
            melds=[[Meld(MeldType.Pon, tiles=[0, 1, 2], opened=True)], [], [], []],
            active_players=[0],
            current_player=0,
            phase=Phase.WaitAct,
            needs_tsumo=False,
            drawn_tile=13,
        )
        action = self.heuristic_action(env.get_observation(0))
        self.assertEqual(__import__("json").loads(action.to_mjai())["type"], "dahai")

    def test_defensive_heuristic_passes_real_call_window_under_multiple_riichi(self) -> None:
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
        response = env.step({0: Action(ActionType.DISCARD, tile=0)})[1]
        public = PublicStateTracker(1)
        public.riichi[0, 2:] = True
        action = self.heuristic_action(response, public, defensive=True)
        self.assertEqual(__import__("json").loads(action.to_mjai())["type"], "none")
