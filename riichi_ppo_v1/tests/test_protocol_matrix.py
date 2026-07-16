"""V5 matrix tests for the public Python state-machine API."""

from __future__ import annotations

import json
import unittest

import numpy as np

try:
    import riichi
except ImportError:  # pragma: no cover
    riichi = None

from riichi_ppo_v1.validation import fixture_snapshot


def start_kyoku() -> dict:
    return {"type": "start_kyoku", "bakaze": "E", "dora_marker": "2p", "kyoku": 1,
            "honba": 0, "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
            "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13]}


@unittest.skipUnless(riichi is not None, "local riichi extension is not installed")
class ProtocolMatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = riichi.MjaiKyokuStateMachineManager(1)
        self.apply_events([{"type": "start_game", "id": 0}, start_kyoku()])

    def apply_events(self, events: list[dict]) -> None:
        self.manager.apply_events_batch([0], [[[json.dumps(event) for event in events], [], [], []]])

    def prepare(self):
        return self.manager.prepare_decisions([0], [[json.dumps({"type": "none"})]], [fixture_snapshot()])

    def test_draw_is_omitted_but_current_state_is_present(self) -> None:
        self.apply_events([{"type": "tsumo", "actor": 0, "pai": "1m"}])
        factors, numeric, lengths, mask, generations = self.prepare()
        factors, numeric = np.asarray(factors), np.asarray(numeric)
        self.assertEqual(factors.shape[-1], 10)
        self.assertEqual(numeric.shape, (*factors.shape[:2], 8))
        # Start-kyoku event plus non-empty state suffix; no tsumo event kind 3.
        used = factors[0, :int(np.asarray(lengths)[0])]
        events = used[(used[:, 0] == 1) & (used[:, 1] == 1)]
        self.assertIn(2, events[:, 2])
        self.assertNotIn(3, events[:, 2])
        self.assertTrue(np.asarray(mask)[0, 0])
        self.assertEqual(int(np.asarray(generations)[0]), 1)

    def test_public_events_have_relative_actor_red_and_tsumogiri_factors(self) -> None:
        self.apply_events([
            {"type": "reach", "actor": 1},
            {"type": "dahai", "actor": 1, "pai": "5mr", "tsumogiri": True},
            {"type": "chi", "actor": 2, "target": 1, "pai": "3m", "consumed": ["4m", "5mr"]},
        ])
        factors, _numeric, lengths, _mask, _generation = self.prepare()
        used = np.asarray(factors)[0, :int(np.asarray(lengths)[0])]
        discard = used[used[:, 2] == 4][0]
        chi = used[used[:, 2] == 5][0]
        self.assertEqual(discard[3], 2)  # shimocha relative to seat 0
        self.assertEqual(discard[6], 1)  # red five
        self.assertEqual(discard[8], 2)  # tsumogiri
        self.assertEqual(chi[3], 3)      # toimen
        self.assertEqual(chi[6], 0)      # called 3m is non-red

    def test_all_four_views_accept_lifecycle_and_keep_terminal_events_out(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(1)
        lifecycle = [start_kyoku(), {"type": "tsumo", "actor": 0, "pai": "5mr"},
            {"type": "reach", "actor": 0}, {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
            {"type": "reach_accepted", "actor": 0}, {"type": "pon", "actor": 2, "target": 1, "pai": "P", "consumed": ["P", "P"]},
            {"type": "dora", "dora_marker": "C"}, {"type": "hora", "actor": 1, "target": 0}, {"type": "end_kyoku"}, {"type": "end_game"}]
        rows = [[json.dumps({"type": "start_game", "id": seat})] + [json.dumps(event) for event in lifecycle] for seat in range(4)]
        end_kyoku, end_game = manager.apply_events_batch([0], [rows])
        self.assertTrue(np.asarray(end_kyoku)[0])
        self.assertTrue(np.asarray(end_game)[0])
        snapshots = [json.dumps({**json.loads(fixture_snapshot()), "player_id": seat}) for seat in range(4)]
        factors, numeric, lengths, masks, generations = manager.prepare_decisions([0, 1, 2, 3], [[json.dumps({"type": "none"})] for _ in range(4)], snapshots)
        self.assertEqual(np.asarray(factors).shape[0], 4)
        self.assertEqual(np.asarray(numeric).shape[-1], 8)
        self.assertTrue(np.all(np.asarray(lengths) > 1))
        self.assertTrue(np.all(np.asarray(masks)[:, 0]))
        self.assertTrue(np.all(np.asarray(generations) == 1))

    def test_three_player_only_events_are_rejected(self) -> None:
        for event in ({"type": "kita", "actor": 0, "pai": "N"}, {"type": "nukidora", "actor": 0, "pai": "N"}):
            with self.assertRaises(ValueError, msg=str(event)):
                self.apply_events([event])
