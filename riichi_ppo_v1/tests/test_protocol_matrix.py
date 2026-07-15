"""Public Python API tests for every four-player MJAI event accepted by the state machine."""

from __future__ import annotations

import json
import unittest

import numpy as np

try:
    import riichi
except ImportError:  # pragma: no cover
    riichi = None

from riichi_ppo_v1.validation import TOKEN_BY_EVENT, fixture_snapshot


def start_kyoku() -> dict:
    return {
        "type": "start_kyoku", "bakaze": "E", "dora_marker": "2p", "kyoku": 1,
        "honba": 0, "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
        "tehais": [
            ["1m", "1m", "2m", "3m", "4m", "5mr", "6m", "7p", "8p", "9p", "E", "S", "C"],
            ["5m", "5m", "5m", "1p", "1p", "1p", "1p", "2p", "2p", "2p", "2p", "3p", "3p"],
            ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N"],
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "P", "F", "C", "E"],
        ],
    }


@unittest.skipUnless(riichi is not None, "local riichi extension is not installed")
class ProtocolMatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = riichi.MjaiKyokuStateMachineManager(1)
        self.apply_events([{"type": "start_game", "id": 0}, start_kyoku()])

    def apply_events(self, events: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        return self.manager.apply_events_batch(
            [0], [[[json.dumps(event) for event in events], [], [], []]]
        )

    def apply(self, event: dict, expected_type: int | None) -> np.ndarray:
        end_kyoku, end_game = self.apply_events([event])
        ids, _attention, _lengths, _mask, _history_lengths, _history_generations = self.manager.prepare_decisions(
            [0], [[json.dumps({"type": "none"})]], [fixture_snapshot()]
        )
        tokens = np.asarray(ids, dtype=np.int64)[0]
        if expected_type is None:
            self.assertEqual(bool(end_kyoku[0]), event["type"] == "end_kyoku")
            self.assertEqual(bool(end_game[0]), event["type"] == "end_game")
        else:
            self.assertTrue((tokens[:, 0] == expected_type).any(), (event, tokens.tolist()))
        self.assertEqual(tokens.ndim, 2)
        self.assertEqual(tokens.shape[1], 8)
        return tokens

    def test_all_event_types_and_protocol_branches(self) -> None:
        self.assertGreater(len(self.apply({"type": "tsumo", "actor": 0, "pai": "1m"}, TOKEN_BY_EVENT["tsumo"])), 0)
        cases = [
            ({"type": "tsumo", "actor": 0, "pai": "5mr"}, "tsumo"),
            ({"type": "tsumo", "actor": 1, "pai": "5p"}, "tsumo"),
            ({"type": "dahai", "actor": 2, "pai": "5pr", "tsumogiri": True}, "dahai"),
            ({"type": "dahai", "actor": 3, "pai": "1s", "tsumogiri": False}, "dahai"),
            ({"type": "chi", "actor": 1, "target": 0, "pai": "5m", "consumed": ["3m", "4m"]}, "chi"),
            ({"type": "chi", "actor": 1, "target": 0, "pai": "5m", "consumed": ["4m", "6m"]}, "chi"),
            ({"type": "chi", "actor": 1, "target": 0, "pai": "5m", "consumed": ["6m", "7m"]}, "chi"),
            ({"type": "pon", "actor": 2, "target": 1, "pai": "5m", "consumed": ["5m", "5mr"]}, "pon"),
            ({"type": "daiminkan", "actor": 3, "target": 2, "pai": "E", "consumed": ["E", "E", "E"]}, "daiminkan"),
            ({"type": "ankan", "actor": 0, "consumed": ["1s", "1s", "1s", "1s"]}, "ankan"),
            ({"type": "kakan", "actor": 1, "pai": "5s", "consumed": ["5s", "5s", "5sr"]}, "kakan"),
            ({"type": "dora", "dora_marker": "7s"}, "dora"),
            ({"type": "reach", "actor": 2}, "reach"),
            ({"type": "reach_accepted", "actor": 2}, "reach_accepted"),
            ({"type": "hora", "actor": 0, "target": 0, "deltas": [12000, -4000, -4000, -4000], "ura_markers": ["3m", "5pr"]}, "hora"),
            ({"type": "hora", "actor": 1, "target": 3}, "hora"),
            ({"type": "ryukyoku", "deltas": [1500, 1500, -1500, -1500]}, "ryukyoku"),
            ({"type": "ryukyoku"}, "ryukyoku"),
        ]
        for event, name in cases:
            self.apply(event, TOKEN_BY_EVENT[name])
        self.apply({"type": "end_kyoku"}, None)
        self.apply({"type": "end_game"}, None)

    def test_non_four_player_events_are_rejected(self) -> None:
        for event in ({"type": "kita", "actor": 0, "pai": "N"}, {"type": "nukidora", "actor": 0, "pai": "N"}):
            with self.assertRaises(ValueError, msg=str(event)):
                self.apply_events([event])

    def test_decision_history_contains_all_confirmed_kyoku_events(self) -> None:
        def prepared() -> tuple[np.ndarray, int]:
            ids, _attention, _lengths, _mask, history_lengths, _history_generations = self.manager.prepare_decisions(
                [0], [[json.dumps({"type": "none"})]], [fixture_snapshot()]
            )
            return np.asarray(ids, dtype=np.int64)[0], int(np.asarray(history_lengths)[0])

        _initial, initial_history = prepared()
        self.apply_events([{"type": "tsumo", "actor": 0, "pai": "1m"}])
        _after_draw, draw_history = prepared()
        self.apply_events([{"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True}])
        ids, discard_history = prepared()

        self.assertLess(initial_history, draw_history)
        self.assertLess(draw_history, discard_history)
        history_types = ids[:discard_history, 0].tolist()
        self.assertIn(TOKEN_BY_EVENT["tsumo"], history_types)
        self.assertIn(TOKEN_BY_EVENT["dahai"], history_types)
