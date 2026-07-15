"""V4 matrix tests for the public Python state-machine API."""

from __future__ import annotations

import json
import unittest

import numpy as np

try:
    import riichi
except ImportError:  # pragma: no cover
    riichi = None

from riichi_ppo_v1.validation import BLOCK_BY_EVENT, fixture_snapshot


def start_kyoku() -> dict:
    return {
        "type": "start_kyoku", "bakaze": "E", "dora_marker": "2p", "kyoku": 1,
        "honba": 0, "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
        "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13],
    }


def snapshot_for(player_id: int) -> str:
    data = json.loads(fixture_snapshot())
    data["player_id"] = player_id
    return json.dumps(data, separators=(",", ":"))


@unittest.skipUnless(riichi is not None, "local riichi extension is not installed")
class ProtocolMatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = riichi.MjaiKyokuStateMachineManager(1)
        self.apply_events([{"type": "start_game", "id": 0}, start_kyoku()])

    def apply_events(self, events: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        return self.manager.apply_events_batch([0], [[[json.dumps(event) for event in events], [], [], []]])

    def prepare(self):
        return self.manager.prepare_decisions([0], [[json.dumps({"type": "none"})]], [fixture_snapshot()])

    def test_draw_is_board_only_and_discard_flushes_into_v4_history(self) -> None:
        self.apply_events([{"type": "tsumo", "actor": 0, "pai": "1m"}])
        kinds, _turn, _meld, board, lengths, _mask, _generation = self.prepare()
        self.assertEqual(int(np.asarray(lengths)[0]), 0)
        self.assertEqual(np.asarray(board).shape, (1, 12, 160))
        # A following meld flushes the pending discard then appends its own block.
        self.apply_events([{"type": "dahai", "actor": 2, "pai": "3p", "tsumogiri": True}, {"type": "pon", "actor": 3, "target": 2, "pai": "3p", "consumed": ["3p", "3p"]}])
        kinds, _turn, meld, _board, lengths, _mask, _generation = self.prepare()
        self.assertEqual(int(np.asarray(lengths)[0]), 2)
        self.assertEqual(np.asarray(kinds)[0, :2].tolist(), [1, 2])
        self.assertEqual(np.asarray(meld)[0, 1, :6].tolist(), [2, 4, 3, 12, 12, 12])

    def test_supported_events_have_the_expected_block_class(self) -> None:
        for event_type, expected in BLOCK_BY_EVENT.items():
            if event_type == "dahai":
                event = {"type": event_type, "actor": 0, "pai": "1m", "tsumogiri": False}
            elif event_type in {"reach", "reach_accepted"}:
                event = {"type": event_type, "actor": 0}
            elif event_type == "dora":
                event = {"type": event_type, "dora_marker": "1m"}
            elif event_type == "chi":
                event = {"type": event_type, "actor": 1, "target": 0, "pai": "3m", "consumed": ["1m", "2m"]}
            elif event_type == "pon":
                event = {"type": event_type, "actor": 1, "target": 0, "pai": "3m", "consumed": ["3m", "3m"]}
            elif event_type == "daiminkan":
                event = {"type": event_type, "actor": 1, "target": 0, "pai": "3m", "consumed": ["3m"] * 3}
            elif event_type == "ankan":
                event = {"type": event_type, "actor": 1, "consumed": ["3m"] * 4}
            else:
                event = {"type": event_type, "actor": 1, "pai": "3m", "consumed": ["3m"] * 3}
            self.apply_events([event])
            if expected == 1:
                self.apply_events([{"type": "ankan", "actor": 1, "consumed": ["4m"] * 4}])
            kinds, *_ = self.prepare()
            self.assertIn(expected, np.asarray(kinds)[0].tolist())

    def test_all_four_views_accept_the_complete_event_lifecycle(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(1)
        lifecycle = [
            start_kyoku(),
            {"type": "tsumo", "actor": 0, "pai": "5mr"},
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
            {"type": "reach_accepted", "actor": 0},
            {"type": "chi", "actor": 1, "target": 0, "pai": "3m", "consumed": ["4m", "5mr"]},
            {"type": "pon", "actor": 2, "target": 1, "pai": "P", "consumed": ["P", "P"]},
            {"type": "daiminkan", "actor": 3, "target": 2, "pai": "9s", "consumed": ["9s"] * 3},
            {"type": "kakan", "actor": 3, "pai": "5pr", "consumed": ["5p", "5p", "5pr"]},
            {"type": "ankan", "actor": 2, "consumed": ["5s", "5s", "5s", "5sr"]},
            {"type": "dora", "dora_marker": "C"},
            {"type": "hora", "actor": 1, "target": 0, "deltas": [-8000, 8000, 0, 0], "ura_markers": ["1m"]},
            {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        rows = []
        for seat in range(4):
            events = [{"type": "start_game", "id": seat}] + lifecycle
            rows.append([json.dumps(event) for event in events])
        end_kyoku, end_game = manager.apply_events_batch([0], [rows])
        self.assertTrue(np.asarray(end_kyoku)[0])
        self.assertTrue(np.asarray(end_game)[0])
        kinds, turn, meld, board, lengths, masks, generations = manager.prepare_decisions(
            [0, 1, 2, 3],
            [[json.dumps({"type": "none"})] for _ in range(4)],
            [snapshot_for(seat) for seat in range(4)],
        )
        self.assertEqual(np.asarray(kinds).shape[0], 4)
        self.assertEqual(np.asarray(turn).shape, (*np.asarray(kinds).shape, 4, 4))
        self.assertEqual(np.asarray(meld).shape, (*np.asarray(kinds).shape, 8))
        self.assertEqual(np.asarray(board).shape, (4, 12, 160))
        self.assertTrue(np.all(np.asarray(lengths) >= 5))
        self.assertTrue(np.all(np.asarray(masks)[:, 0]))
        self.assertTrue(np.all(np.asarray(masks).sum(axis=1) == 1))
        self.assertTrue(np.all(np.asarray(generations) == 1))

    def test_red_called_discard_and_riichi_flag_are_public(self) -> None:
        self.apply_events([
            {"type": "reach", "actor": 1},
            {"type": "dahai", "actor": 1, "pai": "5mr", "tsumogiri": True},
            {"type": "chi", "actor": 2, "target": 1, "pai": "3m", "consumed": ["4m", "5mr"]},
        ])
        _kinds, _turn, _meld, board, _lengths, _mask, _generation = self.prepare()
        board = np.asarray(board)
        # Shimocha's river has red five, tsumogiri, riichi discard and called flags.
        self.assertEqual(int(board[0, 4, 20]), 35)
        self.assertEqual(int(board[0, 4, 52]), 7)
        # Toimen's chi exposes shimocha as source and preserves the red five.
        self.assertEqual(board[0, 8, 20:27].tolist(), [1, 2, 3, 4, 35, 0, 0])

    def test_three_player_only_events_are_rejected(self) -> None:
        for event in ({"type": "kita", "actor": 0, "pai": "N"}, {"type": "nukidora", "actor": 0, "pai": "N"}):
            with self.assertRaises(ValueError, msg=str(event)):
                self.apply_events([event])
