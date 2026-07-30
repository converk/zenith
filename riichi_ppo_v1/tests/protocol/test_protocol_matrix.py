"""Semantic-token matrix tests for the public Python state-machine API."""

from __future__ import annotations

import json
import unittest

import numpy as np

try:
    import riichi
except ImportError:  # pragma: no cover
    riichi = None

from riichi_ppo_v1.model.validation import fixture_snapshot


def start_kyoku(bakaze: str = "E", kyoku: int = 1, oya: int = 0) -> dict:
    return {"type": "start_kyoku", "bakaze": bakaze, "dora_marker": "2p", "kyoku": kyoku,
            "honba": 0, "kyotaku": 0, "oya": oya, "scores": [25000] * 4,
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

    @staticmethod
    def used_rows(factors, lengths, row: int = 0) -> np.ndarray:
        return np.asarray(factors)[row, :int(np.asarray(lengths)[row])]

    @staticmethod
    def relative(observer: int, absolute: int) -> int:
        return 1 + (absolute - observer) % 4

    @staticmethod
    def numeric_features(value: float, field: int) -> np.ndarray:
        periods = (100.0, 1_000.0, 10_000.0, 100_000.0) if field == 1 else (2.0, 8.0, 32.0, 128.0)
        values = []
        for period in periods:
            # Match Rust's f32 `TAU * value / period` evaluation order.
            angle = np.float32(np.float32(np.pi * 2.0) * np.float32(value) / np.float32(period))
            values.extend((np.sin(angle), np.cos(angle)))
        return np.asarray(values)

    def test_draw_is_omitted_but_current_state_is_present(self) -> None:
        self.apply_events([{"type": "tsumo", "actor": 0, "pai": "1m"}])
        factors, numeric, lengths, mask, generations = self.prepare()
        factors, numeric = np.asarray(factors), np.asarray(numeric)
        self.assertEqual(factors.shape[-1], 10)
        self.assertEqual(numeric.shape, (*factors.shape[:2], 8))
        # Start-kyoku event plus non-empty state suffix; no tsumo event kind 3.
        used = self.used_rows(factors, lengths)
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
        used = self.used_rows(factors, lengths)
        discard = used[used[:, 2] == 4][0]
        chi = used[used[:, 2] == 5][0]
        self.assertEqual(discard[3], 2)  # shimocha relative to seat 0
        self.assertEqual(discard[6], 1)  # red five
        self.assertEqual(discard[8], 2)  # tsumogiri
        self.assertEqual(chi[3], 3)      # toimen
        self.assertEqual(chi[6], 0)      # called 3m is non-red

    def test_event_matrix_preserves_public_semantics_for_all_views(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(1)
        events = [
            start_kyoku("S", 4, 1),
            # A hidden opponent draw still consumes one live-wall tile but
            # must not become a tile-bearing history token.
            {"type": "tsumo", "actor": 2, "pai": "?"},
            {"type": "dahai", "actor": 2, "pai": "5mr", "tsumogiri": True},
            {"type": "chi", "actor": 3, "target": 2, "pai": "3p", "consumed": ["4p", "5pr"]},
            {"type": "pon", "actor": 1, "target": 0, "pai": "5s", "consumed": ["5s", "5sr"]},
            {"type": "daiminkan", "actor": 2, "target": 1, "pai": "E", "consumed": ["E", "E", "E"]},
            {"type": "ankan", "actor": 3, "consumed": ["5m", "5m", "5m", "5mr"]},
            {"type": "kakan", "actor": 1, "pai": "5p", "consumed": ["5p", "5p", "5pr"]},
            {"type": "dora", "dora_marker": "5sr"},
            {"type": "reach", "actor": 2},
            {"type": "reach_accepted", "actor": 2},
        ]
        rows = [[json.dumps({"type": "start_game", "id": seat})] + [json.dumps(event) for event in events]
                for seat in range(4)]
        manager.apply_events_batch([0], [rows])
        snapshots = [json.dumps({**json.loads(fixture_snapshot()), "player_id": seat, "oya": 1}) for seat in range(4)]
        factors, numeric, lengths, _masks, generations = manager.prepare_decisions(
            [0, 1, 2, 3], [[json.dumps({"type": "none"})] for _ in range(4)], snapshots,
        )
        self.assertTrue(np.all(np.asarray(generations) == 1))

        # (field, actor, source, suit, rank, red, detail).  Start/dora have
        # no actor; ankan/kakan use SELF as their fixed source.
        expected = [
            (2, None, None, 0, 0, 0, 8),
            (4, 2, None, 1, 5, 1, 2),
            (5, 3, 2, 2, 3, 0, 2),
            (6, 1, 0, 3, 5, 0, 2),
            (7, 2, 1, 4, 1, 0, 1),
            (8, 3, "self", 1, 5, 0, 2),
            (9, 1, "self", 2, 5, 0, 2),
            (10, None, None, 3, 5, 1, 0),
            (11, 2, None, 0, 0, 0, 0),
            (12, 2, None, 0, 0, 0, 0),
        ]
        for observer in range(4):
            used = self.used_rows(factors, lengths, observer)
            history = used[(used[:, 0] == 1) & (used[:, 1] == 1)]
            self.assertEqual(history.shape[0], len(expected))
            for row, (field, actor, source, suit, rank, red, detail) in zip(history, expected, strict=True):
                self.assertEqual(row[2], field)
                self.assertEqual(row[3], 0 if actor is None else self.relative(observer, actor))
                self.assertEqual(row[4:7].tolist(), [suit, rank, red])
                expected_source = 1 if source == "self" else (0 if source is None else self.relative(observer, source))
                self.assertEqual(row[7], expected_source)
                self.assertEqual(row[8], detail)
                self.assertEqual(row[9], 1)

            # The one tsumo above changes only the live-wall counter from 70
            # to 69. Counter numeric features use the field-2 periods.
            live_wall_index = np.flatnonzero((used[:, 0] == 2) & (used[:, 1] == 3) & (used[:, 2] == 5))[0]
            self.assertEqual(used[live_wall_index].tolist(), [2, 3, 5, 0, 0, 0, 0, 0, 0, 0])
            np.testing.assert_allclose(
                np.asarray(numeric)[observer, live_wall_index], self.numeric_features(69.0, 2), atol=1e-6,
            )

    def test_snapshot_suffix_preserves_visible_state_and_hides_opponents(self) -> None:
        snapshot = {
            **json.loads(fixture_snapshot()),
            "oya": 1,
            "round_wind": 2,
            "kyoku_index": 3,
            "honba": 4,
            "riichi_sticks": 5,
            "scores": [12_345, 23_456, 34_567, 45_678],
            "dora_indicators": ["5mr", "C"],
            "hand": ["5mr", "5m", "5m", "E"],
            "drawn_tile": "5pr",
            "riichi_declared": [True, False, False, False],
            "decision_flags": 1,
        }
        factors, numeric, lengths, _mask, _generation = self.manager.prepare_decisions(
            [0], [[json.dumps({"type": "none"})]], [json.dumps(snapshot)],
        )
        used = self.used_rows(factors, lengths)
        numeric = np.asarray(numeric)[0, :len(used)]

        score_rows = np.flatnonzero((used[:, 0] == 2) & (used[:, 1] == 2) & (used[:, 2] == 1))
        self.assertEqual(used[score_rows, 3].tolist(), [1, 2, 3, 4])
        for row, score in zip(score_rows, snapshot["scores"], strict=True):
            np.testing.assert_allclose(numeric[row], self.numeric_features(float(score), 1), atol=1e-6)

        counter_indices = {
            int(row[2]): index for index, row in enumerate(used) if row[0] == 2 and row[1] == 3
        }
        counters = {field: used[index] for field, index in counter_indices.items()}
        self.assertEqual(counters[6].tolist(), [2, 3, 6, 2, 0, 0, 0, 0, 0, 0])  # dealer is shimocha
        self.assertEqual(counters[7].tolist(), [2, 3, 7, 1, 0, 0, 0, 4, 0, 0])  # north self-wind
        self.assertEqual(counters[8][8], 0b111)  # riichi, drawn tile, active decision
        for field, value in ((1, 2), (2, 4), (3, 4), (4, 5), (5, 70)):
            np.testing.assert_allclose(numeric[counter_indices[field]], self.numeric_features(float(value), 2), atol=1e-6)

        hand_rows = used[(used[:, 0] == 2) & (used[:, 1] == 4) & (used[:, 2] == 1)]
        self.assertTrue(any(row.tolist() == [2, 4, 1, 1, 1, 5, 1, 3, 0, 1] for row in hand_rows))
        self.assertTrue(any(row.tolist() == [2, 4, 1, 1, 4, 1, 0, 1, 0, 1] for row in hand_rows))
        self.assertTrue(any(row.tolist() == [2, 4, 3, 0, 1, 5, 1, 0, 0, 1] for row in used))
        self.assertTrue(any(row.tolist() == [2, 4, 5, 1, 2, 5, 1, 1, 0, 1] for row in used))
        self.assertFalse(np.any((used[:, 0] == 2) & (used[:, 1] == 7)))
        self.assertFalse(np.any(used[:, 9] == 2))

    def test_events_clear_stale_pending_actions_and_new_kyoku_bumps_generation(self) -> None:
        self.prepare()
        self.assertEqual(json.loads(self.manager.decode_actions([0], [0])[0]), {"type": "none"})
        self.apply_events([{"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False}])
        with self.assertRaises(ValueError):
            self.manager.decode_actions([0], [0])

        self.apply_events([start_kyoku("W", 3, 0)])
        factors, _numeric, lengths, _mask, generations = self.prepare()
        history = self.used_rows(factors, lengths)
        history = history[(history[:, 0] == 1) & (history[:, 1] == 1)]
        self.assertEqual(history[:, 2].tolist(), [2])
        self.assertEqual(int(np.asarray(generations)[0]), 2)

    def test_all_four_views_accept_lifecycle_and_keep_terminal_events_out(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(1)
        lifecycle = [start_kyoku(), {"type": "tsumo", "actor": 0, "pai": "5mr"},
            {"type": "reach", "actor": 0}, {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
            {"type": "reach_accepted", "actor": 0}, {"type": "pon", "actor": 2, "target": 1, "pai": "P", "consumed": ["P", "P"]},
            {"type": "dora", "dora_marker": "C"}, {"type": "hora", "actor": 1, "target": 0}, {"type": "ryukyoku"},
            {"type": "none"}, {"type": "end_kyoku"}, {"type": "end_game"}]
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
        for row in range(4):
            history = self.used_rows(factors, lengths, row)
            history = history[(history[:, 0] == 1) & (history[:, 1] == 1)]
            self.assertEqual(history[:, 2].tolist(), [2, 11, 4, 12, 6, 10])

    def test_three_player_only_events_are_rejected(self) -> None:
        for event in ({"type": "kita", "actor": 0, "pai": "N"}, {"type": "nukidora", "actor": 0, "pai": "N"}):
            with self.assertRaises(ValueError, msg=str(event)):
                self.apply_events([event])
