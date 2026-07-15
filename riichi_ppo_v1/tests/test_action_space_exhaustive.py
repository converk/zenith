"""Exhaustively validate all 241 fixed V2 action slots through the public binding."""

from __future__ import annotations

import json
import unittest

try:
    import riichi
except ImportError:  # pragma: no cover
    riichi = None

from riichi_ppo_v1.validation import assert_full_action_space, fixture_snapshot


@unittest.skipUnless(riichi is not None, "local riichi extension is not installed")
class ActionSpaceExhaustiveTest(unittest.TestCase):
    def test_every_fixed_action_id_roundtrips(self) -> None:
        assert_full_action_space(riichi.MjaiKyokuStateMachineManager(1))

    def test_invalid_templates_are_rejected(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(1)
        invalid = [
            {"type": "kita", "pai": "N"},
            {"type": "dahai", "pai": "?", "tsumogiri": False},
            {"type": "chi", "pai": "5m", "consumed": ["E", "S"]},
            {"type": "ankan", "consumed": ["1m", "1m", "1m"]},
            {"type": "unknown"},
        ]
        for action in invalid:
            with self.assertRaises(ValueError, msg=str(action)):
                manager.prepare_decisions([0], [[json.dumps(action)]], [fixture_snapshot()])
