"""Exhaustively validate all 241 fixed action slots through the public binding."""

from __future__ import annotations

import json
import unittest

try:
    import riichi
except ImportError:  # pragma: no cover
    riichi = None

from riichi_ppo_v1.model.validation import all_action_templates, assert_full_action_space, canonical
from riichi_ppo_v1.model.encoding_protocol import ACTION_TYPE_CODES, SUPPLIER_REQUIRED_ACTION_TYPES


@unittest.skipUnless(riichi is not None, "local riichi extension is not installed")
class ActionSpaceExhaustiveTest(unittest.TestCase):
    def test_query_action_types_lock_supplier_domains(self) -> None:
        self.assertNotEqual(ACTION_TYPE_CODES["tsumo"], ACTION_TYPE_CODES["ron"])
        self.assertEqual(
            SUPPLIER_REQUIRED_ACTION_TYPES,
            {ACTION_TYPE_CODES[name] for name in ("chi", "pon", "daiminkan", "ron")},
        )

    def test_every_fixed_action_id_roundtrips(self) -> None:
        assert_full_action_space(riichi.MjaiKyokuStateMachineManager(1))

    def test_action_segment_boundaries_keep_their_documented_templates(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(1)
        templates = all_action_templates()
        manager.prepare_decisions(
            [0], [[json.dumps(template, separators=(",", ":")) for template in templates]],
        )
        # First/last slots of every fixed-action segment.  The exhaustive test
        # above proves all ids; this makes an accidental range shift readable.
        for action_id in (0, 1, 74, 75, 76, 132, 133, 169, 170, 171, 204, 205, 238, 239, 240):
            self.assertEqual(canonical(manager.decode_actions([0], [action_id])[0]), canonical(templates[action_id]))

    def test_invalid_templates_are_rejected(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(1)
        invalid = [
            {"type": "kita", "pai": "N"},
            {"type": "dahai", "pai": "?", "tsumogiri": False},
            {"type": "chi", "pai": "5m", "consumed": ["E", "S"]},
            {"type": "chi", "pai": "9m", "consumed": ["1m", "2m"]},
            {"type": "pon", "pai": "1m", "consumed": ["1m", "2m"]},
            {"type": "daiminkan"},
            {"type": "daiminkan", "pai": "1m", "consumed": ["1m", "1m", "2m"]},
            {"type": "ankan", "consumed": ["1m", "1m", "1m"]},
            {"type": "ankan", "consumed": ["1m", "2m", "3m", "4m"]},
            {"type": "kakan", "pai": "1m", "consumed": ["1m", "2m", "3m"]},
            {"type": "unknown"},
        ]
        for action in invalid:
            with self.assertRaises(ValueError, msg=str(action)):
                manager.prepare_decisions([0], [[json.dumps(action)]])

    def test_distinct_actions_cannot_silently_share_one_id(self) -> None:
        manager = riichi.MjaiKyokuStateMachineManager(1)
        collisions = [
            [
                {"type": "chi", "pai": "3m", "consumed": ["4m", "5m"]},
                {"type": "chi", "pai": "6m", "consumed": ["4m", "5m"]},
            ],
            [
                {"type": "daiminkan", "pai": "5m", "consumed": ["5m", "5m", "5m"]},
                {"type": "daiminkan", "pai": "5m", "consumed": ["5mr", "5m", "5m"]},
            ],
        ]
        for actions in collisions:
            with self.assertRaises(ValueError, msg=str(actions)):
                manager.prepare_decisions([0], [[json.dumps(action) for action in actions]])
