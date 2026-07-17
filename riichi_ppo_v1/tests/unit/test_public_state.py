import json

import numpy as np

from riichi_ppo_v1.training.rewards.public_state import PublicStateTracker


def test_public_tracker_uses_events_not_opponent_hands() -> None:
    tracker = PublicStateTracker(1)
    discard = json.dumps({"type": "dahai", "actor": 1, "pai": "5mr"})
    reach = json.dumps({"type": "reach", "actor": 1})
    tracker.update([[[discard, reach], [discard], [], []]])
    # Duplicate observer events are counted once, and red five folds to 5m.
    assert tracker.visible[0, 4] == 1
    assert tracker.has_riichi_threat(0, 0)
    assert tracker.is_genbutsu_to_all_riichi(0, 4)
    remaining = tracker.remaining(0, np.zeros(34, dtype=np.uint8))
    assert remaining[4] == 3
