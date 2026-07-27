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
    assert tracker.discard_counts.tolist() == [1]
    assert tracker.has_riichi_threat(0, 0)
    assert tracker.is_genbutsu_to_all_riichi(0, 4)
    remaining = tracker.remaining(0, np.zeros(34, dtype=np.uint8))
    assert remaining[4] == 3


def test_public_tracker_counts_open_melds_without_counting_kakan_twice() -> None:
    tracker = PublicStateTracker(1)
    events = [
        json.dumps({"type": "pon", "actor": 0, "pai": "1m", "consumed": ["1m", "1m"]}),
        json.dumps({"type": "chi", "actor": 1, "pai": "3p", "consumed": ["1p", "2p"]}),
        json.dumps({"type": "daiminkan", "actor": 1, "pai": "E", "consumed": ["E", "E", "E"]}),
        json.dumps({"type": "ankan", "actor": 2, "consumed": ["5s"] * 4}),
        json.dumps({"type": "kakan", "actor": 0, "pai": "1m", "consumed": ["1m"]}),
    ]
    tracker.update([[events, [], [], []]])

    assert tracker.open_meld_counts[0].tolist() == [1, 2, 0, 0]
    tracker.update([[[json.dumps({"type": "start_kyoku"})], [], [], []]])
    assert tracker.open_meld_counts[0].tolist() == [0, 0, 0, 0]
    assert tracker.discard_counts.tolist() == [0]


def test_public_tracker_snapshots_completed_kyoku_before_the_next_start() -> None:
    tracker = PublicStateTracker(1)
    events = [
        json.dumps({"type": "dahai", "actor": 0, "pai": "1m"}),
        json.dumps({"type": "pon", "actor": 1, "pai": "E", "consumed": ["E", "E"]}),
        json.dumps({"type": "end_kyoku"}),
        json.dumps({"type": "start_kyoku"}),
    ]
    tracker.update([[events, [], [], []]])

    assert tracker.completed_discard_counts.tolist() == [1]
    assert tracker.completed_open_meld_counts.tolist() == [1]
    assert tracker.discard_counts.tolist() == [0]


def test_claim_events_do_not_count_the_called_discard_twice() -> None:
    tracker = PublicStateTracker(1)
    events = [
        json.dumps({"type": "dahai", "actor": 0, "pai": "3m"}),
        json.dumps({"type": "chi", "actor": 1, "target": 0, "pai": "3m", "consumed": ["1m", "2m"]}),
        json.dumps({"type": "dahai", "actor": 2, "pai": "E"}),
        json.dumps({"type": "daiminkan", "actor": 3, "target": 2, "pai": "E", "consumed": ["E"] * 3}),
    ]
    tracker.update([[events, [], [], []]])
    assert tracker.visible[0, :3].tolist() == [1, 1, 1]
    assert tracker.visible[0, 27] == 4


def test_ankan_and_kakan_add_only_newly_visible_tiles() -> None:
    tracker = PublicStateTracker(1)
    events = [
        json.dumps({"type": "ankan", "actor": 0, "consumed": ["2p"] * 4}),
        json.dumps({"type": "dahai", "actor": 1, "pai": "5s"}),
        json.dumps({"type": "pon", "actor": 0, "target": 1, "pai": "5s", "consumed": ["5s", "5sr"]}),
        json.dumps({"type": "kakan", "actor": 0, "pai": "5s", "consumed": ["5s", "5s", "5sr"]}),
    ]
    tracker.update([[events, [], [], []]])
    assert tracker.visible[0, 10] == 4
    assert tracker.visible[0, 22] == 4


def test_initial_and_added_dora_indicators_are_public_tiles() -> None:
    tracker = PublicStateTracker(1)
    rows = [
        json.dumps({"type": "start_kyoku", "dora_marker": "4p"}),
        json.dumps({"type": "dora", "pai": "N"}),
    ]
    tracker.update([[rows, [], [], []]])
    assert tracker.visible[0, 12] == 1
    assert tracker.visible[0, 30] == 1
