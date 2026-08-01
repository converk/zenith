from riichi_ppo_v1.tools.event_statistics import canonical_step_events


def test_event_identity_deduplicates_observers_but_not_steps_or_kyokus() -> None:
    reach = {"type": "reach", "actor": 1}
    observers = [[reach], [reach], [reach], [reach]]
    first = canonical_step_events(
        observers, environment_id=2, hanchan_id=7, kyoku_id=0, step=10,
    )
    later_step = canonical_step_events(
        observers, environment_id=2, hanchan_id=7, kyoku_id=0, step=11,
    )
    later_kyoku = canonical_step_events(
        observers, environment_id=2, hanchan_id=7, kyoku_id=1, step=10,
    )
    assert len(first) == 1
    assert len({first[0][0], later_step[0][0], later_kyoku[0][0]}) == 3


def test_identical_events_within_one_step_keep_real_multiplicity() -> None:
    event = {"type": "hora", "actor": 0, "target": 1}
    rows = canonical_step_events(
        [[event, event], [event, event], [event], [event, event]],
        environment_id=0, hanchan_id=0, kyoku_id=0, step=5,
    )
    assert len(rows) == 2
    assert rows[0][0][-1] == 0
    assert rows[1][0][-1] == 1
