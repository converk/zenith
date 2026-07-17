from riichi_ppo_v1.training.curriculum import Curriculum, checkpoint_cadence


def test_five_fixed_stages_and_monotonic_rank_schedule() -> None:
    curriculum = Curriculum(1_000)
    assert curriculum.snapshot(0).stage.name == "bootstrap"
    assert curriculum.snapshot(200).stage.name == "heuristic"
    assert curriculum.snapshot(500).stage.name == "history_intro"
    assert curriculum.snapshot(680).stage.name == "history_focus"
    final = curriculum.snapshot(840)
    assert final.stage.name == "rank"
    assert final.weights == (0.05, 0.35, 0.6)
    assert checkpoint_cadence(1_000) == 25
