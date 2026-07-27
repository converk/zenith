from riichi_ppo_v1.training.reward_scale import RewardScaleController


def test_reward_controller_stages_and_nonzero_floor() -> None:
    controller = RewardScaleController()
    assert controller.targets(1) == (0.35, 0.15, 1)
    assert controller.targets(1501) == (0.20, 0.10, 2)
    assert controller.targets(3501) == (0.12, 0.05, 3)
    row = {
        "reward_scale/trace_count": 4.0,
        "reward_scale/kyoku_trace_sum_squares": 4.0,
        "reward_scale/discard_trace_sum_squares": 40000.0,
        "reward_scale/call_trace_sum_squares": 40000.0,
    }
    controller.update(4000, [row])
    assert controller.discard_weight == 0.02
    assert controller.call_weight == 0.02


def test_reward_controller_checkpoint_roundtrip() -> None:
    source = RewardScaleController(discard_weight=0.63, call_weight=0.22)
    source.kyoku_rms, source.discard_rms, source.call_rms = 1.2, 0.4, 0.2
    restored = RewardScaleController()
    restored.load_state_dict(source.state_dict())
    assert restored.state_dict() == source.state_dict()
