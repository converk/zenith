from riichi_ppo_v1.training.train import is_better_kyoku_selection


def _window(score: float, deal_in: float, shanten: float) -> dict[str, float]:
    return {
        "eval/kyoku/point_delta_mean": score,
        "eval/kyoku/deal_in_rate": deal_in,
        "eval/efficiency/optimal_shanten_rate": shanten,
    }


def test_kyoku_selection_prioritizes_score_then_defense_then_efficiency() -> None:
    best = _window(0.50, 0.12, 0.80)

    assert is_better_kyoku_selection(_window(0.51, 0.99, 0.00), best)
    assert not is_better_kyoku_selection(_window(0.49, 0.00, 1.00), best)
    assert is_better_kyoku_selection(_window(0.5005, 0.11, 0.70), best)
    assert is_better_kyoku_selection(_window(0.5005, 0.12, 0.81), best)
    assert not is_better_kyoku_selection(_window(0.5005, 0.12, 0.79), best)
    assert is_better_kyoku_selection(best, None)
