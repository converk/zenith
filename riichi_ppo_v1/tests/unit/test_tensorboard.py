from __future__ import annotations

from riichi_ppo_v1.training.tensorboard import (
    CURATED_SCALAR_TAGS,
    TENSORBOARD_DISPLAY_TAGS,
    curated_scalars,
    eval_summary_scalars,
    learner_peak_allocated_mb,
)


def test_curated_scalars_keep_only_the_dashboard_allowlist() -> None:
    values = {name: float(index) for index, name in enumerate(CURATED_SCALAR_TAGS, start=1)}
    values.update({"ppo/loss": 99.0, "rollout/reward_mean": 2.0, "iteration/sps": float("nan")})
    assert set(curated_scalars(values)) == CURATED_SCALAR_TAGS - {"iteration/sps"}


def test_display_tags_match_curated_metrics() -> None:
    assert set(TENSORBOARD_DISPLAY_TAGS) == CURATED_SCALAR_TAGS


def test_fixed_evaluation_dashboard_includes_result_and_style_health() -> None:
    required = {
        "eval/match/mean_rank",
        "eval/match/first_place_rate",
        "eval/match/last_place_rate",
        "eval/match/point_delta_mean",
        "eval/kyoku/win_rate",
        "eval/kyoku/deal_in_rate",
        "eval/kyoku/deal_in_points_mean",
        "eval/kyoku/tsumo_loss_rate",
        "eval/efficiency/optimal_shanten_rate",
        "eval/efficiency/optimal_ukeire_rate",
        "eval/action/riichi_opportunity_accept_rate",
        "eval/action/call_opportunity_accept_rate",
    }
    assert required <= CURATED_SCALAR_TAGS


def test_learner_peak_allocated_mb_uses_the_largest_rank_peak() -> None:
    assert learner_peak_allocated_mb([
        {"gpu/torch_memory_peak_allocated_mb": 320.0},
        {"gpu/torch_memory_peak_allocated_mb": 512.0},
        {"gpu/torch_memory_peak_allocated_mb": float("nan")},
    ]) == 512.0
    assert learner_peak_allocated_mb([{}, {"gpu/torch_memory_allocated_mb": 10.0}]) is None


def test_eval_summary_scalars_projects_model_a_to_eval_prefix() -> None:
    summary = {
        "elapsed_s": 12.5,
        "model_a": {
            "semantic_metrics": {
                "model_a/match/first_place_rate": 0.25,
                "model_a/match/mean_rank": 2.7,
                "model_a/kyoku/win_rate": 0.18,
                "model_a/kyoku/deal_in_rate": 0.12,
            },
        },
        "model_b": {"checkpoint": "/tmp/b.pt"},
    }
    scalars = eval_summary_scalars(summary)
    assert scalars == {
        "eval/match/first_place_rate": 0.25,
        "eval/match/mean_rank": 2.7,
        "eval/kyoku/win_rate": 0.18,
        "eval/kyoku/deal_in_rate": 0.12,
        "eval/performance/elapsed_s": 12.5,
    }
    # 非 model_a 前缀的键与缺省摘要安全跳过。
    assert eval_summary_scalars({}) == {}
    assert eval_summary_scalars({"model_a": {"semantic_metrics": {}}}) == {}
