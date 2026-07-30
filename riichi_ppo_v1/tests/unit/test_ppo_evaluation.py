from riichi_ppo_v1.sft.evaluation_cases import evaluation_cases
from riichi_ppo_v1.training.evaluation import (
    evaluation_shards,
    heuristic_evaluation_config,
    merge_ppo_evaluation_summaries,
    ppo_evaluation_metrics,
    should_run_evaluation,
)


def test_periodic_evaluation_runs_only_on_interval_boundaries() -> None:
    config = {"evaluation_enabled": True, "evaluation_interval_updates": 15}
    assert not should_run_evaluation(config, 0)
    assert not should_run_evaluation(config, 14)
    assert should_run_evaluation(config, 15)
    assert should_run_evaluation(config, 30)
    assert not should_run_evaluation(
        {"evaluation_enabled": False, "evaluation_interval_updates": 15},
        15,
    )


def test_ppo_settings_map_to_the_shared_heuristic_evaluator() -> None:
    mapped = heuristic_evaluation_config({
        "evaluation_hanchan_count": 96,
        "evaluation_parallel_hanchan_count": 48,
        "evaluation_seed_base": 123,
        "evaluation_game_mode": "4p-red-half",
        "evaluation_max_steps": 4000,
        "inference_dtype": "bf16",
    })
    assert mapped["heuristic_evaluation_hanchan_count"] == 96
    assert mapped["heuristic_evaluation_parallel_hanchan_count"] == 48
    assert mapped["heuristic_evaluation_seed_base"] == 123
    assert mapped["heuristic_evaluation_game_mode"] == "4p-red-half"
    assert mapped["heuristic_evaluation_max_steps"] == 4000
    assert mapped["inference_dtype"] == "bf16"


def test_shared_metrics_use_the_ppo_eval_namespace() -> None:
    assert ppo_evaluation_metrics({
        "heuristic_eval/match/mean_rank": 2.25,
        "heuristic_eval/kyoku/win_rate": 0.2,
        "unrelated": 1.0,
    }) == {
        "eval/match/mean_rank": 2.25,
        "eval/kyoku/win_rate": 0.2,
    }


def test_96_hanchans_split_into_two_balanced_gpu_shards() -> None:
    assert evaluation_shards(96, 2) == [(0, 48), (48, 48)]


def test_shards_reconstruct_the_exact_global_seed_seat_and_opponent_schedule() -> None:
    seed_base = 20260717
    combined = []
    for offset, count in evaluation_shards(96, 2):
        combined.extend(evaluation_cases(seed_base + offset, count, cycle=0))
    assert combined == evaluation_cases(seed_base, 96, cycle=0)


def test_shards_require_complete_seat_and_opponent_cycles() -> None:
    import pytest

    with pytest.raises(ValueError):
        evaluation_shards(92, 2)


def test_concurrent_shard_merge_uses_weights_and_wall_clock() -> None:
    merged = merge_ppo_evaluation_summaries([
        {
            "eval/match/count": 48.0,
            "eval/match/mean_rank": 2.0,
            "eval/action/riichi_opportunity_accept_rate": 0.25,
            "eval/action/riichi_opportunity_count": 20.0,
            "eval/performance/elapsed_s": 50.0,
            "eval/performance/hanchan_per_s": 0.96,
        },
        {
            "eval/match/count": 48.0,
            "eval/match/mean_rank": 3.0,
            "eval/action/riichi_opportunity_accept_rate": 0.75,
            "eval/action/riichi_opportunity_count": 60.0,
            "eval/performance/elapsed_s": 60.0,
            "eval/performance/hanchan_per_s": 0.8,
        },
    ])
    assert merged["eval/match/count"] == 96.0
    assert merged["eval/match/mean_rank"] == 2.5
    assert merged["eval/action/riichi_opportunity_accept_rate"] == 0.625
    assert merged["eval/performance/elapsed_s"] == 60.0
    assert merged["eval/performance/hanchan_per_s"] == 1.6
