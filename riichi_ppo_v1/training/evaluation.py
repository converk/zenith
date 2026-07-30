"""Configuration and metric adapters for periodic PPO evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from ..sft.evaluation_cases import merge_evaluation_summaries


def should_run_evaluation(config: Mapping[str, Any], update: int) -> bool:
    """Return whether a completed PPO update should run the fixed evaluation."""
    if not bool(config.get("evaluation_enabled", False)):
        return False
    interval = max(1, int(config.get("evaluation_interval_updates", 15)))
    return int(update) > 0 and int(update) % interval == 0


def evaluation_shards(
    hanchan_count: int,
    actor_count: int,
) -> list[tuple[int, int]]:
    """Return contiguous eight-game-balanced ``(offset, count)`` shards."""
    total = int(hanchan_count)
    actors = int(actor_count)
    if total <= 0 or total % 8:
        raise ValueError("evaluation_hanchan_count must be a positive multiple of 8")
    if actors <= 0:
        raise ValueError("actor_count must be positive")
    blocks = total // 8
    active_actors = min(actors, blocks)
    base, remainder = divmod(blocks, active_actors)
    result: list[tuple[int, int]] = []
    offset = 0
    for rank in range(active_actors):
        count = (base + (1 if rank < remainder else 0)) * 8
        result.append((offset, count))
        offset += count
    return result


def heuristic_evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Map PPO evaluation settings onto the shared SFT evaluator contract."""
    result = dict(config)
    mappings = {
        "evaluation_hanchan_count": "heuristic_evaluation_hanchan_count",
        "evaluation_parallel_hanchan_count": "heuristic_evaluation_parallel_hanchan_count",
        "evaluation_seed_base": "heuristic_evaluation_seed_base",
        "evaluation_game_mode": "heuristic_evaluation_game_mode",
        "evaluation_max_steps": "heuristic_evaluation_max_steps",
        "evaluation_cache_capacity": "heuristic_evaluation_cache_capacity",
    }
    for source, destination in mappings.items():
        if source in config:
            result[destination] = config[source]
    return result


def ppo_evaluation_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Rename shared evaluator metrics into the stable PPO ``eval/`` namespace."""
    prefix = "heuristic_eval/"
    return {
        f"eval/{name.removeprefix(prefix)}": float(value)
        for name, value in metrics.items()
        if name.startswith(prefix)
    }


def merge_ppo_evaluation_summaries(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    """Merge concurrent evaluation shards and retain wall-clock throughput."""
    merged = merge_evaluation_summaries(rows)
    if not rows:
        return merged

    def weighted(metric: str, weight: str) -> None:
        values = [
            (float(row[metric]), float(row.get(weight, 0.0)))
            for row in rows
            if metric in row
        ]
        total = sum(item_weight for _value, item_weight in values)
        if total:
            merged[metric] = sum(
                value * item_weight for value, item_weight in values
            ) / total

    weighted(
        "eval/action/riichi_opportunity_accept_rate",
        "eval/action/riichi_opportunity_count",
    )
    weighted(
        "eval/action/call_opportunity_accept_rate",
        "eval/action/call_opportunity_count",
    )
    for metric in (
        "eval/efficiency/optimal_shanten_rate",
        "eval/efficiency/optimal_ukeire_rate",
    ):
        weighted(metric, "eval/action/decision_count")

    elapsed = max(
        float(row.get("eval/performance/elapsed_s", 0.0))
        for row in rows
    )
    hanchans = sum(float(row.get("eval/match/count", 0.0)) for row in rows)
    merged["eval/performance/elapsed_s"] = elapsed
    merged["eval/performance/hanchan_per_s"] = hanchans / max(elapsed, 1e-9)
    return merged
