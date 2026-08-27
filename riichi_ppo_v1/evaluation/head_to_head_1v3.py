"""确定性 1v3 评测:一个候选座位对阵三个对手座位。

候选模型每半庄占一个座位,其余三席均由对手模型贪心出牌;候选座位在半庄之间
轮转四个位置。上报指标包括一位率、平均名次、相对三名对手平均点差,以及基于
每半庄点差的配对 bootstrap 95% 置信区间。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import time
from typing import Any

# Keep the project-facing device convention consistent with the training entry
# points. This must happen before importing torch.
if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch

from ..model.bridge import BatchedStateBridge, NUM_PLAYERS
from ..training.metrics import SemanticMetrics
from ..training.rewards import PublicStateTracker
from ..training.worker import active_decisions
from ..model.action_groups import action_group as _action_group
from .mechanism import DEFAULT_1V3_HANCHANS_PER_PROCESS, TOTAL_1V3_HANCHANS
from .policy_adapter import load_policy_adapter


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device, non_blocking=device.type == "cuda")


@torch.inference_mode()
def _greedy_actions(
    adapter: Any,
    bridge: BatchedStateBridge,
    decisions: list[Any],
    analysis: Any,
    *,
    metrics: SemanticMetrics | None = None,
    public: PublicStateTracker | None = None,
) -> tuple[list[int], list[Any]]:
    prepared = adapter.prepare(bridge, decisions, analysis)
    logits = adapter.masked_logits(prepared)
    action_ids = logits.argmax(-1).tolist()
    if metrics is not None and public is not None:
        for decision, action_id, legal_row in zip(
            decisions, action_ids, prepared.legal, strict=True,
        ):
            metrics.record_decision(
                int(action_id),
                legal_row,
                threat=public.has_riichi_threat(decision.env_index, decision.seat_id),
                prior_riichi_count=int(public.riichi[decision.env_index].sum()),
                seat=decision.seat_id,
            )
    actions = bridge.decode(decisions, action_ids)
    return action_ids, actions


def evaluate_1v3(
    model_a_path: str | Path,
    model_b_path: str | Path,
    *,
    device: str = "cuda",
    model_a_device: str | None = None,
    model_b_device: str | None = None,
    hanchan_count: int = TOTAL_1V3_HANCHANS,
    parallel_hanchans: int = DEFAULT_1V3_HANCHANS_PER_PROCESS,
    seed_base: int = 0,
    game_mode: str = "4p-red-half",
    max_steps: int = 4000,
) -> dict[str, Any]:
    try:
        import riichi
        from riichienv import BatchedRiichiEnv, HandEvaluator
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "install the local riichi and RiichiEnv extensions before evaluation"
        ) from exc

    def final_tenpai(env_index: int, actions_by_env: list[dict[int, Any]]) -> list[bool | None]:
        """Compute each seat's final tenpai before a pending exhaustive draw."""
        flags: list[bool | None] = [None] * NUM_PLAYERS
        observations_by_env = observations[env_index]
        for seat in range(NUM_PLAYERS):
            obs = observations_by_env[seat]
            hands = getattr(obs, "hands", None)
            melds = getattr(obs, "melds", None)
            if hands is None or melds is None:
                continue
            hand = list(hands[seat])
            meld_list = list(melds[seat])
            tile_count = len(hand) + 3 * len(meld_list)
            if tile_count == 13:
                flags[seat] = HandEvaluator(hand, meld_list).is_tenpai()
            elif tile_count == 14:
                action = actions_by_env[env_index].get(seat)
                tile = getattr(action, "tile", None)
                if tile is not None and int(tile) in hand:
                    remaining = list(hand)
                    remaining.remove(int(tile))
                    flags[seat] = HandEvaluator(remaining, meld_list).is_tenpai()
        return flags

    default_device = torch.device(device)
    device_a = torch.device(model_a_device or default_device)
    device_b = torch.device(model_b_device or default_device)
    if (device_a.type == "cuda" or device_b.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    model_a_path = str(Path(model_a_path).resolve())
    model_b_path = str(Path(model_b_path).resolve())
    adapter_a = load_policy_adapter(model_a_path, device=device_a)
    adapter_b = load_policy_adapter(model_b_path, device=device_b)
    metric_a = SemanticMetrics()
    metric_b = SemanticMetrics()

    batch_size = max(1, min(int(parallel_hanchans), int(hanchan_count)))
    started = time.perf_counter()
    first_places = 0
    top2 = 0
    fourths = 0
    rank_sum = 0
    rank_history: list[int] = []
    next_milestone = 100
    point_diffs: list[float] = []
    completed = 0
    seat_counts = Counter()
    action_counts = {"a": Counter(), "b": Counter()}

    for batch_start in range(0, int(hanchan_count), batch_size):
        batch_size_now = min(batch_size, int(hanchan_count) - batch_start)
        envs = BatchedRiichiEnv(
            batch_size_now,
            seed=int(seed_base) + batch_start,
            step_threads=batch_size_now,
            game_mode=game_mode,
        )
        bridge = BatchedStateBridge(
            riichi.MjaiKyokuStateMachineManager(batch_size_now), batch_size_now,
        )
        observations = list(envs.reset())
        bridge.sync(observations)
        public = PublicStateTracker(batch_size_now)
        public.update(bridge.last_events)
        start_scores = [[int(value) for value in row] for row in envs.scores()]
        candidate_seats = [
            (batch_start + env_index) % NUM_PLAYERS
            for env_index in range(batch_size_now)
        ]
        seat_counts.update(candidate_seats)
        active_envs = set(range(batch_size_now))

        for _step in range(int(max_steps)):
            actions_by_env: list[dict[int, Any]] = [{} for _ in range(batch_size_now)]
            decisions = active_decisions(observations, active_envs)
            for policy_name, adapter in (("a", adapter_a), ("b", adapter_b)):
                policy_decisions = [
                    decision for decision in decisions
                    if (decision.seat_id == candidate_seats[decision.env_index])
                    == (policy_name == "a")
                ]
                if not policy_decisions:
                    continue
                metrics = metric_a if policy_name == "a" else metric_b
                action_ids, actions = _greedy_actions(
                    adapter, bridge, policy_decisions, None,
                    metrics=metrics, public=public,
                )
                action_counts[policy_name].update(int(value) for value in action_ids)
                for decision, action in zip(policy_decisions, actions, strict=True):
                    actions_by_env[decision.env_index][decision.seat_id] = action

            final_tenpai_by_env: dict[int, list[bool | None]] = {}
            for env_index in active_envs:
                tiles_left = min(
                    int(getattr(observations[env_index][seat], "tiles_left", 1))
                    for seat in range(NUM_PLAYERS)
                )
                if tiles_left <= 0:
                    final_tenpai_by_env[env_index] = final_tenpai(
                        env_index, actions_by_env,
                    )

            observations = list(envs.step_batch(actions_by_env))
            bridge.sync(observations)
            public.update(bridge.last_events)
            done = envs.done()
            scores_by_env = envs.scores()
            for env_index in list(active_envs):
                if not bool(done[env_index]):
                    continue
                scores = [int(value) for value in scores_by_env[env_index]]
                seat = candidate_seats[env_index]
                ryukyoku_reason = None
                for rows in bridge.last_events[env_index]:
                    for raw in rows:
                        try:
                            event = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        if event.get("type") == "ryukyoku":
                            ryukyoku_reason = event.get("reason")
                tenpai_flags = final_tenpai_by_env.get(env_index)
                exhaustive_draw = ryukyoku_reason == "exhaustive_draw"
                score_deltas = [
                    scores[player_seat] - start_scores[env_index][player_seat]
                    for player_seat in range(NUM_PLAYERS)
                ]
                metric_a.record_kyoku(
                    [seat],
                    score_deltas,
                    bridge.last_events[env_index],
                    draw_tenpai=(
                        tenpai_flags[seat] if exhaustive_draw and tenpai_flags else None
                    ),
                    exhaustive_draw=exhaustive_draw,
                )
                metric_b.record_kyoku(
                    [player_seat for player_seat in range(NUM_PLAYERS) if player_seat != seat],
                    score_deltas,
                    bridge.last_events[env_index],
                    draw_tenpai=tenpai_flags if exhaustive_draw else None,
                    exhaustive_draw=exhaustive_draw,
                )
                start_scores[env_index] = scores
                ranking = sorted(range(NUM_PLAYERS), key=lambda s: (-scores[s], s))
                rank = ranking.index(seat) + 1
                if rank == 1:
                    first_places += 1
                if rank <= 2:
                    top2 += 1
                if rank == 4:
                    fourths += 1
                rank_sum += rank
                rank_history.append(rank)
                others = [scores[s] for s in range(NUM_PLAYERS) if s != seat]
                point_diff = float(scores[seat] - float(np.mean(others)))
                point_diffs.append(point_diff)
                metric_a.record_match_result(seat, scores, point_delta=point_diff)
                active_envs.remove(env_index)
                completed += 1
            if not active_envs:
                break
        else:
            raise RuntimeError(
                f"1v3 batch {batch_start // batch_size} exceeded {max_steps} steps"
            )
        print(
            f"head_to_head_1v3 completed={completed}/{hanchan_count} "
            f"first_places={first_places} elapsed_s={time.perf_counter() - started:.2f}",
            flush=True,
        )
        while completed >= next_milestone:
            prefix = np.asarray(rank_history[:next_milestone], dtype=np.int64)
            prefix_diffs = np.asarray(point_diffs[:next_milestone], dtype=np.float64)
            print(
                f"1v3_per100 milestone={next_milestone} "
                f"first_rate={float((prefix == 1).mean()):.3f} "
                f"top2_rate={float((prefix <= 2).mean()):.3f} "
                f"four_rate={float((prefix == 4).mean()):.3f} "
                f"mean_rank={float(prefix.mean()):.3f} "
                f"point_diff={float(prefix_diffs.mean()):+.1f}",
                flush=True,
            )
            next_milestone += 100

    elapsed = time.perf_counter() - started
    deltas = np.asarray(point_diffs, dtype=np.float64)
    bootstrap_rng = np.random.default_rng(int(seed_base))
    bootstrap_means = np.asarray([
        float(np.mean(bootstrap_rng.choice(deltas, size=len(deltas), replace=True)))
        for _ in range(2000)
    ], dtype=np.float64)
    ci95 = [
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    ]

    def action_rates(policy: str) -> dict[str, float]:
        grouped = Counter()
        for action_id, count in action_counts[policy].items():
            grouped[_action_group(action_id)] += count
        total = max(sum(grouped.values()), 1)
        return {name: grouped[name] / total for name in (
            "pass", "discard", "reach", "chi", "pon", "kan", "hora", "ryukyoku",
        )}

    def kyoku_metrics(prefix: str, summary: dict[str, float]) -> dict[str, float]:
        return {
            "riichi_rate": summary[f"{prefix}/action/riichi_rate"],
            "riichi_opportunity_accept_rate": summary[
                f"{prefix}/action/riichi_opportunity_accept_rate"
            ],
            "win_rate": summary[f"{prefix}/kyoku/win_rate"],
            "deal_in_rate": summary[f"{prefix}/kyoku/deal_in_rate"],
            "tsumo_loss_rate": summary[f"{prefix}/kyoku/tsumo_loss_rate"],
            "win_points_mean": summary[f"{prefix}/kyoku/win_points_mean"],
            "deal_in_points_mean": summary[f"{prefix}/kyoku/deal_in_points_mean"],
            "draw_tenpai_rate": summary[f"{prefix}/kyoku/draw_tenpai_rate"],
            "kyoku_point_delta_mean": summary[f"{prefix}/kyoku/point_delta_mean"],
            "kyoku_count": summary[f"{prefix}/kyoku/count"],
            "draw_count": summary[f"{prefix}/kyoku/draw_rate"]
            * summary[f"{prefix}/kyoku/count"],
            "exhaustive_draw_count": summary[f"{prefix}/kyoku/exhaustive_draw_count"],
            "draw_tenpai_count": summary[f"{prefix}/kyoku/draw_tenpai_count"],
        }

    summary_a = metric_a.summary("model_a")
    summary_b = metric_b.summary("model_b")
    second_places = sum(rank == 2 for rank in rank_history)
    third_places = sum(rank == 3 for rank in rank_history)

    return {
        "protocol_version": 1,
        "game_mode": game_mode,
        "format": "1v3",
        "hanchan_count": int(hanchan_count),
        "parallel_hanchans": batch_size,
        "seed_base": int(seed_base),
        "candidate_seat_rotation": "i % 4",
        "candidate_seat_counts": {str(seat): int(count) for seat, count in sorted(seat_counts.items())},
        "model_a": {
            "checkpoint": model_a_path,
            "first_place_rate": first_places / int(hanchan_count),
            "first_place_count": first_places,
            "second_place_rate": second_places / int(hanchan_count),
            "second_place_count": second_places,
            "third_place_rate": third_places / int(hanchan_count),
            "third_place_count": third_places,
            "top2_rate": top2 / int(hanchan_count),
            "top2_count": top2,
            "fourth_place_rate": fourths / int(hanchan_count),
            "fourth_place_count": fourths,
            "mean_rank": rank_sum / int(hanchan_count),
            "final_score_mean": summary_a["model_a/match/final_score_mean"],
            "flying_rate": summary_a["model_a/match/flying_rate"],
            "point_diff_vs_mean_opponent_mean": float(deltas.mean()),
            "point_diff_vs_mean_opponent_bootstrap_ci95": ci95,
            "point_diff_samples": [float(value) for value in point_diffs],
            "action_type_rates": action_rates("a"),
            "kyoku_metrics": kyoku_metrics("model_a", summary_a),
            "semantic_metrics": summary_a,
            "metadata": adapter_a.metadata(),
        },
        "model_b": {
            "checkpoint": model_b_path,
            "opponent_seats": NUM_PLAYERS - 1,
            "action_type_rates": action_rates("b"),
            "kyoku_metrics": kyoku_metrics("model_b", summary_b),
            "semantic_metrics": summary_b,
            "metadata": adapter_b.metadata(),
        },
        "elapsed_s": elapsed,
        "hanchan_per_s": int(hanchan_count) / max(elapsed, 1e-9),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--hanchans", type=int, default=TOTAL_1V3_HANCHANS)
    parser.add_argument(
        "--parallel-hanchans", type=int, default=DEFAULT_1V3_HANCHANS_PER_PROCESS
    )
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-a-device")
    parser.add_argument("--model-b-device")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_1v3(
        args.model_a,
        args.model_b,
        device=args.device,
        model_a_device=args.model_a_device,
        model_b_device=args.model_b_device,
        hanchan_count=args.hanchans,
        parallel_hanchans=args.parallel_hanchans,
        seed_base=args.seed_base,
        game_mode=args.game_mode,
        max_steps=args.max_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
