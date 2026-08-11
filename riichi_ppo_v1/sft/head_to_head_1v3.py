"""Deterministic 1v3 evaluation: one candidate seat vs three opponent seats.

The candidate model occupies one seat per hanchan; the other three seats all
play the opponent model greedily.  The candidate seat rotates across the four
positions over hanchans.  Reported metrics: first-place rate, mean rank, mean
point difference against the average of the three opponents, and a paired
bootstrap 95% CI over per-hanchan point differences.
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
from ..training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from ..training.worker import active_decisions
from .head_to_head import _action_group
from .policy_adapter import load_policy_adapter


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device, non_blocking=device.type == "cuda")


@torch.inference_mode()
def _greedy_actions(
    adapter: Any,
    bridge: BatchedStateBridge,
    decisions: list[Any],
    analysis: Any,
) -> tuple[list[int], list[Any]]:
    prepared = adapter.prepare(bridge, decisions, analysis)
    logits = adapter.masked_logits(prepared)
    action_ids = logits.argmax(-1).tolist()
    actions = bridge.decode(decisions, action_ids)
    return action_ids, actions


def evaluate_1v3(
    model_a_path: str | Path,
    model_b_path: str | Path,
    *,
    device: str = "cuda",
    model_a_device: str | None = None,
    model_b_device: str | None = None,
    hanchan_count: int = 500,
    parallel_hanchans: int = 24,
    seed_base: int = 20290000,
    game_mode: str = "4p-red-half",
    max_steps: int = 4000,
) -> dict[str, Any]:
    try:
        import riichi
        from riichienv import BatchedRiichiEnv
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "install the local riichi and RiichiEnv extensions before evaluation"
        ) from exc

    default_device = torch.device(device)
    device_a = torch.device(model_a_device or default_device)
    device_b = torch.device(model_b_device or default_device)
    if (device_a.type == "cuda" or device_b.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    model_a_path = str(Path(model_a_path).resolve())
    model_b_path = str(Path(model_b_path).resolve())
    adapter_a = load_policy_adapter(model_a_path, device=device_a)
    adapter_b = load_policy_adapter(model_b_path, device=device_b)

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
        analyzer = EfficiencyAnalyzer(131_072)
        candidate_seats = [
            (batch_start + env_index) % NUM_PLAYERS
            for env_index in range(batch_size_now)
        ]
        seat_counts.update(candidate_seats)
        active_envs = set(range(batch_size_now))

        for _step in range(int(max_steps)):
            actions_by_env: list[dict[int, Any]] = [{} for _ in range(batch_size_now)]
            decisions = active_decisions(observations, active_envs)
            analysis = (
                DecisionAnalysisBatch.build(
                    decisions, analyzer=analyzer, public=public,
                )
                if decisions
                else None
            )
            for policy_name, adapter in (("a", adapter_a), ("b", adapter_b)):
                policy_decisions = [
                    decision for decision in decisions
                    if (decision.seat_id == candidate_seats[decision.env_index])
                    == (policy_name == "a")
                ]
                if not policy_decisions:
                    continue
                action_ids, actions = _greedy_actions(adapter, bridge, policy_decisions, analysis)
                action_counts[policy_name].update(int(value) for value in action_ids)
                for decision, action in zip(policy_decisions, actions, strict=True):
                    actions_by_env[decision.env_index][decision.seat_id] = action

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
                point_diffs.append(float(scores[seat] - float(np.mean(others))))
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
            "top2_rate": top2 / int(hanchan_count),
            "top2_count": top2,
            "fourth_place_rate": fourths / int(hanchan_count),
            "fourth_place_count": fourths,
            "mean_rank": rank_sum / int(hanchan_count),
            "point_diff_vs_mean_opponent_mean": float(deltas.mean()),
            "point_diff_vs_mean_opponent_bootstrap_ci95": ci95,
            "action_type_rates": action_rates("a"),
            "metadata": adapter_a.metadata(),
        },
        "model_b": {
            "checkpoint": model_b_path,
            "opponent_seats": NUM_PLAYERS - 1,
            "action_type_rates": action_rates("b"),
            "metadata": adapter_b.metadata(),
        },
        "elapsed_s": elapsed,
        "hanchan_per_s": int(hanchan_count) / max(elapsed, 1e-9),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--hanchans", type=int, default=500)
    parser.add_argument("--parallel-hanchans", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=20290000)
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
