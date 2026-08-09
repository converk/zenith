"""Deterministic, seat-balanced 2v2 checkpoint evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
import os
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
from .policy_adapter import load_policy_adapter
from ..training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from ..training.worker import active_decisions


def balanced_team_a_seats(hanchan_count: int) -> list[tuple[int, int]]:
    """Return a two-seat schedule balanced across seats and complementary teams."""
    count = int(hanchan_count)
    if count <= 0 or count % 2:
        raise ValueError("hanchan_count must be a positive even number")
    pairs = list(combinations(range(NUM_PLAYERS), 2))
    schedule: list[tuple[int, int]] = []
    for pair_index in range(count // 2):
        seats = pairs[pair_index % len(pairs)]
        complement = tuple(seat for seat in range(NUM_PLAYERS) if seat not in seats)
        schedule.extend((seats, complement))
    seat_counts = Counter(seat for seats in schedule for seat in seats)
    if len(schedule) != count or set(seat_counts.values()) != {count // 2}:
        raise RuntimeError(f"failed to construct a seat-balanced schedule: {seat_counts}")
    return schedule


def select_winner(
    *,
    model_a: str,
    model_b: str,
    wins_a: int,
    wins_b: int,
    ties: int,
    team_point_diff_sum: int,
    first_places_a: int,
    first_places_b: int,
) -> tuple[str, str]:
    """Select by match win points, then point differential and first places."""
    scored_wins_a = 2 * int(wins_a) + int(ties)
    scored_wins_b = 2 * int(wins_b) + int(ties)
    if scored_wins_a != scored_wins_b:
        return (
            (model_a, "team_win_rate")
            if scored_wins_a > scored_wins_b
            else (model_b, "team_win_rate")
        )
    if int(team_point_diff_sum) != 0:
        return (
            (model_a, "team_point_diff")
            if team_point_diff_sum > 0
            else (model_b, "team_point_diff")
        )
    if int(first_places_a) != int(first_places_b):
        return (
            (model_a, "first_place_count")
            if first_places_a > first_places_b
            else (model_b, "first_place_count")
        )
    return model_a, "stable_model_a_fallback"


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device, non_blocking=device.type == "cuda")


def _load_model(path: str | Path, device: torch.device) -> torch.nn.Module:
    """Compatibility shim for diagnostic tools; loading remains adapter-owned."""
    adapter = load_policy_adapter(path, device=device)
    model = adapter.model
    model.policy_adapter = adapter
    return model


def _bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return bool(torch.cuda.get_device_properties(device).major >= 8)


def _action_group(action_id: int) -> str:
    boundaries = (
        (1, "pass"), (75, "discard"), (76, "reach"), (133, "chi"),
        (170, "pon"), (239, "kan"), (240, "hora"), (241, "ryukyoku"),
    )
    return next(name for end, name in boundaries if int(action_id) < end)


@torch.inference_mode()
def evaluate_2v2(
    model_a_path: str | Path,
    model_b_path: str | Path,
    *,
    device: str = "cuda",
    model_a_device: str | None = None,
    model_b_device: str | None = None,
    hanchan_count: int = 320,
    parallel_hanchans: int = 24,
    seed_base: int = 20260730,
    game_mode: str = "4p-red-half",
    max_steps: int = 4000,
) -> dict[str, Any]:
    """Play greedy 2v2 hanchans and return team and placement statistics."""
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
    if (
        device_a.type == "cuda" or device_b.type == "cuda"
    ) and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    model_a_path = str(Path(model_a_path).resolve())
    model_b_path = str(Path(model_b_path).resolve())
    adapter_a = load_policy_adapter(model_a_path, device=device_a)
    adapter_b = load_policy_adapter(model_b_path, device=device_b)
    schedule = balanced_team_a_seats(hanchan_count)
    base_schedule = schedule[::2]
    parallel_pairs = max(1, min(int(parallel_hanchans) // 2, len(base_schedule)))
    parallel = 2 * parallel_pairs
    started = time.perf_counter()

    wins_a = 0
    wins_b = 0
    ties = 0
    team_point_diff_sum = 0
    first_places_a = 0
    first_places_b = 0
    individual_rank_sum_a = 0
    individual_rank_sum_b = 0
    completed = 0
    paired_point_diffs: dict[int, list[int]] = {}
    action_counts = {"a": Counter(), "b": Counter()}

    for pair_start in range(0, len(base_schedule), parallel_pairs):
        base_seats = base_schedule[pair_start : pair_start + parallel_pairs]
        for swapped in (False, True):
            team_a_by_env = [
                tuple(seat for seat in range(NUM_PLAYERS) if seat not in seats)
                if swapped else seats
                for seats in base_seats
            ]
            batch_size = len(team_a_by_env)
            # Reusing the exact seed range for the swapped pass gives every
            # pair identical walls and initial states.
            envs = BatchedRiichiEnv(
                batch_size,
                seed=int(seed_base) + pair_start,
                step_threads=batch_size,
                game_mode=game_mode,
            )
            bridge = BatchedStateBridge(
                riichi.MjaiKyokuStateMachineManager(batch_size), batch_size,
            )
            observations = list(envs.reset())
            bridge.sync(observations)
            public = PublicStateTracker(batch_size)
            public.update(bridge.last_events)
            analyzer = EfficiencyAnalyzer(131_072)
            active_envs = set(range(batch_size))

            for _step in range(int(max_steps)):
                actions_by_env: list[dict[int, Any]] = [
                    {} for _ in range(batch_size)
                ]
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
                        if (decision.seat_id in team_a_by_env[decision.env_index])
                        == (policy_name == "a")
                    ]
                    if not policy_decisions:
                        continue
                    prepared = adapter.prepare(bridge, policy_decisions, analysis)
                    action_ids = adapter.masked_logits(prepared).argmax(-1).tolist()
                    action_counts[policy_name].update(int(value) for value in action_ids)
                    actions = bridge.decode(policy_decisions, action_ids)
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
                    team_a = set(team_a_by_env[env_index])
                    team_score_a = sum(scores[seat] for seat in team_a)
                    team_score_b = sum(
                        scores[seat] for seat in range(NUM_PLAYERS) if seat not in team_a
                    )
                    point_diff = team_score_a - team_score_b
                    paired_point_diffs.setdefault(pair_start + env_index, []).append(point_diff)
                    team_point_diff_sum += point_diff
                    if point_diff > 0:
                        wins_a += 1
                    elif point_diff < 0:
                        wins_b += 1
                    else:
                        ties += 1
                    ranking = sorted(range(NUM_PLAYERS), key=lambda seat: (-scores[seat], seat))
                    if ranking[0] in team_a:
                        first_places_a += 1
                    else:
                        first_places_b += 1
                    for rank, seat in enumerate(ranking, start=1):
                        if seat in team_a:
                            individual_rank_sum_a += rank
                        else:
                            individual_rank_sum_b += rank
                    active_envs.remove(env_index)
                    completed += 1
                if not active_envs:
                    break
            else:
                raise RuntimeError(
                    f"paired 2v2 batch {pair_start // parallel_pairs} exceeded {max_steps} steps"
                )
        print(
            f"head_to_head completed={completed}/{hanchan_count} "
            f"wins_a={wins_a} wins_b={wins_b} ties={ties} "
            f"elapsed_s={time.perf_counter() - started:.2f}",
            flush=True,
        )

    selected, selection_reason = select_winner(
        model_a=model_a_path,
        model_b=model_b_path,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        team_point_diff_sum=team_point_diff_sum,
        first_places_a=first_places_a,
        first_places_b=first_places_b,
    )
    elapsed = time.perf_counter() - started
    scored_wins_a = wins_a + 0.5 * ties
    scored_wins_b = wins_b + 0.5 * ties
    if any(len(values) != 2 for values in paired_point_diffs.values()):
        raise RuntimeError("paired evaluation did not complete both seat-swapped games")
    paired = np.asarray(
        [np.mean(paired_point_diffs[index]) for index in sorted(paired_point_diffs)],
        dtype=np.float64,
    )
    paired_se = float(paired.std(ddof=1) / np.sqrt(len(paired))) if len(paired) > 1 else 0.0
    bootstrap_rng = np.random.default_rng(int(seed_base))
    bootstrap_means = np.asarray([
        float(np.mean(bootstrap_rng.choice(paired, size=len(paired), replace=True)))
        for _ in range(2000)
    ], dtype=np.float64)
    paired_bootstrap_ci95 = [
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
        "protocol_version": 2,
        "game_mode": game_mode,
        "hanchan_count": int(hanchan_count),
        "parallel_hanchans": parallel,
        "seed_base": int(seed_base),
        "greedy": True,
        "paired_walls": True,
        "seat_swap_within_pair": True,
        "paired_point_diff_mean": float(paired.mean()),
        "paired_point_diff_standard_error": paired_se,
        "paired_point_diff_95ci": [
            float(paired.mean() - 1.96 * paired_se),
            float(paired.mean() + 1.96 * paired_se),
        ],
        "paired_point_diff_bootstrap_ci95": paired_bootstrap_ci95,
        "model_a_device": str(device_a),
        "model_b_device": str(device_b),
        "team_win_definition": "higher sum of the two teammates' final scores; tie counts 0.5",
        "selection_order": [
            "team_win_rate",
            "team_point_diff",
            "first_place_count",
            "stable_model_a_fallback",
        ],
        "model_a": {
            "checkpoint": model_a_path,
            "team_wins": wins_a,
            "team_ties": ties,
            "team_win_rate": scored_wins_a / hanchan_count,
            "team_point_diff_mean": team_point_diff_sum / hanchan_count,
            "team_point_diff_paired_bootstrap_ci95": paired_bootstrap_ci95,
            "first_place_count": first_places_a,
            "first_place_rate": first_places_a / hanchan_count,
            "individual_mean_rank": individual_rank_sum_a / (2 * hanchan_count),
            "action_type_rates": action_rates("a"),
            "metadata": adapter_a.metadata(),
        },
        "model_b": {
            "checkpoint": model_b_path,
            "team_wins": wins_b,
            "team_ties": ties,
            "team_win_rate": scored_wins_b / hanchan_count,
            "team_point_diff_mean": -team_point_diff_sum / hanchan_count,
            "team_point_diff_paired_bootstrap_ci95": [
                -paired_bootstrap_ci95[1],
                -paired_bootstrap_ci95[0],
            ],
            "first_place_count": first_places_b,
            "first_place_rate": first_places_b / hanchan_count,
            "individual_mean_rank": individual_rank_sum_b / (2 * hanchan_count),
            "action_type_rates": action_rates("b"),
            "metadata": adapter_b.metadata(),
        },
        "selected_checkpoint": selected,
        "selection_reason": selection_reason,
        "elapsed_s": elapsed,
        "hanchan_per_s": hanchan_count / max(elapsed, 1e-9),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--hanchans", type=int, default=320)
    parser.add_argument("--parallel-hanchans", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=20260730)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-a-device")
    parser.add_argument("--model-b-device")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_2v2(
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
