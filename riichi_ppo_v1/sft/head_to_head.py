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

from ..model import KyokuTransformerActorCritic, ModelConfig
from ..model.feature_schema import DECISION_ANALYSIS_VERSION, RUST_ANALYSIS_VERSION, feature_schema_sha256
from ..model.bridge import BatchedStateBridge, NUM_PLAYERS
from ..model.schema import TOKEN_SCHEMA_VERSION
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
    schedule = pairs * (count // len(pairs))
    remainder = count % len(pairs)
    if remainder:
        # An even remainder can be filled by complementary seat pairs, keeping
        # every physical seat equally represented for both checkpoints.
        balanced_remainder = (
            (0, 1),
            (2, 3),
            (0, 2),
            (1, 3),
        )
        schedule.extend(balanced_remainder[:remainder])
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


def _load_model(path: str | Path, device: torch.device) -> KyokuTransformerActorCritic:
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schema = int(payload.get("token_schema_version", 0))
    if schema not in {11, TOKEN_SCHEMA_VERSION}:
        raise RuntimeError(
            f"{checkpoint} uses unsupported token schema {schema}"
        )
    if schema == TOKEN_SCHEMA_VERSION:
        if payload.get("feature_schema_sha256") != feature_schema_sha256():
            raise RuntimeError(f"{checkpoint} has a missing or incompatible v13 feature hash")
        if int(payload.get("rust_analysis_version", -1)) != RUST_ANALYSIS_VERSION:
            raise RuntimeError(f"{checkpoint} has an incompatible Rust analysis version")
        if int(payload.get("decision_analysis_version", -1)) != DECISION_ANALYSIS_VERSION:
            raise RuntimeError(f"{checkpoint} has an incompatible decision-analysis version")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise RuntimeError(f"{checkpoint} is missing model_config")
    model_config = dict(model_config)
    model_config.setdefault(
        "policy_head_type", "legacy_fixed" if schema == 11 else "isolated_action_query"
    )
    state = payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"{checkpoint} is missing model state")
    model = KyokuTransformerActorCritic(ModelConfig(**model_config))
    model.token_schema_version = schema
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device, non_blocking=device.type == "cuda")


def _bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return bool(torch.cuda.get_device_properties(device).major >= 8)


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
    model_a = _load_model(model_a_path, device_a)
    model_b = _load_model(model_b_path, device_b)
    use_bf16_a = _bf16_supported(device_a)
    use_bf16_b = _bf16_supported(device_b)
    schedule = balanced_team_a_seats(hanchan_count)
    parallel = max(1, min(int(parallel_hanchans), int(hanchan_count)))
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

    for batch_start in range(0, hanchan_count, parallel):
        team_a_by_env = schedule[batch_start : batch_start + parallel]
        batch_size = len(team_a_by_env)
        envs = BatchedRiichiEnv(
            batch_size,
            seed=int(seed_base) + batch_start,
            step_threads=batch_size,
            game_mode=game_mode,
        )
        bridge = BatchedStateBridge(
            riichi.MjaiKyokuStateMachineManager(batch_size),
            batch_size,
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
                    decisions,
                    analyzer=analyzer,
                    public=public,
                )
                if decisions
                else None
            )
            for policy_name, model, model_device, use_bf16 in (
                ("a", model_a, device_a, use_bf16_a),
                ("b", model_b, device_b, use_bf16_b),
            ):
                policy_decisions = [
                    decision
                    for decision in decisions
                    if (
                        decision.seat_id in team_a_by_env[decision.env_index]
                    )
                    == (policy_name == "a")
                ]
                if not policy_decisions:
                    continue
                (
                    factors,
                    numeric,
                    lengths,
                    legal,
                    _generations,
                    _critic,
                    _critic_lengths,
                ) = bridge.prepare(
                    policy_decisions, analysis,
                    token_schema_version=int(getattr(model, "token_schema_version", TOKEN_SCHEMA_VERSION)),
                )
                with torch.autocast(
                    device_type=model_device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    output = model.forward_policy(
                        _tensor(factors, model_device),
                        _tensor(numeric, model_device),
                        _tensor(legal, model_device),
                        _tensor(lengths, model_device),
                    )
                action_ids = output["policy_logits"].argmax(-1).tolist()
                actions = bridge.decode(policy_decisions, action_ids)
                for decision, action in zip(
                    policy_decisions, actions, strict=True,
                ):
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
                    scores[seat] for seat in range(NUM_PLAYERS)
                    if seat not in team_a
                )
                point_diff = team_score_a - team_score_b
                team_point_diff_sum += point_diff
                if point_diff > 0:
                    wins_a += 1
                elif point_diff < 0:
                    wins_b += 1
                else:
                    ties += 1
                ranking = sorted(
                    range(NUM_PLAYERS),
                    key=lambda seat: (-scores[seat], seat),
                )
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
                f"2v2 batch {batch_start // parallel} exceeded {max_steps} steps"
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
    return {
        "schema_version": 1,
        "game_mode": game_mode,
        "hanchan_count": int(hanchan_count),
        "parallel_hanchans": parallel,
        "seed_base": int(seed_base),
        "greedy": True,
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
            "first_place_count": first_places_a,
            "first_place_rate": first_places_a / hanchan_count,
            "individual_mean_rank": individual_rank_sum_a / (2 * hanchan_count),
        },
        "model_b": {
            "checkpoint": model_b_path,
            "team_wins": wins_b,
            "team_ties": ties,
            "team_win_rate": scored_wins_b / hanchan_count,
            "team_point_diff_mean": -team_point_diff_sum / hanchan_count,
            "first_place_count": first_places_b,
            "first_place_rate": first_places_b / hanchan_count,
            "individual_mean_rank": individual_rank_sum_b / (2 * hanchan_count),
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
