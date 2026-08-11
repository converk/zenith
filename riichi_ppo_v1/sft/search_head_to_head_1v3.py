"""1v3 tree-search evaluation: one searched seat vs three greedy opponents.

The candidate model occupies one seat per hanchan and uses root-branch pMCPA
search (same parameters as the offline search validation); the other three
seats play a greedy opponent model.  Hanchan walls come from an explicit
random seed list (one seed per hanchan), so the evaluation is not a contiguous
seed range.  Metrics: first/top2/third/fourth place rates, mean rank, point
difference against the mean opponent, bootstrap CI, per-seat breakdown and
search statistics (searches, override rate, per-decision latency).
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from ..model.bridge import BatchedStateBridge, Decision, NUM_PLAYERS
from ..grp.model import RankPredictor
from ..training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from ..training.worker import active_decisions
from .head_to_head import _action_group
from .policy_adapter import load_policy_adapter
from .search_head_to_head import (
    SearchStats,
    is_searchable_decision,
    run_root_search,
)


@torch.inference_mode()
def _forward_full(
    model: torch.nn.Module,
    device: torch.device,
    prepared: tuple[np.ndarray, ...],
) -> dict[str, torch.Tensor]:
    factors, numeric, lengths, legal, _gen, critic_factors, critic_lengths = prepared

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value).to(device, non_blocking=device.type == "cuda")

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
        output = model(
            tensor(factors),
            tensor(numeric),
            tensor(legal),
            tensor(lengths),
            critic_factors=tensor(critic_factors),
            critic_lengths=tensor(critic_lengths),
        )
    return {
        "policy_logits": output["policy_logits"].float(),
        "value": output["value"].float(),
    }


def evaluate_1v3_search(
    model_path: str | Path,
    opponent_model_path: str | Path,
    seeds: list[int],
    *,
    device: str = "cuda",
    grp_model_path: str | Path,
    search_width: int = 3,
    search_depth: int = 3,
    search_rollouts: int = 2,
    temperature: float = 1.0,
    search_rng_seed: int = 20260811,
    game_mode: str = "4p-red-half",
    max_steps: int = 4000,
) -> dict[str, Any]:
    try:
        import riichi
        from riichienv import BatchedRiichiEnv, RiichiEnv
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "install the local riichi and RiichiEnv extensions before evaluation"
        ) from exc

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    model_path = str(Path(model_path).resolve())
    opponent_model_path = str(Path(opponent_model_path).resolve())
    adapter = load_policy_adapter(model_path, device=device)
    opponent_adapter = load_policy_adapter(opponent_model_path, device=device)
    model = adapter.model
    model.eval()
    grp_model = RankPredictor.from_checkpoint(str(grp_model_path))
    grp_model.eval()
    search_rng = torch.Generator(device="cpu").manual_seed(int(search_rng_seed))
    stats = SearchStats()
    action_counts = {"search": Counter(), "opponent": Counter()}

    started = time.perf_counter()
    hanchan_records: list[dict[str, Any]] = []
    total_searches = 0
    total_overrides = 0
    search_wall_s = 0.0

    for hanchan_index, seed in enumerate(seeds):
        candidate_seat = hanchan_index % NUM_PLAYERS
        envs = BatchedRiichiEnv(1, seed=int(seed), step_threads=1, game_mode=game_mode)
        mirror = RiichiEnv(game_mode=game_mode, seed=int(seed))
        mirror.reset()
        bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
        observations = list(envs.reset())
        bridge.sync(observations)
        public = PublicStateTracker(1)
        public.update(bridge.last_events)
        analyzer = EfficiencyAnalyzer(131_072)
        hanchan_searches = 0
        hanchan_overrides = 0

        for _step in range(int(max_steps)):
            actions_by_env: list[dict[int, Any]] = [{}]
            decisions = active_decisions(observations)
            analysis = (
                DecisionAnalysisBatch.build(
                    decisions, analyzer=analyzer, public=public,
                )
                if decisions
                else None
            )
            candidate_decisions = [
                decision for decision in decisions
                if decision.seat_id == candidate_seat
            ]
            opponent_decisions = [
                decision for decision in decisions
                if decision.seat_id != candidate_seat
            ]

            chosen_map: dict[int, int] = {}
            if candidate_decisions:
                prepared = bridge.prepare(candidate_decisions, analysis)
                output = _forward_full(model, device, prepared)
                logits = output["policy_logits"]
                values = output["value"]
                greedy_ids = logits.argmax(-1).tolist()
                chosen_map = {
                    decision.batch_index: int(action_id)
                    for decision, action_id in zip(candidate_decisions, greedy_ids, strict=True)
                }
                searchable: list[Decision] = []
                search_rows: list[int] = []
                for row, decision in enumerate(candidate_decisions):
                    if is_searchable_decision(decision.observation):
                        searchable.append(decision)
                        search_rows.append(row)
                stats.total_decisions += len(candidate_decisions)
                stats.searchable += len(searchable)
                if searchable:
                    prepared_rows = {
                        "factors": prepared[0][search_rows],
                        "numeric": prepared[1][search_rows],
                        "lengths": prepared[2][search_rows],
                        "legal": prepared[3][search_rows],
                    }
                    chosen = run_root_search(
                        decisions=searchable,
                        logits=logits[search_rows],
                        values=values[search_rows],
                        prepared_rows=prepared_rows,
                        main_bridge=bridge,
                        mirrors=[mirror],
                        team_seats_by_env={0: (candidate_seat,)},
                        grp_model=grp_model,
                        grp_pts_weight=(10, 4, -4, -10),
                        model=model,
                        device=device,
                        value_mode="seat",
                        search_width=search_width,
                        search_depth=search_depth,
                        rollouts=search_rollouts,
                        temperature=temperature,
                        search_rng=search_rng,
                        stats=stats,
                    )
                    chosen_map.update(chosen)
                    hanchan_searches += len(searchable)
                    for decision in searchable:
                        if chosen.get(decision.batch_index) != greedy_ids[
                            candidate_decisions.index(decision)
                        ]:
                            hanchan_overrides += 1

            if candidate_decisions:
                ids = [chosen_map[decision.batch_index] for decision in candidate_decisions]
                actions = bridge.decode(candidate_decisions, ids)
                action_counts["search"].update(int(value) for value in ids)
                for decision, action in zip(candidate_decisions, actions, strict=True):
                    actions_by_env[0][decision.seat_id] = action
            if opponent_decisions:
                prepared = opponent_adapter.prepare(bridge, opponent_decisions, analysis)
                opp_ids = opponent_adapter.masked_logits(prepared).argmax(-1).tolist()
                opp_actions = bridge.decode(opponent_decisions, opp_ids)
                action_counts["opponent"].update(int(value) for value in opp_ids)
                for decision, action in zip(opponent_decisions, opp_actions, strict=True):
                    actions_by_env[0][decision.seat_id] = action

            observations = list(envs.step_batch(actions_by_env))
            if actions_by_env[0]:
                mirror.step(actions_by_env[0])
            bridge.sync(observations)
            public.update(bridge.last_events)
            if bool(envs.done()[0]):
                scores = [int(value) for value in envs.scores()[0]]
                ranking = sorted(range(NUM_PLAYERS), key=lambda s: (-scores[s], s))
                rank = ranking.index(candidate_seat) + 1
                others = [scores[s] for s in range(NUM_PLAYERS) if s != candidate_seat]
                point_diff = float(scores[candidate_seat] - float(np.mean(others)))
                hanchan_records.append({
                    "hanchan_index": hanchan_index,
                    "seed": int(seed),
                    "candidate_seat": int(candidate_seat),
                    "rank": int(rank),
                    "point_diff_vs_mean_opponent": point_diff,
                    "searches": int(hanchan_searches),
                    "overrides": int(hanchan_overrides),
                })
                total_searches += hanchan_searches
                total_overrides += hanchan_overrides
                break
        else:
            raise RuntimeError(f"hanchan {hanchan_index} (seed {seed}) exceeded {max_steps} steps")
        if (hanchan_index + 1) % 50 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"search_1v3 completed={hanchan_index + 1}/{len(seeds)} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    search_wall_s = stats.search_time_s
    elapsed = time.perf_counter() - started
    ranks = np.asarray([r["rank"] for r in hanchan_records], dtype=np.int64)
    diffs = np.asarray([r["point_diff_vs_mean_opponent"] for r in hanchan_records], dtype=np.float64)
    n = len(hanchan_records)
    bootstrap_rng = np.random.default_rng(int(search_rng_seed))
    bootstrap_means = np.asarray([
        float(np.mean(bootstrap_rng.choice(diffs, size=n, replace=True)))
        for _ in range(2000)
    ], dtype=np.float64)
    ci95 = [
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    ]
    per_seat: dict[int, list[int]] = {seat: [] for seat in range(NUM_PLAYERS)}
    for record in hanchan_records:
        per_seat[record["candidate_seat"]].append(record["rank"])

    return {
        "protocol_version": 1,
        "format": "1v3_search",
        "hanchan_count": n,
        "seeds": [int(seed) for seed in seeds],
        "candidate_seat_rotation": "i % 4",
        "search_config": {
            "width": int(search_width),
            "depth": int(search_depth),
            "rollouts": int(search_rollouts),
            "temperature": float(temperature),
            "value_mode": "seat",
            "grp_model": str(grp_model_path),
        },
        "model": {
            "checkpoint": model_path,
            "first_place_rate": float(np.mean(ranks == 1)),
            "second_place_rate": float(np.mean(ranks == 2)),
            "third_place_rate": float(np.mean(ranks == 3)),
            "fourth_place_rate": float(np.mean(ranks == 4)),
            "top2_rate": float(np.mean(ranks <= 2)),
            "mean_rank": float(ranks.mean()),
            "point_diff_vs_mean_opponent_mean": float(diffs.mean()),
            "point_diff_vs_mean_opponent_bootstrap_ci95": ci95,
            "per_seat_first_place_rate": {
                str(seat): float(np.mean(np.asarray(values) == 1))
                for seat, values in per_seat.items()
            },
            "action_type_rates": _action_rates(action_counts["search"]),
        },
        "opponent": {
            "checkpoint": opponent_model_path,
            "opponent_seats": NUM_PLAYERS - 1,
            "action_type_rates": _action_rates(action_counts["opponent"]),
            "metadata": opponent_adapter.metadata(),
        },
        "search_stats": stats.as_dict(),
        "search_per_decision_ms": (
            search_wall_s * 1000.0 / max(stats.searched, 1)
        ),
        "elapsed_s": elapsed,
        "hanchan_per_s": n / max(elapsed, 1e-9),
        "hanchans": hanchan_records,
    }


def _action_rates(counter: Any) -> dict[str, float]:
    grouped = Counter()
    for action_id, count in counter.items():
        grouped[_action_group(action_id)] += count
    total = max(sum(grouped.values()), 1)
    return {name: grouped[name] / total for name in (
        "pass", "discard", "reach", "chi", "pon", "kan", "hora", "ryukyoku",
    )}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--opponent-model", required=True)
    parser.add_argument("--seeds-file", required=True)
    parser.add_argument("--grp-model", required=True)
    parser.add_argument("--search-width", type=int, default=3)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--search-rollouts", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--search-rng-seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    seeds = json.loads(Path(args.seeds_file).read_text(encoding="utf-8"))
    if not isinstance(seeds, list) or not all(isinstance(value, int) for value in seeds):
        raise ValueError("seeds-file must contain a JSON list of integers")
    result = evaluate_1v3_search(
        args.model,
        args.opponent_model,
        seeds,
        device=args.device,
        grp_model_path=args.grp_model,
        search_width=args.search_width,
        search_depth=args.search_depth,
        search_rollouts=args.search_rollouts,
        temperature=args.temperature,
        search_rng_seed=args.search_rng_seed,
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
