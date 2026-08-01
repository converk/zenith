"""320-hanchan 2v2 head-to-head with detailed skill metrics.

Plays a series of seat-balanced 2v2 hanchans between a "candidate" checkpoint
(model A) and a "baseline" checkpoint (model B), then records for *each seat
group* (candidate vs baseline):

- win-rate & point-diff (team-level)
- first-place-rate, mean_rank, rank-distribution (1-4 位率)
- 胡牌次数与率（和牌：tsumo+ron 总和，区分自摸 vs 荣和）
- 放铳次数与率（ron 中 target 在本方）
- 立直宣告率与立直后和牌率
- 被飞率（负分终局比例）

两份 checkpoint 在 GPU0 和 GPU3 双卡上跑两组 seed：默认 seed_base 用一组
"随机"值（基于脚本实际运行时间或显式 --seed-base）。

Usage (from workspace root):

    CUDA_DEVICE=0 python riichi_ppo_v1/tools/benchmark_ppo_vs_sft_detailed.py \\
        --model-a checkpoints/train_riichi_v11_ppo_selected/checkpoint_00050.pt \\
        --model-b checkpoints/train_riichi_v11_sft_40pct_2v2_selection/best_heuristic.snapshot.pt \\
        --hanchans 320 --parallel-hanchans 24 \\
        --seed-base 20260801 \\
        --output checkpoints/train_riichi_v11_ppo_selected/ppo_vs_sft_2v2_detailed.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from riichi_ppo_v1.model.bridge import BatchedStateBridge, NUM_PLAYERS
from riichi_ppo_v1.sft.head_to_head import (
    _bf16_supported,
    _load_model,
    _tensor,
    balanced_team_a_seats,
    select_winner,
)
from riichi_ppo_v1.training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from riichi_ppo_v1.training.rewards.decision import action_kind
from riichi_ppo_v1.training.worker import active_decisions
from riichi_ppo_v1.tools.event_statistics import canonical_step_events


def _summarize_metrics(metrics: dict, total_seats_total: int) -> dict[str, float]:
    """Normalize per-team cumulative counters into rates / averages."""
    hanchans = metrics["hanchans"]
    seat_total = hanchans * 2  # 2 seats for each team per hanchan
    point_diffs = np.asarray(metrics["point_diffs"], dtype=np.float64)
    if len(point_diffs):
        rng = np.random.default_rng(20260801)
        indices = rng.integers(0, len(point_diffs), size=(10_000, len(point_diffs)))
        bootstrap = point_diffs[indices].mean(axis=1)
        point_diff_ci = [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]
    else:
        point_diff_ci = [0.0, 0.0]
    return {
        "team_wins": metrics["team_wins"],
        "team_win_rate": (2 * metrics["team_wins"] + metrics["team_ties"]) / (2 * hanchans),
        "team_ties": metrics["team_ties"],
        "team_point_diff_mean": metrics["team_point_diff_sum"] / hanchans,
        "team_point_diff_paired_bootstrap_ci95": point_diff_ci,
        "individual_hanchans": seat_total,
        "first_places": metrics["first_places"],
        "first_place_rate": metrics["first_places"] / seat_total,
        "individual_rank_sum": metrics["rank_sum"],
        "individual_mean_rank": metrics["rank_sum"] / seat_total,
        "tsumo_count": metrics["tsumo"],
        "ron_count": metrics["ron"],
        "agari_count": metrics["tsumo"] + metrics["ron"],
        "player_kyokus": metrics["player_kyokus"],
        "agari_per_player_kyoku": (metrics["tsumo"] + metrics["ron"]) / max(metrics["player_kyokus"], 1),
        "tsumo_per_player_kyoku": metrics["tsumo"] / max(metrics["player_kyokus"], 1),
        "ron_per_player_kyoku": metrics["ron"] / max(metrics["player_kyokus"], 1),
        "dealin_count": metrics["dealin"],
        "dealin_per_player_kyoku": metrics["dealin"] / max(metrics["player_kyokus"], 1),
        "reach_count": metrics["reach"],
        "reach_per_player_kyoku": metrics["reach"] / max(metrics["player_kyokus"], 1),
        "reach_per_player_hanchan": metrics["reach"] / seat_total,
        "reach_given_opportunity": metrics["reach"] / max(metrics["reach_opportunities"], 1),
        "reach_opportunities": metrics["reach_opportunities"],
        "first_reach_count": metrics["first_reach"],
        "chase_reach_count": metrics["chase_reach"],
        "calls_per_player_kyoku": metrics["calls"] / max(metrics["player_kyokus"], 1),
        "furo_player_kyoku_rate": metrics["player_kyokus_with_call"] / max(metrics["player_kyokus"], 1),
        "topseat_count": metrics["rank_dist"][1],
        "topseat_rate": metrics["rank_dist"][1] / seat_total,
        "fourth_seat_count": metrics["rank_dist"][4],
        "fourth_seat_rate": metrics["rank_dist"][4] / seat_total,
        "rank_distribution": dict(metrics["rank_dist"]),
        "bitten_count": metrics["bitten"],
        "bitten_rate": metrics["bitten"] / hanchans,
    }


def _new_metrics(hanchans: int) -> dict:
    return {
        "team_wins": 0,
        "team_ties": 0,
        "team_point_diff_sum": 0,
        "first_places": 0,
        "rank_sum": 0,
        "rank_dist": Counter(),
        "tsumo": 0,
        "ron": 0,
        "dealin": 0,
        "reach": 0,
        "reach_opportunities": 0,
        "first_reach": 0,
        "chase_reach": 0,
        "calls": 0,
        "player_kyokus_with_call": 0,
        "player_kyokus": 0,
        "bitten": 0,
        "hanchans": hanchans,
        "point_diffs": [],
    }


def _collect_env_events(bridge) -> list[list[list[dict]]]:
    """Collect parsed events per env → seat → list[dict], deduped seat-level."""
    result: list[list[list[dict]]] = []
    for env_index in range(len(bridge.last_events)):
        per_seat: list[list[dict]] = []
        for seat in range(NUM_PLAYERS):
            parsed: list[dict] = []
            for raw in bridge.last_events[env_index][seat]:
                try:
                    parsed.append(json.loads(raw))
                except (ValueError, TypeError):
                    pass
            per_seat.append(parsed)
        result.append(per_seat)
    return result


def _ingest_metrics(
    metrics_a: dict,
    metrics_b: dict,
    team_a: set[int],
    scores: list[int],
) -> None:
    """Aggregate this completed hanchan's events into the per-team metrics."""
    # ranking
    ranking = sorted(range(NUM_PLAYERS), key=lambda s: (-scores[s], s))
    first_seat = ranking[0]
    if first_seat in team_a:
        metrics_a["first_places"] += 1
    else:
        metrics_b["first_places"] += 1
    for rank, seat in enumerate(ranking, start=1):
        if seat in team_a:
            metrics_a["rank_sum"] += rank
            metrics_a["rank_dist"][rank] += 1
        else:
            metrics_b["rank_sum"] += rank
            metrics_b["rank_dist"][rank] += 1
        # 被飞
        if scores[seat] < 0:
            if seat in team_a:
                metrics_a["bitten"] += 1
            else:
                metrics_b["bitten"] += 1

    # Team score diff
    score_a = sum(scores[s] for s in team_a)
    score_b = sum(scores[s] for s in range(NUM_PLAYERS) if s not in team_a)
    diff = score_a - score_b
    metrics_a["team_point_diff_sum"] += diff
    metrics_b["team_point_diff_sum"] += -diff
    metrics_a["point_diffs"].append(diff)
    metrics_b["point_diffs"].append(-diff)
    if diff > 0:
        metrics_a["team_wins"] += 1
    elif diff < 0:
        metrics_b["team_wins"] += 1
    else:
        metrics_a["team_ties"] += 1
        metrics_b["team_ties"] += 1

def _ingest_action_events(
    metrics_a: dict, metrics_b: dict, team_a: set[int], events: list[dict],
    reached_this_kyoku: set[int], called_this_kyoku: set[int],
) -> None:
    for ev in events:
        kind = str(ev.get("type", ""))
        if kind == "hora":
            actor = int(ev.get("actor", -1))
            target = int(ev.get("target", -1))
            actor_metrics = metrics_a if actor in team_a else metrics_b
            actor_metrics["tsumo" if target == actor else "ron"] += 1
            if target != actor:
                if target in team_a and actor not in team_a:
                    metrics_a["dealin"] += 1
                elif target not in team_a and actor in team_a:
                    metrics_b["dealin"] += 1
        elif kind == "reach":
            actor = int(ev.get("actor", -1))
            target_metrics = metrics_a if actor in team_a else metrics_b
            target_metrics["reach"] += 1
            target_metrics["chase_reach" if reached_this_kyoku else "first_reach"] += 1
            reached_this_kyoku.add(actor)
        elif kind in {"chi", "pon", "daiminkan"}:
            actor = int(ev.get("actor", -1))
            (metrics_a if actor in team_a else metrics_b)["calls"] += 1
            called_this_kyoku.add(actor)


@torch.inference_mode()
def evaluate_2v2_detailed(
    model_a_path: str | Path,
    model_b_path: str | Path,
    *,
    device: str = "cuda",
    model_a_device: str | None = None,
    model_b_device: str | None = None,
    hanchan_count: int = 320,
    parallel_hanchans: int = 24,
    seed_base: int = 20260801,
    game_mode: str = "4p-red-half",
    max_steps: int = 4000,
) -> dict:
    try:
        import riichi
        from riichienv import BatchedRiichiEnv
    except ImportError as exc:
        raise RuntimeError("install local riichi and RiichiEnv extensions before evaluation") from exc

    default_device = torch.device(device)
    device_a = torch.device(model_a_device or default_device)
    device_b = torch.device(model_b_device or default_device)
    if (device_a.type == "cuda" or device_b.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    model_a_path = str(Path(model_a_path).resolve())
    model_b_path = str(Path(model_b_path).resolve())
    model_a = _load_model(model_a_path, device_a)
    model_b = _load_model(model_b_path, device_b)
    use_bf16_a, use_bf16_b = _bf16_supported(device_a), _bf16_supported(device_b)

    schedule = balanced_team_a_seats(hanchan_count)
    parallel = max(1, min(int(parallel_hanchans), int(hanchan_count)))
    started = time.perf_counter()

    metrics_a = _new_metrics(hanchan_count)
    metrics_b = _new_metrics(hanchan_count)

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
        bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(batch_size), batch_size)
        observations = list(envs.reset())
        bridge.sync(observations)
        public = PublicStateTracker(batch_size)
        public.update(bridge.last_events)
        analyzer = EfficiencyAnalyzer(131_072)
        active_envs = set(range(batch_size))
        kyoku_ids = [0] * batch_size
        reached_this_kyoku = [set() for _ in range(batch_size)]
        called_this_kyoku = [set() for _ in range(batch_size)]
        seen_event_keys: set[tuple[int, int, int, int, int]] = set()

        for environment_step in range(int(max_steps)):
            actions_by_env: list[dict[int, object]] = [{} for _ in range(batch_size)]
            decisions = active_decisions(observations, active_envs)
            analysis = (
                DecisionAnalysisBatch.build(decisions, analyzer=analyzer, public=public)
                if decisions
                else None
            )
            for decision in decisions:
                if any(action_kind(action) == "reach" for action in decision.observation.legal_actions()):
                    target_metrics = (
                        metrics_a if decision.seat_id in team_a_by_env[decision.env_index]
                        else metrics_b
                    )
                    target_metrics["reach_opportunities"] += 1
            for policy_name, model, model_device, use_bf16 in (
                ("a", model_a, device_a, use_bf16_a),
                ("b", model_b, device_b, use_bf16_b),
            ):
                policy_decisions = [
                    d for d in decisions
                    if (d.seat_id in team_a_by_env[d.env_index]) == (policy_name == "a")
                ]
                if not policy_decisions:
                    continue
                factors, numeric, lengths, legal, _g, _c, _cl = bridge.prepare(
                    policy_decisions, analysis,
                    token_schema_version=int(getattr(model, "token_schema_version", 13)),
                )
                with torch.autocast(device_type=model_device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    output = model.forward_policy(
                        _tensor(factors, model_device),
                        _tensor(numeric, model_device),
                        _tensor(legal, model_device),
                        _tensor(lengths, model_device),
                    )
                action_ids = output["policy_logits"].argmax(-1).tolist()
                actions = bridge.decode(policy_decisions, action_ids)
                for decision, action in zip(policy_decisions, actions, strict=True):
                    actions_by_env[decision.env_index][decision.seat_id] = action

            observations = list(envs.step_batch(actions_by_env))
            ended_kyoku, _ended_game = bridge.sync(observations)
            public.update(bridge.last_events)
            done = envs.done()
            scores_by_env = envs.scores()
            env_events_all = _collect_env_events(bridge)
            for env_index in list(active_envs):
                keyed_events = canonical_step_events(
                    env_events_all[env_index], environment_id=env_index,
                    hanchan_id=batch_start + env_index, kyoku_id=kyoku_ids[env_index],
                    step=environment_step,
                )
                unique_events: list[dict] = []
                for key, event in keyed_events:
                    if key in seen_event_keys:
                        continue
                    seen_event_keys.add(key)
                    unique_events.append(event)
                _ingest_action_events(
                    metrics_a, metrics_b, set(team_a_by_env[env_index]), unique_events,
                    reached_this_kyoku[env_index], called_this_kyoku[env_index],
                )
                if bool(ended_kyoku[env_index]):
                    metrics_a["player_kyokus"] += 2
                    metrics_b["player_kyokus"] += 2
                    metrics_a["player_kyokus_with_call"] += sum(
                        seat in team_a_by_env[env_index] for seat in called_this_kyoku[env_index]
                    )
                    metrics_b["player_kyokus_with_call"] += sum(
                        seat not in team_a_by_env[env_index] for seat in called_this_kyoku[env_index]
                    )
                    kyoku_ids[env_index] += 1
                    reached_this_kyoku[env_index].clear()
                    called_this_kyoku[env_index].clear()
                if not bool(done[env_index]):
                    continue
                scores = [int(v) for v in scores_by_env[env_index]]
                _ingest_metrics(
                    metrics_a,
                    metrics_b,
                    set(team_a_by_env[env_index]),
                    scores,
                )
                active_envs.remove(env_index)
                completed += 1
            if not active_envs:
                break
        else:
            raise RuntimeError(f"2v2 batch {batch_start // parallel} exceeded {max_steps} steps")
        print(
            f"head_to_head completed={completed}/{hanchan_count} "
            f"a_wins={metrics_a['team_wins']} b_wins={metrics_b['team_wins']} "
            f"ties={metrics_a['team_ties']} a_agari={metrics_a['tsumo']+metrics_a['ron']} "
            f"b_agari={metrics_b['tsumo']+metrics_b['ron']} "
            f"elapsed_s={time.perf_counter() - started:.2f}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    selected, reason = select_winner(
        model_a=model_a_path,
        model_b=model_b_path,
        wins_a=metrics_a["team_wins"],
        wins_b=metrics_b["team_wins"],
        ties=metrics_a["team_ties"],
        team_point_diff_sum=metrics_a["team_point_diff_sum"],
        first_places_a=metrics_a["first_places"],
        first_places_b=metrics_b["first_places"],
    )

    summary_a = _summarize_metrics(metrics_a, hanchan_count * 2)
    summary_b = _summarize_metrics(metrics_b, hanchan_count * 2)
    summary_a["checkpoint"] = model_a_path
    summary_b["checkpoint"] = model_b_path
    return {
        "schema_version": 2,
        "game_mode": game_mode,
        "hanchan_count": int(hanchan_count),
        "parallel_hanchans": parallel,
        "seed_base": int(seed_base),
        "greedy": True,
        "model_a": summary_a,
        "model_b": summary_b,
        "selected_checkpoint": selected,
        "selection_reason": reason,
        "elapsed_s": elapsed,
        "hanchan_per_s": hanchan_count / max(elapsed, 1e-9),
    }


def _format_summary_table(result: dict) -> str:
    rows = [
        ("Win rate", "team_win_rate"),
        ("Point diff (mean)", "team_point_diff_mean"),
        ("First-place rate", "first_place_rate"),
        ("Mean rank", "individual_mean_rank"),
        ("Top seat rate (1st)", "topseat_rate"),
        ("Fourth seat rate (4th)", "fourth_seat_rate"),
        ("Agari / player-kyoku", "agari_per_player_kyoku"),
        ("Tsumo / player-kyoku", "tsumo_per_player_kyoku"),
        ("Ron / player-kyoku", "ron_per_player_kyoku"),
        ("Dealin / player-kyoku", "dealin_per_player_kyoku"),
        ("Reach / player-kyoku", "reach_per_player_kyoku"),
        ("Reach / player-hanchan", "reach_per_player_hanchan"),
        ("Reach / opportunity", "reach_given_opportunity"),
        ("Calls / player-kyoku", "calls_per_player_kyoku"),
        ("Furo player-kyoku rate", "furo_player_kyoku_rate"),
        ("Bitten rate", "bitten_rate"),
        ("Hanchans", "individual_hanchans"),
        ("Rank distribution", "rank_distribution"),
    ]
    lines = []
    a, b = result["model_a"], result["model_b"]
    name_a = Path(a["checkpoint"]).name
    name_b = Path(b["checkpoint"]).name
    lines.append(
        f"{'metric':<26}  {name_a:<32}  {name_b:<32}"
    )
    lines.append("-" * (26 + 2 + 32 + 2 + 32))
    for label, key in rows:
        va = a.get(key, "—")
        vb = b.get(key, "—")
        if isinstance(va, float):
            va_str = f"{va:.4f}"
        elif isinstance(va, dict):
            va_str = " ".join(f"{k}:{v}" for k, v in sorted(va.items()))
        else:
            va_str = str(va)
        if isinstance(vb, float):
            vb_str = f"{vb:.4f}"
        elif isinstance(vb, dict):
            vb_str = " ".join(f"{k}:{v}" for k, v in sorted(vb.items()))
        else:
            vb_str = str(vb)
        lines.append(f"{label:<26}  {va_str:<32}  {vb_str:<32}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True, help="PPO candidate checkpoint.")
    parser.add_argument("--model-b", required=True, help="SFT baseline checkpoint.")
    parser.add_argument("--hanchans", type=int, default=320)
    parser.add_argument("--parallel-hanchans", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=20260801, help="Random base seed.")
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-a-device", default=None)
    parser.add_argument("--model-b-device", default=None)
    parser.add_argument("--output", required=True, help="JSON output path.")
    args = parser.parse_args()

    print(
        f"[benchmark] a={args.model_a}, b={args.model_b}, "
        f"hanchans={args.hanchans}, seed_base={args.seed_base}",
        flush=True,
    )
    result = evaluate_2v2_detailed(
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
    table = _format_summary_table(result)
    print("\n" + "=" * 80)
    print(table)
    print("=" * 80)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[benchmark] results written to {output}")
    print(f"[benchmark] elapsed {result['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
