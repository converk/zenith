"""Standalone GRP V2 pairwise long-term value-ranking verification.

Pipeline (matches audit/reports/grp_ranker_20260811/GOAL_PROMPT.md):

  generate -> select -> grp -> mc -> merge-mc -> pairs

* ``generate``  plays full hanchans with the configured policy and records
  every non-final kyoku-ending state together with a minimal resume spec
  (scores, oya, round wind, honba, kyotaku).
* ``select``    picks a diverse subset of states across kyoku stages and
  score-spread situations.
* ``grp``       computes GRP V2 expected utility for every seat.
* ``mc``        runs adaptive Monte-Carlo continuations from each state to
  the end of the hanchan (batched, resumable, shardable across GPUs).
* ``merge-mc``  combines sharded MC output into one state_values table.
* ``pairs``     constructs a-priori pairs (hard + easy sanity pairs) and
  writes pairwise_results.csv, calibration.csv and summary.json.

The Monte-Carlo ground truth is the final rank reward used by the GRP
system: rank 1 -> 10, rank 2 -> 4, rank 3 -> -4, rank 4 -> -10.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichienv import RiichiEnv  # noqa: E402
import riichi  # noqa: E402

from riichi_ppo_v1.grp.model import (  # noqa: E402
    RankPredictor,
    grp_features_from_scores,
    reward_from_rank_probs,
)
from riichi_ppo_v1.model.bridge import BatchedStateBridge  # noqa: E402
from riichi_ppo_v1.sft.policy_adapter import load_policy_adapter  # noqa: E402
from riichi_ppo_v1.training.rewards import (  # noqa: E402
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from riichi_ppo_v1.training.worker import active_decisions  # noqa: E402


@dataclass
class ExperimentConfig:
    seed_base: int = 20260811
    policy_checkpoint: str = (
        "checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt"
    )
    grp_checkpoint: str = (
        "checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt"
    )
    grp_pts_weight: tuple[float, float, float, float] = (10.0, 4.0, -4.0, -10.0)
    game_mode: str = "4p-red-half"

    # State generation / selection
    generator_games: int = 36
    states_per_ordinal: int = 6
    max_states: int = 48

    # MC continuation budget (adaptive)
    mc_min_samples: int = 64
    mc_increment: int = 64
    mc_max_samples: int = 192
    mc_se_threshold: float = 0.65  # stop once max per-seat SE <= threshold
    mc_batch_size: int = 96
    mc_analysis_cache_capacity: int = 262_144

    # Pair construction (a-priori, observable-score based)
    hard_score_gap: int = 10_000
    easy_score_gap: int = 15_000
    hard_pairs_per_group_seat: int = 8
    easy_pairs_per_group_seat: int = 2

    # Statistics
    z_determined: float = 1.96  # 95% CI
    z_determined_secondary: float = 1.28  # 80% CI
    tie_delta_threshold: float = 0.5  # |deltaMC| below this counts as tie
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260812


STAGE_GROUPS: dict[str, tuple[int, int]] = {
    "early": (0, 1),
    "middle": (2, 4),
    "late": (5, 999),
}


def stage_group(ordinal: int) -> str:
    for name, (lo, hi) in STAGE_GROUPS.items():
        if lo <= int(ordinal) <= hi:
            return name
    return "late"


def rank_reward(rank: int, weights: Iterable[float]) -> float:
    return float(list(weights)[int(rank) - 1])


def _seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed % (2**32))


class BatchPlayer:
    """Plays a batch of individual RiichiEnv tables with the loaded policy."""

    def __init__(
        self,
        adapter: Any,
        envs: list[RiichiEnv],
        *,
        analysis_cache_capacity: int = 262_144,
    ) -> None:
        self.adapter = adapter
        self.envs = envs
        self.n = len(envs)
        self.bridge = BatchedStateBridge(
            riichi.MjaiKyokuStateMachineManager(self.n), self.n
        )
        self.public = PublicStateTracker(self.n)
        self.analyzer = EfficiencyAnalyzer(analysis_cache_capacity)
        self.observations = [env.get_observations() for env in envs]
        self.bridge.sync(self.observations)
        self.public.update(self.bridge.last_events)

    def step(self, active: set[int]) -> tuple[np.ndarray, np.ndarray, int]:
        decisions = active_decisions(self.observations, active)
        actions_by_env: list[dict[int, Any]] = [{} for _ in range(self.n)]
        if decisions:
            analysis = DecisionAnalysisBatch.build(
                decisions,
                analyzer=self.analyzer,
                public=self.public,
            )
            prepared = self.adapter.prepare(self.bridge, decisions, analysis)
            logits = self.adapter.masked_logits(prepared)
            probs = torch.softmax(logits, dim=-1)
            action_ids = torch.multinomial(probs, 1).squeeze(1).tolist()
            actions = self.bridge.decode(decisions, action_ids)
            for decision, action in zip(decisions, actions, strict=True):
                actions_by_env[decision.env_index][decision.seat_id] = action
        for env, row in zip(self.envs, actions_by_env, strict=True):
            if row:
                env.step(row)
        self.observations = [env.get_observations() for env in self.envs]
        end_kyoku, end_game = self.bridge.sync(self.observations)
        self.public.update(self.bridge.last_events)
        return end_kyoku, end_game, len(decisions)


def _field(observation: Any) -> tuple[int, int, int, int]:
    return (
        int(observation.round_wind),
        int(observation.kyoku_index),
        int(observation.honba),
        int(observation.riichi_sticks),
    )


def _ranks(scores: list[int]) -> list[int]:
    order = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    result = [0] * 4
    for position, seat in enumerate(order):
        result[seat] = position + 1
    return result


def _fresh_env(seed: int) -> RiichiEnv:
    env = RiichiEnv(game_mode="4p-red-half", seed=seed, round_wind=0)
    env.reset(oya=0, honba=0, kyotaku=0, scores=[25000] * 4, round_wind=0, seed=seed)
    return env


def _resume_env(state: dict[str, Any], seed: int) -> RiichiEnv:
    env = RiichiEnv(
        game_mode="4p-red-half",
        seed=seed,
        round_wind=int(state["next_round_wind"]),
    )
    env.reset(
        oya=int(state["next_oya"]),
        honba=int(state["next_honba"]),
        kyotaku=int(state["next_kyotaku"]),
        scores=[int(x) for x in state["round_end_scores"]],
        round_wind=int(state["next_round_wind"]),
        seed=seed,
    )
    return env


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import csv

    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def generate_states(config: ExperimentConfig, out_dir: Path) -> None:
    """Play full hanchans with the policy and record kyoku-ending states."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter = load_policy_adapter(config.policy_checkpoint, device=device)
    n_games = int(config.generator_games)
    envs = [_fresh_env(int(config.seed_base) + i) for i in range(n_games)]
    player = BatchPlayer(adapter, envs, analysis_cache_capacity=config.mc_analysis_cache_capacity)
    start_scores = [env.scores() for env in envs]
    kyoku_fields = [_field(obs[0]) for obs in player.observations]
    kyoku_counters = [0] * n_games
    active = set(range(n_games))
    states: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    decisions = 0
    started = time.perf_counter()
    step_count = 0
    while active:
        end_kyoku, end_game, step_decisions = player.step(active)
        decisions += step_decisions
        step_count += 1
        scores_by_env = [env.scores() for env in envs]
        done_indices: list[int] = []
        for env_index in list(active):
            scores = scores_by_env[env_index]
            if bool(end_kyoku[env_index]):
                chang, ju, ben, sticks = kyoku_fields[env_index]
                ordinal = int(chang) * 4 + int(ju)
                if not bool(end_game[env_index]):
                    final_ranks = _ranks(scores)
                    states.append(
                        {
                            "state_id": f"g{int(config.seed_base) + env_index:06d}_ky{ordinal:02d}",
                            "game_seed": int(config.seed_base) + env_index,
                            "kyoku_ordinal": ordinal,
                            "chang": chang,
                            "ju": ju,
                            "honba": ben,
                            "riichi_sticks": sticks,
                            "round_initial_scores": start_scores[env_index],
                            "round_end_scores": scores,
                            "next_oya": int(envs[env_index].oya),
                            "next_round_wind": int(envs[env_index].round_wind),
                            "next_honba": int(envs[env_index].honba),
                            "next_kyotaku": int(envs[env_index].riichi_sticks),
                            "end_rank_seat0": final_ranks[0],
                            "end_rank_seat1": final_ranks[1],
                            "end_rank_seat2": final_ranks[2],
                            "end_rank_seat3": final_ranks[3],
                            "score_spread": int(max(scores) - min(scores)),
                            "stage_group": stage_group(ordinal),
                        }
                    )
                kyoku_counters[env_index] += 1
                start_scores[env_index] = scores
                kyoku_fields[env_index] = _field(player.observations[env_index][0])
            if bool(end_game[env_index]):
                final_ranks = _ranks(scores)
                games.append(
                    {
                        "game_seed": int(config.seed_base) + env_index,
                        "final_scores": scores,
                        "final_rank_seat0": final_ranks[0],
                        "final_rank_seat1": final_ranks[1],
                        "final_rank_seat2": final_ranks[2],
                        "final_rank_seat3": final_ranks[3],
                        "kyoku_count": kyoku_counters[env_index],
                    }
                )
                active.remove(env_index)
        if step_count % 100 == 0:
            print(
                f"[generate] step={step_count} active={len(active)} states={len(states)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "generator_games.csv", games)
    _write_csv(out_dir / "generated_states.csv", states)
    summary = {
        "games": n_games,
        "states_recorded": len(states),
        "elapsed_s": round(elapsed, 3),
        "hanchan_per_s": round(n_games / elapsed, 4),
    }
    (out_dir / "generate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("[generate] done:", json.dumps(summary))


def select_states(config: ExperimentConfig, out_dir: Path) -> None:
    rows = _read_csv(out_dir / "generated_states.csv")
    for row in rows:
        for key in (
            "round_initial_scores",
            "round_end_scores",
            "score_spread",
            "next_oya",
            "next_round_wind",
            "next_honba",
            "next_kyotaku",
            "kyoku_ordinal",
            "honba",
            "riichi_sticks",
            "chang",
            "ju",
        ):
            if key in ("round_initial_scores", "round_end_scores"):
                row[key] = [int(x) for x in json.loads(row[key])]
            elif key in {"score_spread", "next_oya", "next_round_wind", "next_honba", "next_kyotaku", "kyoku_ordinal", "honba", "riichi_sticks", "chang", "ju"}:
                row[key] = int(row[key])
    by_ordinal: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ordinal[int(row["kyoku_ordinal"])].append(row)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal in sorted(by_ordinal):
        group = sorted(by_ordinal[ordinal], key=lambda r: (int(r["score_spread"]), r["state_id"]))
        take = min(int(config.states_per_ordinal), len(group))
        if take == 0:
            continue
        if take == 1:
            indices = [0]
        else:
            indices = sorted(
                {round(i * (len(group) - 1) / (take - 1)) for i in range(take)}
            )
        for index in indices:
            if len(selected) >= int(config.max_states):
                break
            row = group[index]
            if row["state_id"] not in seen:
                seen.add(row["state_id"])
                selected.append(row)
        if len(selected) >= int(config.max_states):
            break
    # Fill remaining budget with extra diversity (sorted by spread percentile).
    if len(selected) < int(config.max_states):
        remaining = [row for row in rows if row["state_id"] not in seen]
        ordinals = sorted({int(row["kyoku_ordinal"]) for row in remaining})
        for ordinal in ordinals:
            group = sorted(
                (row for row in remaining if int(row["kyoku_ordinal"]) == ordinal),
                key=lambda r: (int(r["score_spread"]), r["state_id"]),
            )
            for row in group:
                if len(selected) >= int(config.max_states):
                    break
                seen.add(row["state_id"])
                selected.append(row)
    selected.sort(key=lambda r: r["state_id"])
    _write_csv(out_dir / "selected_states.csv", selected)
    per_ordinal = defaultdict(int)
    per_group = defaultdict(int)
    for row in selected:
        per_ordinal[int(row["kyoku_ordinal"])] += 1
        per_group[row["stage_group"]] += 1
    summary = {
        "selected": len(selected),
        "per_ordinal": dict(sorted(per_ordinal.items())),
        "per_stage_group": dict(per_group),
    }
    (out_dir / "select_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("[select] done:", json.dumps(summary))


def compute_grp(config: ExperimentConfig, out_dir: Path, device_str: str) -> None:
    rows = _read_csv(out_dir / "selected_states.csv")
    for row in rows:
        row["round_end_scores"] = [int(x) for x in json.loads(row["round_end_scores"])]
        row["round_initial_scores"] = [int(x) for x in json.loads(row["round_initial_scores"])]
    device = torch.device(device_str)
    model = RankPredictor.from_checkpoint(config.grp_checkpoint)
    model.to(device)
    model.eval()
    batch = []
    for row in rows:
        for seat in range(4):
            batch.append(
                grp_features_from_scores(
                    row["round_initial_scores"],
                    row["round_end_scores"],
                    chang=int(row["chang"]),
                    ju=int(row["ju"]),
                    ben=int(row["honba"]),
                    liqibang=int(row["riichi_sticks"]),
                    player=seat,
                )
            )
    features = np.stack(batch)
    probs = model.predict_rank_probs(features, device=device)
    for row_index, row in enumerate(rows):
        for seat in range(4):
            p = probs[row_index * 4 + seat]
            value = reward_from_rank_probs(p, config.grp_pts_weight)
            row[f"grp_p1_seat{seat}"] = float(p[0])
            row[f"grp_p2_seat{seat}"] = float(p[1])
            row[f"grp_p3_seat{seat}"] = float(p[2])
            row[f"grp_p4_seat{seat}"] = float(p[3])
            row[f"grp_value_seat{seat}"] = value
    _write_csv(out_dir / "grp_values.csv", rows)
    print(f"[grp] computed values for {len(rows)} states")


def run_mc(
    config: ExperimentConfig,
    out_dir: Path,
    device_str: str,
    *,
    shard_id: int,
    num_shards: int,
) -> None:
    rows = _read_csv(out_dir / "selected_states.csv")
    for row in rows:
        row["round_end_scores"] = [int(x) for x in json.loads(row["round_end_scores"])]
    states = [row for index, row in enumerate(rows) if index % int(num_shards) == int(shard_id)]
    device = torch.device(device_str)
    adapter = load_policy_adapter(config.policy_checkpoint, device=device)
    cont_path = out_dir / f"mc_continuations_shard{shard_id}.csv"
    values_path = out_dir / f"state_values_mc_shard{shard_id}.csv"
    samples_by_state: dict[str, list[dict[str, Any]]] = {}
    if cont_path.exists():
        for row in _read_csv(cont_path):
            sid = row["state_id"]
            samples_by_state.setdefault(sid, []).append(row)
    for key in samples_by_state:
        samples_by_state[key].sort(key=lambda r: int(r["sample_idx"]))
    started_all = time.perf_counter()
    total_continuations = 0
    total_decisions = 0
    for state_index, state in enumerate(states):
        sid = state["state_id"]
        samples = samples_by_state.setdefault(sid, [])
        target = int(config.mc_min_samples)
        while len(samples) < int(config.mc_max_samples):
            batch_size = min(
                int(config.mc_batch_size),
                target - len(samples),
            )
            if batch_size <= 0:
                break
            seeds = [
                (
                    1_000_003 * int(config.seed_base)
                    + 17 * (int(shard_id) * 1_000 + state_index)
                    + len(samples)
                    + sample_offset
                )
                % (2**31)
                for sample_offset in range(batch_size)
            ]
            envs = [_resume_env(state, seed) for seed in seeds]
            player = BatchPlayer(
                adapter,
                envs,
                analysis_cache_capacity=config.mc_analysis_cache_capacity,
            )
            active = set(range(len(envs)))
            batch_rewards = [None] * len(envs)
            batch_final_scores = [None] * len(envs)
            batch_decisions = 0
            while active:
                _end_kyoku, end_game, step_decisions = player.step(active)
                batch_decisions += step_decisions
                for env_index in list(active):
                    if bool(end_game[env_index]):
                        final_scores = envs[env_index].scores()
                        final_ranks = _ranks(final_scores)
                        batch_final_scores[env_index] = final_scores
                        batch_rewards[env_index] = [
                            rank_reward(final_ranks[seat], config.grp_pts_weight)
                            for seat in range(4)
                        ]
                        active.remove(env_index)
            for sample_offset in range(len(envs)):
                row = {
                    "state_id": sid,
                    "sample_idx": len(samples),
                    "seed": seeds[sample_offset],
                    "final_scores": batch_final_scores[sample_offset],
                    "final_rank_seat0": _ranks(batch_final_scores[sample_offset])[0],
                    "final_rank_seat1": _ranks(batch_final_scores[sample_offset])[1],
                    "final_rank_seat2": _ranks(batch_final_scores[sample_offset])[2],
                    "final_rank_seat3": _ranks(batch_final_scores[sample_offset])[3],
                    "reward_seat0": batch_rewards[sample_offset][0],
                    "reward_seat1": batch_rewards[sample_offset][1],
                    "reward_seat2": batch_rewards[sample_offset][2],
                    "reward_seat3": batch_rewards[sample_offset][3],
                }
                samples.append(row)
                total_continuations += 1
            total_decisions += batch_decisions
            _flush_mc(
                out_dir,
                shard_id,
                cont_path,
                values_path,
                state,
                samples_by_state,
            )
            se_by_seat = _mc_se(samples)
            max_se = max(se_by_seat.values())
            print(
                f"[mc] shard={shard_id} state={sid} n={len(samples)} "
                f"se={ {seat: round(se_by_seat[seat], 3) for seat in range(4)} }",
                flush=True,
            )
            if max_se <= float(config.mc_se_threshold) and len(samples) >= int(config.mc_min_samples):
                break
            target += int(config.mc_increment)
    elapsed = time.perf_counter() - started_all
    summary = {
        "shard_id": shard_id,
        "states": len(states),
        "continuations": total_continuations,
        "elapsed_s": round(elapsed, 3),
        "decisions": total_decisions,
    }
    (out_dir / f"mc_summary_shard{shard_id}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"[mc] shard={shard_id} done:", json.dumps(summary))


def _mc_se(samples: list[dict[str, Any]]) -> dict[int, float]:
    result = {}
    for seat in range(4):
        values = np.asarray([float(sample[f"reward_seat{seat}"]) for sample in samples])
        result[seat] = (
            float(values.std(ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
    return result


def _flush_mc(
    out_dir: Path,
    shard_id: int,
    cont_path: Path,
    values_path: Path,
    state: dict[str, Any],
    samples_by_state: dict[str, list[dict[str, Any]]],
) -> None:
    all_rows = sorted(
        (
            dict(row)
            for rows in samples_by_state.values()
            for row in rows
        ),
        key=lambda row: (row["state_id"], int(row["sample_idx"])),
    )
    _write_csv(cont_path, all_rows)
    value_rows = []
    for sid, samples in sorted(samples_by_state.items()):
        value_row = {"state_id": sid, "n": len(samples)}
        for seat in range(4):
            values = np.asarray(
                [float(sample[f"reward_seat{seat}"]) for sample in samples]
            )
            value_row[f"mc_mean_seat{seat}"] = (
                float(values.mean()) if len(values) else None
            )
            value_row[f"mc_std_seat{seat}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            value_row[f"mc_se_seat{seat}"] = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        value_rows.append(value_row)
    _write_csv(values_path, value_rows)


def merge_mc(out_dir: Path) -> None:
    """Merge sharded state values and continuation tables."""
    value_rows: list[dict[str, Any]] = []
    cont_rows: list[dict[str, Any]] = []
    shard_files = sorted(out_dir.glob("state_values_mc_shard*.csv"))
    cont_files = sorted(out_dir.glob("mc_continuations_shard*.csv"))
    for path in shard_files:
        value_rows.extend(_read_csv(path))
    for path in cont_files:
        cont_rows.extend(_read_csv(path))
    value_rows.sort(key=lambda row: row["state_id"])
    cont_rows.sort(key=lambda row: (row["state_id"], int(row["sample_idx"])))
    _write_csv(out_dir / "state_values_mc.csv", value_rows)
    _write_csv(out_dir / "mc_continuations.csv", cont_rows)
    selected = {row["state_id"]: row for row in _read_csv(out_dir / "selected_states.csv")}
    grp = {row["state_id"]: row for row in _read_csv(out_dir / "grp_values.csv")}
    mc = {row["state_id"]: row for row in value_rows}
    combined = []
    for state_id in sorted(selected):
        row = dict(selected[state_id])
        for source in (grp, mc):
            row.update(
                {key: value for key, value in source.get(state_id, {}).items() if key not in row}
            )
        combined.append(row)
    _write_csv(out_dir / "state_values.csv", combined)
    print(
        f"[merge-mc] states={len(value_rows)} continuations={len(cont_rows)} "
        f"from {len(shard_files)} value shards / {len(cont_files)} continuation shards"
    )


def construct_pairs(config: ExperimentConfig, out_dir: Path) -> list[dict[str, Any]]:
    states = _read_csv(out_dir / "selected_states.csv")
    grp = {row["state_id"]: row for row in _read_csv(out_dir / "grp_values.csv")}
    mc = {row["state_id"]: row for row in _read_csv(out_dir / "state_values_mc.csv")}
    for row in states:
        row["round_end_scores"] = [int(x) for x in json.loads(row["round_end_scores"])]
        row["score_spread"] = int(row["score_spread"])
    pairs: list[dict[str, Any]] = []
    for seat in range(4):
        for group in ("early", "middle", "late"):
            group_states = [
                row for row in states
                if row["stage_group"] == group
                and row["state_id"] in grp
                and row["state_id"] in mc
            ]
            hard: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
            easy: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
            for i in range(len(group_states)):
                for j in range(i + 1, len(group_states)):
                    a, b = group_states[i], group_states[j]
                    if a["game_seed"] == b["game_seed"]:
                        continue
                    diff = abs(
                        int(a["round_end_scores"][seat])
                        - int(b["round_end_scores"][seat])
                    )
                    if diff <= int(config.hard_score_gap):
                        hard.append((diff, a, b))
                    elif diff >= int(config.easy_score_gap):
                        easy.append((diff, a, b))
            hard.sort(key=lambda item: (item[0], item[1]["state_id"], item[2]["state_id"]))
            easy.sort(key=lambda item: (-item[0], item[1]["state_id"], item[2]["state_id"]))
            for diff, a, b in hard[: int(config.hard_pairs_per_group_seat)]:
                pairs.append(
                    _pair_row(a, b, seat, "hard", diff, grp, mc)
                )
            for diff, a, b in easy[: int(config.easy_pairs_per_group_seat)]:
                pairs.append(
                    _pair_row(a, b, seat, "easy", diff, grp, mc)
                )
    _write_csv(out_dir / "pairwise_results.csv", pairs)
    print(f"[pairs] constructed {len(pairs)} pairs")
    return pairs


def _pair_row(
    a: dict[str, Any],
    b: dict[str, Any],
    seat: int,
    kind: str,
    score_diff: int,
    grp: dict[str, dict[str, Any]],
    mc: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ga = float(grp[a["state_id"]][f"grp_value_seat{seat}"])
    gb = float(grp[b["state_id"]][f"grp_value_seat{seat}"])
    ma = float(mc[a["state_id"]][f"mc_mean_seat{seat}"])
    mb = float(mc[b["state_id"]][f"mc_mean_seat{seat}"])
    sea = float(mc[a["state_id"]][f"mc_se_seat{seat}"])
    seb = float(mc[b["state_id"]][f"mc_se_seat{seat}"])
    return {
        "pair_id": f"{a['state_id']}|{b['state_id']}|s{seat}",
        "pair_key": f"{a['state_id']}|{b['state_id']}",
        "state_a": a["state_id"],
        "state_b": b["state_id"],
        "seat": seat,
        "pair_kind": kind,
        "stage_group": a["stage_group"],
        "kyoku_ordinal_a": int(a["kyoku_ordinal"]),
        "kyoku_ordinal_b": int(b["kyoku_ordinal"]),
        "score_diff": int(score_diff),
        "grp_value_a": ga,
        "grp_value_b": gb,
        "mc_value_a": ma,
        "mc_value_b": mb,
        "mc_se_a": sea,
        "mc_se_b": seb,
        "delta_grp": ga - gb,
        "delta_mc": ma - mb,
        "delta_mc_se": math.sqrt(sea * sea + seb * seb),
        "n_a": int(mc[a["state_id"]]["n"]),
        "n_b": int(mc[b["state_id"]]["n"]),
    }


def compute_statistics(
    config: ExperimentConfig,
    out_dir: Path,
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    from scipy import stats as scipy_stats

    z1 = float(config.z_determined)
    z2 = float(config.z_determined_secondary)

    def determined(row: dict[str, Any], z: float = z1) -> bool:
        return abs(row["delta_mc"]) > z * row["delta_mc_se"]

    def tie(row: dict[str, Any]) -> bool:
        return abs(row["delta_mc"]) < float(config.tie_delta_threshold)

    def accuracy(subset: list[dict[str, Any]]) -> float | None:
        rows = [row for row in subset if determined(row)]
        if not rows:
            return None
        return float(
            np.mean([np.sign(row["delta_grp"]) == np.sign(row["delta_mc"]) for row in rows])
        )

    def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (centre - margin, centre + margin)

    def clustered_bootstrap_ci(
        rows: list[dict[str, Any]],
        determined_filter: bool = True,
    ) -> tuple[float, float]:
        selected = [row for row in rows if determined(row)] if determined_filter else rows
        if not selected:
            return (0.0, 0.0)
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_pair[row["pair_key"]].append(row)
        pair_keys = sorted(by_pair)
        rng = np.random.default_rng(int(config.bootstrap_seed))
        accs: list[float] = []
        for _ in range(int(config.bootstrap_replicates)):
            sampled_keys = rng.choice(pair_keys, size=len(pair_keys), replace=True)
            sampled = [row for key in sampled_keys for row in by_pair[key]]
            accs.append(
                float(
                    np.mean(
                        [np.sign(row["delta_grp"]) == np.sign(row["delta_mc"]) for row in sampled]
                    )
                )
            )
        accs = np.asarray(accs)
        return (
            float(np.percentile(accs, 2.5)),
            float(np.percentile(accs, 97.5)),
        )

    def pearson_spearman(subset: list[dict[str, Any]]) -> dict[str, float | None]:
        if len(subset) < 3:
            return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None, "n": len(subset)}
        dx = np.asarray([row["delta_grp"] for row in subset], dtype=float)
        dy = np.asarray([row["delta_mc"] for row in subset], dtype=float)
        pr = scipy_stats.pearsonr(dx, dy)
        sr = scipy_stats.spearmanr(dx, dy)
        return {
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
            "n": len(subset),
        }

    overall_determined = [row for row in pairs if determined(row)]
    overall_secondary = [row for row in pairs if determined(row, z2)]
    hard = [row for row in pairs if row["pair_kind"] == "hard"]
    easy = [row for row in pairs if row["pair_kind"] == "easy"]

    overall_acc = accuracy(pairs)
    def accuracy_z(subset: list[dict[str, Any]], z: float) -> float | None:
        rows = [row for row in subset if determined(row, z)]
        if not rows:
            return None
        return float(np.mean([np.sign(row["delta_grp"]) == np.sign(row["delta_mc"]) for row in rows]))

    overall_acc_secondary = accuracy_z(pairs, z2)
    hard_acc = accuracy(hard)
    easy_acc = accuracy(easy)
    per_stage: dict[str, dict[str, Any]] = {}
    per_ordinal: dict[str, dict[str, Any]] = {}
    for stage in ("early", "middle", "late"):
        subset = [row for row in pairs if row["stage_group"] == stage]
        acc = accuracy(subset)
        det = [row for row in subset if determined(row)]
        per_stage[stage] = {
            "n": len(subset),
            "n_determined95": len(det),
            "accuracy95": acc,
            "accuracy95_secondary": accuracy_z(subset, z2),
        }
    for ordinal in sorted({row["kyoku_ordinal_a"] for row in pairs}):
        subset = [
            row for row in pairs
            if row["kyoku_ordinal_a"] == ordinal and row["kyoku_ordinal_b"] == ordinal
        ]
        if len(subset) >= 8:
            per_ordinal[str(ordinal)] = {
                "n": len(subset),
                "n_determined95": len([row for row in subset if determined(row)]),
                "accuracy95": accuracy(subset),
            }

    # Calibration buckets by |delta_grp| quantiles (all pairs).
    abs_grp = np.asarray([abs(row["delta_grp"]) for row in pairs], dtype=float)
    quantiles = [0.2, 0.4, 0.6, 0.8]
    edges = list(np.quantile(abs_grp, quantiles))
    edges = sorted(set(edges))
    buckets: list[dict[str, Any]] = []
    bounds = [(-math.inf, edges[0])] if edges else [(-math.inf, math.inf)]
    bounds += [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    bounds.append((edges[-1], math.inf))
    for lo, hi in bounds:
        subset = [
            row for row in pairs
            if abs(row["delta_grp"]) >= lo and abs(row["delta_grp"]) < hi
        ]
        det = [row for row in subset if determined(row)]
        buckets.append(
            {
                "bucket": f"{'-inf' if lo == -math.inf else round(lo, 4)}..{'+inf' if hi == math.inf else round(hi, 4)}",
                "n": len(subset),
                "n_determined95": len(det),
                "mean_abs_delta_grp": float(np.mean([abs(row["delta_grp"]) for row in subset])) if subset else None,
                "mean_delta_grp": float(np.mean([row["delta_grp"] for row in subset])) if subset else None,
                "mean_delta_mc": float(np.mean([row["delta_mc"] for row in subset])) if subset else None,
                "mean_abs_delta_mc": float(np.mean([abs(row["delta_mc"]) for row in subset])) if subset else None,
                "accuracy95": accuracy(subset),
            }
        )

    # High-confidence subsets by |delta_grp| percentile.
    high_conf: dict[str, dict[str, Any]] = {}
    for label, percentile in (("top50", 50), ("top25", 75), ("top10", 90)):
        threshold = float(np.percentile(abs_grp, percentile))
        subset = [row for row in pairs if abs(row["delta_grp"]) >= threshold]
        det = [row for row in subset if determined(row)]
        high_conf[label] = {
            "threshold_abs_delta_grp": threshold,
            "n": len(subset),
            "coverage": len(subset) / max(len(pairs), 1),
            "n_determined95": len(det),
            "accuracy95": accuracy(subset),
            "accuracy95_secondary": accuracy_z(subset, z2),
            "mean_delta_mc": float(np.mean([row["delta_mc"] for row in subset])) if subset else None,
            "mean_abs_delta_mc": float(np.mean([abs(row["delta_mc"]) for row in subset])) if subset else None,
            "mean_abs_delta_grp": float(np.mean([abs(row["delta_grp"]) for row in subset])) if subset else None,
        }

    summary: dict[str, Any] = {
        "n_pairs": len(pairs),
        "n_hard": len(hard),
        "n_easy": len(easy),
        "n_pairs_by_stage": {stage: len([r for r in pairs if r["stage_group"] == stage]) for stage in ("early", "middle", "late")},
        "mc_uncertain95_rate": len(pairs) - len(overall_determined),
        "mc_uncertain95_frac": (len(pairs) - len(overall_determined)) / max(len(pairs), 1),
        "mc_uncertain80_rate": len(pairs) - len(overall_secondary),
        "mc_uncertain80_frac": (len(pairs) - len(overall_secondary)) / max(len(pairs), 1),
        "mc_tie_rate": sum(tie(row) for row in pairs),
        "mc_tie_frac": sum(tie(row) for row in pairs) / max(len(pairs), 1),
        "overall": {
            "n": len(pairs),
            "n_determined95": len(overall_determined),
            "accuracy95": overall_acc,
            "accuracy95_secondary": overall_acc_secondary,
            "wilson95_ci": wilson_ci(
                int(round(overall_acc * len(overall_determined))),
                len(overall_determined),
            ) if overall_determined and overall_acc is not None else None,
            "clustered_bootstrap95_ci": clustered_bootstrap_ci(pairs),
            "correlation": pearson_spearman(pairs),
            "correlation_determined95": pearson_spearman(overall_determined),
        },
        "hard": {
            "n": len(hard),
            "n_determined95": len([row for row in hard if determined(row)]),
            "accuracy95": hard_acc,
            "accuracy95_secondary": accuracy_z(hard, z2),
            "wilson95_ci": wilson_ci(
                int(round(hard_acc * len([row for row in hard if determined(row)]))),
                len([row for row in hard if determined(row)]),
            ) if hard_acc is not None else None,
            "clustered_bootstrap95_ci": clustered_bootstrap_ci(hard),
            "correlation": pearson_spearman(hard),
        },
        "easy": {
            "n": len(easy),
            "n_determined95": len([row for row in easy if determined(row)]),
            "accuracy95": easy_acc,
        },
        "per_stage": per_stage,
        "per_ordinal": per_ordinal,
        "calibration_buckets": buckets,
        "high_confidence": high_conf,
    }
    _write_csv(
        out_dir / "calibration.csv",
        [{"bucket": b["bucket"], **{k: v for k, v in b.items() if k != "bucket"}} for b in buckets],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    parser.add_argument("--config", default=None, help="optional JSON config overrides")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    sub.add_parser("select")
    sub.add_parser("grp").add_argument("--device", default="cuda:0")
    mc_parser = sub.add_parser("mc")
    mc_parser.add_argument("--device", default="cuda:0")
    mc_parser.add_argument("--shard-id", type=int, default=0)
    mc_parser.add_argument("--num-shards", type=int, default=1)
    sub.add_parser("merge-mc")
    sub.add_parser("pairs")
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.config:
        overrides = json.loads(Path(args.config).read_text())
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_rng(int(config.seed_base))
    (out_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False)
    )

    if args.command == "generate":
        generate_states(config, out_dir)
    elif args.command == "select":
        select_states(config, out_dir)
    elif args.command == "grp":
        compute_grp(config, out_dir, args.device)
    elif args.command == "mc":
        run_mc(config, out_dir, args.device, shard_id=args.shard_id, num_shards=args.num_shards)
    elif args.command == "merge-mc":
        merge_mc(out_dir)
    elif args.command == "pairs":
        if not (out_dir / "state_values_mc.csv").exists():
            merge_mc(out_dir)
        pairs = construct_pairs(config, out_dir)
        summary = compute_statistics(config, out_dir, pairs)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
