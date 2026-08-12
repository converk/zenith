"""Step 2.5 full-hanchan Monte Carlo teacher-override audit.

For every audited decision (Step-2 high-confidence override or matched
conservative-keep control):

1. reconstruct the information state ``I`` by replaying the raw MJAI kyoku;
2. verify state-reconstruction fidelity (legal set, Policy Top1 action,
   Teacher-Best / strongest-challenger action, unambiguous identity);
3. sample hidden worlds from the Step-2 uniform baseline sampler (seeded, so
   future kyoku walls are branch-matched);
4. in each world force A = Policy Top1 and B = audited action, then continue
   greedily (PPO v2, argmax) with the SAME policy for every player until the
   entire hanchan ends (renchan / ryuukyoku / honba / kyotaku / south entry /
   tobi / final settlement handled by RiichiEnv ``4p-red-half``);
5. record the target seat's actual final rank and utility
   (1st=+10, 2nd=+4, 3rd=-4, 4th=-10) and the paired difference
   ``D = R_B - R_A`` per world.

Budget is adaptive: 64 -> 128 -> 256 -> 512 worlds for overrides
(64/128/256 for keep controls), stopping when the z=1.96 CI of ``D`` has
excluded 0 for two consecutive waves or the cap is reached.  The GRP V2 leaf
value is never used in this phase.

Run two shards with ``CUDA_DEVICE=0`` and ``CUDA_DEVICE=2`` (physical GPUs 0
and 3) on this host.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

REPO_ROOT = Path(__file__).resolve().parents[4]
STEP2_DIR = REPO_ROOT / "audit/reports/grp_ranker_20260811/step2_top4_rollout"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STEP2_DIR) not in sys.path:
    sys.path.insert(0, str(STEP2_DIR))

from riichienv import RiichiEnv  # noqa: E402

import paired_rollout as pr  # noqa: E402


@dataclass
class ExperimentConfig:
    seed_base: int = 20260813
    continuation_policy: str = (
        "checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt"
    )
    game_mode: str = "4p-red-half"
    raw_index: str = "datasets/tenhou_sft_2024_2025/index.jsonl.gz"
    raw_validation_dir: str = "datasets/tenhou_sft_2024_2025/validation"

    min_worlds: int = 64
    world_increment: int = 64
    max_worlds_override: int = 256
    max_worlds_keep: int = 256
    stability_waves: int = 2
    z_determined: float = 1.96
    max_wave_steps: int = 4000
    max_world_failure_rate: float = 0.25
    near_tie_effect: float = 0.5
    analysis_cache_capacity: int = 262_144


UTILITY = (10.0, 4.0, -4.0, -10.0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _i(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def _delta_stats(deltas: list[float], z: float) -> dict[str, float]:
    if not deltas:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
        }
    n = len(deltas)
    arr = np.asarray(deltas, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1) if n > 1 else 0.0)
    se = std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_lo": mean - z * se,
        "ci95_hi": mean + z * se,
    }


def classify_delta(stats: dict[str, float], *, near_tie_effect: float) -> tuple[str, str]:
    """Return (resolution, unresolved_type).

    ``SUPPORTED``: CI entirely > 0 (TeacherBest > PolicyTop1 in full-hanchan
    expected utility); ``REJECTED``: CI entirely < 0.  If the CI includes 0,
    the decision is unresolved; near-tie is singled out when the point
    estimate is within ``near_tie_effect`` utility points of zero.
    """
    if stats["n"] == 0 or not math.isfinite(stats["mean"]):
        return "UNRESOLVED", "no_data"
    if stats["ci95_lo"] > 0:
        return "SUPPORTED", ""
    if stats["ci95_hi"] < 0:
        return "REJECTED", ""
    if abs(stats["mean"]) <= near_tie_effect:
        return "UNRESOLVED", "near_tie"
    return "UNRESOLVED", "high_variance"


def keep_resolution(stats: dict[str, float], *, near_tie_effect: float) -> tuple[str, str]:
    """Keep-control resolution (challenger minus Top1).

    A negative Δ means Top1 is better, so the Step-2 keep verdict is
    confirmed.  ``SUPPORTED``/``REJECTED`` here are mapped to the keep
    decision itself (not the challenger's direction).
    """
    resolution, detail = classify_delta(stats, near_tie_effect=near_tie_effect)
    if resolution == "SUPPORTED":
        return "KEEP_REJECTED", detail  # challenger better -> keep was wrong
    if resolution == "REJECTED":
        return "KEEP_SUPPORTED", detail  # Top1 better -> keep was right
    return "KEEP_UNRESOLVED", detail


def _world_seed(config: ExperimentConfig, sid: str, world_idx: int) -> int:
    sid_hash = pr._stable_int(sid)
    return (
        int(config.seed_base) * 7919
        + sid_hash * 104729
        + int(world_idx) * 15485863
    ) % (2**31)


def reconstruct(
    row: dict[str, Any],
    store: pr.RawKyokuStore,
) -> tuple[RiichiEnv, Any, dict[str, Any], list[dict[str, Any]]]:
    content = store.read(str(row["game_id"]), int(row["kyoku_index"]))
    env, obs, _kyoku_initial, meta = pr.replay_to_decision(
        content,
        seat=int(row["seat"]),
        decision_index=int(row["decision_index"]),
    )
    events = [json.loads(line) for line in content.splitlines()]
    return env, obs, meta, events


def verify_reconstruction(
    row: dict[str, Any],
    obs: Any,
) -> tuple[bool, dict[str, Any]]:
    """Check legal-set / action-identity fidelity for the audited pair."""
    legal_ids = [pr.action_id(action, obs) for action in obs.legal_actions()]
    legal_ids = [int(value) for value in legal_ids if value is not None]
    top1_id = int(row["top1_action"])
    b_rank = int(row["audit_b_rank"])
    b_id = int(row["audit_b_action"])
    detail = {
        "legal_action_count": len(legal_ids),
        "top1_in_legal": top1_id in legal_ids,
        "b_in_legal": b_id in legal_ids,
        "top1_action": top1_id,
        "audit_b_rank": b_rank,
        "audit_b_action": b_id,
        "reconstruction_ok": top1_id in legal_ids and b_id in legal_ids,
    }
    return detail["reconstruction_ok"], detail


def _run_full_hanchan_wave(
    player: pr.BatchPlayer,
    envs: list[RiichiEnv],
    *,
    max_steps: int,
    progress: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], int]:
    """Drive all branches to end_game (whole hanchan), never to end_kyoku."""
    active = set(range(len(envs)))
    outcomes: dict[int, dict[str, Any]] = {}
    steps = 0
    kyoku_boundaries: Counter = Counter()
    while active:
        end_kyoku, end_game, decisions_this_step = player.step(active)
        steps += 1
        progress["policy_decisions"] += decisions_this_step
        for env_index in list(active):
            if bool(end_kyoku[env_index]):
                kyoku_boundaries[env_index] += 1
            if bool(end_game[env_index]) or bool(envs[env_index].done()):
                outcomes[env_index] = {
                    "scores": [int(value) for value in envs[env_index].scores()],
                    "ranks": [int(value) for value in envs[env_index].ranks()],
                    "done": bool(envs[env_index].done()),
                    "round_wind": int(envs[env_index].round_wind),
                    "kyoku_idx": int(envs[env_index].kyoku_idx),
                    "honba": int(envs[env_index].honba),
                    "riichi_sticks": int(envs[env_index].riichi_sticks),
                    "n_kyoku": int(kyoku_boundaries[env_index]),
                }
                active.remove(env_index)
        if steps > max_steps:
            raise RuntimeError(
                f"full-hanchan wave did not end within {max_steps} steps "
                f"(active={len(active)})"
            )
    if len(outcomes) != len(envs):
        raise RuntimeError("full-hanchan wave ended without all branch outcomes")
    return outcomes, steps


def evaluate_decision(
    row: dict[str, Any],
    store: pr.RawKyokuStore,
    adapter: Any,
    config: ExperimentConfig,
    *,
    progress: dict[str, Any],
) -> dict[str, Any]:
    sid = str(row["decision_id"])
    seat = int(row["seat"])
    b_rank = int(row["audit_b_rank"])
    is_override = str(row["group"]) == "override"
    max_worlds = (
        int(config.max_worlds_override)
        if is_override
        else int(config.max_worlds_keep)
    )

    env, obs, meta, events = reconstruct(row, store)
    recon_ok, recon_detail = verify_reconstruction(row, obs)
    if not recon_ok:
        progress["reconstruction_failures"] += 1
        return {
            "summary": {
                **row,
                "status": "reconstruction_failed",
                "n_worlds": 0,
                **recon_detail,
            },
            "branch_outcomes": [],
            "paired_deltas": [],
            "stability": [],
        }

    legal_actions = obs.legal_actions()
    forced: list[tuple[int, Any]] = []
    for rank, action_id_value in ((1, int(row["top1_action"])), (b_rank, int(row["audit_b_action"]))):
        found = next(
            (
                action
                for action in legal_actions
                if pr.action_id(action, obs) == action_id_value
            ),
            None,
        )
        if found is None:
            raise RuntimeError(f"{sid}: audited action {action_id_value} rank {rank} not forceable")
        forced.append((rank, found))

    target_rank_reward = [float(value) for value in UTILITY]
    deltas: list[float] = []
    branch_outcomes: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    wave_verdicts: deque[str] = deque(maxlen=int(config.stability_waves))
    n_worlds = 0
    n_worlds_failed = 0
    target_n = int(config.min_worlds)
    resolution = "UNRESOLVED"
    unresolved_type = ""

    while n_worlds < max_worlds:
        wave = min(target_n - n_worlds, max_worlds - n_worlds)
        world_indices = list(range(n_worlds, n_worlds + wave))
        envs: list[RiichiEnv] = []
        branch_meta: list[tuple[int, int]] = []  # (world_idx, audited rank)
        wave_failures = 0
        for world_idx in world_indices:
            world_seed = _world_seed(config, sid, world_idx)
            try:
                world, _rewritten = pr.sample_world(
                    env,
                    seat,
                    np.random.default_rng(world_seed),
                    events,
                    decision_index=int(row["decision_index"]),
                    decision_event_index=int(meta["decision_event_index"]),
                    world_seed=world_seed,
                )
            except Exception as exc:  # pragma: no cover - defensive
                wave_failures += 1
                progress["world_sampling_failures"] += 1
                print(f"[world] {sid} w{world_idx} sampling failed: {exc}", flush=True)
                continue
            for rank, action in forced:
                branch = world.clone()
                branch.step({seat: action})
                envs.append(branch)
                branch_meta.append((world_idx, rank))
        if not envs:
            raise RuntimeError(f"{sid}: no worlds could be sampled in wave")
        if (
            wave
            and wave_failures / wave > float(config.max_world_failure_rate)
        ):
            progress["world_sampling_failures"] += 1
            return {
                "summary": {
                    **row,
                    "status": "sampling_failed",
                    "n_worlds": n_worlds,
                    "n_worlds_failed": n_worlds_failed + wave_failures,
                    **recon_detail,
                },
                "branch_outcomes": [],
                "paired_deltas": [],
                "stability": [],
            }
        n_worlds_failed += wave_failures

        player = pr.BatchPlayer(
            adapter,
            envs,
            env_labels=[
                f"w{world_idx}-r{rank}"
                for world_idx, rank in branch_meta
            ],
            analysis_cache_capacity=config.analysis_cache_capacity,
        )
        try:
            outcomes, wave_steps = _run_full_hanchan_wave(
                player, envs, max_steps=int(config.max_wave_steps), progress=progress
            )
        except Exception as exc:
            progress["branch_failures"] += 1
            print(f"[branch] {sid} wave failed: {exc}", flush=True)
            raise

        progress["rollouts"] += len(envs)
        progress["waves"] += 1
        progress["wave_steps"] += wave_steps
        wave_values: dict[int, list[float]] = {}
        for (world_idx, rank), env_index in zip(branch_meta, range(len(envs)), strict=True):
            outcome = outcomes[env_index]
            target_rank = int(outcome["ranks"][seat])
            utility = target_rank_reward[target_rank - 1]
            wave_values.setdefault(world_idx, {})[rank] = utility
            branch_outcomes.append(
                {
                    "decision_id": sid,
                    "world_idx": world_idx,
                    "branch": "A" if rank == 1 else "B",
                    "candidate_rank": rank,
                    "action_id": int(row["top1_action"]) if rank == 1 else int(row["audit_b_action"]),
                    "final_scores": json.dumps(outcome["scores"], separators=(",", ":")),
                    "final_rank": target_rank,
                    "final_utility": utility,
                    "round_wind": outcome["round_wind"],
                    "kyoku_idx": outcome["kyoku_idx"],
                    "honba": outcome["honba"],
                    "riichi_sticks": outcome["riichi_sticks"],
                    "n_kyoku": outcome["n_kyoku"],
                    "done": outcome["done"],
                }
            )
        for world_idx in world_indices:
            if world_idx not in wave_values or 1 not in wave_values[world_idx] or b_rank not in wave_values[world_idx]:
                continue
            delta = wave_values[world_idx][b_rank] - wave_values[world_idx][1]
            deltas.append(delta)
            paired_rows.append(
                {
                    "decision_id": sid,
                    "world_idx": world_idx,
                    "r_a": wave_values[world_idx][1],
                    "r_b": wave_values[world_idx][b_rank],
                    "delta_full": delta,
                }
            )
        n_worlds += wave

        stats = _delta_stats(deltas, float(config.z_determined))
        resolution, unresolved_type = classify_delta(
            stats, near_tie_effect=float(config.near_tie_effect)
        )
        wave_verdicts.append(resolution)
        stability_rows.append(
            {
                "decision_id": sid,
                "n_worlds": n_worlds,
                "mean_delta_full": stats["mean"],
                "se_delta_full": stats["se"],
                "ci95_lo": stats["ci95_lo"],
                "ci95_hi": stats["ci95_hi"],
                "resolution": resolution,
                "unresolved_type": unresolved_type,
            }
        )
        stable = (
            len(wave_verdicts) == int(config.stability_waves)
            and all(value == wave_verdicts[-1] for value in wave_verdicts)
            and wave_verdicts[-1] != "UNRESOLVED"
        )
        print(
            f"[fullmc] {sid} group={row['group']} n={n_worlds} "
            f"delta={stats['mean']:.3f}±{stats['se']:.3f} "
            f"ci=[{stats['ci95_lo']:.3f},{stats['ci95_hi']:.3f}] "
            f"resolution={resolution} elapsed={time.perf_counter() - progress['started']:.1f}s",
            flush=True,
        )
        if stable or n_worlds >= max_worlds:
            break
        target_n = min(n_worlds + int(config.world_increment), max_worlds)

    stats = _delta_stats(deltas, float(config.z_determined))
    resolution, unresolved_type = classify_delta(
        stats, near_tie_effect=float(config.near_tie_effect)
    )
    mean_r_a, std_r_a = _mean_std(
        [float(row["r_a"]) for row in paired_rows]
    )
    mean_r_b, std_r_b = _mean_std(
        [float(row["r_b"]) for row in paired_rows]
    )
    teacher_rank = int(row["teacher_best"])
    teacher_delta_key = {2: "delta_ba_mean", 3: "delta_ca_mean", 4: "delta_da_mean"}.get(teacher_rank)
    teacher_se_key = {2: "delta_ba_se", 3: "delta_ca_se", 4: "delta_da_se"}.get(teacher_rank)
    strongest_delta, strongest_rank = float("-inf"), 1
    for rank, key in ((2, "delta_ba_mean"), (3, "delta_ca_mean"), (4, "delta_da_mean")):
        value = _f(row.get(key))
        if math.isfinite(value) and value > strongest_delta:
            strongest_delta, strongest_rank = value, rank
    if is_override:
        predicted_delta = _f(row.get(teacher_delta_key)) if teacher_delta_key else float("nan")
        predicted_se = _f(row.get(teacher_se_key)) if teacher_se_key else float("nan")
    else:
        predicted_delta = (
            _f(row.get({2: "delta_ba_mean", 3: "delta_ca_mean", 4: "delta_da_mean"}[strongest_rank]))
            if strongest_rank in (2, 3, 4)
            else float("nan")
        )
        predicted_se = (
            _f(row.get({2: "delta_ba_se", 3: "delta_ca_se", 4: "delta_da_se"}[strongest_rank]))
            if strongest_rank in (2, 3, 4)
            else float("nan")
        )

    summary = {
        **row,
        "status": "ok",
        "n_worlds": n_worlds,
        "n_worlds_failed": n_worlds_failed,
        "mean_r_a": mean_r_a,
        "mean_r_b": mean_r_b,
        "std_r_a": std_r_a,
        "std_r_b": std_r_b,
        "mean_delta_full": stats["mean"],
        "median_delta_full": float(np.median(deltas)) if deltas else float("nan"),
        "std_delta_full": stats["std"],
        "se_delta_full": stats["se"],
        "ci95_lo": stats["ci95_lo"],
        "ci95_hi": stats["ci95_hi"],
        "resolution": resolution,
        "unresolved_type": unresolved_type,
        "keep_resolution": (
            keep_resolution(stats, near_tie_effect=float(config.near_tie_effect))[0]
            if not is_override
            else ""
        ),
        "predicted_delta_grp": predicted_delta,
        "predicted_delta_grp_se": predicted_se,
        "predicted_delta_grp_ci_lo": (
            predicted_delta - 1.96 * predicted_se if math.isfinite(predicted_se) else float("nan")
        ),
        "predicted_delta_grp_ci_hi": (
            predicted_delta + 1.96 * predicted_se if math.isfinite(predicted_se) else float("nan")
        ),
        "strongest_challenger_rank": strongest_rank,
        "strongest_challenger_delta_grp": strongest_delta,
        "round_wind_meta": int(meta.get("round_wind", -1)),
        "kyoku_meta": int(meta.get("kyoku_index", -1)),
        "honba_meta": int(meta.get("honba", -1)),
        **recon_detail,
    }
    return {
        "summary": summary,
        "branch_outcomes": branch_outcomes,
        "paired_deltas": paired_rows,
        "stability": stability_rows,
    }


class GpuSampler:
    """Background nvidia-smi sampler for the physical GPU used by this shard."""

    PHYSICAL_INDEX = {"0": 0, "1": 1, "2": 3, "3": 4}

    def __init__(self) -> None:
        self.device = os.environ.get("CUDA_DEVICE", "0")
        self.physical = self.PHYSICAL_INDEX.get(self.device, 0)
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(self.physical),
        ]
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=10
                ).stdout.strip()
                parts = output.replace("%", "").split(",")
                if len(parts) == 2:
                    self.samples.append((float(parts[0]), float(parts[1])))
            except Exception:
                pass
            self._stop.wait(15.0)

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=20)
        if not self.samples:
            return {"gpu_util_pct": float("nan"), "gpu_mem_used_mb": float("nan")}
        return {
            "gpu_util_pct": round(float(np.mean([s[0] for s in self.samples])), 1),
            "gpu_mem_used_mb": round(float(np.mean([s[1] for s in self.samples])), 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--only-ids", default=None,
        help="comma-separated decision ids (applied after sharding)",
    )
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or (Path(__file__).resolve().parent / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig()
    if args.config:
        overrides = json.loads(Path(args.config).read_text())
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    pr._seed_rng(int(config.seed_base) + int(args.shard_id))
    (out_dir / "fullmc_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False)
    )

    samples = _read_csv(out_dir / "audit_samples.csv")
    samples = [
        row for index, row in enumerate(samples)
        if index % int(args.num_shards) == int(args.shard_id)
    ]
    if args.resume:
        summary_path = out_dir / f"audit_summary_shard{args.shard_id}.csv"
        if summary_path.exists():
            done = {row["decision_id"] for row in _read_csv(summary_path)}
            skipped = sum(1 for row in samples if row["decision_id"] in done)
            samples = [row for row in samples if row["decision_id"] not in done]
            print(f"[fullmc] resume: skipped {skipped} completed decisions")
    if args.limit:
        samples = samples[: int(args.limit)]
    if args.only_ids:
        allowed = set(args.only_ids.split(","))
        samples = [row for row in samples if row["decision_id"] in allowed]
        print(f"[fullmc] only-ids filter: {len(samples)} decisions")
    print(f"[fullmc] shard={args.shard_id} decisions={len(samples)}")

    summaries: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    if args.append:
        for name, sink in (
            (f"audit_summary_shard{args.shard_id}.csv", summaries),
            (f"branch_outcomes_shard{args.shard_id}.csv", branch_rows),
            (f"paired_deltas_shard{args.shard_id}.csv", delta_rows),
            (f"stability_shard{args.shard_id}.csv", stability_rows),
        ):
            path = out_dir / name
            if path.exists():
                sink.extend(_read_csv(path))

    device = torch.device(args.device)
    adapter = pr.load_policy_adapter(
        str(REPO_ROOT / config.continuation_policy), device=device
    )
    store = pr.RawKyokuStore(
        REPO_ROOT / config.raw_index,
        REPO_ROOT / config.raw_validation_dir,
    )
    store.load_needed({str(row["game_id"]) for row in samples})
    print(f"[fullmc] raw kyoku locations loaded: {len(store.locations)}")

    sampler = GpuSampler()
    sampler.start()
    progress: dict[str, Any] = {
        "started": time.perf_counter(),
        "cpu_start": time.process_time(),
        "policy_decisions": 0,
        "rollouts": 0,
        "waves": 0,
        "wave_steps": 0,
        "reconstruction_failures": 0,
        "world_sampling_failures": 0,
        "branch_failures": 0,
    }
    for index, row in enumerate(samples):
        result = evaluate_decision(
            row, store, adapter, config, progress=progress
        )
        summaries.append(result["summary"])
        branch_rows.extend(result["branch_outcomes"])
        delta_rows.extend(result["paired_deltas"])
        stability_rows.extend(result["stability"])
        if (index + 1) % 2 == 0 or index + 1 == len(samples):
            _write_csv(
                out_dir / f"audit_summary_shard{args.shard_id}.csv", summaries
            )
            _write_csv(
                out_dir / f"branch_outcomes_shard{args.shard_id}.csv",
                branch_rows,
            )
            _write_csv(
                out_dir / f"paired_deltas_shard{args.shard_id}.csv",
                delta_rows,
            )
            _write_csv(
                out_dir / f"stability_shard{args.shard_id}.csv",
                stability_rows,
            )
            elapsed = time.perf_counter() - progress["started"]
            print(
                f"[fullmc] shard={args.shard_id} progress={index + 1}/{len(samples)} "
                f"elapsed={elapsed:.1f}s policy_decisions/s="
                f"{progress['policy_decisions'] / elapsed:.1f} "
                f"rollouts={progress['rollouts']}",
                flush=True,
            )
    gpu_stats = sampler.stop()
    store.close()
    _write_csv(out_dir / f"audit_summary_shard{args.shard_id}.csv", summaries)
    _write_csv(out_dir / f"branch_outcomes_shard{args.shard_id}.csv", branch_rows)
    _write_csv(out_dir / f"paired_deltas_shard{args.shard_id}.csv", delta_rows)
    _write_csv(out_dir / f"stability_shard{args.shard_id}.csv", stability_rows)
    elapsed = time.perf_counter() - progress["started"]
    cpu_elapsed = time.process_time() - progress["cpu_start"]
    summary = {
        "shard_id": args.shard_id,
        "decisions": len(samples),
        "worlds": sum(
            int(_f(row.get("n_worlds")) or 0)
            for row in summaries
        ),
        "policy_decisions": progress["policy_decisions"],
        "rollouts": progress["rollouts"],
        "waves": progress["waves"],
        "wave_steps": progress["wave_steps"],
        "reconstruction_failures": progress["reconstruction_failures"],
        "world_sampling_failures": progress["world_sampling_failures"],
        "elapsed_s": round(elapsed, 3),
        "policy_decisions_per_s": round(progress["policy_decisions"] / elapsed, 2),
        "rollouts_per_s": round(progress["rollouts"] / elapsed, 2),
        "avg_policy_batch_size": round(
            progress["policy_decisions"] / max(1, progress["wave_steps"]), 2
        ),
        "cpu_seconds": round(cpu_elapsed, 3),
        "cpu_core_util_pct": round(100 * cpu_elapsed / elapsed, 1),
        **gpu_stats,
    }
    (out_dir / f"fullmc_summary_shard{args.shard_id}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"[fullmc] shard={args.shard_id} done:", json.dumps(summary))


if __name__ == "__main__":
    main()
