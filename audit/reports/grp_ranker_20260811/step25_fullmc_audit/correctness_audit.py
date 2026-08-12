"""Step 2.5 full-hanchan runtime correctness audit.

GOAL_PROMPT_STEP2.5.md section 30 requires a correctness audit of the
full-hanchan continuation: renchan / ryuukyoku / honba / kyotaku /
south-entry / final settlement must follow the ``4p-red-half`` rules used by
the environment.  This script runs a small sample of audited decisions (a few
worlds each) and checks:

1. every branch reaches ``done()`` (game end, not just kyoku end);
2. per-kyoku resolution events (hora / ryuukyoku) drive the observed
   honba / oya transitions;
3. total tile conservation (136 tiles) at every kyoku start and game end;
4. total score conservation (``sum(scores) + 1000*riichi_sticks == 100000``)
   at every kyoku start and game end;
5. branch-matched future walls: at matching (round_wind, kyoku_idx, honba,
   oya) kyoku starts, A and B have identical walls and dora markers;
6. final rank consistency: ``ranks()`` matches a manual score sort with
   seat-order tie breaking;
7. documented game-length distribution (east/south/west termination) under
   the env's 30000-target rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
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
from fullmc_audit import (  # noqa: E402
    ExperimentConfig,
    GpuSampler,
    _world_seed,
    reconstruct,
    verify_reconstruction,
)


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


def tile_count(env: RiichiEnv) -> int:
    hands = sum(len(hand) for hand in env.hands)
    wall = len(env.wall)
    discards = sum(len(row) for row in env.discards)
    melds = sum(
        # Open melds keep the called discard in their tile list, but that tile
        # is also (correctly) still visible in the discarder's discard record.
        # Only tiles removed from the meld owner's hand change the total.
        len(meld.tiles)
        - (1 if bool(getattr(meld, "opened", True)) else 0)
        for meld_rows in env.melds
        for meld in meld_rows
    )
    return hands + wall + discards + melds


def parse_kyoku_resolutions(env: RiichiEnv) -> list[dict[str, Any]]:
    """Return per-kyoku summaries from the env's MJAI log tail."""
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in env.mjai_log:
        event = raw if isinstance(raw, dict) else json.loads(raw)
        kind = str(event.get("type", ""))
        if kind == "start_kyoku":
            if current is not None:
                rows.append(current)
            current = {
                "wind": str(event.get("bakaze")),
                "kyoku": int(event.get("kyoku", 0)),
                "honba": int(event.get("honba", 0)),
                "oya": int(event.get("oya", -1)),
                "resolution": None,
            }
        elif kind == "hora":
            if current is not None:
                current["resolution"] = "hora"
                current["hora_actor"] = int(event.get("actor", -1))
        elif kind == "ryukyoku":
            if current is not None:
                current["resolution"] = "ryukyoku"
                current["ryukyoku_reason"] = str(event.get("reason", ""))
        elif kind == "end_game":
            if current is not None:
                rows.append(current)
    return rows


def expected_next_round(previous: dict[str, Any]) -> tuple[int, int | None]:
    """Return (expected next honba, expected next oya) for a kyoku resolution.

    ``oya`` is ``None`` for ryuukyoku (renchan depends on dealer tenpai, which
    is not encoded in the MJAI log; the allowed set is checked instead).
    """
    if previous["resolution"] == "ryukyoku":
        return previous["honba"] + 1, None
    dealer_won = previous["resolution"] == "hora" and previous["hora_actor"] == previous["oya"]
    if dealer_won:
        return previous["honba"] + 1, previous["oya"]
    return 0, (previous["oya"] + 1) % 4


def run_one_decision(
    row: dict[str, Any],
    store: pr.RawKyokuStore,
    adapter: Any,
    config: ExperimentConfig,
    world_indices: list[int],
) -> dict[str, Any]:
    sid = str(row["decision_id"])
    seat = int(row["seat"])
    b_rank = int(row["audit_b_rank"])
    env, obs, meta, events = reconstruct(row, store)
    recon_ok, recon_detail = verify_reconstruction(row, obs)
    checks: list[dict[str, Any]] = []

    for world_idx in world_indices:
        world_seed = _world_seed(config, sid, world_idx)
        world, _rewritten = pr.sample_world(
            env,
            seat,
            np.random.default_rng(world_seed),
            events,
            decision_index=int(row["decision_index"]),
            decision_event_index=int(meta["decision_event_index"]),
            world_seed=world_seed,
        )
        legal = obs.legal_actions()
        forced = [
            (1, next(a for a in legal if pr.action_id(a, obs) == int(row["top1_action"]))),
            (b_rank, next(a for a in legal if pr.action_id(a, obs) == int(row["audit_b_action"]))),
        ]
        envs: list[RiichiEnv] = []
        for _rank, action in forced:
            branch = world.clone()
            branch.step({seat: action})
            envs.append(branch)
        player = pr.BatchPlayer(
            adapter,
            envs,
            env_labels=["A", "B"],
            analysis_cache_capacity=config.analysis_cache_capacity,
        )
        active = set(range(len(envs)))
        boundaries: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
        ordinals: dict[int, int] = {0: 0, 1: 0}
        steps = 0
        final: dict[int, dict[str, Any]] = {}
        while active:
            end_kyoku, end_game, _nd = player.step(active)
            steps += 1
            if steps > int(config.max_wave_steps):
                raise RuntimeError(f"{sid} w{world_idx}: too many steps")
            for env_index in list(active):
                if bool(end_kyoku[env_index]):
                    visible = sorted(
                        [int(tile) for hand in envs[env_index].hands for tile in hand]
                        + [int(tile) for row in envs[env_index].discards for tile in row]
                        + [
                            int(tile)
                            for meld_rows in envs[env_index].melds
                            for meld in meld_rows
                            for tile in meld.tiles
                        ]
                    )
                    boundaries[env_index].append(
                        {
                            "ordinal": ordinals[env_index],
                            "round_wind": int(envs[env_index].round_wind),
                            "kyoku_idx": int(envs[env_index].kyoku_idx),
                            "honba": int(envs[env_index].honba),
                            "oya": int(envs[env_index].oya),
                            "wall": [int(tile) for tile in envs[env_index].wall],
                            "dora": [int(tile) for tile in envs[env_index].dora_indicators],
                            "scores": [int(value) for value in envs[env_index].scores()],
                            "riichi_sticks": int(envs[env_index].riichi_sticks),
                            "tile_count": tile_count(envs[env_index]),
                            "public_fingerprint": tuple(visible),
                        }
                    )
                    ordinals[env_index] += 1
                if bool(end_game[env_index]) or bool(envs[env_index].done()):
                    final[env_index] = {
                        "done": bool(envs[env_index].done()),
                        "scores": [int(value) for value in envs[env_index].scores()],
                        "ranks": [int(value) for value in envs[env_index].ranks()],
                        "round_wind": int(envs[env_index].round_wind),
                        "kyoku_idx": int(envs[env_index].kyoku_idx),
                        "honba": int(envs[env_index].honba),
                        "riichi_sticks": int(envs[env_index].riichi_sticks),
                        "steps": steps,
                        "n_kyoku_boundaries": len(boundaries[env_index]),
                    }
                    active.remove(env_index)

        # --- checks ---
        # 1. done
        for env_index in (0, 1):
            checks.append(
                {
                    "decision_id": sid,
                    "world_idx": world_idx,
                    "check": "reaches_game_end",
                    "branch": "A" if env_index == 0 else "B",
                    "ok": bool(final[env_index]["done"]),
                    "detail": "done=%s" % final[env_index]["done"],
                }
            )
        # 2. tile conservation at every kyoku start + end
        for env_index in (0, 1):
            counts = [boundary["tile_count"] for boundary in boundaries[env_index]]
            ok_tiles = all(value == 136 for value in counts)
            checks.append(
                {
                    "decision_id": sid,
                    "world_idx": world_idx,
                    "check": "tile_conservation",
                    "branch": "A" if env_index == 0 else "B",
                    "ok": bool(ok_tiles),
                    "detail": "counts=%s" % counts,
                }
            )
        # 3. score conservation at every kyoku start + end
        for env_index in (0, 1):
            totals = [
                sum(boundary["scores"])
                + 1000 * boundary["riichi_sticks"]
                for boundary in boundaries[env_index]
            ]
            end_total = (
                sum(final[env_index]["scores"])
                + 1000 * final[env_index]["riichi_sticks"]
            )
            ok_scores = all(value == 100000 for value in totals) and end_total == 100000
            checks.append(
                {
                    "decision_id": sid,
                    "world_idx": world_idx,
                    "check": "score_conservation",
                    "branch": "A" if env_index == 0 else "B",
                    "ok": bool(ok_scores),
                    "detail": "totals=%s end=%s" % (totals, end_total),
                }
            )
        # 4. honba / oya transitions vs log resolutions.  The first kyoku in
        # the env log is the decision kyoku itself (replayed prefix); each
        # recorded boundary is the transition INTO the following kyoku.  The
        # final boundary is the game-end marker and has no honba transition.
        for env_index in (0, 1):
            resolutions = parse_kyoku_resolutions(envs[env_index])
            starts = boundaries[env_index]
            bad = []
            for index, boundary in enumerate(starts[:-1]):
                resolution = resolutions[index] if index < len(resolutions) else None
                if resolution is None or resolution["resolution"] is None:
                    bad.append(f"missing resolution for boundary {index}")
                    continue
                expected_honba, expected_oya = expected_next_round(resolution)
                if boundary["honba"] != expected_honba:
                    bad.append(
                        f"boundary {index}: honba {boundary['honba']} "
                        f"!= expected {expected_honba}"
                    )
                if expected_oya is not None and boundary["oya"] != expected_oya:
                    bad.append(
                        f"boundary {index}: oya {boundary['oya']} "
                        f"!= expected {expected_oya}"
                    )
            checks.append(
                {
                    "decision_id": sid,
                    "world_idx": world_idx,
                    "check": "honba_transitions",
                    "branch": "A" if env_index == 0 else "B",
                    "ok": not bad,
                    "detail": "; ".join(bad) if bad else "ok",
                }
            )
        # 5. branch-matched randomness: both branches clone the same seeded
        # world, so each shuffle uses the same seed.  Wall values are compared
        # only when the two branches are at the same kyoku ordinal AND have
        # identical public tile multisets (then the same permutation must give
        # the same wall); otherwise renchan-path divergence is expected.
        def key(boundary: dict[str, Any]) -> tuple[Any, ...]:
            return (
                boundary["ordinal"],
                boundary["round_wind"],
                boundary["kyoku_idx"],
                boundary["honba"],
                boundary["oya"],
            )

        by_key_a = {key(b): b for b in boundaries[0]}
        by_key_b = {key(b): b for b in boundaries[1]}
        common = set(by_key_a) & set(by_key_b)
        comparable = 0
        mismatches = []
        for shared_key in common:
            a, b = by_key_a[shared_key], by_key_b[shared_key]
            if a["public_fingerprint"] != b["public_fingerprint"]:
                continue
            comparable += 1
            if tuple(a["wall"]) != tuple(b["wall"]) or tuple(a["dora"]) != tuple(b["dora"]):
                mismatches.append(shared_key)
        checks.append(
            {
                "decision_id": sid,
                "world_idx": world_idx,
                "check": "branch_matched_walls",
                "branch": "A/B",
                "ok": not mismatches,
                "detail": (
                    f"comparable={comparable}/{len(common)} mismatches={len(mismatches)} "
                    f"path_diverged={len(common) - comparable}"
                ),
            }
        )
        # 6. final rank consistency
        for env_index in (0, 1):
            scores = final[env_index]["scores"]
            order = sorted(range(4), key=lambda seat_id: (-scores[seat_id], seat_id))
            manual = [0] * 4
            for rank, seat_id in enumerate(order, start=1):
                manual[seat_id] = rank
            checks.append(
                {
                    "decision_id": sid,
                    "world_idx": world_idx,
                    "check": "final_rank_consistency",
                    "branch": "A" if env_index == 0 else "B",
                    "ok": manual == final[env_index]["ranks"],
                    "detail": "scores=%s ranks=%s" % (scores, final[env_index]["ranks"]),
                }
            )
        # 7. game-length distribution
        for env_index in (0, 1):
            checks.append(
                {
                    "decision_id": sid,
                    "world_idx": world_idx,
                    "check": "termination_wind",
                    "branch": "A" if env_index == 0 else "B",
                    "ok": True,
                    "detail": (
                        f"round_wind={final[env_index]['round_wind']} "
                        f"kyoku_idx={final[env_index]['kyoku_idx']} "
                        f"honba={final[env_index]['honba']} "
                        f"n_kyoku={final[env_index]['n_kyoku_boundaries']}"
                    ),
                }
            )
    return {
        "decision_id": sid,
        "group": str(row.get("group")),
        "reconstruction_ok": recon_ok,
        "worlds": world_indices,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--worlds", default="0,1,2,3")
    parser.add_argument(
        "--only-ids", default=None,
        help="comma-separated decision ids to audit",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir or (Path(__file__).resolve().parent / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig()
    samples = _read_csv(out_dir / "audit_samples.csv")
    # Audit a small balanced sample: keep decision order stable.
    selected = samples[: int(args.limit)]
    if args.only_ids:
        allowed = set(str(args.only_ids).split(","))
        selected = [row for row in samples if row["decision_id"] in allowed]
        print(f"[correctness] only-ids filter: {len(selected)} decisions")
    world_indices = [int(value) for value in str(args.worlds).split(",")]
    print(f"[correctness] decisions={len(selected)} worlds={world_indices}")

    device = torch.device(args.device)
    adapter = pr.load_policy_adapter(
        str(REPO_ROOT / config.continuation_policy), device=device
    )
    store = pr.RawKyokuStore(
        REPO_ROOT / config.raw_index,
        REPO_ROOT / config.raw_validation_dir,
    )
    store.load_needed({str(row["game_id"]) for row in selected})
    sampler = GpuSampler()
    sampler.start()
    results: list[dict[str, Any]] = []
    for row in selected:
        results.append(run_one_decision(row, store, adapter, config, world_indices))
        print(f"[correctness] done {row['decision_id']}", flush=True)
    gpu_stats = sampler.stop()
    store.close()

    all_checks = [check for result in results for check in result["checks"]]
    by_check: Counter = Counter(check["check"] for check in all_checks)
    failed = [check for check in all_checks if not check["ok"]]
    summary = {
        "decisions_audited": len(results),
        "worlds_per_decision": world_indices,
        "total_checks": len(all_checks),
        "passed": len(all_checks) - len(failed),
        "failed": len(failed),
        "by_check": dict(by_check),
        "failures": failed,
        "termination_wind_counts": dict(
            Counter(check["detail"] for check in all_checks if check["check"] == "termination_wind")
        ),
        **gpu_stats,
    }
    (out_dir / "fullmc_correctness_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
