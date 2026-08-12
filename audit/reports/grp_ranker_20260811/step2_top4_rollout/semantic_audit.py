"""Business-semantics audit for the Step-2 sampled data and rollout pipeline.

Checks, on a sample of selected decisions:

1. Reconstruction fidelity: replaying the raw kyoku to the target decision and
   re-encoding it through the batched state machine must reproduce the policy
   Top4 action ids (and near-identical probabilities) that were used to build
   ``selected_decisions.csv`` from the precomputed validation encoding.
2. World-consistency invariants: tile-type conservation across hands + melds
   + discards + dora + wall, dora-slot preservation, hand sizes, and a couple
   of greedy rollouts that terminate at ``end_kyoku`` with finite GRP values.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import riichi

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from paired_rollout import (  # noqa: E402
    BatchPlayer,
    ExperimentConfig,
    GRPBatcher,
    RawKyokuStore,
    _build_unknown_pool,
    _mjai_to_tile_id,
    build_pre_streams,
    decision_id,
    replay_to_decision,
    sample_world,
)
from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision  # noqa: E402
from riichi_ppo_v1.sft.policy_adapter import load_policy_adapter  # noqa: E402
from riichi_ppo_v1.training.rewards import (  # noqa: E402
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from riichi_ppo_v1.training.rewards.decision import action_id  # noqa: E402
from riichi_ppo_v1.training.worker import active_decisions  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, object]]:
    import csv

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_reconstruction_fidelity(
    rows: list[dict[str, object]],
    store: RawKyokuStore,
    adapter: object,
    device: torch.device,
    *,
    limit: int = 8,
) -> list[dict[str, object]]:
    """Re-encode a decision through the bridge and compare with the CSV row."""
    results: list[dict[str, object]] = []
    for row in rows[:limit]:
        content = store.read(str(row["game_id"]), int(row["kyoku_index"]))
        env, obs, _initial, meta = replay_to_decision(
            content,
            seat=int(row["seat"]),
            decision_index=int(row["decision_index"]),
        )
        seat = int(row["seat"])
        events = [json.loads(line) for line in content.splitlines()]
        streams = build_pre_streams(events, int(meta["decision_event_index"]))
        state_machine = riichi.MjaiKyokuStateMachineManager(1)
        state_machine.apply_events_batch([0], [streams])
        bridge = BatchedStateBridge(state_machine, 1)
        public = PublicStateTracker(1)
        public.update([streams])
        analyzer = EfficiencyAnalyzer()
        for seat_index in range(4):
            env.get_observations()[seat_index].new_events()
        observations = [env.get_observations()]
        bridge.observations_by_env = observations
        bridge.last_events = [[]]
        decisions = active_decisions(observations, {0})
        match = next((d for d in decisions if d.seat_id == seat), None)
        if match is None:
            results.append({
                "decision_id": decision_id(row),
                "ok": False,
                "error": "reconstructed state has no decision for target seat",
            })
            continue
        analysis = DecisionAnalysisBatch.build(
            [match], analyzer=analyzer, public=public
        )
        prepared = adapter.prepare(bridge, [match], analysis)
        logits = adapter.masked_logits(prepared)[0].float()
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        order = np.argsort(-probs, kind="stable")
        legal_ids = [
            action_id(a, obs) for a in obs.legal_actions()
        ]
        top_ids = [int(a) for a in order if int(a) in legal_ids][:4]
        csv_top = [
            int(row["top1_action"]), int(row["top2_action"]),
            int(row["top3_action"]), int(row["top4_action"]),
        ]
        ok_ids = top_ids == csv_top
        pi = [float(probs[a]) for a in top_ids]
        csv_pi = [float(row[f"pi{i}"]) for i in (1, 2, 3, 4)]
        max_pi_err = max(abs(p - q) for p, q in zip(pi, csv_pi, strict=True)) if ok_ids else None
        top1_agrees = top_ids[0] == csv_top[0]
        top4_set_agrees = sorted(top_ids) == sorted(csv_top)
        results.append({
            "decision_id": decision_id(row),
            "ok": bool(ok_ids),
            "top1_agrees": bool(top1_agrees),
            "top4_set_agrees": bool(top4_set_agrees),
            "top_ids_replayed": top_ids,
            "top_ids_csv": csv_top,
            "pi_replayed": [round(p, 5) for p in pi],
            "pi_csv": [round(p, 5) for p in csv_pi],
            "max_pi_error": None if max_pi_err is None else round(float(max_pi_err), 5),
            "legal_count": len(legal_ids),
        })
    return results


def check_world_invariants(
    rows: list[dict[str, object]],
    store: RawKyokuStore,
    *,
    limit: int = 5,
    worlds_per_decision: int = 3,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in rows[:limit]:
        content = store.read(str(row["game_id"]), int(row["kyoku_index"]))
        env, obs, _initial, meta = replay_to_decision(
            content,
            seat=int(row["seat"]),
            decision_index=int(row["decision_index"]),
        )
        events = [json.loads(line) for line in content.splitlines()]
        seat = int(row["seat"])
        sid = decision_id(row)
        for world_index in range(worlds_per_decision):
            rng = np.random.default_rng(10_000 + world_index)
            try:
                world, _rewritten = sample_world(
                    env, seat, rng, events,
                    decision_index=int(row["decision_index"]),
                    decision_event_index=int(meta["decision_event_index"]),
                )
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "decision_id": sid,
                    "world_index": world_index,
                    "ok": False,
                    "error": f"sample_world: {type(exc).__name__}: {exc}",
                })
                continue
            # Tile-type conservation: every type 0..33 appears exactly 4 times
            # across hands + melds + discards + wall.  Revealed dora indicator
            # tiles are physically part of the wall (their dead-wall slots are
            # preserved in ``world.wall``), so dora_indicators must not be
            # counted again separately.
            total = Counter()
            for hand in world.hands:
                total.update(int(t) // 4 for t in hand)
            for meld_rows in world.melds:
                for meld in meld_rows:
                    tiles = [int(t) for t in meld.tiles]
                    if bool(getattr(meld, "opened", True)):
                        # The first tile of an open meld is the called discard,
                        # which is already counted in the discarder's river.
                        total.update(t // 4 for t in tiles[1:])
                    else:
                        total.update(t // 4 for t in tiles)
            for discards in world.discards:
                total.update(int(t) // 4 for t in discards)
            total.update(int(t) // 4 for t in world.wall)
            conserved = all(total[t] == 4 for t in range(34))
            # Dora slots preserved.
            dora_ok = True
            for index, dora_tile in enumerate(env.dora_indicators):
                slot = 4 + 2 * index - env.rinshan_draw_count
                dora_ok = dora_ok and int(world.wall[slot]) // 4 == int(dora_tile) // 4
            # Hand sizes match the pre-world state.
            sizes_ok = [
                len(world.hands[i]) == len(env.hands[i]) for i in range(4)
            ]
            results.append({
                "decision_id": sid,
                "world_index": world_index,
                "ok": bool(conserved and dora_ok and all(sizes_ok)),
                "conserved": bool(conserved),
                "dora_slots_preserved": bool(dora_ok),
                "hand_sizes_match": sizes_ok,
                "wall_len": len(world.wall),
                "violations": {
                    str(t): int(total[t])
                    for t in range(34)
                    if total[t] != 4
                },
            })
    return results


def check_rollout_termination(
    rows: list[dict[str, object]],
    store: RawKyokuStore,
    player_adapter: object,
    grp: GRPBatcher,
    config: ExperimentConfig,
    *,
    limit: int = 3,
    worlds: int = 2,
) -> list[dict[str, object]]:
    """Run a few greedy branches to kyoku end; check end_kyoku and finite GRP."""
    results: list[dict[str, object]] = []
    for row in rows[:limit]:
        content = store.read(str(row["game_id"]), int(row["kyoku_index"]))
        env, obs, _initial, meta = replay_to_decision(
            content,
            seat=int(row["seat"]),
            decision_index=int(row["decision_index"]),
        )
        events = [json.loads(line) for line in content.splitlines()]
        seat = int(row["seat"])
        sid = decision_id(row)
        envs = []
        branch_meta = []
        for world_index in range(worlds):
            world, _rewritten = sample_world(
                env, seat, np.random.default_rng(20_000 + world_index), events,
                decision_index=int(row["decision_index"]),
                decision_event_index=int(meta["decision_event_index"]),
            )
            for rank in (1, 2):
                action = next(
                    a for a in obs.legal_actions()
                    if action_id(a, obs) == int(row[f"top{rank}_action"])
                )
                branch = world.clone()
                branch.step({seat: action})
                envs.append(branch)
                branch_meta.append((world_index, rank))
        player = BatchPlayer(player_adapter, envs)
        active = set(range(len(envs)))
        end_kyoku_flags = [False] * len(envs)
        steps = 0
        while active and steps < 500:
            end_kyoku, _end_game, _n = player.step(active)
            steps += 1
            for index in list(active):
                if bool(end_kyoku[index]):
                    end_kyoku_flags[index] = True
                    active.remove(index)
        metas = []
        scores = []
        seats = []
        for index, (world_index, rank) in enumerate(branch_meta):
            honba = envs[index].honba
            sticks = envs[index].riichi_sticks
            metas.append({
                "kyoku_initial_scores": meta["kyoku_initial_scores"],
                "round_wind": meta["round_wind"],
                "kyoku_index": meta["kyoku_index"],
                "honba": honba,
                "riichi_sticks": sticks,
            })
            scores.append(list(envs[index].scores()))
            seats.append(seat)
        values = grp.evaluate(metas, scores, seats)
        results.append({
            "decision_id": sid,
            "ok": bool(all(end_kyoku_flags) and all(np.isfinite(v) for v in values)),
            "ended": end_kyoku_flags,
            "steps": steps,
            "grp_values": [round(float(v), 3) for v in values],
            "end_scores": scores,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.results or (Path(__file__).resolve().parent / "results"))
    device = torch.device(args.device)
    config = ExperimentConfig()
    rows = _read_csv(out_dir / "selected_decisions.csv")
    if args.limit:
        rows = rows[: int(args.limit)]
    store = RawKyokuStore(
        Path(__file__).resolve().parents[4] / config.raw_index,
        Path(__file__).resolve().parents[4] / config.raw_validation_dir,
    )
    store.load_needed({str(row["game_id"]) for row in rows})

    sft_adapter = load_policy_adapter(
        str(Path(__file__).resolve().parents[4] / "checkpoints/train_riichi_v13_sft/best_heuristic.pt"),
        device=device,
    )
    player_adapter = load_policy_adapter(
        str(Path(__file__).resolve().parents[4] / config.continuation_policy),
        device=device,
    )
    grp = GRPBatcher(config, device)

    fidelity = check_reconstruction_fidelity(
        rows, store, sft_adapter, device, limit=8
    )
    worlds = check_world_invariants(rows, store, limit=5, worlds_per_decision=3)
    rollouts = check_rollout_termination(
        rows, store, player_adapter, grp, config, limit=3, worlds=2
    )

    report = {
        "fidelity": {
            "checked": len(fidelity),
            "all_ok": all(row["ok"] for row in fidelity),
            "top1_agreement": (
                float(np.mean([row["top1_agrees"] for row in fidelity]))
                if fidelity else None
            ),
            "top4_set_agreement": (
                float(np.mean([row["top4_set_agrees"] for row in fidelity]))
                if fidelity else None
            ),
            "rows": fidelity,
        },
        "world_invariants": {
            "checked": len(worlds),
            "all_ok": all(row["ok"] for row in worlds),
            "rows": worlds,
        },
        "rollout_termination": {
            "checked": len(rollouts),
            "all_ok": all(row["ok"] for row in rollouts),
            "rows": rollouts,
        },
    }
    (out_dir / "semantic_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
