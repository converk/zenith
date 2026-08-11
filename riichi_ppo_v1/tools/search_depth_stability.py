"""Tree-search depth stability quick check for the deterministic Oracle Search.

The script samples real root-player decision states, then runs the same
Oracle Search with ``root_player_decision_depth`` in {2, 3, 5} and compares
the candidate Q vectors, best actions, final executed actions, override
judgements and Q deltas.

Search semantics implemented here (fully deterministic when the environment
clone is deterministic):

* the root player's current and future decisions are expanded Top-K and
  backed up with max (plus the ``search_override_threshold`` policy-anchored
  rule used by the current project search);
* all opponent decisions use the PPO greedy action (unique path);
* draws / wall advancement / ordinary environment events are deterministic
  and pass through as the single child value;
* a terminal kyoku is scored with the GRP V2 reward from the root player's
  perspective;
* a root decision reached at the depth cutoff is scored with the root
  player's PPO critic value.

Before the stability experiment, the script verifies whether the search is
strictly deterministic by repeating the same state + search (both a pure
greedy path and the full Top-K tree) and comparing event sequences, action
sequences and leaf values.

Note on the existing harness: ``sft/search_head_to_head.py``'s
``run_root_search`` currently samples *all* seats from the policy with
``torch.multinomial`` and averages rollout values; it does not implement the
deterministic Oracle Search described above.  This tool therefore implements
the deterministic Oracle Search itself, and reports the verified mode
separately.
"""

from __future__ import annotations

import argparse
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

from ..grp.model import (
    RankPredictor,
    grp_features_from_scores,
    reward_from_rank_probs,
)
from ..model.bridge import BatchedStateBridge, Decision, NUM_PLAYERS
from ..sft.policy_adapter import load_policy_adapter
from ..sft.search_head_to_head import is_searchable_decision
from ..training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from ..training.worker import active_decisions


DEFAULT_MODEL = "checkpoints/train_riichi_ppo_next_e5a_kl010/checkpoint_00100.pt"
DEFAULT_GRP = "checkpoints/train_riichi_ppo_goal_grp_v2/grp_rank_predictor.pt"
DEFAULT_OUT = "audit/reports/ppo_rl_next_goal_20260810/search_depth_stability"
GRP_PTS_WEIGHT = (10.0, 4.0, -4.0, -10.0)
EXPERIMENT_VERSION = 1


def _field_from_observation(observation: Any) -> tuple[int, int, int, int]:
    """Full field tuple (includes riichi sticks, as the GRP feature expects)."""
    return (
        int(getattr(observation, "round_wind", 0)),
        int(getattr(observation, "kyoku_index", 0)),
        int(getattr(observation, "honba", 0)),
        int(getattr(observation, "riichi_sticks", 0)),
    )


def _kyoku_signature(observation: Any) -> tuple[int, int, int]:
    """Round identity used to detect a kyoku end.

    ``riichi_sticks`` is intentionally excluded: it increments mid-kyoku when
    a player's riichi is accepted, so it cannot be used as an end signal.
    """
    return (
        int(getattr(observation, "round_wind", 0)),
        int(getattr(observation, "kyoku_index", 0)),
        int(getattr(observation, "honba", 0)),
    )


def _grp_terminal_reward(
    grp_model: RankPredictor | None,
    pts_weight: tuple[float, ...],
    start_scores: list[int],
    end_scores: list[int],
    fields: tuple[int, int, int, int],
    player: int,
) -> float:
    if grp_model is None:
        return 0.0
    rows = np.stack([
        grp_features_from_scores(
            start_scores,
            end_scores,
            chang=fields[0],
            ju=fields[1],
            ben=fields[2],
            liqibang=fields[3],
            player=player,
        )
    ])
    probs = grp_model.predict_rank_probs(rows)
    return reward_from_rank_probs(probs[0], pts_weight)


@torch.inference_mode()
def _model_forward(
    model: torch.nn.Module,
    device: torch.device,
    factors: np.ndarray,
    numeric: np.ndarray,
    lengths: np.ndarray,
    legal: np.ndarray,
    critic_factors: np.ndarray,
    critic_lengths: np.ndarray,
) -> dict[str, torch.Tensor]:
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


def _legal_ids(logits_row: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isfinite(logits_row))


def _softmax_probs(logits_row: np.ndarray, legal_ids: np.ndarray) -> list[float]:
    values = np.asarray([float(logits_row[i]) for i in legal_ids], dtype=np.float64)
    values -= float(np.max(values))
    probs = np.exp(values)
    probs /= float(np.sum(probs))
    return [float(value) for value in probs]


def _action_json(action: Any) -> str:
    return json.dumps(action.to_dict(), sort_keys=True, ensure_ascii=False, default=str)


def _copy_public_tracker(src: PublicStateTracker) -> PublicStateTracker:
    """Copy one row of public tile accounting (per search-tree node)."""
    dst = PublicStateTracker(1)
    dst.visible = src.visible.copy()
    dst.discard_masks = src.discard_masks.copy()
    dst.riichi = src.riichi.copy()
    dst.post_riichi_safe_masks = src.post_riichi_safe_masks.copy()
    dst.discard_counts = src.discard_counts.copy()
    dst.completed_discard_counts = src.completed_discard_counts.copy()
    dst.discard_counts_by_seat = src.discard_counts_by_seat.copy()
    dst.completed_discard_counts_by_seat = src.completed_discard_counts_by_seat.copy()
    dst.open_meld_counts = src.open_meld_counts.copy()
    dst.completed_open_meld_counts = src.completed_open_meld_counts.copy()
    dst.completed_open_meld_counts_by_seat = src.completed_open_meld_counts_by_seat.copy()
    dst.events = src.events
    return dst


class _PostRiichiMaskView:
    """Array-like view over per-node trackers for DecisionAnalysisBatch."""

    def __init__(self, trackers: dict[int, PublicStateTracker]) -> None:
        self._trackers = trackers

    def __getitem__(self, key: tuple[int, int]) -> int:
        env_index, opponent = key
        return self._trackers[env_index].post_riichi_safe_masks[0, opponent]


class _RowPublic:
    """Per-batch-row facade over the per-node public trackers."""

    def __init__(self, trackers: dict[int, PublicStateTracker]) -> None:
        self._trackers = trackers
        self.post_riichi_safe_masks = _PostRiichiMaskView(trackers)

    def genbutsu_coverage(self, env_index: int, tile: int) -> int:
        return int(self._trackers[env_index].genbutsu_coverage(0, tile))


class _Node:
    __slots__ = (
        "slot", "env", "tracker", "root_count", "path", "parent", "candidate_id",
        "is_root", "seeded", "pre_seeded", "manager_seeded", "events_by_seat",
        "children", "child_values", "candidate_ids", "candidate_logits",
        "pending", "value", "expanded", "root_q",
    )

    def __init__(
        self,
        *,
        slot: int,
        env: Any,
        tracker: PublicStateTracker,
        root_count: int,
        path: list[tuple[int, Any]],
        parent: "_Node | None",
        candidate_id: int | None,
        is_root: bool,
        seeded: bool,
        pre_seeded: bool,
        manager_seeded: bool,
        events_by_seat: list[list[str]],
    ) -> None:
        self.slot = slot
        self.env = env
        self.tracker = tracker
        self.root_count = root_count
        self.path = list(path)
        self.parent = parent
        self.candidate_id = candidate_id
        self.is_root = is_root
        self.seeded = seeded
        self.pre_seeded = pre_seeded
        self.manager_seeded = manager_seeded
        self.events_by_seat = events_by_seat
        self.children: list[int] = []
        self.child_values: dict[int, float] = {}
        self.candidate_ids: list[int] = []
        self.candidate_logits: list[float] = []
        self.pending = 0
        self.value: float | None = None
        self.expanded = False
        self.root_q: list[float] | None = None


def _leaf_fingerprint(node: _Node, value: float) -> dict[str, Any]:
    return {
        "value": float(value),
        "actions": [
            [int(seat), _action_json(action)] for seat, action in node.path
        ],
        "events": node.env.mjai_log,
    }


def _fingerprint_str(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)


def _chosen_child_index(
    parent: _Node,
    override_threshold: float,
) -> int:
    """Select the backed-up child of an inner root decision node.

    Candidates are stored in policy-prior order, so index 0 is the PPO Top1.
    With the default threshold 0.0 this reduces to argmax over Q with a
    deterministic policy-prior tie-break.
    """
    ordered = [parent.child_values[slot] for slot in parent.children]
    best = max(
        range(len(ordered)),
        key=lambda index: (ordered[index], parent.candidate_logits[index]),
    )
    if best != 0 and ordered[best] - ordered[0] > float(override_threshold):
        return best
    return 0


@torch.inference_mode()
def oracle_search(
    *,
    snapshot: Any,
    root_seat: int,
    depth: int,
    top_k: int,
    model: torch.nn.Module,
    device: torch.device,
    grp_model: RankPredictor | None,
    grp_pts_weight: tuple[float, ...],
    start_scores: list[int],
    grp_fields: tuple[int, int, int, int],
    kyoku_signature: tuple[int, int, int],
    snapshot_tracker: PublicStateTracker | None,
    snapshot_full_events: list[list[str]] | None,
    policy_top1: int,
    override_threshold: float,
    analyzer: EfficiencyAnalyzer,
    collect_fingerprints: bool = False,
) -> dict[str, Any]:
    """Run one deterministic Oracle Search from a frozen state snapshot.

    ``depth`` is the root player's own decision count: the root candidate
    action is decision #1, and the search stops (PPO critic leaf) when the
    root player reaches their ``depth``-th decision.  Opponents always follow
    the PPO greedy action, so the tree only branches at root decisions.
    """
    try:
        import riichi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Oracle Search requires the local riichi extension") from exc

    if int(depth) < 2:
        raise ValueError("depth must be >= 2 (root action is decision #1)")
    started = time.perf_counter()

    if snapshot_full_events is None:
        raise RuntimeError("oracle_search requires the snapshot full event history")
    top_k = max(1, int(top_k))
    capacity = (
        (top_k ** int(depth) - 1) // (top_k - 1) + 4
        if top_k > 1
        else int(depth) + 4
    )
    manager = riichi.MjaiKyokuStateMachineManager(capacity)
    bridge = BatchedStateBridge(manager, capacity)

    nodes: list[_Node] = []
    root_node = _Node(
        slot=0,
        env=snapshot.clone(),
        tracker=(
            _copy_public_tracker(snapshot_tracker)
            if snapshot_tracker is not None
            else PublicStateTracker(1)
        ),
        root_count=1,
        path=[],
        parent=None,
        candidate_id=None,
        is_root=True,
        seeded=snapshot_tracker is not None,
        pre_seeded=True,
        manager_seeded=False,
        events_by_seat=[list(row) for row in snapshot_full_events],
    )
    nodes.append(root_node)
    active = {0}
    fingerprints: list[dict[str, Any]] = [] if collect_fingerprints else None
    root_expansions = 0
    leaf_count = 0
    terminal_count = 0
    critic_count = 0
    env_steps = 0
    forward_calls = 0
    waves = 0

    def resolve(node: _Node, value: float, *, record_leaf: bool) -> None:
        nonlocal leaf_count
        node.value = float(value)
        if record_leaf:
            leaf_count += 1
            if collect_fingerprints:
                fingerprints.append(_leaf_fingerprint(node, float(value)))
        parent = node.parent
        if parent is None:
            return
        parent.child_values[node.slot] = float(value)
        parent.pending -= 1
        if parent.pending == 0:
            if parent.is_root:
                parent.root_q = [
                    parent.child_values[slot] for slot in parent.children
                ]
                return
            chosen = _chosen_child_index(parent, override_threshold)
            chosen_slot = parent.children[chosen]
            resolve(parent, parent.child_values[chosen_slot], record_leaf=False)

    while active:
        waves += 1
        active_list = [nodes[index] for index in sorted(active)]
        observations_all = [node.env.get_observations() for node in active_list]
        observations_by_capacity: list[dict[int, Any]] = [
            {} for _ in range(capacity)
        ]
        for node, observations in zip(active_list, observations_all, strict=True):
            observations_by_capacity[node.slot] = observations
        bridge.observations_by_env = observations_by_capacity

        apply_slots: list[int] = []
        apply_events: list[list[list[str]]] = []
        for node, observations in zip(active_list, observations_all, strict=True):
            if node.pre_seeded and not node.manager_seeded:
                delta = None
            else:
                delta = [
                    list(observations[seat].new_events())
                    for seat in range(NUM_PLAYERS)
                ]
                node.events_by_seat = [
                    list(node.events_by_seat[seat]) + delta[seat]
                    for seat in range(NUM_PLAYERS)
                ]
            if not node.manager_seeded:
                apply_slots.append(node.slot)
                apply_events.append(node.events_by_seat)
                node.manager_seeded = True
            elif delta is not None:
                apply_slots.append(node.slot)
                apply_events.append(delta)
            if not node.seeded and delta is not None:
                node.tracker.update([delta])
        if apply_slots:
            manager.apply_events_batch(apply_slots, apply_events)
        public = _RowPublic(
            {node.slot: node.tracker for node in active_list}
        )

        terminal_slots: list[int] = []
        root_info: list[dict[str, Any]] = []
        advance_rows: list[tuple[_Node, list[Decision]]] = []
        for node, observations in zip(active_list, observations_all, strict=True):
            env = node.env
            obs = observations
            if env.done() or _kyoku_signature(obs[0]) != kyoku_signature:
                terminal_slots.append(node.slot)
                continue
            root_legal = bool(obs[root_seat].legal_actions())
            others = [
                Decision(node.slot, seat, observation)
                for seat, observation in obs.items()
                if seat != root_seat and observation.legal_actions()
            ]
            if root_legal:
                root_info.append({
                    "node": node,
                    "observations": observations,
                    "others": others,
                })
            else:
                if not others:
                    raise RuntimeError(
                        f"search node {node.slot} is not done but has no decisions "
                        f"(root_seat={root_seat}, wave={waves})"
                    )
                advance_rows.append((node, others))

        forward_decisions: list[Decision] = []
        forward_meta: list[tuple[str, int, int]] = []
        for info in root_info:
            node = info["node"]
            role = "leaf" if node.root_count >= int(depth) else "expand"
            forward_decisions.append(
                Decision(node.slot, root_seat, info["observations"][root_seat])
            )
            forward_meta.append((role, node.slot, root_seat))
            for other in info["others"]:
                forward_decisions.append(other)
                forward_meta.append(("expand_other", node.slot, other.seat_id))
        for node, decisions in advance_rows:
            for decision in decisions:
                forward_decisions.append(decision)
                forward_meta.append(("advance", node.slot, decision.seat_id))

        logits_by_row: np.ndarray | None = None
        values_by_row: np.ndarray | None = None
        if forward_decisions:
            analysis = DecisionAnalysisBatch.build(
                forward_decisions, analyzer=analyzer, public=public,
            )
            prepared = bridge.prepare(forward_decisions, analysis)
            factors, numeric, lengths, legal, _gen, critic_factors, critic_lengths = prepared
            output = _model_forward(
                model, device, factors, numeric, lengths, legal,
                critic_factors, critic_lengths,
            )
            logits_by_row = output["policy_logits"].float().cpu().numpy()
            values_by_row = output["value"].float().cpu().numpy()
            forward_calls += 1
        row_by_slot_seat: dict[tuple[int, int], int] = {}
        for row, (_role, slot, seat) in enumerate(forward_meta):
            row_by_slot_seat[(slot, seat)] = row

        # 1) Terminal nodes -> GRP V2 reward from the root player's viewpoint.
        for slot in terminal_slots:
            node = nodes[slot]
            end_scores = [int(value) for value in node.env.scores()]
            value = _grp_terminal_reward(
                grp_model,
                grp_pts_weight,
                start_scores,
                end_scores,
                grp_fields,
                root_seat,
            )
            terminal_count += 1
            resolve(node, value, record_leaf=True)
            active.discard(node.slot)

        # 2) Root decision at the depth cutoff -> PPO critic leaf.
        for info in root_info:
            node = info["node"]
            if node.root_count < int(depth):
                continue
            row = row_by_slot_seat[(node.slot, root_seat)]
            value = float(values_by_row[row])
            critic_count += 1
            resolve(node, value, record_leaf=True)
            active.discard(node.slot)

        # 3) Root decisions below the cutoff -> expand Top-K.
        for info in root_info:
            node = info["node"]
            if node.root_count >= int(depth):
                continue
            row = row_by_slot_seat[(node.slot, root_seat)]
            logits_row = logits_by_row[row]
            legal = _legal_ids(logits_row)
            ranked = legal[np.argsort(-logits_row[legal])]
            candidates = [int(value) for value in ranked[:max(1, int(top_k))]]
            node.candidate_ids = candidates
            node.candidate_logits = [float(logits_row[candidate]) for candidate in candidates]
            node.expanded = True
            root_expansions += 1

            other_decoded: dict[int, Any] = {}
            for other in info["others"]:
                other_row = row_by_slot_seat[(node.slot, other.seat_id)]
                greedy_id = int(np.argmax(logits_by_row[other_row]))
                other_decoded[other.seat_id] = bridge.decode([other], [greedy_id])[0]

            root_decision = Decision(
                node.slot, root_seat, info["observations"][root_seat],
            )
            for candidate_id in candidates:
                root_action = bridge.decode([root_decision], [candidate_id])[0]
                child_env = node.env.clone()
                actions: dict[int, Any] = {root_seat: root_action, **other_decoded}
                child_env.step(actions)
                env_steps += 1
                child_slot = len(nodes)
                child = _Node(
                    slot=child_slot,
                    env=child_env,
                    tracker=_copy_public_tracker(node.tracker),
                    root_count=node.root_count + 1,
                    path=[
                        *node.path,
                        *sorted(actions.items()),
                    ],
                    parent=node,
                    candidate_id=candidate_id,
                    is_root=False,
                    seeded=False,
                    pre_seeded=False,
                    manager_seeded=False,
                    events_by_seat=[list(row) for row in node.events_by_seat],
                )
                nodes.append(child)
                node.children.append(child_slot)
                active.add(child_slot)
            node.pending = len(candidates)
            node.child_values = {}
            active.discard(node.slot)

        # 4) Non-root nodes -> greedy PPO actions (single unique path).
        for node, decisions in advance_rows:
            actions: dict[int, Any] = {}
            for decision in decisions:
                row = row_by_slot_seat[(node.slot, decision.seat_id)]
                greedy_id = int(np.argmax(logits_by_row[row]))
                actions[decision.seat_id] = bridge.decode([decision], [greedy_id])[0]
            node.path.extend(sorted(actions.items()))
            node.env.step(actions)
            env_steps += 1

    if root_node.root_q is None:
        raise RuntimeError(
            "Oracle Search failed to resolve the root node: "
            f"expanded={root_node.expanded} pending={root_node.pending} "
            f"children={len(root_node.children)} "
            f"terminal_count={terminal_count} critic_count={critic_count} "
            f"waves={waves} active_now={sorted(active)}"
        )
    q_values = list(root_node.root_q)
    candidates = list(root_node.candidate_ids)
    if not candidates:
        raise RuntimeError("Oracle Search produced no root candidates")
    best_index = max(
        range(len(q_values)),
        key=lambda index: (q_values[index], root_node.candidate_logits[index]),
    )
    top1 = int(policy_top1)
    if top1 not in candidates:
        root_obs = root_node.env.get_observations()
        root_legal = len(root_obs[root_seat].legal_actions())
        raise RuntimeError(
            f"PPO Top1 {top1} is not among search candidates {candidates}: "
            f"root_legal_actions={root_legal} candidate_logits={root_node.candidate_logits} "
            f"current_player={root_node.env.current_player} phase={root_node.env.phase} "
            f"turn={root_node.env.turn_count} kyoku={root_node.env.kyoku_idx} "
            f"riichi_stage={root_node.env.riichi_stage} "
            f"drawn={root_node.env.drawn_tile} needs_tsumo={root_node.env.needs_tsumo}"
        )
    top1_index = candidates.index(top1)
    best_id = int(candidates[best_index])
    delta_q = float(q_values[best_index] - q_values[top1_index])
    if best_id != top1 and delta_q > float(override_threshold):
        final_id = best_id
        override = True
    else:
        final_id = top1
        override = False

    return {
        "candidates": candidates,
        "q_values": [float(value) for value in q_values],
        "best_index": int(best_index),
        "best_id": best_id,
        "best_q": float(q_values[best_index]),
        "policy_top1": top1,
        "final_id": final_id,
        "override": bool(override),
        "delta_q": delta_q,
        "stats": {
            "root_expansions": int(root_expansions),
            "leaf_nodes": int(leaf_count),
            "terminal_leaves": int(terminal_count),
            "critic_leaves": int(critic_count),
            "nodes_created": int(len(nodes)),
            "env_steps": int(env_steps),
            "model_forward_calls": int(forward_calls),
            "waves": int(waves),
            "wall_s": round(time.perf_counter() - started, 6),
        },
        "fingerprints": fingerprints if collect_fingerprints else None,
    }


def _evaluate_state(
    state: dict[str, Any],
    *,
    depths: list[int],
    top_k: int,
    override_threshold: float,
    model: torch.nn.Module,
    device: torch.device,
    grp_model: RankPredictor | None,
    grp_pts_weight: tuple[float, ...],
    analyzer: EfficiencyAnalyzer,
) -> dict[str, Any]:
    policy_top1 = int(np.argmax(state["logits"]))
    legal = np.asarray(state["legal_ids"], dtype=np.int64)
    ranked = legal[np.argsort(-state["logits"][legal])]
    top_k_ids = [int(value) for value in ranked[:top_k]]
    probs_by_id = dict(zip(state["legal_ids"], state["probs"], strict=True))
    record: dict[str, Any] = {
        "state_id": int(state["state_id"]),
        "game_id": int(state["game_id"]),
        "seat": int(state["seat"]),
        "step": int(state["step"]),
        "policy_top1": policy_top1,
        "policy_top_k": top_k_ids,
        "policy_probs_top3": [float(probs_by_id[action_id]) for action_id in top_k_ids],
        "legal_ids": [int(value) for value in legal],
        "policy_probs_legal": [float(value) for value in state["probs"]],
        "depths": {},
    }
    for depth in depths:
        result = oracle_search(
            snapshot=state["snapshot"],
            root_seat=state["seat"],
            depth=depth,
            top_k=top_k,
            model=model,
            device=device,
            grp_model=grp_model,
            grp_pts_weight=grp_pts_weight,
            start_scores=state["start_scores"],
            grp_fields=state["grp_fields"],
            kyoku_signature=state["kyoku_signature"],
            snapshot_tracker=state["public_tracker"],
            snapshot_full_events=state["full_events"],
            policy_top1=policy_top1,
            override_threshold=override_threshold,
            analyzer=analyzer,
        )
        result.pop("fingerprints", None)
        record["depths"][str(depth)] = result
    return record


def sample_states(
    *,
    model: torch.nn.Module,
    device: torch.device,
    game_mode: str,
    quota: int,
    seed_base: int,
    shard_index: int,
    max_hanchans: int,
    max_steps: int,
    include_responses: bool,
    analyzer: EfficiencyAnalyzer,
) -> list[dict[str, Any]]:
    """Play hanchans and freeze valid root-player decision states."""
    try:
        import riichi
        from riichienv import BatchedRiichiEnv, RiichiEnv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sampling requires riichi and RiichiEnv extensions") from exc

    states: list[dict[str, Any]] = []
    hanchan_count = 0
    while len(states) < quota and hanchan_count < max_hanchans:
        seed = int(seed_base) + shard_index * 100_000 + hanchan_count
        candidate_seat = hanchan_count % NUM_PLAYERS
        envs = BatchedRiichiEnv(1, seed=seed, step_threads=1, game_mode=game_mode)
        mirror = RiichiEnv(game_mode=game_mode, seed=seed)
        mirror.reset()
        bridge = BatchedStateBridge(riichi.MjaiKyokuStateMachineManager(1), 1)
        observations = list(envs.reset())
        bridge.sync(observations)
        history_by_seat: list[list[str]] = [
            list(bridge.last_events[0][seat]) for seat in range(NUM_PLAYERS)
        ]
        public = PublicStateTracker(1)
        public.update(bridge.last_events)

        for step in range(int(max_steps)):
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
            chosen: dict[int, int] = {}

            if candidate_decisions:
                prepared = bridge.prepare(candidate_decisions, analysis)
                factors, numeric, lengths, legal, _gen, critic_factors, critic_lengths = prepared
                output = _model_forward(
                    model, device, factors, numeric, lengths, legal,
                    critic_factors, critic_lengths,
                )
                logits = output["policy_logits"].float().cpu().numpy()
                for row, decision in enumerate(candidate_decisions):
                    legal = _legal_ids(logits[row])
                    if (
                        len(legal) >= 2
                        and is_searchable_decision(
                            decision.observation, include_responses=include_responses,
                        )
                    ):
                        states.append({
                            "snapshot": mirror.clone(),
                            "seat": int(candidate_seat),
                            "logits": logits[row].copy(),
                            "legal_ids": [int(value) for value in legal],
                            "probs": _softmax_probs(logits[row], legal),
                            "start_scores": [int(value) for value in mirror.scores()],
                            "grp_fields": _field_from_observation(decision.observation),
                            "kyoku_signature": _kyoku_signature(decision.observation),
                            "public_tracker": _copy_public_tracker(public),
                            "full_events": [
                                list(row) for row in history_by_seat
                            ],
                            "game_id": int(seed),
                            "step": int(step),
                            "state_id": len(states),
                        })
                        if len(states) >= quota:
                            break
                    chosen[decision.batch_index] = int(np.argmax(logits[row]))

            if len(states) >= quota:
                break
            if opponent_decisions:
                prepared = bridge.prepare(opponent_decisions, analysis)
                factors, numeric, lengths, legal, _gen, critic_factors, critic_lengths = prepared
                output = _model_forward(
                    model, device, factors, numeric, lengths, legal,
                    critic_factors, critic_lengths,
                )
                logits = output["policy_logits"].float().cpu().numpy()
                for row, decision in enumerate(opponent_decisions):
                    chosen[decision.batch_index] = int(np.argmax(logits[row]))

            all_decisions = [*candidate_decisions, *opponent_decisions]
            if not all_decisions:
                raise RuntimeError(
                    f"hanchan {hanchan_count} (seed {seed}) step {step} has no decisions "
                    "but is not done"
                )
            ids = [chosen[decision.batch_index] for decision in all_decisions]
            actions = bridge.decode(all_decisions, ids)
            actions_by_env: list[dict[int, Any]] = [{
                decision.seat_id: action
                for decision, action in zip(all_decisions, actions, strict=True)
            }]
            observations = list(envs.step_batch(actions_by_env))
            mirror.step(actions_by_env[0])
            bridge.sync(observations)
            for seat in range(NUM_PLAYERS):
                history_by_seat[seat].extend(bridge.last_events[0][seat])
            public.update(bridge.last_events)
            if bool(envs.done()[0]):
                break
        else:
            raise RuntimeError(
                f"hanchan {hanchan_count} (seed {seed}) exceeded {max_steps} steps"
            )
        hanchan_count += 1
        print(
            f"[shard {shard_index}] hanchan {hanchan_count} done "
            f"(seed {seed}, seat {candidate_seat}), states={len(states)}",
            flush=True,
        )
    return states


def run_determinism_check(
    states: list[dict[str, Any]],
    *,
    depth: int,
    top_k: int,
    greedy_repeats: int,
    full_repeats: int,
    model: torch.nn.Module,
    device: torch.device,
    grp_model: RankPredictor | None,
    grp_pts_weight: tuple[float, ...],
    override_threshold: float,
    analyzer: EfficiencyAnalyzer,
    max_states: int = 10,
) -> dict[str, Any]:
    """Repeat the search on identical states and compare traces/values."""
    checks: list[dict[str, Any]] = []
    for state in states[:max_states]:
        common = {
            "snapshot": state["snapshot"],
            "root_seat": state["seat"],
            "depth": depth,
            "model": model,
            "device": device,
            "grp_model": grp_model,
            "grp_pts_weight": grp_pts_weight,
            "start_scores": state["start_scores"],
            "grp_fields": state["grp_fields"],
            "kyoku_signature": state["kyoku_signature"],
            "snapshot_tracker": state["public_tracker"],
            "snapshot_full_events": state["full_events"],
            "policy_top1": int(np.argmax(state["logits"])),
            "override_threshold": override_threshold,
            "analyzer": analyzer,
            "collect_fingerprints": True,
        }

        # A) single greedy path (top_k=1): same root action, one unique future.
        greedy_runs = [
            oracle_search(top_k=1, **common) for _ in range(int(greedy_repeats))
        ]
        greedy_fps = [
            sorted(_fingerprint_str(record) for record in run["fingerprints"])
            for run in greedy_runs
        ]
        greedy_q = [run["q_values"] for run in greedy_runs]
        greedy_ok = (
            all(fps == greedy_fps[0] for fps in greedy_fps[1:])
            and all(q == greedy_q[0] for q in greedy_q[1:])
        )

        # B) full Top-K tree.
        full_runs = [
            oracle_search(top_k=top_k, **common) for _ in range(int(full_repeats))
        ]
        full_fps = [
            sorted(_fingerprint_str(record) for record in run["fingerprints"])
            for run in full_runs
        ]
        full_q = [run["q_values"] for run in full_runs]
        full_ok = (
            all(fps == full_fps[0] for fps in full_fps[1:])
            and all(q == full_q[0] for q in full_q[1:])
        )
        checks.append({
            "state_id": int(state["state_id"]),
            "game_id": int(state["game_id"]),
            "seat": int(state["seat"]),
            "greedy_path_deterministic": bool(greedy_ok),
            "full_search_deterministic": bool(full_ok),
        })
        if not greedy_ok or not full_ok:
            print(
                f"[determinism] MISMATCH state_id={state['state_id']} "
                f"greedy_ok={greedy_ok} full_ok={full_ok}",
                flush=True,
            )

    all_ok = all(
        check["greedy_path_deterministic"] and check["full_search_deterministic"]
        for check in checks
    )
    return {
        "mode": "deterministic" if all_ok else "stochastic",
        "checked_states": len(checks),
        "depth": int(depth),
        "top_k": int(top_k),
        "greedy_repeats": int(greedy_repeats),
        "full_search_repeats": int(full_repeats),
        "checks": checks,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _bucket_label(value: float, edges: list[float]) -> str:
    if value < 0.02:
        return "<0.02"
    if value < 0.05:
        return "[0.02,0.05)"
    if value < 0.10:
        return "[0.05,0.10)"
    if value < 0.20:
        return "[0.10,0.20)"
    return ">=0.20"


def summarize_records(
    records: list[dict[str, Any]],
    *,
    depths: list[int],
    determinism: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    if depths != [2, 3, 5]:
        raise ValueError("summarize_records is defined for depths [2,3,5]")
    depth_keys = [str(depth) for depth in depths]
    deepest = depth_keys[-1]
    pairs = [(a, b) for a, b in [
        *((a, deepest) for a in depth_keys[:-1]),
        (depth_keys[0], depth_keys[1]),
    ] if a != b]

    best_counts = {pair: 0 for pair in pairs}
    final_counts = {pair: 0 for pair in pairs}
    override_counts = {key: 0 for key in depth_keys}
    override_agreement = {pair: 0 for pair in pairs}
    delta_sign_agreement = {pair: 0 for pair in pairs}
    delta_abs_diff = {pair: 0.0 for pair in pairs}
    pairwise_agreement = {pair: 0 for pair in pairs}
    pairwise_total = {pair: 0 for pair in pairs}
    delta_abs5: list[float] = []
    timing = {key: [] for key in depth_keys}
    tree_stats = {
        key: {
            "root_expansions": 0,
            "leaf_nodes": 0,
            "terminal_leaves": 0,
            "critic_leaves": 0,
            "nodes_created": 0,
            "env_steps": 0,
            "model_forward_calls": 0,
        }
        for key in depth_keys
    }
    conflicts: list[dict[str, Any]] = []

    for record in records:
        results = record["depths"]
        for key in depth_keys:
            result = results[key]
            override_counts[key] += int(result["override"])
            stats = result["stats"]
            timing[key].append(float(stats["wall_s"]))
            for field in tree_stats[key]:
                tree_stats[key][field] += int(stats[field])
        delta_abs5.append(abs(float(results[deepest]["delta_q"])))
        for pair in pairs:
            a, b = pair
            best_counts[pair] += int(results[a]["best_id"] == results[b]["best_id"])
            final_counts[pair] += int(results[a]["final_id"] == results[b]["final_id"])
            override_agreement[pair] += int(
                bool(results[a]["override"]) == bool(results[b]["override"])
            )
            delta_sign_agreement[pair] += int(
                np.sign(results[a]["delta_q"]) == np.sign(results[b]["delta_q"])
            )
            delta_abs_diff[pair] += abs(
                float(results[a]["delta_q"]) - float(results[b]["delta_q"])
            )
            for i in range(len(results[a]["candidates"])):
                for j in range(i + 1, len(results[a]["candidates"])):
                    sign_a = np.sign(
                        float(results[a]["q_values"][i]) - float(results[a]["q_values"][j])
                    )
                    sign_b = np.sign(
                        float(results[b]["q_values"][i]) - float(results[b]["q_values"][j])
                    )
                    pairwise_total[pair] += 1
                    pairwise_agreement[pair] += int(sign_a == sign_b)

        conflict = (
            results["2"]["best_id"] != results["5"]["best_id"]
            or results["3"]["best_id"] != results["5"]["best_id"]
            or results["2"]["final_id"] != results["5"]["final_id"]
            or results["3"]["final_id"] != results["5"]["final_id"]
        ) if depths == [2, 3, 5] else False
        if conflict:
            conflicts.append({
                "state_id": int(record["state_id"]),
                "game_id": int(record["game_id"]),
                "seat": int(record["seat"]),
                "step": int(record["step"]),
                "policy_top1": int(record["policy_top1"]),
                "policy_top_k": [int(value) for value in record["policy_top_k"]],
                "policy_probs_top3": [
                    float(value) for value in record["policy_probs_top3"]
                ],
                "q2": [float(value) for value in results["2"]["q_values"]],
                "q3": [float(value) for value in results["3"]["q_values"]],
                "q5": [float(value) for value in results["5"]["q_values"]],
                "best2": int(results["2"]["best_id"]),
                "best3": int(results["3"]["best_id"]),
                "best5": int(results["5"]["best_id"]),
                "final_action2": int(results["2"]["final_id"]),
                "final_action3": int(results["3"]["final_id"]),
                "final_action5": int(results["5"]["final_id"]),
                "override2": bool(results["2"]["override"]),
                "override3": bool(results["3"]["override"]),
                "override5": bool(results["5"]["override"]),
                "delta_q2": float(results["2"]["delta_q"]),
                "delta_q3": float(results["3"]["delta_q"]),
                "delta_q5": float(results["5"]["delta_q"]),
            })

    n = len(records)
    bucket_edges = [0.02, 0.05, 0.10, 0.20]
    bucket_labels = [
        "<0.02", "[0.02,0.05)", "[0.05,0.10)", "[0.10,0.20)", ">=0.20",
    ]
    buckets: dict[str, dict[str, Any]] = {
        label: {"count": 0, "final_agreement_2_vs_5": 0, "final_agreement_3_vs_5": 0}
        for label in bucket_labels
    }
    for record, delta5 in zip(records, delta_abs5, strict=True):
        label = _bucket_label(delta5, bucket_edges)
        buckets[label]["count"] += 1
        buckets[label]["final_agreement_2_vs_5"] += int(
            record["depths"]["2"]["final_id"] == record["depths"]["5"]["final_id"]
        )
        buckets[label]["final_agreement_3_vs_5"] += int(
            record["depths"]["3"]["final_id"] == record["depths"]["5"]["final_id"]
        )

    def rate(count: int) -> float:
        return round(100.0 * count / max(n, 1), 2)

    summary = {
        "valid_states": n,
        "search_mode": (
            determinism["mode"] if determinism is not None else "unknown"
        ),
        "best_action_agreement": {
            f"{a}_vs_{b}": rate(best_counts[(a, b)]) for a, b in pairs
        },
        "final_action_agreement": {
            f"{a}_vs_{b}": rate(final_counts[(a, b)]) for a, b in pairs
        },
        "pairwise_agreement": {
            f"{a}_vs_{b}": (
                round(
                    100.0 * pairwise_agreement[(a, b)] / max(pairwise_total[(a, b)], 1),
                    2,
                )
            )
            for a, b in pairs
        },
        "override_counts": {
            key: int(override_counts[key]) for key in depth_keys
        },
        "override_agreement": {
            f"{a}_vs_{b}": rate(override_agreement[(a, b)]) for a, b in pairs
        },
        "delta_q": {
            "sign_agreement": {
                f"{a}_vs_{b}": rate(delta_sign_agreement[(a, b)]) for a, b in pairs
            },
            "mean_abs_diff": {
                f"{a}_vs_{b}": round(delta_abs_diff[(a, b)] / max(n, 1), 6)
                for a, b in pairs
            },
            "abs_delta5_distribution": {
                "min": round(min(delta_abs5), 6) if delta_abs5 else None,
                "median": round(_percentile(delta_abs5, 50), 6) if delta_abs5 else None,
                "p90": round(_percentile(delta_abs5, 90), 6) if delta_abs5 else None,
                "p95": round(_percentile(delta_abs5, 95), 6) if delta_abs5 else None,
                "max": round(max(delta_abs5), 6) if delta_abs5 else None,
            },
        },
        "buckets": {
            "definitions": {
                "<0.02": "|delta_q5| < 0.02",
                "[0.02,0.05)": "0.02 <= |delta_q5| < 0.05",
                "[0.05,0.10)": "0.05 <= |delta_q5| < 0.10",
                "[0.10,0.20)": "0.10 <= |delta_q5| < 0.20",
                ">=0.20": "|delta_q5| >= 0.20",
            },
            "edges": bucket_edges,
            "bins": {
                label: {
                    "count": int(bins["count"]),
                    "final_agreement_2_vs_5": round(
                        100.0 * bins["final_agreement_2_vs_5"] / max(bins["count"], 1), 2
                    ),
                    "final_agreement_3_vs_5": round(
                        100.0 * bins["final_agreement_3_vs_5"] / max(bins["count"], 1), 2
                    ),
                }
                for label, bins in buckets.items()
            },
        },
        "timing_ms": {
            key: {
                "mean": round(1000.0 * float(np.mean(timing[key])), 3)
                if timing[key]
                else None,
                "p50": round(1000.0 * _percentile(timing[key], 50), 3)
                if timing[key]
                else None,
                "p95": round(1000.0 * _percentile(timing[key], 95), 3)
                if timing[key]
                else None,
            }
            for key in depth_keys
        },
        "tree_stats": tree_stats,
        "conflict_counts": {
            "depth2_vs_depth5": sum(
                int(record["depths"]["2"]["final_id"] != record["depths"]["5"]["final_id"])
                for record in records
            ),
            "depth3_vs_depth5": sum(
                int(record["depths"]["3"]["final_id"] != record["depths"]["5"]["final_id"])
                for record in records
            ),
        } if depths == [2, 3, 5] else {},
        "determinism": determinism,
        "config": config,
    }
    return {"summary": summary, "conflicts": conflicts}


def _load_models(
    model_path: str,
    grp_model_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, RankPredictor | None]:
    adapter = load_policy_adapter(model_path, device=device)
    model = adapter.model
    model.eval()
    grp_model = (
        RankPredictor.from_checkpoint(grp_model_path) if grp_model_path else None
    )
    if grp_model is not None:
        grp_model.eval()
    return model, grp_model


def run_shard(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    quota = int(np.ceil(int(args.num_states) / max(1, int(args.num_shards))))
    depths = [int(value) for value in str(args.depths).split(",")]
    if depths != [2, 3, 5]:
        raise ValueError("this experiment is defined for depths [2,3,5]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, grp_model = _load_models(args.model, args.grp_model, device)
    analyzer = EfficiencyAnalyzer(131_072)
    config = {
        "version": EXPERIMENT_VERSION,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "num_states_requested": int(args.num_states),
        "per_shard_quota": int(quota),
        "depths": depths,
        "top_k": int(args.top_k),
        "override_threshold": float(args.override_threshold),
        "model": str(Path(args.model).resolve()),
        "grp_model": str(Path(args.grp_model).resolve()),
        "grp_pts_weight": [float(value) for value in GRP_PTS_WEIGHT],
        "seed_base": int(args.seed_base),
        "game_mode": args.game_mode,
        "device": str(device),
    }
    print(f"[shard {args.shard_index}] sampling up to {quota} states", flush=True)
    states = sample_states(
        model=model,
        device=device,
        game_mode=args.game_mode,
        quota=quota,
        seed_base=args.seed_base,
        shard_index=int(args.shard_index),
        max_hanchans=int(args.max_hanchans),
        max_steps=int(args.max_steps),
        include_responses=bool(args.include_responses),
        analyzer=analyzer,
    )
    print(
        f"[shard {args.shard_index}] sampled {len(states)} states; "
        "running determinism check",
        flush=True,
    )
    determinism = run_determinism_check(
        states,
        depth=3,
        top_k=int(args.top_k),
        greedy_repeats=int(args.determinism_repeats),
        full_repeats=max(3, int(args.determinism_repeats) - 2),
        model=model,
        device=device,
        grp_model=grp_model,
        grp_pts_weight=GRP_PTS_WEIGHT,
        override_threshold=float(args.override_threshold),
        analyzer=analyzer,
        max_states=int(args.determinism_states),
    )
    print(
        f"[shard {args.shard_index}] determinism={determinism['mode']} "
        f"({len(determinism['checks'])} states checked)",
        flush=True,
    )
    records = [
        _evaluate_state(
            state,
            depths=depths,
            top_k=int(args.top_k),
            override_threshold=float(args.override_threshold),
            model=model,
            device=device,
            grp_model=grp_model,
            grp_pts_weight=GRP_PTS_WEIGHT,
            analyzer=analyzer,
        )
        for state in states
    ]
    result = summarize_records(
        records, depths=depths, determinism=determinism, config=config,
    )
    shard_payload = {
        "summary": result["summary"],
        "records": records,
        "conflicts": result["conflicts"],
    }
    shard_path = out_dir / f"shard_{int(args.shard_index)}.json"
    temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(shard_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(shard_path)
    print(f"[shard {args.shard_index}] wrote {shard_path}", flush=True)
    _print_summary(result["summary"], prefix=f"[shard {args.shard_index}]")


def merge_shards(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    shard_paths = sorted(out_dir.glob("shard_*.json"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.json files under {out_dir}")
    merged_records: list[dict[str, Any]] = []
    determinism_checks: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for path in shard_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        merged_records.extend(payload.get("records", []))
        configs.append(summary["config"])
        if summary.get("determinism") is not None:
            determinism_checks.append(summary["determinism"])
        print(
            f"[merge] loaded {path} (states={summary['valid_states']}, "
            f"records={len(payload.get('records', []))})"
        )

    if not merged_records:
        raise RuntimeError(
            "shard files do not contain per-state records; re-run the shards "
            "with this script version"
        )
    depths = [2, 3, 5]
    determinism = None
    if determinism_checks:
        all_ok = all(
            check["mode"] == "deterministic" for check in determinism_checks
        )
        determinism = {
            "mode": "deterministic" if all_ok else "stochastic",
            "shards": determinism_checks,
        }
    config = {
        "version": EXPERIMENT_VERSION,
        "num_states_requested": int(configs[0]["num_states_requested"]) if configs else 0,
        "depths": depths,
        "top_k": int(configs[0]["top_k"]) if configs else 3,
        "override_threshold": (
            float(configs[0]["override_threshold"]) if configs else 0.0
        ),
        "model": configs[0]["model"] if configs else None,
        "grp_model": configs[0]["grp_model"] if configs else None,
        "grp_pts_weight": configs[0]["grp_pts_weight"] if configs else None,
        "seed_base": configs[0]["seed_base"] if configs else None,
        "game_mode": configs[0]["game_mode"] if configs else None,
        "shards": [
            {
                "index": int(config["shard_index"]),
                "states": int(config["per_shard_quota"]),
                "seed_base": int(config["seed_base"]),
            }
            for config in configs
        ],
    }
    result = summarize_records(
        merged_records, depths=depths, determinism=determinism, config=config,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "search_depth_stability.json"
    conflict_path = out_dir / "search_depth_disagreements.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": result["summary"],
                "records": merged_records,
                "conflicts": result["conflicts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    conflict_path.write_text(
        json.dumps(
            {
                "conflict_count": len(result["conflicts"]),
                "conflicts": result["conflicts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[merge] wrote {report_path}")
    print(f"[merge] wrote {conflict_path}")
    _print_summary(result["summary"], prefix="[merge]")


def _print_summary(summary: dict[str, Any], *, prefix: str) -> None:
    n = summary["valid_states"]
    print(f"{prefix} 有效状态数: {n}")
    print(f"{prefix} 搜索模式: {summary['search_mode']}")
    print(
        f"{prefix} best一致率: "
        f"2v5={summary['best_action_agreement']['2_vs_5']}% "
        f"3v5={summary['best_action_agreement']['3_vs_5']}%"
    )
    print(
        f"{prefix} 最终动作一致率: "
        f"2v5={summary['final_action_agreement']['2_vs_5']}% "
        f"3v5={summary['final_action_agreement']['3_vs_5']}%"
    )
    print(
        f"{prefix} pairwise一致率: "
        f"2v5={summary['pairwise_agreement']['2_vs_5']}% "
        f"3v5={summary['pairwise_agreement']['3_vs_5']}%"
    )
    print(
        f"{prefix} override一致率: "
        f"2v5={summary['override_agreement']['2_vs_5']}% "
        f"3v5={summary['override_agreement']['3_vs_5']}%"
    )
    print(
        f"{prefix} delta-Q同号率: "
        f"2v5={summary['delta_q']['sign_agreement']['2_vs_5']}% "
        f"3v5={summary['delta_q']['sign_agreement']['3_vs_5']}%"
    )
    print(
        f"{prefix} 平均|ΔQ差|: "
        f"2v5={summary['delta_q']['mean_abs_diff']['2_vs_5']} "
        f"3v5={summary['delta_q']['mean_abs_diff']['3_vs_5']}"
    )
    print(f"{prefix} 耗时(ms):")
    for key in ("2", "3", "5"):
        timing = summary["timing_ms"][key]
        print(
            f"{prefix}   depth{key}: mean={timing['mean']} / "
            f"P50={timing['p50']} / P95={timing['p95']}"
        )
    conflicts = summary.get("conflict_counts", {})
    if conflicts:
        print(
            f"{prefix} 冲突状态: 2v5={conflicts['depth2_vs_depth5']} "
            f"3v5={conflicts['depth3_vs_depth5']}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--grp-model", default=DEFAULT_GRP)
    parser.add_argument("--num-states", type=int, default=150)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=20260811)
    parser.add_argument("--depths", default="2,3,5")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--override-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-hanchans", type=int, default=8)
    parser.add_argument("--include-responses", action="store_true")
    parser.add_argument("--determinism-repeats", type=int, default=5)
    parser.add_argument("--determinism-states", type=int, default=10)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--merge", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.merge:
        merge_shards(args)
        return
    run_shard(args)


if __name__ == "__main__":
    main()
