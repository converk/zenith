"""Offline root-branch pMCPA search evaluation for 2v2 head-to-head.

This harness plays the same seat-balanced, paired-wall 2v2 protocol as
``head_to_head``, but lets one team replace its greedy policy actions with
actions selected by a light root-branch Monte-Carlo search (pMCPA-style):

* at a searchable decision of the search team, the top-``search_width`` legal
  actions by policy probability are the root candidates (default 3: in the
  SFT-initialized model the top-3 already covers the favorable decisions);
* each candidate is applied to a deep clone of the current table, followed by
  ``search_depth`` rollout steps in which all seats sample from the base
  policy.  ``depth_mode="round"`` counts full-table decision rounds (default,
  backwards compatible); ``depth_mode="own"`` counts the searching player's own
  decisions (the root candidate action is the first one);
* a rollout is scored with the GRP terminal reward when its kyoku ends, and
  otherwise with the base policy's critic value for the searching seat;
* the candidate with the best mean rollout value replaces the greedy action.

The harness is CPU-heavy but uses one idle GPU for policy/value inference.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import time
from typing import Any, Iterable

# Keep the project-facing device convention consistent with the training entry
# points. This must happen before importing torch.
if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch

from ..model.bridge import BatchedStateBridge, Decision, NUM_PLAYERS
from ..grp.model import (
    RankPredictor,
    grp_features_from_scores,
    reward_from_rank_probs,
)
from ..training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from ..training.worker import active_decisions
from .head_to_head import (
    _action_group,
    balanced_team_a_seats,
    select_winner,
)
from .policy_adapter import load_policy_adapter


def _action_kinds(observation: Any) -> set[str]:
    from ..training.rewards.decision import action_kind

    return {action_kind(action) for action in observation.legal_actions()}


def is_searchable_decision(
    observation: Any,
    *,
    include_responses: bool = False,
) -> bool:
    """Return whether a decision window is worth root search.

    Default targets 立直/切牌 (discard/reach) windows.  ``include_responses``
    additionally covers call/win/ryukyoku response windows.
    """
    kinds = _action_kinds(observation)
    if kinds & {"dahai", "reach"}:
        return True
    if include_responses and kinds & {
        "chi", "pon", "daiminkan", "ankan", "kakan",
        "hora", "ron", "tsumo", "ryukyoku", "kyushukyuhai",
    }:
        return True
    return False


def _field_from_observation(observation: Any) -> tuple[int, int, int, int]:
    return (
        int(getattr(observation, "round_wind", 0)),
        int(getattr(observation, "kyoku_index", 0)),
        int(getattr(observation, "honba", 0)),
        int(getattr(observation, "riichi_sticks", 0)),
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


def _choose_actions(
    logits: torch.Tensor,
    *,
    greedy: bool,
    temperature: float,
    rng: torch.Generator | None,
) -> tuple[list[int], list[float]]:
    logits = logits.float()
    if greedy:
        chosen = logits.argmax(-1)
        log_probs = torch.log_softmax(logits, dim=-1).gather(
            1, chosen[:, None]
        ).squeeze(1)
        return chosen.cpu().tolist(), log_probs.cpu().tolist()
    else:
        scaled = (logits / max(float(temperature), 1e-6)).cpu()
        probs = torch.softmax(scaled, dim=-1)
        chosen = torch.multinomial(probs, 1, generator=rng).squeeze(1)
        log_probs = torch.log_softmax(scaled, dim=-1).gather(
            1, chosen[:, None]
        ).squeeze(1)
        return chosen.tolist(), log_probs.tolist()


def _decode_actions(
    bridge: BatchedStateBridge,
    decisions: list[Decision],
    action_ids: list[int],
) -> list[Any]:
    return bridge.decode(decisions, action_ids)


class SearchStats:
    def __init__(self) -> None:
        self.total_decisions = 0
        self.searchable = 0
        self.searched = 0
        self.overrides = 0
        self.search_time_s = 0.0
        self.rollout_steps = 0
        self.rollout_clones = 0
        self.terminal_scored = 0
        self.leaf_scored = 0
        self.chosen_values: list[float] = []
        self.greedy_values: list[float] = []
        self.value_gaps: list[float] = []
        self.override_by_group: Counter[str] = Counter()

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "searchable_decisions": self.searchable,
            "searched_decisions": self.searched,
            "override_count": self.overrides,
            "override_rate": self.overrides / max(self.searched, 1),
            "search_wall_s": round(self.search_time_s, 3),
            "rollout_steps": self.rollout_steps,
            "rollout_clones": self.rollout_clones,
            "terminal_scored": self.terminal_scored,
            "leaf_scored": self.leaf_scored,
            "mean_chosen_value": (
                float(np.mean(self.chosen_values)) if self.chosen_values else None
            ),
            "mean_greedy_value": (
                float(np.mean(self.greedy_values)) if self.greedy_values else None
            ),
            "mean_chosen_minus_greedy": (
                float(np.mean(self.value_gaps)) if self.value_gaps else None
            ),
            "override_by_action_group": {
                name: int(count) for name, count in sorted(self.override_by_group.items())
            },
        }


class DistillRecorder:
    """Accumulate search-decision states and target distributions to npz parts."""

    def __init__(self, directory: str | Path, *, tau: float = 1.0) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tau = max(float(tau), 1e-6)
        self.part_index = 0
        self.pending: list[dict[str, Any]] = []

    def add(
        self,
        *,
        factors: np.ndarray,
        numeric: np.ndarray,
        length: int,
        legal: np.ndarray,
        target_probs: np.ndarray,
    ) -> None:
        self.pending.append({
            "factors": np.ascontiguousarray(factors, dtype=np.uint8),
            "numeric": np.ascontiguousarray(numeric, dtype=np.float32),
            "length": int(length),
            "legal": np.ascontiguousarray(legal, dtype=np.bool_),
            "target_probs": np.ascontiguousarray(target_probs, dtype=np.float32),
        })
        if len(self.pending) >= 4096:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        max_length = max(row["length"] for row in self.pending)
        count = len(self.pending)
        factors = np.zeros((count, max_length, 10), dtype=np.uint8)
        numeric = np.zeros((count, max_length, 8), dtype=np.float32)
        lengths = np.zeros(count, dtype=np.int64)
        legal = np.zeros((count, 241), dtype=np.bool_)
        target_probs = np.zeros((count, 241), dtype=np.float32)
        for row_index, row in enumerate(self.pending):
            factors[row_index, : row["length"]] = row["factors"]
            numeric[row_index, : row["length"]] = row["numeric"]
            lengths[row_index] = row["length"]
            legal[row_index] = row["legal"]
            target_probs[row_index] = row["target_probs"]
        path = self.directory / f"part_{self.part_index:05d}.npz"
        np.savez_compressed(
            path,
            factors=factors,
            numeric=numeric,
            lengths=lengths,
            legal=legal,
            target_probs=target_probs,
        )
        self.part_index += 1
        self.pending = []

    def close(self, **meta: Any) -> None:
        self.flush()
        (self.directory / "meta.json").write_text(
            json.dumps(
                {"tau": self.tau, "parts": self.part_index, **meta},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def run_root_search(
    *,
    decisions: list[Decision],
    logits: torch.Tensor,
    values: torch.Tensor,
    prepared_rows: dict[str, np.ndarray] | None,
    main_bridge: BatchedStateBridge,
    mirrors: list[Any],
    team_seats_by_env: dict[int, tuple[int, ...]],
    grp_model: RankPredictor | None,
    grp_pts_weight: tuple[float, ...],
    model: torch.nn.Module,
    device: torch.device,
    value_mode: str,
    search_width: int,
    search_depth: int,
    depth_mode: str = "round",
    rollouts: int,
    temperature: float,
    search_rng: torch.Generator,
    stats: SearchStats,
    recorder: DistillRecorder | None = None,
) -> dict[int, int]:
    """Run one synchronized root-search batch and return chosen action ids."""
    if not decisions:
        return {}
    started = time.perf_counter()
    try:
        import riichi
        from riichienv import RiichiEnv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "search evaluation requires riichi and RiichiEnv extensions"
        ) from exc

    batch_size = len(decisions)
    legal_masks = np.zeros((batch_size, 241), dtype=np.bool_)
    # prepared legal masks are not passed in; rebuild from the model output by
    # evaluating each decision's observation.
    logits_cpu = logits.float().cpu().numpy()
    values_cpu = values.float().cpu().numpy()
    candidates_by_decision: list[list[int]] = []
    for row, decision in enumerate(decisions):
        legal_ids = np.flatnonzero(np.isfinite(logits_cpu[row]))
        ranked = legal_ids[np.argsort(-logits_cpu[row][legal_ids])]
        width = max(1, min(int(search_width), len(ranked)))
        candidates_by_decision.append([int(value) for value in ranked[:width]])

    # Build one clone per (decision, candidate, rollout).
    clones: list[RiichiEnv] = []
    clone_meta: list[dict[str, Any]] = []
    root_value: list[float] = []
    start_scores_by_clone: list[list[int]] = []
    fields_by_clone: list[tuple[int, int, int, int]] = []
    search_seat_by_clone: list[int] = []
    team_seats_by_clone: list[tuple[int, ...]] = []

    for row, decision in enumerate(decisions):
        mirror = mirrors[decision.env_index]
        actions = [
            _decode_actions(main_bridge, [decision], [candidate_id])[0]
            for candidate_id in candidates_by_decision[row]
        ]
        start_scores = [int(value) for value in mirror.scores()]
        fields = _field_from_observation(decision.observation)
        for candidate_id, action in zip(
            candidates_by_decision[row], actions, strict=True,
        ):
            for _rollout in range(max(1, int(rollouts))):
                clone = mirror.clone()
                clone.step({decision.seat_id: action})
                clones.append(clone)
                clone_meta.append({
                    "decision_row": row,
                    "candidate_id": candidate_id,
                })
                root_value.append(float(values_cpu[row]))
                start_scores_by_clone.append(list(start_scores))
                fields_by_clone.append(fields)
                search_seat_by_clone.append(int(decision.seat_id))
                team_seats_by_clone.append(team_seats_by_env[decision.env_index])
                stats.rollout_clones += 1
    num_clones = len(clones)
    manager = riichi.MjaiKyokuStateMachineManager(num_clones)
    rollout_bridge = BatchedStateBridge(manager, num_clones)
    rollout_public = PublicStateTracker(num_clones)
    analyzer = EfficiencyAnalyzer(131_072)
    active_clones = set(range(num_clones))
    terminal_values: list[float | None] = [None] * num_clones
    last_values: list[dict[int, float]] = [{} for _ in range(num_clones)]
    # The root candidate action is the searching player's first own decision.
    own_decision_counts = [1] * num_clones

    wave = 0
    while True:
        if depth_mode == "own":
            if not active_clones:
                break
            pending = [
                clone_index for clone_index in sorted(active_clones)
                if own_decision_counts[clone_index] < max(1, int(search_depth))
            ]
            if not pending:
                break
        elif wave >= max(1, int(search_depth)):
            break
        if not active_clones:
            break
        observations_all = [clone.get_observations() for clone in clones]
        rollout_bridge.sync(observations_all)
        rollout_public.update(rollout_bridge.last_events)
        decisions_all: list[Decision] = []
        for clone_index in sorted(active_clones):
            for seat, observation in observations_all[clone_index].items():
                if observation.legal_actions():
                    decisions_all.append(Decision(clone_index, seat, observation))
        if not decisions_all:
            # No active decision (e.g. every table is in an end window); the
            # remaining waves cannot change the value.
            break
        analysis = DecisionAnalysisBatch.build(
            decisions_all, analyzer=analyzer, public=rollout_public,
        )
        prepared = rollout_bridge.prepare(decisions_all, analysis)
        factors, numeric, lengths, legal, _generations, critic_factors, critic_lengths = prepared
        output = _model_forward(
            model, device, factors, numeric, lengths, legal,
            critic_factors, critic_lengths,
        )
        action_ids, _log_probs = _choose_actions(
            output["policy_logits"], greedy=False, temperature=temperature,
            rng=search_rng,
        )
        rollout_values = output["value"].cpu().tolist()
        actions = _decode_actions(rollout_bridge, decisions_all, action_ids)
        actions_by_env: list[dict[int, Any]] = [{} for _ in range(num_clones)]
        row_by_clone_seat: dict[tuple[int, int], int] = {
            (decision.env_index, decision.seat_id): row
            for row, decision in enumerate(decisions_all)
        }
        if depth_mode == "own":
            for clone_index in active_clones:
                if (clone_index, search_seat_by_clone[clone_index]) in row_by_clone_seat:
                    own_decision_counts[clone_index] += 1
        for decision, action in zip(decisions_all, actions, strict=True):
            actions_by_env[decision.env_index][decision.seat_id] = action
        for (clone_index, seat), row in row_by_clone_seat.items():
            last_values[clone_index][seat] = float(rollout_values[row])

        ended: set[int] = set()
        for clone_index in sorted(active_clones):
            row_actions = actions_by_env[clone_index]
            if not row_actions:
                continue
            clone = clones[clone_index]
            clone.step(row_actions)
            stats.rollout_steps += 1
            if clone.done():
                ended.add(clone_index)
                continue
            observation0 = clone.get_observation(0)
            new_fields = _field_from_observation(observation0)
            if new_fields != fields_by_clone[clone_index]:
                ended.add(clone_index)

        for clone_index in ended:
            end_scores = [int(value) for value in clones[clone_index].scores()]
            if value_mode == "team":
                terminal_values[clone_index] = sum(
                    _grp_terminal_reward(
                        grp_model,
                        grp_pts_weight,
                        start_scores_by_clone[clone_index],
                        end_scores,
                        fields_by_clone[clone_index],
                        seat,
                    )
                    for seat in team_seats_by_clone[clone_index]
                )
            else:
                terminal_values[clone_index] = _grp_terminal_reward(
                    grp_model,
                    grp_pts_weight,
                    start_scores_by_clone[clone_index],
                    end_scores,
                    fields_by_clone[clone_index],
                    search_seat_by_clone[clone_index],
                )
            stats.terminal_scored += 1
            active_clones.discard(clone_index)
        wave += 1

    # Score every clone (terminal value, else last search-seat critic value,
    # else the root critic value as fallback).
    values_by_clone: list[float] = []
    for clone_index in range(num_clones):
        if terminal_values[clone_index] is not None:
            values_by_clone.append(float(terminal_values[clone_index]))
        elif last_values[clone_index]:
            if value_mode == "team":
                team_seats = team_seats_by_clone[clone_index]
                recorded = {
                    seat: value
                    for seat, value in last_values[clone_index].items()
                    if seat in team_seats
                }
                if len(recorded) == len(team_seats):
                    values_by_clone.append(float(sum(recorded.values())))
                elif recorded:
                    values_by_clone.append(2.0 * float(sum(recorded.values()) / len(recorded)))
                else:
                    values_by_clone.append(2.0 * float(root_value[clone_index]))
            else:
                seat = search_seat_by_clone[clone_index]
                values_by_clone.append(
                    float(last_values[clone_index].get(seat, root_value[clone_index]))
                )
            stats.leaf_scored += 1
        else:
            values_by_clone.append(float(root_value[clone_index]))
            stats.leaf_scored += 1

    # Aggregate per (decision, candidate) and choose the argmax mean value.
    value_by_key: dict[tuple[int, int], list[float]] = {}
    for clone_index, meta in enumerate(clone_meta):
        key = (meta["decision_row"], meta["candidate_id"])
        value_by_key.setdefault(key, []).append(values_by_clone[clone_index])
    chosen: dict[int, int] = {}
    for row, decision in enumerate(decisions):
        candidates = candidates_by_decision[row]
        means = [
            float(np.mean(value_by_key[(row, candidate_id)]))
            for candidate_id in candidates
        ]
        # Deterministic tie-break by policy prior (logits).
        best_index = max(
            range(len(candidates)),
            key=lambda index: (means[index], float(logits_cpu[row][candidates[index]])),
        )
        chosen_id = candidates[best_index]
        chosen[decision.batch_index] = chosen_id
        greedy_id = int(np.argmax(logits_cpu[row]))
        stats.chosen_values.append(means[best_index])
        if greedy_id in candidates:
            greedy_mean = means[candidates.index(greedy_id)]
            stats.greedy_values.append(greedy_mean)
            stats.value_gaps.append(means[best_index] - greedy_mean)
        if chosen_id != greedy_id:
            stats.overrides += 1
            group = _action_group(chosen_id)
            stats.override_by_group[group] += 1
        if recorder is not None and prepared_rows is not None:
            target_probs = np.zeros(241, dtype=np.float32)
            scaled = np.asarray(means, dtype=np.float64) / recorder.tau
            scaled -= float(np.max(scaled))
            weights = np.exp(scaled)
            weights /= float(np.sum(weights))
            target_probs[np.asarray(candidates, dtype=np.int64)] = weights.astype(
                np.float32
            )
            length = int(prepared_rows["lengths"][row])
            recorder.add(
                factors=prepared_rows["factors"][row, :length],
                numeric=prepared_rows["numeric"][row, :length],
                length=length,
                legal=prepared_rows["legal"][row],
                target_probs=target_probs,
            )

    stats.searched += len(decisions)
    stats.search_time_s += time.perf_counter() - started
    return chosen


@torch.inference_mode()
def evaluate_2v2_search(
    model_a_path: str | Path,
    model_b_path: str | Path,
    *,
    device: str = "cuda",
    model_a_device: str | None = None,
    model_b_device: str | None = None,
    hanchan_count: int = 240,
    parallel_hanchans: int = 24,
    seed_base: int = 20260730,
    game_mode: str = "4p-red-half",
    max_steps: int = 4000,
    search_depth: int = 3,
    search_width: int = 3,
    depth_mode: str = "round",
    rollouts: int = 2,
    temperature: float = 1.0,
    include_responses: bool = False,
    value_mode: str = "seat",
    grp_model_path: str | None = None,
    grp_pts_weight: Iterable[float] = (10, 4, -4, -10),
    search_team: str = "a",
    search_rng_seed: int = 20260810,
    record_distill_dir: str | Path | None = None,
    distill_tau: float = 1.0,
) -> dict[str, Any]:
    """Play greedy 2v2 hanchans with one team optionally using root search."""
    try:
        import riichi
        from riichienv import BatchedRiichiEnv, RiichiEnv
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
    model_a = adapter_a.model
    model_b = adapter_b.model
    model_a.eval()
    model_b.eval()
    pts_weight = tuple(float(value) for value in grp_pts_weight)
    grp_model = (
        RankPredictor.from_checkpoint(str(grp_model_path)) if grp_model_path else None
    )
    if grp_model is not None:
        grp_model.eval()

    schedule = balanced_team_a_seats(hanchan_count)
    base_schedule = schedule[::2]
    parallel_pairs = max(1, min(int(parallel_hanchans) // 2, len(base_schedule)))
    parallel = 2 * parallel_pairs
    started = time.perf_counter()
    search_rng = torch.Generator(device="cpu").manual_seed(int(search_rng_seed))

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
    stats = SearchStats()
    recorder = (
        DistillRecorder(record_distill_dir, tau=distill_tau)
        if record_distill_dir
        else None
    )

    for pair_start in range(0, len(base_schedule), parallel_pairs):
        base_seats = base_schedule[pair_start : pair_start + parallel_pairs]
        for swapped in (False, True):
            team_a_by_env = [
                tuple(seat for seat in range(NUM_PLAYERS) if seat not in seats)
                if swapped else seats
                for seats in base_seats
            ]
            batch_size = len(team_a_by_env)
            envs = BatchedRiichiEnv(
                batch_size,
                seed=int(seed_base) + pair_start,
                step_threads=batch_size,
                game_mode=game_mode,
            )
            mirrors: list[RiichiEnv] = []
            for env_index in range(batch_size):
                mirror = RiichiEnv(
                    game_mode=game_mode, seed=int(seed_base) + pair_start + env_index,
                )
                mirror.reset()
                mirrors.append(mirror)
            bridge = BatchedStateBridge(
                riichi.MjaiKyokuStateMachineManager(batch_size), batch_size,
            )
            team_seats_by_env = {
                env_index: team_a_by_env[env_index]
                for env_index in range(batch_size)
            }
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
                output_by_policy: dict[str, dict[str, torch.Tensor]] = {}
                prepared_by_policy: dict[str, tuple[np.ndarray, ...]] = {}
                policy_decisions_by_policy: dict[str, list[Decision]] = {}
                chosen_by_policy: dict[str, dict[int, int]] = {}
                for policy_name, adapter, model in (
                    ("a", adapter_a, model_a),
                    ("b", adapter_b, model_b),
                ):
                    policy_decisions = [
                        decision for decision in decisions
                        if (decision.seat_id in team_a_by_env[decision.env_index])
                        == (policy_name == "a")
                    ]
                    if not policy_decisions:
                        chosen_by_policy[policy_name] = {}
                        policy_decisions_by_policy[policy_name] = []
                        continue
                    prepared = bridge.prepare(policy_decisions, analysis)
                    factors, numeric, lengths, legal, _generations, critic_factors, critic_lengths = prepared
                    output = _model_forward(
                        model, device_b if policy_name == "b" else device_a,
                        factors, numeric, lengths, legal,
                        critic_factors, critic_lengths,
                    )
                    output_by_policy[policy_name] = output
                    prepared_by_policy[policy_name] = prepared
                    policy_decisions_by_policy[policy_name] = policy_decisions
                    greedy_ids, _log_probs = _choose_actions(
                        output["policy_logits"], greedy=True, temperature=1.0,
                        rng=None,
                    )
                    chosen_by_policy[policy_name] = {
                        decision.batch_index: int(action_id)
                        for decision, action_id in zip(
                            policy_decisions, greedy_ids, strict=True,
                        )
                    }
                    action_counts[policy_name].update(int(value) for value in greedy_ids)

                if search_team == "a" and policy_decisions_by_policy.get("a"):
                    team_a_decisions = policy_decisions_by_policy["a"]
                    searchable: list[Decision] = []
                    search_rows: list[int] = []
                    for row, decision in enumerate(team_a_decisions):
                        if is_searchable_decision(
                            decision.observation, include_responses=include_responses,
                        ):
                            searchable.append(decision)
                            search_rows.append(row)
                    stats.total_decisions += len(team_a_decisions)
                    stats.searchable += len(searchable)
                    if searchable:
                        # rows of ``searchable`` within the team-a decision
                        # batch: rebuild logits/values only for searched rows.
                        output_a = output_by_policy["a"]
                        logits = output_a["policy_logits"][search_rows]
                        values = output_a["value"][search_rows]
                        prepared_a = prepared_by_policy["a"]
                        prepared_rows = (
                            {
                                "factors": prepared_a[0][search_rows],
                                "numeric": prepared_a[1][search_rows],
                                "lengths": prepared_a[2][search_rows],
                                "legal": prepared_a[3][search_rows],
                            }
                            if recorder is not None
                            else None
                        )
                        chosen = run_root_search(
                            decisions=searchable,
                            logits=logits,
                            values=values,
                            prepared_rows=prepared_rows,
                            main_bridge=bridge,
                            mirrors=mirrors,
                            team_seats_by_env=team_seats_by_env,
                            grp_model=grp_model,
                            grp_pts_weight=pts_weight,
                            model=model_a,
                            device=device_a,
                            value_mode=value_mode,
                            search_width=search_width,
                            search_depth=search_depth,
                            depth_mode=depth_mode,
                            rollouts=rollouts,
                            temperature=temperature,
                            search_rng=search_rng,
                            stats=stats,
                            recorder=recorder,
                        )
                        chosen_by_policy["a"].update(chosen)

                for policy_name, chosen_map in chosen_by_policy.items():
                    policy_decisions = policy_decisions_by_policy[policy_name]
                    if not policy_decisions:
                        continue
                    ids = [chosen_map[decision.batch_index] for decision in policy_decisions]
                    actions = _decode_actions(bridge, policy_decisions, ids)
                    for decision, action in zip(policy_decisions, actions, strict=True):
                        actions_by_env[decision.env_index][decision.seat_id] = action

                observations = list(envs.step_batch(actions_by_env))
                for env_index in range(batch_size):
                    if actions_by_env[env_index]:
                        mirrors[env_index].step(actions_by_env[env_index])
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
                    f"paired 2v2 search batch {pair_start // parallel_pairs} exceeded {max_steps} steps"
                )
        print(
            f"search_head_to_head completed={completed}/{hanchan_count} "
            f"wins_a={wins_a} wins_b={wins_b} ties={ties} "
            f"searched={stats.searched} overrides={stats.overrides} "
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
    if recorder is not None:
        recorder.close(
            hanchan_count=int(hanchan_count),
            seed_base=int(seed_base),
            search_depth=int(search_depth),
            search_width=int(search_width),
            rollouts=int(rollouts),
            value_mode=value_mode,
            searched=int(stats.searched),
        )

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
        "search_team": search_team,
        "search_config": {
            "search_depth": int(search_depth),
            "depth_mode": depth_mode,
            "search_width": int(search_width),
            "rollouts": int(rollouts),
            "temperature": float(temperature),
            "include_responses": bool(include_responses),
            "value_mode": value_mode,
            "grp_model": str(grp_model_path) if grp_model_path else None,
            "grp_pts_weight": [float(value) for value in pts_weight],
            "search_rng_seed": int(search_rng_seed),
        },
        "search_stats": stats.as_dict(),
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
    parser.add_argument("--hanchans", type=int, default=240)
    parser.add_argument("--parallel-hanchans", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=20260730)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-a-device")
    parser.add_argument("--model-b-device")
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--depth-mode", default="round", choices=("round", "own"))
    parser.add_argument("--search-width", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-responses", action="store_true")
    parser.add_argument("--value-mode", default="seat", choices=("seat", "team"))
    parser.add_argument("--grp-model")
    parser.add_argument("--search-team", default="a", choices=("a", "b"))
    parser.add_argument("--search-rng-seed", type=int, default=20260810)
    parser.add_argument("--record-distill-dir")
    parser.add_argument("--distill-tau", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_2v2_search(
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
        search_depth=args.search_depth,
        depth_mode=args.depth_mode,
        search_width=args.search_width,
        rollouts=args.rollouts,
        temperature=args.temperature,
        include_responses=args.include_responses,
        value_mode=args.value_mode,
        grp_model_path=args.grp_model,
        search_team=args.search_team,
        search_rng_seed=args.search_rng_seed,
        record_distill_dir=args.record_distill_dir,
        distill_tau=args.distill_tau,
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
