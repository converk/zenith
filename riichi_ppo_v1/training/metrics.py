"""Semantic, privacy-safe training metrics for Riichi PPO.

The counters in this module deliberately consume only selected actions, public
MJAI events and settled scores.  In particular, no concealed hand, wall or
critic-only feature is retained by the metric stream.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

METRIC_SCHEMA_VERSION = 1


def action_kind(action_id: int) -> str:
    """Map the stable 241-action protocol to a compact public category."""
    value = int(action_id)
    if value == 0:
        return "pass"
    if 1 <= value <= 74:
        return "tsumogiri" if value % 2 == 0 else "discard"
    if value == 75:
        return "riichi"
    if 76 <= value <= 132:
        return "chi"
    if 133 <= value <= 169:
        return "pon"
    if value == 170:
        return "daiminkan"
    if 171 <= value <= 204:
        return "ankan"
    if 205 <= value <= 238:
        return "kakan"
    if value == 239:
        return "hora"
    if value == 240:
        return "kyushu"
    raise ValueError(f"invalid action id {action_id!r}")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


@dataclass
class SemanticMetrics:
    """One finite window of learner-centric rollout or evaluation metrics."""

    decisions: int = 0
    legal_actions: list[float] = field(default_factory=list)
    actions: Counter[str] = field(default_factory=Counter)
    riichi_opportunities: int = 0
    call_opportunities: int = 0
    efficiency_rewards: list[float] = field(default_factory=list)
    optimal_shanten: list[float] = field(default_factory=list)
    optimal_ukeire: list[float] = field(default_factory=list)
    shanten_gaps: list[float] = field(default_factory=list)
    ukeire_losses: list[float] = field(default_factory=list)
    threat_discards: int = 0
    genbutsu_all: int = 0
    genbutsu_coverage: list[float] = field(default_factory=list)
    kyoku_points: list[float] = field(default_factory=list)
    win_points: list[float] = field(default_factory=list)
    deal_in_points: list[float] = field(default_factory=list)
    draw_points: list[float] = field(default_factory=list)
    kyoku_discard_counts: list[float] = field(default_factory=list)
    kyoku_open_melds: list[float] = field(default_factory=list)
    wins: int = 0
    tsumo_wins: int = 0
    ron_wins: int = 0
    deal_ins: int = 0
    tsumo_losses: int = 0
    draws: int = 0
    exhaustive_draws: int = 0
    draw_tenpai: int = 0
    selected_shanten: list[float] = field(default_factory=list)
    selected_effective_shanten: list[float] = field(default_factory=list)
    first_riichis: int = 0
    chase_riichis: int = 0
    dangerous_calls: int = 0
    bad_kans: int = 0
    accepted_kans: int = 0
    decision_groups: Counter[str] = field(default_factory=Counter)
    action_groups: Counter[str] = field(default_factory=Counter)
    kyoku_group_points: dict[str, list[float]] = field(default_factory=dict)
    kyoku_group_wins: Counter[str] = field(default_factory=Counter)
    kyoku_group_deal_ins: Counter[str] = field(default_factory=Counter)
    kyoku_group_tsumo_losses: Counter[str] = field(default_factory=Counter)
    kyoku_group_draws: Counter[str] = field(default_factory=Counter)
    rewards: list[float] = field(default_factory=list)
    reward_efficiency: list[float] = field(default_factory=list)
    reward_kyoku: list[float] = field(default_factory=list)
    reward_weighted_efficiency: list[float] = field(default_factory=list)
    reward_weighted_kyoku: list[float] = field(default_factory=list)
    completed_matches: int = 0
    match_ranks: list[int] = field(default_factory=list)
    match_point_deltas: list[float] = field(default_factory=list)
    match_kyoku_lengths: list[float] = field(default_factory=list)
    match_discard_counts: list[float] = field(default_factory=list)
    policy_decisions: Counter[str] = field(default_factory=Counter)
    policy_seats: Counter[str] = field(default_factory=Counter)
    structural_tenpai: int = 0
    open_no_yaku_tenpai: int = 0
    furiten_tenpai: int = 0
    effective_ron_tiles: list[float] = field(default_factory=list)
    accepted_calls: int = 0
    bad_calls: int = 0
    structural_optimal: list[float] = field(default_factory=list)
    rule_tenpai_preference: list[float] = field(default_factory=list)

    def record_decision(self, action_id: int, legal_mask: np.ndarray, *, threat: bool = False,
                        genbutsu_to_all: bool = False, genbutsu_count: int = 0,
                        seat: int | None = None, dealer: bool | None = None,
                        phase: str | None = None, prior_riichi_count: int = 0) -> None:
        self.decisions += 1
        kind = action_kind(action_id)
        self.actions[kind] += 1
        legal = np.asarray(legal_mask, dtype=np.bool_)
        self.legal_actions.append(float(legal.sum()))
        self.riichi_opportunities += int(bool(legal[75]))
        self.call_opportunities += int(bool(legal[76:171].any()))
        if threat and action_kind(action_id) in {"discard", "tsumogiri"}:
            self.threat_discards += 1
            self.genbutsu_all += int(genbutsu_to_all)
            self.genbutsu_coverage.append(float(genbutsu_count))
        if kind == "riichi":
            self.first_riichis += int(prior_riichi_count == 0)
            self.chase_riichis += int(prior_riichi_count > 0)
        groups: list[str] = []
        if seat is not None:
            groups.append(f"seat_{int(seat)}")
        if dealer is not None:
            groups.append("dealer" if dealer else "nondealer")
        if phase:
            groups.append(f"phase_{phase}")
        for group in groups:
            self.decision_groups[group] += 1
            self.action_groups[f"{group}/{kind}"] += 1

    def record_efficiency(self, *, reward: float, shanten_gap: int | None = None,
                          ukeire_loss: float | None = None,
                          selected_shanten: int | None = None,
                          selected_effective_shanten: int | None = None) -> None:
        self.efficiency_rewards.append(float(reward))
        if shanten_gap is not None:
            self.shanten_gaps.append(float(shanten_gap))
            self.optimal_shanten.append(float(shanten_gap == 0))
        if ukeire_loss is not None:
            self.ukeire_losses.append(float(ukeire_loss))
            self.optimal_ukeire.append(float(ukeire_loss == 0.0))
        if selected_shanten is not None:
            self.selected_shanten.append(float(selected_shanten))
        if selected_effective_shanten is not None:
            self.selected_effective_shanten.append(float(selected_effective_shanten))

    def record_rule_action(
        self, action_id: int, *, threat: bool, bad_kan: bool = False,
    ) -> None:
        kind = action_kind(action_id)
        is_call = kind in {"chi", "pon", "daiminkan"}
        is_kan = kind in {"daiminkan", "ankan", "kakan"}
        self.dangerous_calls += int(is_call and threat)
        self.accepted_kans += int(is_kan)
        self.bad_kans += int(is_kan and bad_kan)

    def record_transition_reward(self, transition: object) -> None:
        efficiency = float(getattr(transition, "discard_regret", 0.0))
        efficiency_weight = float(getattr(transition, "discard_weight", 0.0))
        kyoku = float(getattr(transition, "kyoku_reward"))
        self.rewards.append(float(getattr(transition, "reward")))
        self.reward_efficiency.append(efficiency); self.reward_kyoku.append(kyoku)
        self.reward_weighted_efficiency.append(efficiency_weight * efficiency)
        self.reward_weighted_kyoku.append(kyoku)

    def record_lineup(self, policies: Iterable[str], learner_seats: Iterable[int]) -> None:
        del learner_seats
        for _seat, policy in enumerate(policies):
            self.policy_seats[str(policy)] += 1
            # Approximate per-seat decision share by lineup occupancy so mixed
            # opponent lineups (E2) report a meaningful current fraction.
            self.policy_decisions[str(policy)] += 1

    def record_rule_quality(
        self,
        candidate: object,
        *,
        accepted_call: bool,
        bad_call: bool,
        best_rank: tuple[int, ...] | None = None,
        alternatives: Iterable[object] = (),
    ) -> None:
        if best_rank is not None:
            rank = tuple(getattr(candidate, "rank"))
            self.structural_optimal.append(float(rank[0] == best_rank[0]))
            relevant = [
                item for item in alternatives
                if int(getattr(item, "structural_shanten", 99)) == 0
            ]
            if len({(
                int(getattr(item, "effective_shanten", 99)),
                bool(getattr(item, "furiten", False)),
                bool(getattr(item, "open_no_yaku", False)),
            ) for item in relevant}) > 1:
                self.rule_tenpai_preference.append(float(rank == best_rank))
        if int(getattr(candidate, "structural_shanten", 99)) == 0:
            self.structural_tenpai += 1
            self.open_no_yaku_tenpai += int(bool(getattr(candidate, "open_no_yaku", False)))
            self.furiten_tenpai += int(bool(getattr(candidate, "furiten", False)))
            self.effective_ron_tiles.append(float(getattr(candidate, "live_ron", 0)))
        self.accepted_calls += int(accepted_call)
        self.bad_calls += int(accepted_call and bad_call)

    def record_kyoku(
        self,
        learner_seats: Iterable[int],
        score_deltas: Iterable[float],
        events: Iterable[Iterable[str]],
        *,
        discard_count: int | None = None,
        open_meld_count: int | None = None,
        dealer_seat: int | None = None,
        phase: str | None = None,
        draw_tenpai: bool | Iterable[bool | None] | None = None,
        exhaustive_draw: bool = False,
    ) -> None:
        seats = set(int(seat) for seat in learner_seats)
        deltas = [float(value) / 1000.0 for value in score_deltas]
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for source in events:
            for raw in source:
                if raw in seen:
                    continue
                seen.add(raw)
                try:
                    rows.append(json.loads(raw))
                except (TypeError, ValueError):
                    continue
        horas = [row for row in rows if row.get("type") == "hora"]
        is_draw = any(row.get("type") == "ryukyoku" for row in rows)
        tenpai_by_seat: dict[int, bool] = {}
        if draw_tenpai is not None:
            if isinstance(draw_tenpai, (list, tuple, np.ndarray)):
                for seat_index, flag in enumerate(draw_tenpai):
                    if flag is not None:
                        tenpai_by_seat[seat_index] = bool(flag)
            else:
                tenpai_by_seat = {seat: bool(draw_tenpai) for seat in seats}
        if open_meld_count is not None:
            self.kyoku_open_melds.append(float(open_meld_count))
        for seat in seats:
            point = deltas[seat]
            self.kyoku_points.append(point)
            if discard_count is not None:
                self.kyoku_discard_counts.append(float(discard_count))
            won = [row for row in horas if int(row.get("actor", -1)) == seat]
            dealt = [row for row in horas if int(row.get("target", -1)) == seat and int(row.get("actor", -1)) != seat]
            lost_to_tsumo = any(
                int(row.get("actor", -1)) != seat
                and int(row.get("target", -2)) == int(row.get("actor", -1))
                for row in horas
            )
            if won:
                self.wins += 1
                self.win_points.append(point)
                self.tsumo_wins += sum(int(row.get("target", -1)) == int(row.get("actor", -2)) for row in won)
                self.ron_wins += sum(int(row.get("target", -1)) != int(row.get("actor", -2)) for row in won)
            if dealt:
                self.deal_ins += 1
                self.deal_in_points.append(point)
            if lost_to_tsumo:
                self.tsumo_losses += 1
            if is_draw:
                self.draws += 1
                self.exhaustive_draws += int(exhaustive_draw)
                self.draw_points.append(point)
                if seat in tenpai_by_seat:
                    self.draw_tenpai += int(tenpai_by_seat[seat])
            groups = [f"seat_{seat}"]
            if dealer_seat is not None:
                groups.append("dealer" if seat == int(dealer_seat) else "nondealer")
            if phase:
                groups.append(f"phase_{phase}")
            for group in groups:
                self.kyoku_group_points.setdefault(group, []).append(point)
                self.kyoku_group_wins[group] += int(bool(won))
                self.kyoku_group_deal_ins[group] += int(bool(dealt))
                self.kyoku_group_tsumo_losses[group] += int(lost_to_tsumo)
                self.kyoku_group_draws[group] += int(is_draw)

    def record_match_length(self, kyoku_count: int, *, discard_count: int | None = None) -> None:
        """Record one completed physical hanchan without assigning a seat rank."""
        self.completed_matches += 1
        self.match_kyoku_lengths.append(float(kyoku_count))
        if discard_count is not None:
            self.match_discard_counts.append(float(discard_count))

    def record_match_result(self, learner_seat: int, final_scores: Iterable[float], *, kyoku_count: int | None = None,
                            discard_count: int | None = None, point_delta: float | None = None) -> None:
        """Record the candidate's final placement for evaluation reporting only.

        Equal final scores are broken by seat id to give every completed match
        exactly one placement.  Evaluation rotates the candidate through all
        four seats, so this deterministic fallback cannot favour one policy.
        These values are never used by rollout rewards or checkpoint selection.
        """
        scores = [float(score) for score in final_scores]
        seat = int(learner_seat)
        if not 0 <= seat < len(scores):
            raise ValueError(f"learner seat {seat} is outside final scores")
        ranking = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        self.match_ranks.append(ranking.index(seat) + 1)
        self.completed_matches += 1
        if point_delta is not None:
            self.match_point_deltas.append(float(point_delta))
        if kyoku_count is not None:
            self.match_kyoku_lengths.append(float(kyoku_count))
        if discard_count is not None:
            self.match_discard_counts.append(float(discard_count))

    def summary(self, prefix: str = "train") -> dict[str, float]:
        kyokus = len(self.kyoku_points)
        matches = self.completed_matches
        result = {
            f"{prefix}/kyoku/count": float(kyokus),
            f"{prefix}/kyoku/point_delta_mean": _mean(self.kyoku_points),
            f"{prefix}/kyoku/point_delta_p10": _percentile(self.kyoku_points, 10),
            f"{prefix}/kyoku/point_delta_p50": _percentile(self.kyoku_points, 50),
            f"{prefix}/kyoku/point_delta_p90": _percentile(self.kyoku_points, 90),
            f"{prefix}/kyoku/win_rate": _rate(self.wins, kyokus),
            f"{prefix}/kyoku/tsumo_rate": _rate(self.tsumo_wins, kyokus),
            f"{prefix}/kyoku/ron_rate": _rate(self.ron_wins, kyokus),
            f"{prefix}/kyoku/deal_in_rate": _rate(self.deal_ins, kyokus),
            f"{prefix}/kyoku/tsumo_loss_rate": _rate(self.tsumo_losses, kyokus),
            f"{prefix}/kyoku/win_points_mean": _mean(self.win_points),
            f"{prefix}/kyoku/deal_in_points_mean": _mean(self.deal_in_points),
            f"{prefix}/kyoku/draw_rate": _rate(self.draws, kyokus),
            f"{prefix}/kyoku/draw_point_delta_mean": _mean(self.draw_points),
            f"{prefix}/kyoku/draw_tenpai_count": float(self.draw_tenpai),
            f"{prefix}/kyoku/exhaustive_draw_count": float(self.exhaustive_draws),
            f"{prefix}/kyoku/draw_tenpai_rate": _rate(self.draw_tenpai, self.exhaustive_draws),
            f"{prefix}/kyoku/discard_count_mean": _mean(self.kyoku_discard_counts),
            f"{prefix}/kyoku/open_melds_mean": _mean(self.kyoku_open_melds),
            f"{prefix}/action/decision_count": float(self.decisions),
            f"{prefix}/action/legal_count_mean": _mean(self.legal_actions),
            f"{prefix}/action/riichi_opportunity_count": float(self.riichi_opportunities),
            f"{prefix}/action/call_opportunity_count": float(self.call_opportunities),
            f"{prefix}/action/riichi_opportunity_accept_rate": _rate(self.actions["riichi"], self.riichi_opportunities),
            f"{prefix}/action/call_opportunity_accept_rate": _rate(
                sum(self.actions[key] for key in ("chi", "pon", "daiminkan")),
                self.call_opportunities,
            ),
            f"{prefix}/action/first_riichi_rate": _rate(self.first_riichis, kyokus),
            f"{prefix}/action/chase_riichi_rate": _rate(self.chase_riichis, kyokus),
            f"{prefix}/efficiency/reward_mean": _mean(self.efficiency_rewards),
            f"{prefix}/efficiency/optimal_shanten_rate": _mean(self.optimal_shanten),
            f"{prefix}/efficiency/optimal_ukeire_rate": _mean(self.optimal_ukeire),
            f"{prefix}/efficiency/shanten_gap_mean": _mean(self.shanten_gaps),
            f"{prefix}/efficiency/ukeire_loss_mean": _mean(self.ukeire_losses),
            f"{prefix}/efficiency/selected_shanten_mean": _mean(self.selected_shanten),
            f"{prefix}/efficiency/selected_effective_shanten_mean": _mean(
                self.selected_effective_shanten,
            ),
            f"{prefix}/defense/threat_discard_count": float(self.threat_discards),
            f"{prefix}/defense/genbutsu_all_rate": _rate(self.genbutsu_all, self.threat_discards),
            f"{prefix}/defense/genbutsu_coverage_mean": _mean(self.genbutsu_coverage),
            f"{prefix}/rules/open_no_yaku_tenpai_rate": _rate(
                self.open_no_yaku_tenpai, self.structural_tenpai,
            ),
            f"{prefix}/rules/furiten_tenpai_rate": _rate(
                self.furiten_tenpai, self.structural_tenpai,
            ),
            f"{prefix}/rules/effective_ron_tiles_mean": _mean(self.effective_ron_tiles),
            f"{prefix}/rules/bad_call_rate": _rate(self.bad_calls, self.accepted_calls),
            f"{prefix}/rules/dangerous_call_rate": _rate(
                self.dangerous_calls, self.accepted_calls,
            ),
            f"{prefix}/rules/bad_kan_rate": _rate(self.bad_kans, self.accepted_kans),
            f"{prefix}/fixed/structural_optimal_rate": _mean(self.structural_optimal),
            f"{prefix}/fixed/rule_tenpai_preference_accuracy": _mean(
                self.rule_tenpai_preference,
            ),
            f"{prefix}/reward/total_mean": _mean(self.rewards),
            f"{prefix}/match/count": float(matches),
            f"{prefix}/match/first_place_rate": _rate(sum(rank == 1 for rank in self.match_ranks), matches),
            f"{prefix}/match/mean_rank": _mean([float(rank) for rank in self.match_ranks]),
            f"{prefix}/match/top2_rate": _rate(sum(rank <= 2 for rank in self.match_ranks), matches),
            f"{prefix}/match/last_place_rate": _rate(sum(rank == 4 for rank in self.match_ranks), matches),
            f"{prefix}/match/point_delta_mean": _mean(self.match_point_deltas),
            f"{prefix}/match/point_delta_p10": _percentile(self.match_point_deltas, 10),
            f"{prefix}/match/point_delta_p50": _percentile(self.match_point_deltas, 50),
            f"{prefix}/match/point_delta_p90": _percentile(self.match_point_deltas, 90),
            f"{prefix}/match/positive_point_delta_rate": _rate(
                sum(delta > 0.0 for delta in self.match_point_deltas), matches,
            ),
            f"{prefix}/match/length_kyokus_mean": _mean(self.match_kyoku_lengths),
            f"{prefix}/match/discard_count_mean": _mean(self.match_discard_counts),
        }
        for kind in ("pass", "discard", "tsumogiri", "riichi", "chi", "pon", "daiminkan", "ankan", "kakan", "hora", "kyushu"):
            result[f"{prefix}/action/{kind}_rate"] = _rate(self.actions[kind], self.decisions)
        for group, count in sorted(self.decision_groups.items()):
            result[f"{prefix}/breakdown/{group}/decision_count"] = float(count)
            for kind in ("pass", "discard", "tsumogiri", "riichi", "chi", "pon", "daiminkan", "ankan", "kakan"):
                result[f"{prefix}/breakdown/{group}/{kind}_rate"] = _rate(
                    self.action_groups[f"{group}/{kind}"], count,
                )
        for group, points in sorted(self.kyoku_group_points.items()):
            count = len(points)
            result[f"{prefix}/breakdown/{group}/kyoku_count"] = float(count)
            result[f"{prefix}/breakdown/{group}/point_delta_mean"] = _mean(points)
            result[f"{prefix}/breakdown/{group}/win_rate"] = _rate(
                self.kyoku_group_wins[group], count,
            )
            result[f"{prefix}/breakdown/{group}/deal_in_rate"] = _rate(
                self.kyoku_group_deal_ins[group], count,
            )
            result[f"{prefix}/breakdown/{group}/tsumo_loss_rate"] = _rate(
                self.kyoku_group_tsumo_losses[group], count,
            )
            result[f"{prefix}/breakdown/{group}/draw_rate"] = _rate(
                self.kyoku_group_draws[group], count,
            )
        for name, values in (("efficiency", self.reward_efficiency), ("kyoku", self.reward_kyoku),
                             ("weighted_efficiency", self.reward_weighted_efficiency), ("weighted_kyoku", self.reward_weighted_kyoku)):
            result[f"{prefix}/reward/{name}_mean"] = _mean(values)
        total_seats = sum(self.policy_seats.values())
        total_decisions = sum(self.policy_decisions.values())
        result[f"{prefix}/opponents/current_seat_fraction"] = _rate(self.policy_seats["current"], total_seats)
        result[f"{prefix}/opponents/current_decision_fraction"] = _rate(self.policy_decisions["current"], total_decisions)
        return result


class RollingKyokuMetrics:
    """Small bounded history used only for online health trends."""
    def __init__(self, capacity: int = 1000) -> None:
        self.points: deque[float] = deque(maxlen=max(1, int(capacity)))

    def update(self, values: Iterable[float]) -> dict[str, float]:
        self.points.extend(float(value) for value in values)
        return {"train_rolling/kyoku/window_count": float(len(self.points)), "train_rolling/kyoku/point_delta_mean": _mean(list(self.points))}


def ppo_buffer_metrics(transitions: Iterable[object]) -> dict[str, float]:
    rows = list(transitions)
    values = np.asarray([float(item.value) for item in rows], dtype=np.float64)
    returns = np.asarray([float(item.return_) for item in rows], dtype=np.float64)
    advantages = np.asarray([float(item.advantage) for item in rows], dtype=np.float64)
    variance = float(np.var(returns))
    explained = 0.0 if variance <= 1e-12 else 1.0 - float(np.var(returns - values)) / variance
    result = {"explained_variance": explained}
    for name, values_ in (("old_value", values), ("return", returns), ("advantage", advantages)):
        result[f"buffer/{name}_mean"] = float(values_.mean()) if len(values_) else 0.0
        result[f"buffer/{name}_std"] = float(values_.std()) if len(values_) else 0.0
    return result


def append_metric_jsonl(path: str | Path, *, update: int, global_decisions: int, global_kyokus: int,
                        source: str, metrics: Mapping[str, float], metadata: Mapping[str, object] | None = None) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    row = {"schema_version": METRIC_SCHEMA_VERSION, "update": int(update), "global_decisions": int(global_decisions),
           "global_kyokus": int(global_kyokus), "source": str(source), "metrics": {key: float(value) for key, value in metrics.items() if math.isfinite(float(value))}}
    if metadata:
        row["metadata"] = dict(metadata)
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def metric_counters(path: str | Path) -> tuple[int, int]:
    """Recover cumulative counters on resume without loading training state."""
    destination = Path(path)
    if not destination.exists():
        return 0, 0
    try:
        with destination.open("rb") as file:
            file.seek(0, 2)
            end = file.tell()
            file.seek(max(0, end - 65_536))
            rows = [row for row in file.read().decode("utf-8").splitlines() if row]
        latest = json.loads(rows[-1]) if rows else {}
        return int(latest.get("global_decisions", 0)), int(latest.get("global_kyokus", 0))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return 0, 0
