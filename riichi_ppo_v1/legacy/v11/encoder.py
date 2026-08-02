"""Frozen online feature encoder used only for v11 checkpoint evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...model.bridge import Decision, _action_jsons_and_decision_flag, snapshot_json
from ...model.critic_features import collect_visible_table_state, encode_public_summary
from ...training.rewards.decision import DecisionAnalysisBatch, _action_type_code
from .contract import V11_REPLAY_RUNTIME_ID


def assert_runtime() -> None:
    import riichienv

    if getattr(riichienv, "REPLAY_SEMANTICS_VERSION", None) != V11_REPLAY_RUNTIME_ID:
        raise RuntimeError("RiichiEnv runtime is incompatible with the frozen v11 encoder")


def _candidate_tokens(
    analysis_batch: DecisionAnalysisBatch,
    decisions: list[Decision],
    legal_masks: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    factor_rows: list[np.ndarray] = []
    numeric_rows: list[np.ndarray] = []
    for row_index, decision in enumerate(decisions):
        analysis = analysis_batch.for_decision(decision)
        by_id = {candidate.action_id: candidate for candidate in analysis.candidates}
        legal_ids = np.flatnonzero(legal_masks[row_index])
        factors = np.zeros((len(legal_ids), 10), dtype=np.uint8)
        numeric = np.zeros((len(legal_ids), 8), dtype=np.float32)
        threats = sum(bool(value) for value in getattr(decision.observation, "riichi_declared", ()))
        best_shanten = min((item.structural_shanten for item in analysis.candidates), default=0)
        for token, aid_value in enumerate(legal_ids):
            aid = int(aid_value)
            candidate = by_id.get(aid)
            factors[token, :4] = (7, _action_type_code(aid), min(aid + 1, 255), min(threats, 3))
            if candidate is None:
                continue
            factors[token, 4] = min(candidate.structural_shanten + 1, 7)
            factors[token, 5] = min(candidate.effective_shanten + 1, 15)
            factors[token, 6] = int(candidate.has_yaku) + 2 * int(candidate.riichi_route)
            factors[token, 7] = (
                int(candidate.open_no_yaku) + 2 * int(candidate.furiten)
                + 4 * int(candidate.closed)
            )
            factors[token, 8] = (
                min(candidate.ron_points // 1000, 15)
                + 16 * min(candidate.tsumo_points // 1000, 15)
            )
            factors[token, 9] = int(candidate.four_visible)
            numeric[token] = (
                candidate.structural_shanten / 6.0,
                candidate.effective_shanten / 6.0,
                (candidate.structural_shanten - best_shanten) / 3.0,
                candidate.ukeire / 40.0,
                candidate.live_ron / 16.0,
                candidate.live_tsumo / 16.0,
                max(candidate.ron_points, candidate.tsumo_points) / 12000.0,
                candidate.genbutsu_coverage / 3.0,
            )
        factor_rows.append(factors)
        numeric_rows.append(numeric)
    return factor_rows, numeric_rows


def prepare_v11(
    bridge: Any,
    decisions: list[Decision],
    analysis: DecisionAnalysisBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode base history + legacy candidates + public summary, in that order."""
    assert_runtime()
    if not decisions:
        raise ValueError("cannot prepare an empty v11 decision batch")
    rows = [_action_jsons_and_decision_flag(item.observation) for item in decisions]
    prepared = bridge.state_machine.prepare_decisions(
        [item.batch_index for item in decisions],
        [actions for actions, _flag in rows],
        [
            snapshot_json(item.observation, flag)
            for item, (_actions, flag) in zip(decisions, rows, strict=True)
        ],
    )
    factors = np.asarray(prepared[0], dtype=np.uint8)
    numeric = np.asarray(prepared[1], dtype=np.float32)
    lengths = np.asarray(prepared[2], dtype=np.int64)
    legal = np.asarray(prepared[3], dtype=np.bool_)
    generations = np.asarray(prepared[4], dtype=np.int64)
    legacy_factors, legacy_numeric = _candidate_tokens(analysis, decisions, legal)
    if bridge.observations_by_env is None:
        public = [np.zeros((0, 10), dtype=np.uint8) for _item in decisions]
    else:
        tables = {
            env_index: collect_visible_table_state(
                bridge.observations_by_env[env_index], include_public_state=True,
            )
            for env_index in {item.env_index for item in decisions}
        }
        public = [
            encode_public_summary(tables[item.env_index], item.seat_id).factors
            for item in decisions
        ]
    new_lengths = lengths + np.asarray(
        [len(a) + len(b) for a, b in zip(legacy_factors, public, strict=True)],
        dtype=np.int64,
    )
    if np.any(new_lengths + 1 > 4096):
        raise RuntimeError("v11 tokens overflow the frozen 4096-token context")
    out_factors = np.zeros((len(decisions), int(new_lengths.max()), 10), dtype=np.uint8)
    out_numeric = np.zeros((len(decisions), int(new_lengths.max()), 8), dtype=np.float32)
    for row, (candidate_factors, candidate_numeric, public_factors) in enumerate(
        zip(legacy_factors, legacy_numeric, public, strict=True)
    ):
        base = int(lengths[row])
        candidate_end = base + len(candidate_factors)
        out_factors[row, :base] = factors[row, :base]
        out_numeric[row, :base] = numeric[row, :base]
        out_factors[row, base:candidate_end] = candidate_factors
        out_numeric[row, base:candidate_end] = candidate_numeric
        out_factors[row, candidate_end:new_lengths[row]] = public_factors
    if legal.shape != (len(decisions), 241) or not np.all(legal.any(axis=1)):
        raise RuntimeError("v11 encoder received a malformed legal mask")
    return out_factors, out_numeric, new_lengths, legal, generations
