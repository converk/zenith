"""Single-seat RiichiEnv observation to semantic-token bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

# The bot deliberately imports the *training-side* decision-analysis stack
# rather than re-implementing candidate-token encoding.  ``DecisionAnalysisBatch``
# produces the segment=7 candidate tokens (one per legal action) that the model
# sees during SFT, PPO rollout, and online evaluation
# (``riichi_ppo_v1/sft/head_to_head.py``, ``heuristic_evaluation.py``).
# Without these tokens the model's input distribution shifts → argmax drifts on
# ~43% of decisions in empirical tests (see
# ``riichi_lab_bot/tools/verify_candidate_token_drift.py``).
from riichi_ppo_v1.model.bridge import (
    NUM_PLAYERS,
    Decision,
    action_jsons_and_decision_flag,
    snapshot_json,
)
from riichi_ppo_v1.model.semantic_validation import assert_actor_token_semantics
from riichi_ppo_v1.training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)

from .features import encode_public_summary
from .model import NUM_ACTIONS, NUMERIC_WIDTH, TOKEN_WIDTH
from .observation import ObservationView, ThreatSnapshotTracker

_MODEL_EVENT_TYPES = frozenset(
    {
        "start_game",
        "start_kyoku",
        "tsumo",
        "dahai",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
        "dora",
        "reach",
        "reach_accepted",
        "hora",
        "ryukyoku",
        "end_kyoku",
        "end_game",
    }
)


@dataclass(frozen=True)
class EventContext:
    last_type: str | None = None
    actor: int | None = None
    pai: str | None = None


@dataclass(frozen=True)
class PreparedDecision:
    observation: Any
    seat: int
    token_factors: np.ndarray
    token_numeric: np.ndarray
    token_length: int
    legal_mask: np.ndarray
    legal_jsons: tuple[str, ...]
    event_context: EventContext


def _parse_event_context(events: list[str]) -> EventContext:
    context = EventContext()
    for raw in events:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        event_type = value.get("type")
        # Kakan and (in the kokushi variant) ankan expose a ron target for
        # chankan/rob-a-kan just like a discard does.
        if event_type in {"tsumo", "dahai", "kakan", "ankan"}:
            actor = value.get("actor")
            context = EventContext(
                str(event_type),
                int(actor) if isinstance(actor, int) else None,
                value.get("pai") if isinstance(value.get("pai"), str) else None,
            )
    return context


class OnlineStateBridge:
    """Stateful bridge for exactly one bot seat in one connection."""

    def __init__(self, seat: int) -> None:
        if not 0 <= int(seat) < NUM_PLAYERS:
            raise ValueError("4-player seat must be in [0, 3]")
        try:
            import riichi
        except ImportError as exc:
            raise RuntimeError(
                "the local riichi native extension is not installed"
            ) from exc
        self.seat = int(seat)
        self.manager = riichi.MjaiKyokuStateMachineManager(1)
        self.event_context = EventContext()
        # The bot mirrors the training path's candidate-token injection by
        # keeping an efficiency analyzer and public-state tracker that are
        # fed model events on every ``prepare`` call.
        self.analyzer = EfficiencyAnalyzer(131_072)
        self.public = PublicStateTracker(1)
        self.threats = ThreatSnapshotTracker()

    def prepare(self, observation: Any) -> PreparedDecision:
        if int(observation.player_id) != self.seat:
            raise ValueError(
                "Observation seat mismatch: "
                f"expected {self.seat}, got {observation.player_id}"
            )
        legal_actions = list(observation.legal_actions())
        if not legal_actions:
            raise ValueError("request_action observation has no legal actions")

        raw_events = list(observation.new_events())
        accepted_events: list[str] = []
        saw_start_kyoku = False
        for raw in raw_events:
            try:
                event_type = json.loads(raw).get("type")
            except (TypeError, json.JSONDecodeError):
                continue
            if event_type in _MODEL_EVENT_TYPES:
                accepted_events.append(raw)
                if event_type == "start_kyoku":
                    saw_start_kyoku = True
        self.threats.apply_events(accepted_events)
        # Reset cross-kyoku event_context residue: when a new kyoku starts the
        # previous kyoku's last tsumo/dahai is no longer a valid ron target,
        # so ``safety.action_to_response`` must not reuse it as ``target``.
        if saw_start_kyoku:
            self.event_context = EventContext()
        new_context = _parse_event_context(accepted_events)
        if new_context.last_type is not None:
            self.event_context = new_context

        derived_fields = self.threats.fields()
        riichi_overrides = {
            "riichi_declared": derived_fields["riichi_declared"],
            "riichi_accepted": derived_fields["riichi_accepted"],
            "riichi_declaration_indices": derived_fields[
                "riichi_declaration_indices"
            ],
            "riichi_sutehais": derived_fields["riichi_sutehais"],
            "tsumogiri_flags": derived_fields["tsumogiri_flags"],
        }
        if isinstance(observation, ObservationView):
            observation.set_fields(
                {
                    name: value
                    for name, value in derived_fields.items()
                    if name in observation.missing_fields
                }
            )
            observation.set_fields(riichi_overrides)
        else:
            current = {
                name: getattr(observation, name, None)
                for name in riichi_overrides
            }
            if current != riichi_overrides:
                observation = ObservationView(observation, riichi_overrides)

        events_by_player = [
            accepted_events if player == self.seat else []
            for player in range(NUM_PLAYERS)
        ]
        self.manager.apply_events_batch([0], [events_by_player])
        # PublicStateTracker reads ``actor`` out of each MJAI event payload, so
        # feeding the bot's own event delta is enough to advance the public view
        # of every seat (rivers, melds, riichi, dora).
        self.public.update([events_by_player])

        legal_jsons, decision_flag = action_jsons_and_decision_flag(observation)
        factors, numeric, lengths, mask, _generation = (
            self.manager.prepare_decisions(
                [self.seat],
                [legal_jsons],
                [snapshot_json(observation, decision_flag)],
            )
        )
        factors_a = np.asarray(factors, dtype=np.uint8)
        numeric_a = np.asarray(numeric, dtype=np.float32)
        lengths_a = np.asarray(lengths, dtype=np.int64)
        mask_a = np.asarray(mask, dtype=np.bool_)
        if factors_a.ndim != 3 or factors_a.shape[0] != 1:
            raise RuntimeError(
                f"state machine returned malformed factors {factors_a.shape}"
            )
        if factors_a.shape[2] != TOKEN_WIDTH:
            raise RuntimeError("state machine returned wrong token width")
        if numeric_a.shape != (*factors_a.shape[:2], NUMERIC_WIDTH):
            raise RuntimeError("state machine returned malformed numeric data")
        if lengths_a.shape != (1,) or mask_a.shape != (1, NUM_ACTIONS):
            raise RuntimeError("state machine returned malformed metadata")
        if not mask_a[0].any():
            raise RuntimeError("state machine returned an empty legal mask")

        base_length = int(lengths_a[0])

        # Inject the v13 state rows and segment=7 candidate tokens in the same
        # order as the training path
        # (``riichi_ppo_v1/model/bridge.py:BatchedStateBridge.prepare``):
        # base history/state -> public summary -> six state rows -> candidate
        # query pairs.
        decision = Decision(0, self.seat, observation)
        analysis = DecisionAnalysisBatch.build(
            [decision], analyzer=self.analyzer, public=self.public,
        )
        state_factors, state_numeric = analysis.state_tokens([decision])
        candidate_factors, candidate_numeric = analysis.candidate_tokens(
            [decision], mask_a,
        )
        state_factor_row = np.asarray(state_factors[0], dtype=np.uint8)
        state_numeric_row = (
            np.asarray(state_numeric[0], dtype=np.float32)
            if state_numeric
            else np.zeros((0, NUMERIC_WIDTH), dtype=np.float32)
        )
        candidate_factor_row = np.asarray(candidate_factors[0], dtype=np.uint8)
        candidate_numeric_row = (
            np.asarray(candidate_numeric[0], dtype=np.float32)
            if candidate_numeric
            else np.zeros((0, NUMERIC_WIDTH), dtype=np.float32)
        )
        candidate_count = int(candidate_factor_row.shape[0])

        public = encode_public_summary(observation, self.seat)
        state_count = int(state_factor_row.shape[0])
        total_length = (
            base_length + len(public) + state_count + candidate_count
        )
        if total_length > 4096:
            raise RuntimeError(
                f"token context overflow: {total_length} > 4096"
            )
        output_factors = np.zeros(
            (total_length, TOKEN_WIDTH), dtype=np.uint8
        )
        output_numeric = np.zeros(
            (total_length, NUMERIC_WIDTH), dtype=np.float32
        )
        output_factors[:base_length] = factors_a[0, :base_length]
        output_numeric[:base_length] = numeric_a[0, :base_length]
        cursor = base_length
        if len(public):
            output_factors[cursor : cursor + len(public)] = public
            cursor += len(public)
        if state_count:
            output_factors[cursor : cursor + state_count] = state_factor_row
            output_numeric[cursor : cursor + state_count] = state_numeric_row
            cursor += state_count
        if candidate_count:
            output_factors[cursor : cursor + candidate_count] = candidate_factor_row
            output_numeric[cursor : cursor + candidate_count] = candidate_numeric_row
            cursor += candidate_count
        if cursor != total_length:
            raise RuntimeError(
                "token assembly length mismatch: "
                f"{cursor} != {total_length}"
            )
        assert_actor_token_semantics(
            output_factors[None],
            output_numeric[None],
            np.asarray([total_length], dtype=np.int64),
        )
        return PreparedDecision(
            observation=observation,
            seat=self.seat,
            token_factors=output_factors,
            token_numeric=output_numeric,
            token_length=total_length,
            legal_mask=mask_a[0].copy(),
            legal_jsons=tuple(legal_jsons),
            event_context=self.event_context,
        )

    def decode(self, prepared: PreparedDecision, action_id: int) -> Any:
        if not prepared.legal_mask[int(action_id)]:
            raise ValueError(f"model selected masked action id {action_id}")
        mjai = self.manager.decode_actions(
            [self.seat], [int(action_id)]
        )[0]
        action = prepared.observation.select_action_from_mjai(mjai)
        if action is None:
            raise RuntimeError(
                f"RiichiEnv rejected decoded action id={action_id}: {mjai}"
            )
        return action
