"""Run one hanchan of PPO (vs SFT 2v2) and write a human-readable decision trace.

The output format matches ``checkpoints/train_riichi_v9/best_score_hanchan_trace_seed3.md``:
each PPO decision is rendered as a Markdown section showing the round/honba
state, hand, drawn tile, visible table (rivers + melds for every seat), and the
full list of legal actions with the policy's probability, marking the selected
action.

Usage (from the workspace root):

    CUDA_DEVICE=3 conda run -n Mahjong-AI python \
        riichi_ppo_v1/tools/record_ppo_actions.py \
        --ppo checkpoints/train_riichi_v11_ppo_selected/best.pt \
        --baseline checkpoints/train_riichi_v11_sft_40pct/best_heuristic.pt \
        --output checkpoints/train_riichi_v11_ppo_selected/best_hanchan_trace_seed20260730.md

PPO (model_a) occupies seats 0 and 1; SFT (model_b) occupies seats 2 and 3.
The hanchan is deterministic given the fixed ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Keep the project-facing device convention consistent with the training entry
# points. This must happen before importing torch.
if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

# Make sure the project package is importable when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from riichi_ppo_v1.model.bridge import (
    BatchedStateBridge,
    Decision,
    NUM_PLAYERS,
    NUM_ACTIONS,
    tile_id_to_mjai,
)
from riichi_ppo_v1.sft.head_to_head import _bf16_supported, _load_model, _tensor
from riichi_ppo_v1.training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from riichi_ppo_v1.training.rewards.decision import action_id as compute_action_id
from riichi_ppo_v1.training.worker import active_decisions


# Mapping from round_wind / kyoku_index to the Mahjong round notation used in
# the trace file (E1, E2, ..., S1, S2, ...).
_WIND_CHARS = "ESWN"


def _round_label(round_wind: int, kyoku_index: int) -> str:
    """Return the canonical ``E1``/``E4``/``S2`` style round label."""
    wind = _WIND_CHARS[int(round_wind) % 4]
    kyoku = int(kyoku_index) + 1
    return f"{wind}{kyoku}"


def _seat_label(seat: int) -> str:
    return f"Seat {int(seat)}"


def _meld_type_name(meld_type: int) -> str:
    return {
        0: "meldtype.chi",
        1: "meldtype.pon",
        2: "meldtype.daiminkan",
        3: "meldtype.ankan",
        4: "meldtype.kakan",
    }.get(int(meld_type), f"meldtype.{int(meld_type)}")


def _format_meld(meld: Any) -> str:
    """Render one meld as ``meldtype.chi(2s 3s 4s; called=3s; from=Seat 0)``."""
    mt = _meld_type_name(int(meld.meld_type))
    tiles = [tile_id_to_mjai(int(t)) for t in meld.tiles]
    called = meld.called_tile
    if called is None:
        called_str = "-"
    else:
        called_str = tile_id_to_mjai(int(called))
    opened = "opened" if bool(meld.opened) else "closed"
    from_who = meld.from_who
    from_label = _seat_label(int(from_who)) if int(from_who) >= 0 else " Seat -"
    return f"{mt}({' '.join(tiles)}; called={called_str}; {opened}; from={from_label})"


def _format_hand(observation: Any) -> str:
    pid = int(observation.player_id)
    raw = [int(t) for t in observation.hand]
    drawn = getattr(observation, "drawn_tile", None)
    drawn_int = int(drawn) if drawn is not None else None
    # The hand from Observation already contains the drawn tile; keep it inside
    # the hand list so the length stays at 14 on a tsumo turn, matching the
    # trace file convention.
    tiles = [tile_id_to_mjai(t) for t in raw]
    return "[" + " ".join(tiles) + "]"


def _format_dora(dora_indicators: list[int]) -> str:
    if not dora_indicators:
        return "[]"
    return "[" + " ".join(tile_id_to_mjai(int(t)) for t in dora_indicators) + "]"


def _format_scores(scores: list[int]) -> str:
    return "[" + ", ".join(str(int(s)) for s in scores) + "]"


def _drawn_tile_str(observation: Any) -> str:
    drawn = getattr(observation, "drawn_tile", None)
    if drawn is None:
        return "-"
    return tile_id_to_mjai(int(drawn)) or "-"


def _format_legal_option(
    action_index: int,
    action: Any,
    observation: Any,
    action_global_id: int | None = None,
) -> str:
    """Render a single legal action like ``dahai 3s (tedashi)``."""
    tsumogiri = _tsumogiri_for(action, observation, action_global_id)
    mjai_str = action.to_mjai()
    mjai = json.loads(mjai_str)
    atype = str(mjai.get("type", ""))
    # discard-style rendering matches the reference trace.
    if atype == "dahai":
        marker = "tsumogiri" if tsumogiri else "tedashi"
        return f"dahai {mjai['pai']} ({marker})  {mjai_str}"
    if atype == "none":
        return f"none  {mjai_str}"
    if atype == "reach":
        return f"reach  {mjai_str}"
    if atype == "chi":
        return f"chi {mjai['pai']}  {mjai_str}"
    if atype == "pon":
        return f"pon {mjai['pai']}  {mjai_str}"
    if atype == "daiminkan":
        return f"daiminkan {mjai['pai']}  {mjai_str}"
    if atype == "ankan":
        return f"ankan {mjai['pai']}  {mjai_str}"
    if atype == "kakan":
        return f"kakan {mjai['pai']}  {mjai_str}"
    if atype == "hora":
        return f"hora  {mjai_str}"
    if atype == "ryukyoku":
        return f"ryukyoku  {mjai_str}"
    return f"{atype}  {mjai_str}"


def _format_table_state(observation: Any) -> list[str]:
    """Render the per-seat river/meld lines used by the trace."""
    lines: list[str] = []
    for seat in range(NUM_PLAYERS):
        river = [tile_id_to_mjai(int(t)) for t in observation.discards[seat]]
        melds = observation.melds[seat]
        if melds:
            meld_str = ", ".join(_format_meld(m) for m in melds)
        else:
            meld_str = "-"
        river_str = "[" + " ".join(river) + "]" if river else "[]"
        lines.append(f"- {_seat_label(seat)}: river={river_str} melds=[{meld_str}]")
    return lines


def _tsumogiri_for(action: Any, observation: Any, action_global_id: int | None) -> bool:
    """Determine whether a discard action is a tsumogiri.

    Prefer the canonical policy-head action id (odd = tsumogiri) so that two
    discards of the same PAI are disambiguated correctly; fall back to the
    drawn-tile comparison for actions whose id is unavailable.
    """
    if action_global_id is not None and action_global_id >= 1 and action_global_id <= 74:
        # 1..74 are dahai ids; odd ids are tsumogiri.
        return (action_global_id - 1) % 2 == 1
    drawn = getattr(observation, "drawn_tile", None)
    return (
        action.tile is not None
        and drawn is not None
        and int(action.tile) == int(drawn)
        and action.action_type == 0  # DISCARD
    )


def _is_selected(
    action: Any,
    action_global_id: int | None,
    chosen_action_id: int,
) -> bool:
    """True if this legal action is the one the policy selected."""
    if action_global_id is not None:
        return int(action_global_id) == int(chosen_action_id)
    return False


def render_decision(
    decision: Decision,
    *,
    action_id: int,
    chosen_mjai: str,
    probs: list[float],
    legal_actions: list[Any],
    action_ids: list[int],
    index: int,
) -> str:
    """Render one decision section in the Markdown trace format."""
    obs = decision.observation
    seat = int(decision.seat_id)
    round_label = _round_label(int(obs.round_wind), int(obs.kyoku_index))
    dealer_seat = int(obs.oya)
    scores = _format_scores([int(s) for s in obs.scores])
    dora = _format_dora([int(t) for t in obs.dora_indicators])
    hand = _format_hand(obs)
    drawn = _drawn_tile_str(obs)
    table_lines = _format_table_state(obs)

    # Sort legal actions by their global action id, matching the ordering the
    # reference trace uses.
    triples = sorted(zip(action_ids, legal_actions, probs), key=lambda t: int(t[0]))

    lines: list[str] = []
    lines.append(f"## Decision {index}: Seat {seat}")
    lines.append("")
    lines.append(
        f"Round: {round_label}; dealer={_seat_label(dealer_seat)}; "
        f"honba={int(obs.honba)}; riichi_sticks={int(obs.riichi_sticks)}; "
        f"scores={scores}; dora={dora}"
    )
    lines.append("")
    lines.append(f"Hand: {hand}")
    lines.append("")
    lines.append(f"Drawn tile: {drawn}")
    lines.append("")
    lines.append("Visible table:")
    lines.append("")
    lines.extend(table_lines)
    lines.append("")
    lines.append("Legal options (policy probability):")
    lines.append("")
    for aid, action, prob in triples:
        aid_int = int(aid) if aid is not None else None
        rendered = _format_legal_option(aid_int or 0, action, obs, aid_int)
        is_sel = _is_selected(action, aid_int, action_id)
        marker = " ← SELECTED" if is_sel else ""
        lines.append(f"- {aid_int}: {rendered}; p={float(prob) * 100.0:.4f}%{marker}")
    lines.append("")
    selected_action = _action_for_id(action_ids, legal_actions, action_id)
    selected_aid = next(
        (aid for aid, action in zip(action_ids, legal_actions) if action is selected_action),
        action_id,
    )
    lines.append(
        f"Selected (action_id={action_id}): "
        f"{_format_legal_option(action_id, selected_action, obs, int(selected_aid) if selected_aid is not None else action_id)}"
    )
    lines.append("")
    return "\n".join(lines)


def _action_for_id(
    action_ids: list[int], legal_actions: list[Any], action_id: int
) -> Any:
    for aid, action in zip(action_ids, legal_actions):
        if int(aid) == int(action_id):
            return action
    raise ValueError(f"action_id {action_id} not among legal action ids")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo", required=True, help="PPO checkpoint path (model_a).")
    parser.add_argument("--baseline", required=True, help="SFT baseline checkpoint path (model_b).")
    parser.add_argument(
        "--output",
        default="checkpoints/train_riichi_v11_ppo_selected/best_hanchan_trace_seed20260730.md",
        help="Markdown trace file to write.",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ppo-seats",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Seats occupied by the PPO checkpoint.",
    )
    args = parser.parse_args()

    try:
        import riichi
        from riichienv import BatchedRiichiEnv
    except ImportError as exc:
        raise RuntimeError(
            "install the local riichi and RiichiEnv extensions before evaluation"
        ) from exc

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    ppo_path = str(Path(args.ppo).resolve())
    baseline_path = str(Path(args.baseline).resolve())
    ppo = _load_model(ppo_path, device)
    baseline = _load_model(baseline_path, device)
    use_bf16 = _bf16_supported(device)
    ppo_seats = set(int(seat) for seat in args.ppo_seats)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 1
    envs = BatchedRiichiEnv(
        batch_size,
        seed=int(args.seed),
        step_threads=batch_size,
        game_mode=args.game_mode,
    )
    bridge = BatchedStateBridge(
        riichi.MjaiKyokuStateMachineManager(batch_size),
        batch_size,
    )
    observations = list(envs.reset())
    bridge.sync(observations)
    public = PublicStateTracker(batch_size)
    public.update(bridge.last_events)
    analyzer = EfficiencyAnalyzer(131_072)

    active_envs = {0}
    step_index = 0
    decision_count = 0
    started = time.perf_counter()

    print(f"[record] ppo={ppo_path}", flush=True)
    print(f"[record] baseline={baseline_path}", flush=True)
    print(f"[record] ppo_seats={sorted(ppo_seats)} seed={args.seed}", flush=True)
    print(f"[record] writing to {output_path}", flush=True)

    sections: list[str] = []

    for step_index in range(args.max_steps):
        actions_by_env: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        decisions = active_decisions(observations, active_envs)
        analysis = (
            DecisionAnalysisBatch.build(
                decisions,
                analyzer=analyzer,
                public=public,
            )
            if decisions
            else None
        )

        for policy_name, model, model_device, use_bf16 in (
            ("ppo", ppo, device, use_bf16),
            ("baseline", baseline, device, use_bf16),
        ):
            policy_decisions = [
                decision
                for decision in decisions
                if (decision.seat_id in ppo_seats) == (policy_name == "ppo")
            ]
            if not policy_decisions:
                continue
            (
                factors,
                numeric,
                lengths,
                legal,
                _generations,
                _critic,
                _critic_lengths,
            ) = bridge.prepare(policy_decisions, analysis)
            with torch.autocast(
                device_type=model_device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model.forward_policy(
                    _tensor(factors, model_device),
                    _tensor(numeric, model_device),
                    _tensor(legal, model_device),
                    _tensor(lengths, model_device),
                )
            logits = output["policy_logits"]
            action_ids = logits.argmax(-1).tolist()
            mjai_actions = bridge.state_machine.decode_actions(
                [decision.batch_index for decision in policy_decisions],
                [int(action_id) for action_id in action_ids],
            )
            # Softmax over the masked legal logits to obtain the policy
            # probabilities shown in the trace.  ``-inf`` entries are masked
            # out automatically by the normalization.
            logits_t = output["policy_logits"].to(dtype=torch.float32)
            probs = torch.softmax(logits_t, dim=-1).tolist()
            # legal is a numpy boolean mask of shape [batch, NUM_ACTIONS].
            legal_mask = np.asarray(legal, dtype=bool)
            legal_action_ids = [np.nonzero(row)[0].tolist() for row in legal_mask]
            for decision, action_id, mjai_str, prob_row, legal_ids in zip(
                policy_decisions,
                action_ids,
                mjai_actions,
                probs,
                legal_action_ids,
                strict=True,
            ):
                action = decision.observation.select_action_from_mjai(mjai_str)
                if action is None:
                    raise RuntimeError(
                        f"MJAI action was rejected: seat={decision.seat_id} "
                        f"action_id={action_id} mjai={mjai_str}"
                    )
                actions_by_env[decision.env_index][decision.seat_id] = action

                if policy_name == "ppo":
                    decision_count += 1
                    legal_actions = decision.observation.legal_actions()
                    # Compute the canonical 241-way action id for every legal
                    # action (matching the policy head's action space).
                    paired_ids = [
                        compute_action_id(action, decision.observation)
                        for action in legal_actions
                    ]
                    # Two legal discards of the same PAI collapse to one
                    # policy-head action id; keep only one representative so
                    # the trace does not list duplicate ids.
                    seen: set[int] = set()
                    deduped: list[tuple[int | None, Any]] = []
                    for aid, action in zip(paired_ids, legal_actions):
                        key = aid if aid is not None else id(action)
                        if key in seen:
                            continue
                        seen.add(key)
                        deduped.append((aid, action))
                    ordered = sorted(
                        deduped,
                        key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0),
                    )
                    ordered_ids = [aid for aid, _ in ordered]
                    ordered_actions = [action for _, action in ordered]
                    full_probs = [
                        float(prob_row[aid]) if aid is not None else 0.0
                        for aid in ordered_ids
                    ]
                    section = render_decision(
                        decision,
                        action_id=int(action_id),
                        chosen_mjai=mjai_str,
                        probs=full_probs,
                        legal_actions=ordered_actions,
                        action_ids=ordered_ids,
                        index=decision_count,
                    )
                    sections.append(section)
                    print(
                        f"[record] step={step_index} seat={decision.seat_id} "
                        f"action_id={action_id} mjai={mjai_str}",
                        flush=True,
                    )

        observations = list(envs.step_batch(actions_by_env))
        bridge.sync(observations)
        public.update(bridge.last_events)
        done = envs.done()
        scores_by_env = envs.scores()
        if bool(done[0]):
            break
    else:
        raise RuntimeError(f"hanchan exceeded {args.max_steps} steps")

    scores = [int(value) for value in scores_by_env[0]]
    team_ppo = sum(scores[seat] for seat in ppo_seats)
    team_baseline = sum(scores[seat] for seat in range(NUM_PLAYERS) if seat not in ppo_seats)
    elapsed = time.perf_counter() - started

    header_lines = [
        "# Checkpoint decision trace",
        "",
        f"Checkpoint: `{args.ppo}`  ",
        f"Baseline: `{args.baseline}`  ",
        f"Seed: `{args.seed}`  ",
        f"PPO seats: `{sorted(ppo_seats)}`  "
        f"(baseline occupies the remaining seats, in seat order).",
        "",
    ]
    footer_lines = [
        "## Hanchan summary",
        "",
        f"- Steps: {step_index + 1}",
        f"- Final scores (seats 0-3): {scores}",
        f"- PPO team score = {team_ppo}",
        f"- Baseline team score = {team_baseline}",
        f"- Point diff (PPO - baseline) = {team_ppo - team_baseline:+d}",
        f"- PPO decisions recorded = {decision_count}",
        f"- Elapsed: {elapsed:.1f}s",
        "",
    ]
    output_path.write_text(
        "\n".join(header_lines) + "\n".join(sections) + "\n" + "\n".join(footer_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 80, flush=True)
    print(f"[record] hanchan finished in {step_index + 1} steps", flush=True)
    print(f"[record] final scores (seats 0-3): {scores}", flush=True)
    print(f"[record] PPO team score = {team_ppo}", flush=True)
    print(f"[record] baseline team score = {team_baseline}", flush=True)
    print(f"[record] point_diff (PPO - baseline) = {team_ppo - team_baseline:+d}", flush=True)
    print(f"[record] PPO decisions recorded = {decision_count}", flush=True)
    print(f"[record] trace written to {output_path}", flush=True)
    print(f"[record] elapsed {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
