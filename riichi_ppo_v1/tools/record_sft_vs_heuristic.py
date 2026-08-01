"""Play one hanchan per parallel env of an SFT model vs three heuristic opponents
and record every model decision in a Markdown trace.

The model (``--model``) occupies a fixed candidate seat per hanchan (default
Seat 0).  The three remaining seats are filled by a rotating schedule of the
efficiency / defensive heuristics implemented in
``riichi_ppo_v1.training.opponents.heuristic`` — the same opponent set used
by ``riichi_ppo_v1.sft.heuristic_evaluation``.

For every step on which the model is asked to decide, the script appends a
Markdown section rendering the round state, the model's hand + drawn tile,
the river/meld table for all four seats, every legal action with the policy
softmax probability, and the selected action — mirroring the format of the
existing ``checkpoints/train_riichi_v9/best_score_hanchan_trace_seed3.md``
trace.

Usage (from the workspace root):

    CUDA_DEVICE=3 conda run -n Mahjong-AI python \\
        riichi_ppo_v1/tools/record_sft_vs_heuristic.py \\
        --model checkpoints/train_riichi_v11_sft_40pct/best_heuristic.pt \\
        --hanchans 2 \\
        --output checkpoints/train_riichi_v11_sft_40pct/heuristic_hanchan_trace.md
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
    tile_id_to_mjai,
)
from riichi_ppo_v1.sft.evaluation_cases import DEFENSE, EFFICIENCY
from riichi_ppo_v1.sft.head_to_head import _bf16_supported, _load_model, _tensor
from riichi_ppo_v1.training.opponents.heuristic import HeuristicPolicy
from riichi_ppo_v1.training.rewards import (
    DecisionAnalysisBatch,
    EfficiencyAnalyzer,
    PublicStateTracker,
)
from riichi_ppo_v1.training.rewards.decision import action_id as compute_action_id
from riichi_ppo_v1.training.worker import active_decisions


_WIND_CHARS = "ESWN"
_RECIPES = (
    (EFFICIENCY, DEFENSE, EFFICIENCY),
    (DEFENSE, EFFICIENCY, DEFENSE),
)


def _round_label(round_wind: int, kyoku_index: int) -> str:
    wind = _WIND_CHARS[int(round_wind) % 4]
    return f"{wind}{int(kyoku_index) + 1}"


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
    mt = _meld_type_name(int(meld.meld_type))
    tiles = [tile_id_to_mjai(int(t)) for t in meld.tiles]
    called = meld.called_tile
    called_str = tile_id_to_mjai(int(called)) if called is not None else "-"
    opened = "opened" if bool(meld.opened) else "closed"
    from_who = meld.from_who
    from_label = _seat_label(int(from_who)) if int(from_who) >= 0 else "Seat -"
    return (
        f"{mt}({' '.join(tiles)}; called={called_str}; {opened}; from={from_label})"
    )


def _format_hand(observation: Any) -> str:
    raw = [int(t) for t in observation.hand]
    sorted_hand = sorted(raw, key=_tile_sort_key)
    drawn_int = getattr(observation, "drawn_tile", None)
    drawn = tile_id_to_mjai(int(drawn_int)) if drawn_int is not None else None
    rendered: list[str] = []
    marked = False
    for tile_str in reversed(
        [tile_id_to_mjai(t) for t in sorted(raw)]
    ):
        if not marked and drawn is not None and tile_str == drawn:
            rendered.append(f"{tile_str}>")
            marked = True
        else:
            rendered.append(tile_str)
    rendered.reverse()
    return "[" + " ".join(rendered) + "]"


_TILES_ORDER = {c: i for i, c in enumerate(
    list(f"{n}{s}" for s in "mps" for n in range(1, 10)) + ["5mr", "5pr", "5sr"]
    + ["E", "S", "W", "N", "P", "F", "C"]
)}


def _tile_sort_key(tile_id: int) -> int:
    mjai = tile_id_to_mjai(int(tile_id))
    return _TILES_ORDER.get(mjai, 999)


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


def _tsumogiri_for(action: Any, observation: Any, action_global_id: int | None) -> bool:
    if action_global_id is not None and 1 <= action_global_id <= 74:
        return (action_global_id - 1) % 2 == 1
    drawn = getattr(observation, "drawn_tile", None)
    return (
        action.tile is not None
        and drawn is not None
        and int(action.tile) == int(drawn)
        and action.action_type == 0
    )


def _format_legal_option(
    action: Any,
    observation: Any,
    action_global_id: int | None = None,
) -> str:
    tsumogiri = _tsumogiri_for(action, observation, action_global_id)
    mjai_str = action.to_mjai()
    mjai = json.loads(mjai_str)
    atype = str(mjai.get("type", ""))
    if atype == "dahai":
        marker = "tsumogiri" if tsumogiri else "tedashi"
        return f"dahai {mjai['pai']} ({marker})  {mjai_str}"
    if atype == "none":
        return f"none  {mjai_str}"
    if atype == "reach":
        return f"reach  {mjai_str}"
    if atype in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
        return f"{atype} {mjai.get('pai', '-')}  {mjai_str}"
    if atype in {"hora", "ron", "tsumo"}:
        return f"{atype}  {mjai_str}"
    if atype in {"ryukyoku", "kyushukyuhai", "kyushu_kyuhai"}:
        return f"{atype}  {mjai_str}"
    return f"{atype}  {mjai_str}"


def _format_table_state(observation: Any) -> list[str]:
    lines: list[str] = []
    for seat in range(NUM_PLAYERS):
        river = [tile_id_to_mjai(int(t)) for t in observation.discards[seat]]
        melds = observation.melds[seat]
        meld_str = ", ".join(_format_meld(m) for m in melds) if melds else "-"
        river_str = "[" + " ".join(river) + "]" if river else "[]"
        lines.append(
            f"- {_seat_label(seat)}: river={river_str} melds=[{meld_str}]"
        )
    return lines


def _is_selected(action: Any, action_global_id: int | None, chosen_aid: int) -> bool:
    if action_global_id is not None:
        return int(action_global_id) == int(chosen_aid)
    return False


def _action_for_id(
    action_ids: list[int], legal_actions: list[Any], action_id: int
) -> Any:
    for aid, action in zip(action_ids, legal_actions):
        if int(aid) == int(action_id):
            return action
    raise ValueError(f"action_id {action_id} not among legal action ids")


def render_decision(
    decision: Decision,
    *,
    hanchan_index: int,
    action_id: int,
    probs: list[float],
    legal_actions: list[Any],
    action_ids: list[int | None],
    index: int,
) -> str:
    obs = decision.observation
    seat = int(decision.seat_id)
    round_label = _round_label(int(obs.round_wind), int(obs.kyoku_index))
    dealer_seat = int(obs.oya)
    scores = _format_scores([int(s) for s in obs.scores])
    dora = _format_dora([int(t) for t in obs.dora_indicators])
    hand = _format_hand(obs)
    drawn = _drawn_tile_str(obs)
    table_lines = _format_table_state(obs)

    triples = sorted(
        zip(action_ids, legal_actions, probs),
        key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0),
    )

    lines: list[str] = []
    lines.append(f"## Hanchan {hanchan_index + 1} — Decision {index}: {_seat_label(seat)}")
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
        rendered = _format_legal_option(action, obs, aid_int)
        is_sel = _is_selected(action, aid_int, action_id)
        marker = " ← SELECTED" if is_sel else ""
        lines.append(f"- {aid_int}: {rendered}; p={float(prob) * 100.0:.4f}%{marker}")
    lines.append("")
    selected_action = _action_for_id(
        [a or 0 for a in action_ids], legal_actions, action_id
    )
    sel_aid = next(
        (aid for aid, action in zip(action_ids, legal_actions) if action is selected_action),
        action_id,
    )
    lines.append(
        f"Selected (action_id={action_id}): "
        f"{_format_legal_option(selected_action, obs, int(sel_aid) if sel_aid is not None else action_id)}"
    )
    lines.append("")
    return "\n".join(lines)


def _build_lineups(hanchan_count: int) -> list[list[str]]:
    """Return a per-hanchan lineup: ``("candidate", opp0, opp1, opp2)``.

    The candidate seat is always Seat 0; the seat-balanced schedule rotates the
    two opponent recipes so Seat 0 alternates between defensive-heavy and
    efficiency-heavy opposition over consecutive hanchans.
    """
    lineups: list[list[str]] = []
    for index in range(hanchan_count):
        recipe = _RECIPES[(index // NUM_PLAYERS) % len(_RECIPES)]
        # Seat 0 is the candidate.  Seats 1/2/3 take the three opponents in the
        # documented recipe order.
        lineups.append(["candidate", recipe[0], recipe[1], recipe[2]])
    return lineups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Path to the SFT checkpoint to play."
    )
    parser.add_argument(
        "--hanchans", type=int, default=2, help="Number of hanchans to play (defaults to 2)."
    )
    parser.add_argument(
        "--seed-base", type=int, default=20260730, help="Base seed for the env."
    )
    parser.add_argument(
        "--game-mode", default="4p-red-half", help="Game mode passed to BatchedRiichiEnv."
    )
    parser.add_argument(
        "--max-steps", type=int, default=4000, help="Per-hanchan step cap (default 4000)."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown file to write the trace to (defaults to stdout).",
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

    model_path = str(Path(args.model).resolve())
    model = _load_model(model_path, device)
    use_bf16 = _bf16_supported(device)

    hanchan_count = int(args.hanchans)
    lineups = _build_lineups(hanchan_count)
    candidate_seat = 0
    ppo_seats = {candidate_seat}

    output_path = Path(args.output).resolve() if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    print(f"[record] model={model_path}", flush=True)
    print(f"[record] hanchans={hanchan_count} seed_base={args.seed_base}", flush=True)
    print(f"[record] candidate_seat=Seat {candidate_seat}", flush=True)
    print(f"[record] opponents schedule: {lineups[0]}", flush=True)
    if output_path is not None:
        print(f"[record] writing to {output_path}", flush=True)

    for hanchan_index in range(hanchan_count):
        lineup = lineups[hanchan_index]
        batch_size = 1
        envs = BatchedRiichiEnv(
            batch_size,
            seed=int(args.seed_base) + hanchan_index,
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
        heuristics = {
            EFFICIENCY: HeuristicPolicy(analyzer, public, defensive=False),
            DEFENSE: HeuristicPolicy(analyzer, public, defensive=True),
        }

        active_envs = {0}
        decision_count = 0
        hanchan_start = time.perf_counter()

        for step_index in range(int(args.max_steps)):
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

            # 1. Heuristic opponents pick first.
            for policy_name, policy in heuristics.items():
                policy_decisions = [
                    decision
                    for decision in decisions
                    if lineup[decision.seat_id] == policy_name
                ]
                if not policy_decisions:
                    continue
                for decision, action in zip(
                    policy_decisions,
                    policy.select_batch(policy_decisions, analysis),
                    strict=True,
                ):
                    actions_by_env[decision.env_index][decision.seat_id] = action

            # 2. Model picks.
            model_decisions = [
                decision
                for decision in decisions
                if decision.seat_id in ppo_seats
            ]
            if model_decisions:
                (
                    factors,
                    numeric,
                    lengths,
                    legal,
                    _generations,
                    _critic,
                    _critic_lengths,
                ) = bridge.prepare(model_decisions, analysis)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    output = model.forward_policy(
                        _tensor(factors, device),
                        _tensor(numeric, device),
                        _tensor(legal, device),
                        _tensor(lengths, device),
                    )
                logits = output["policy_logits"].to(dtype=torch.float32)
                probs = torch.softmax(logits, dim=-1).tolist()
                action_ids_chosen = logits.argmax(-1).tolist()
                legal_mask = np.asarray(legal, dtype=bool)
                legal_action_ids_list = [
                    np.nonzero(row)[0].tolist() for row in legal_mask
                ]
                chosen_mjai = bridge.state_machine.decode_actions(
                    [d.batch_index for d in model_decisions],
                    [int(a) for a in action_ids_chosen],
                )
                for decision, action_id, mjai_str, prob_row, legal_ids in zip(
                    model_decisions,
                    action_ids_chosen,
                    chosen_mjai,
                    probs,
                    legal_action_ids_list,
                    strict=True,
                ):
                    action = decision.observation.select_action_from_mjai(mjai_str)
                    if action is None:
                        raise RuntimeError(
                            f"MJAI action was rejected: seat={decision.seat_id} "
                            f"action_id={action_id} mjai={mjai_str}"
                        )
                    actions_by_env[decision.env_index][decision.seat_id] = action

                    decision_count += 1
                    legal_actions = decision.observation.legal_actions()
                    paired_ids = [
                        compute_action_id(action, decision.observation)
                        for action in legal_actions
                    ]
                    seen: set[int] = set()
                    deduped: list[tuple[int | None, Any]] = []
                    for aid, act in zip(paired_ids, legal_actions):
                        key = aid if aid is not None else id(act)
                        if key in seen:
                            continue
                        seen.add(key)
                        deduped.append((aid, act))
                    ordered = sorted(
                        deduped,
                        key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0),
                    )
                    ordered_ids = [aid for aid, _ in ordered]
                    ordered_actions = [act for _, act in ordered]
                    full_probs = [
                        float(prob_row[aid]) if aid is not None else 0.0
                        for aid in ordered_ids
                    ]
                    section = render_decision(
                        decision,
                        hanchan_index=hanchan_index,
                        action_id=int(action_id),
                        probs=full_probs,
                        legal_actions=ordered_actions,
                        action_ids=ordered_ids,
                        index=decision_count,
                    )
                    sections.append(section)
                    print(
                        f"[record] hanchan={hanchan_index + 1} step={step_index} "
                        f"seat={decision.seat_id} action_id={action_id} "
                        f"mjai={mjai_str}",
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
            raise RuntimeError(f"hanchan {hanchan_index + 1} exceeded {args.max_steps} steps")

        scores = [int(value) for value in scores_by_env[0]]
        elapsed_h = time.perf_counter() - hanchan_start
        summary_rows.append({
            "hanchan": hanchan_index + 1,
            "steps": step_index + 1,
            "scores": scores,
            "decisions": decision_count,
            "elapsed_s": round(elapsed_h, 2),
        })
        print(
            f"[record] hanchan {hanchan_index + 1} done: scores={scores} "
            f"decisions={decision_count} elapsed={elapsed_h:.1f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - started

    header_lines = [
        "# SFT model vs three heuristics — decision trace",
        "",
        f"Model: `{args.model}`  ",
        f"Hanchans: `{hanchan_count}`  ",
        f"Seed base: `{args.seed_base}`  ",
        f"Candidate seat: `{_seat_label(candidate_seat)}`  ",
        f"Opponents (seats 1/2/3): {lineups[0][1:]}",
        "",
        "## Hanchan summary",
        "",
        "| hanchan | steps | final scores (seats 0-3) | model decisions | elapsed (s) |",
        "|---|---|---|---|---|",
    ]
    for row in summary_rows:
        header_lines.append(
            f"| {row['hanchan']} | {row['steps']} | {row['scores']} | "
            f"{row['decisions']} | {row['elapsed_s']} |"
        )
    header_lines.append("")
    footer_lines = [
        "",
        "## Trace",
        "",
    ]

    body = (
        "\n".join(header_lines)
        + "\n".join(sections)
        + "\n".join(footer_lines)
    )

    if output_path is not None:
        output_path.write_text(body, encoding="utf-8")
        print(f"[record] trace written to {output_path}", flush=True)
    else:
        print(body)

    print("\n" + "=" * 80, flush=True)
    print(f"[record] total decisions recorded = {sum(r['decisions'] for r in summary_rows)}", flush=True)
    print(f"[record] total elapsed {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
