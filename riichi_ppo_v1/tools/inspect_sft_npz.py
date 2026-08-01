"""Inspect SFT training samples from an encoded .npz shard.

The encoded ``riichi-sft-encoded-v1`` shards deliberately store only the
model-visible actor input (public history + actor state + public summary +
deterministic candidate tokens) plus the 241-way ``action`` id selected by
the expert.  The expert's private hand is *not* materialized into the shard,
so this script reports, for each inspected sample:

- ``action_id`` and the canonical MJAI payload it decodes to (e.g.
  ``dahai 3s (tedashi)`` or ``none``);
- every legal action id that was available at this decision (extracted from
  the candidate-token segment whose factor[2] == aid + 1) together with its
  MJAI spelling — this list is the closest available proxy for "what the
  expert could have done" inside the stored sample.

Usage (from the workspace root):

    conda run -n Mahjong-AI python \\
        riichi_ppo_v1/tools/inspect_sft_npz.py \\
        --shard datasets/tenhou_sft_2024_2025_encoded_40pct_v4_batched/train/train-00000-000-r0.npz \\
        --n 100 \\
        --output checkpoints/inspect_sft_100.md

``--n`` defaults to 100.  With ``--output`` the report is written as
Markdown, otherwise it is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make sure the project package is importable when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from riichi_ppo_v1.model.validation import all_action_templates


CANDIDATE_SEGMENT = 7


def _templates_by_id() -> list[dict[str, Any]]:
    templates = all_action_templates()
    if len(templates) != 241:
        raise RuntimeError(f"expected 241 action templates, got {len(templates)}")
    return templates


def _render_template(template: dict[str, Any]) -> str:
    """Render a fixed action template in the MJAI style used by the project."""
    kind = str(template.get("type", ""))
    if kind == "dahai":
        marker = "tsumogiri" if bool(template.get("tsumogiri")) else "tedashi"
        return f"dahai {template['pai']} ({marker})"
    if kind == "none":
        return "none"
    if kind == "reach":
        return "reach"
    if kind in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
        pai = template.get("pai", "-")
        consumed = template.get("consumed", [])
        return f"{kind} {pai} (consumed={consumed})"
    if kind in {"hora", "ron", "tsumo"}:
        return f"{kind}"
    if kind in {"ryukyoku", "kyushukyuhai", "kyushu_kyuhai"}:
        return f"{kind}"
    return json.dumps(template, sort_keys=True)


def _legal_action_ids(factors: np.ndarray) -> list[int]:
    """Extract the legal action ids from the candidate-token segment.

    Each candidate token carries ``factor[0] == CANDIDATE_SEGMENT`` and
    ``factor[2] == min(aid + 1, 255)``.  The +1 offset encodes id 0
    distinguishably and 241 ids fit inside the uint8 capacity without
    saturation, so the raw value uniquely identifies one legal action id.
    """
    rows = factors[factors[:, 0] == CANDIDATE_SEGMENT]
    ids: list[int] = []
    for row in rows:
        offset = int(row[2])
        if offset <= 0:
            continue
        aid = offset - 1
        if 0 <= aid < 241:
            ids.append(aid)
    return ids


def render_sample(
    index: int,
    *,
    action_id: int,
    legal_ids: list[int],
    templates: list[dict[str, Any]],
    hand_note: str,
) -> str:
    chosen_template = templates[int(action_id)] if 0 <= int(action_id) < 241 else {"type": "unknown"}
    chosen_render = _render_template(chosen_template)
    lines: list[str] = []
    lines.append(f"## Sample {index + 1}")
    lines.append("")
    lines.append(f"- Expert action: action_id={int(action_id)} → `{chosen_render}`")
    lines.append(f"- {hand_note}")
    lines.append(f"- Legal action ids at this decision: {len(legal_ids)}")
    lines.append("")
    lines.append("  | id | action |")
    lines.append("  |---|---|")
    for aid in sorted(legal_ids):
        template = templates[aid]
        marker = " ← expert" if int(aid) == int(action_id) else ""
        lines.append(f"  | {aid} | {_render_template(template)}{marker} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        required=True,
        help="Path to the riichi-sft-encoded-v1 .npz shard to inspect.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of samples to inspect (defaults to 100).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional Markdown file to write the report to.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of leading samples to skip before inspecting.",
    )
    args = parser.parse_args()

    shard_path = Path(args.shard).resolve()
    templates = _templates_by_id()
    with np.load(shard_path, allow_pickle=False) as data:
        offsets = data["offsets"]
        factors_all = data["factors"]
        actions = data["actions"]
    total = int(actions.shape[0])
    start = max(0, int(args.skip))
    end = min(total, start + max(0, int(args.n)))
    if end <= start:
        raise RuntimeError(
            f"no samples available in [{start}, {end}); shard has {total} samples"
        )

    header_lines = [
        "# SFT shard inspection",
        "",
        f"Shard: `{shard_path}`  ",
        f"Samples in shard: `{total}`  ",
        f"Inspected range: samples `[{start}, {end})` ({end - start} samples)",
        "",
        "## Note on hand reconstruction",
        "",
        "The encoded shards are `actor_only: true`.  Expert private hands are",
        "not materialized: the stored factors contain only public history, the",
        "actor's state, a public river/meld summary, and one deterministic",
        "candidate token per legal action.  The legal-action list below is the",
        "best available proxy for what the expert could have discarded.",
        "",
    ]

    sections: list[str] = []
    for i in range(start, end):
        row_start = int(offsets[i])
        row_end = int(offsets[i + 1])
        factors = factors_all[row_start:row_end]
        action_id = int(actions[i])
        legal_ids = _legal_action_ids(factors)
        # Categorize legal actions to surface the discard candidates.
        discard_pais: list[str] = []
        for aid in legal_ids:
            template = templates[aid]
            if str(template.get("type")) == "dahai":
                marker = "(tsumogiri)" if bool(template.get("tsumogiri")) else "(tedashi)"
                discard_pais.append(f"{template['pai']} {marker}")
        hand_note = (
            f"Discard candidates recoverable from tokens ({len(discard_pais)}): "
            + (", ".join(sorted(discard_pais)) if discard_pais else "(none)")
        )
        sections.append(render_sample(
            i,
            action_id=action_id,
            legal_ids=legal_ids,
            templates=templates,
            hand_note=hand_note,
        ))

    body = "\n".join(header_lines) + "\n" + "\n".join(sections)
    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
        print(
            f"[inspect] wrote {end - start} samples to {out_path}",
            flush=True,
        )
    else:
        print(body)
        print(f"\n[inspect] inspected {end - start} samples", flush=True)


if __name__ == "__main__":
    main()
