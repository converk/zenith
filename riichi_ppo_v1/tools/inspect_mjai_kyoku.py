"""Inspect SFT training data by replaying raw mjai kyokus.

The encoded ``_encoded_*.npz`` shards are actor-only and materialize neither
the expert's private hand nor the original event stream, so they cannot be
used to audit "what hand did the expert hold?" or "what discard actually
happened at the table?".  This script instead walks the source mjai tar
shards under ``datasets/tenhou_sft_2024_2025/{train,validation}/`` and replays
each kyoku by hand, reporting — for every decision the expert made — the
**hand**, **drawn tile**, and the **selected discard/action**, alongside the
public river and any open melds.  The output format matches the existing
decision-trace style so the result is easy to read.

Usage (from the workspace root):

    conda run -n Mahjong-AI python \\
        riichi_ppo_v1/tools/inspect_mjai_kyoku.py \\
        --shard datasets/tenhou_sft_2024_2025/train/train-00000.tar \\
        --n 100 \\
        --output checkpoints/inspect_mjai_100.md

``--n`` defaults to 100 decisions.  Because each tar shard holds 4096 kyokus
(≈1.5M decisions), the tool by default inspects decisions across the first
``--kyokus`` kyokus (default 50) until ``--n`` decisions are collected; use
``--kyokus 1`` to inspect only the very first kyoku in the shard.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import sys
import tarfile
from typing import Any

# Make sure the project package is importable when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


HONORS = {"E": 108, "S": 112, "W": 116, "N": 120, "P": 124, "F": 128, "C": 132}


def tile_to_str(pai: str) -> str:
    """Normalize MJAI pai spelling (``5mr`` stays red; ``0m`` normalized to ``5mr``)."""
    return pai


def seat_label(seat: int) -> str:
    return f"Seat {int(seat)}"


def render_discard_river(river: list[dict[str, Any]]) -> str:
    """Render one seat's river as a sequence of tile spellings."""
    tiles: list[str] = []
    for item in river:
        if item.get("type") == "dora":
            continue
        pai = item.get("pai")
        if pai is None:
            continue
        if item.get("tsumogiri"):
            tiles.append(f"{pai}>")
        elif item.get("riichi"):
            tiles.append(f"{pai}*")
        else:
            tiles.append(str(pai))
    return "[" + " ".join(tiles) + "]"


def render_meld(meld_event: dict[str, Any]) -> str:
    kind = str(meld_event.get("type", ""))
    pai = meld_event.get("pai", "-")
    consumed = meld_event.get("consumed", [])
    from_seat = meld_event.get("target")
    if from_seat is not None:
        from_label = seat_label(int(from_seat))
    else:
        from_label = "-"
    tiles = " ".join(str(t) for t in (consumed or [pai]))
    name = {
        "chi": "meldtype.chi",
        "pon": "meldtype.pon",
        "daiminkan": "meldtype.daiminkan",
        "ankan": "meldtype.ankan",
        "kakan": "meldtype.kakan",
        "kakan_extracted": "meldtype.kakan",
    }.get(kind, f"meldtype.{kind}")
    return f"{name}({tiles}; called={pai}; from={from_label})"


def render_hand(hand: list[str], drawn: str | None) -> str:
    """Render a sorted hand.  The drawn tile is already part of ``hand`` (the
    tsumo event appended it), so it is only marked with a trailing ``>`` to
    highlight which tile was just drawn — it is *not* appended a second time.
    """
    sorted_hand = sorted(hand, key=_tile_sort_key)
    rendered: list[str] = []
    # Mark at most one tile as the drawn tile; if several physical tiles share
    # the same spelling, picking the last one matches the natural "newly drawn"
    # position at the end of the hand before sorting.
    marked = False
    for tile in reversed(sorted_hand):
        if not marked and drawn is not None and tile == drawn:
            rendered.append(f"{tile}>")
            marked = True
        else:
            rendered.append(str(tile))
    rendered.reverse()
    return "[" + " ".join(rendered) + "]"


def render_action(action: dict[str, Any], drawn_tile: str | None) -> str:
    kind = str(action.get("type", ""))
    pai = action.get("pai")
    if kind == "dahai":
        tsumogiri = bool(action.get("tsumogiri", False))
        # Riichi-flagged discards carry ``riichi`` boolean.
        riichi = bool(action.get("riichi", False))
        marker = "tsumogiri" if tsumogiri else "tedashi"
        suffix = " (reach)" if riichi else ""
        return f"dahai {pai} ({marker}){suffix}"
    if kind in {"pon", "chi", "daiminkan", "kakan", "ankan"}:
        consumed = action.get("consumed", [])
        return f"{kind} {pai} (consumed={consumed})"
    if kind in {"hora", "ron", "tsumo"}:
        return f"{kind} {pai}"
    if kind == "reach":
        return "reach"
    if kind == "none":
        return "none"
    if kind in {"ryukyoku", "kyushukyuhai", "kyushu_kyuhai"}:
        return kind
    return json.dumps(action, sort_keys=True, ensure_ascii=False)


class KyokuState:
    """A hand-rolled mjai parser that tracks hands, rivers, melds per seat.

    This parser deliberately understands only the subset of MJAI needed for
    an inspection report: tsumo, dahai, chi/pon/kan/open-kan, reach, hora,
    ryukyoku, and the start/end events.  Anything else is recorded as raw
    event metadata without changing hand state, which is sufficient for
    describing the expert's decisions.
    """

    def __init__(self, start_event: dict[str, Any]) -> None:
        self.bakaze = str(start_event.get("bakaze", "E"))
        self.kyoku = int(start_event.get("kyoku", 1)) - 1
        self.honba = int(start_event.get("honba", 0))
        self.kyotaku = int(start_event.get("kyotaku", 0))
        self.oya = int(start_event.get("oya", 0))
        self.scores = [int(s) for s in start_event.get("scores", [25000] * 4)]
        dora_raw = start_event.get("dora_marker", [])
        if isinstance(dora_raw, str):
            self.dora_indicators: list[str] = [dora_raw]
        else:
            self.dora_indicators = [str(t) for t in dora_raw if t is not None]
        tehais = start_event.get("tehais", [[], [], [], []])
        self.hands: list[list[str]] = [[str(t) for t in hand] for hand in tehais]
        self.rivers: list[list[dict[str, Any]]] = [[], [], [], []]
        self.melds: list[list[dict[str, Any]]] = [[], [], [], []]
        self.riichi_declared = [False, False, False, False]
        self.drawn_tile: str | None = None
        self.last_discard: tuple[int, str] | None = None
        self.turn_count = 0
        self.events: list[dict[str, Any]] = []

    @property
    def round_label(self) -> str:
        wind = {"E": "E", "S": "S", "W": "W", "N": "N"}.get(self.bakaze, self.bakaze)
        return f"{wind}{self.kyoku + 1}"

    def owns_tile(self, seat: int, pai: str) -> bool:
        return pai in self.hands[seat]

    def remove_tile_from_hand(self, seat: int, pai: str, *, allow_red: bool = True) -> None:
        # Prefer a non-red copy first when the request is an ordinary 5.
        candidates = [tile for tile in self.hands[seat] if tile == pai or (pai[0] == "5" and tile == pai + "r")]
        if not candidates:
            target = pai
            if target not in self.hands[seat]:
                return
        else:
            target = candidates[0]
        self.hands[seat].remove(target)

    def apply(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        kind = str(event.get("type", ""))
        actor = int(event.get("actor", -1)) if "actor" in event else None
        if kind == "tsumo" and actor is not None:
            pai = str(event["pai"])
            self.hands[actor].append(pai)
            self.drawn_tile = pai
            self.last_discard = None
            self.turn_count += 1
            return
        if kind == "dahai" and actor is not None:
            pai = str(event["pai"])
            self.remove_tile_from_hand(actor, pai)
            self.rivers[actor].append(event)
            self.drawn_tile = None
            self.last_discard = (actor, pai)
            return
        if kind in {"chi", "pon", "kakan", "daiminkan"} and actor is not None:
            pai = str(event.get("pai", ""))
            consumed = list(event.get("consumed", []))
            target = event.get("target")
            # Two MJAI payload conventions exist for the called tile:
            #   * tenhou→mjai: ``consumed`` lists only the tiles taken from the
            #     actor's hand; the called tile lives in ``pai`` (so for
            #     ``pon W consumed=['W','W']`` both entries come from the hand).
            #   * offline replay: ``consumed`` may include the called tile as an
            #     extra entry (e.g. ``chi 3p consumed=['2p','3p','4p']``).
            # ``consumed_tiles`` in riichi_ppo_v1 only strips the extra copy when
            # its length is ``expected + 1``.  Mirror that exact rule here so
            # hand state stays consistent.
            expected = {"chi": 2, "pon": 2, "daiminkan": 3}.get(kind, 0)
            if expected and len(consumed) == expected + 1:
                consumed.remove(pai)
            for tile in consumed:
                self.remove_tile_from_hand(actor, str(tile))
            # Remove the called tile from the target's river.
            if target is not None and pai is not None:
                for item in list(self.rivers[int(target)]):
                    if item.get("pai") == pai and item.get("type") == "dahai":
                        self.rivers[int(target)].remove(item)
                        break
            # Add the meld to the actor's meld list.
            self.melds[actor].append(event)
            self.drawn_tile = None
            return
        if kind == "ankan" and actor is not None:
            pai = str(event.get("pai", ""))
            consumed = list(event.get("consumed", [pai, pai, pai, pai]))
            for tile in consumed:
                self.remove_tile_from_hand(actor, str(tile))
            self.melds[actor].append(event)
            self.drawn_tile = None
            return
        if kind == "reach" and actor is not None:
            self.riichi_declared[actor] = True
            return
        if kind in {"hora", "ryukyoku"}:
            return
        # Unknown / structural event (start_game, end_kyoku, ...) — ignored.

    def snapshot_seat_hand(self, seat: int) -> list[str]:
        """Return the current raw hand of a seat, sorted in MJAI order."""
        return sorted(self.hands[seat], key=_tile_sort_key)


_TILES_ORDER = {c: i for i, c in enumerate(
    list(f"{n}{s}" for s in "mps" for n in range(1, 10)) + ["5mr", "5pr", "5sr"]
    + ["E", "S", "W", "N", "P", "F", "C"]
)}


def _tile_sort_key(tile: str) -> int:
    return _TILES_ORDER.get(tile, 999)


def extract_decisions(content: str) -> list[dict[str, Any]]:
    """Parse one mjai kyoku JSONL string into a per-decision view.

    Each returned dict contains:
      ``state``: the ``KyokuState`` immediately before the decision;
      ``event``: the decision event itself (dahai / chi / pon / kan / reach /
                 hora / ryukyoku / none / tsumo-agari);
      ``seat``: the acting seat.
    """
    decisions: list[dict[str, Any]] = []
    state = None
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = str(event.get("type", ""))
        if kind == "start_kyoku":
            state = KyokuState(event)
            continue
        if state is None:
            continue
        if kind in {"tsumo"}:
            state.apply(event)
            continue
        if kind in {"dahai", "chi", "pon", "kakan", "ankan", "daiminkan", "reach", "hora", "ryukyoku"}:
            actor = int(event.get("actor", -1)) if "actor" in event else -1
            if actor >= 0:
                # Snapshot the table state at the moment of the decision so
                # later events do not rewrite this decision's view (the state
                # object is shared across the whole kyoku, so the report must
                # retain its own independent copy of hands/rivers/melds).
                decisions.append({"state": copy.deepcopy(state), "event": copy.deepcopy(event), "seat": actor})
            state.apply(event)
            continue
        state.apply(event)
    return decisions


def render_decision(
    *,
    index: int,
    state: KyokuState,
    event: dict[str, Any],
    seat: int,
    kyoku_name: str,
) -> str:
    hand = state.snapshot_seat_hand(seat)
    drawn = state.drawn_tile if state.drawn_tile is not None else None
    river_rows = [render_discard_river(state.rivers[s]) for s in range(4)]
    meld_rows: list[str] = []
    for s in range(4):
        melds = state.melds[s]
        if melds:
            meld_rows.append(f"{seat_label(s)}: melds=[{', '.join(render_meld(m) for m in melds)}]")
        else:
            meld_rows.append(f"{seat_label(s)}: melds=[-]")
    lines: list[str] = []
    lines.append(f"## Decision {index}: {seat_label(seat)} (kyoku {kyoku_name})")
    lines.append("")
    lines.append(
        f"Round: {state.round_label}; dealer={seat_label(state.oya)}; "
        f"honba={state.honba}; riichi_sticks={state.kyotaku}; "
        f"scores=[{', '.join(str(s) for s in state.scores)}]; "
        f"dora=[{', '.join(state.dora_indicators)}]"
    )
    lines.append("")
    lines.append(f"Hand: {render_hand(hand, drawn)}")
    lines.append("")
    lines.append(f"Drawn tile: {drawn if drawn is not None else '-'}")
    lines.append("")
    lines.append("Visible table:")
    lines.append("")
    for s in range(4):
        lines.append(f"- {seat_label(s)}: river={river_rows[s]} melds=[{((', '.join(render_meld(m) for m in state.melds[s])) if state.melds[s] else '-')}]")
    lines.append("")
    lines.append(f"Expert action: {render_action(event, drawn)}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        required=True,
        help="Path to the source mjai kyoku tar shard (e.g. .../train/train-00000.tar).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of decisions to report (defaults to 100).",
    )
    parser.add_argument(
        "--kyokus",
        type=int,
        default=40,
        help="Maximum number of kyokus to scan from the shard (defaults to 40).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional Markdown file to write the report to.",
    )
    parser.add_argument(
        "--skip-kyokus",
        type=int,
        default=0,
        help="Number of kyokus to skip before inspecting.",
    )
    args = parser.parse_args()

    shard_path = Path(args.shard).resolve()
    target_decisions = int(args.n)
    max_kyokus = int(args.kyokus)
    skip_kyokus = int(args.skip_kyokus)

    collected: list[str] = []
    seen_kyokus = 0
    scanned_kyokus = 0
    with tarfile.open(shard_path, "r") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        for member in members:
            if scanned_kyokus < skip_kyokus:
                scanned_kyokus += 1
                continue
            if seen_kyokus >= max_kyokus:
                break
            seen_kyokus += 1
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            content = (
                gzip.decompress(payload).decode("utf-8")
                if payload[:2] == b"\x1f\x8b"
                else payload.decode("utf-8")
            )
            decisions = extract_decisions(content)
            for idx, decision in enumerate(decisions):
                if len(collected) >= target_decisions:
                    break
                collected.append(
                    render_decision(
                        index=len(collected) + 1,
                        state=decision["state"],
                        event=decision["event"],
                        seat=decision["seat"],
                        kyoku_name=member.name,
                    )
                )
            if len(collected) >= target_decisions:
                break

    header_lines = [
        "# SFT mjai kyoku inspection",
        "",
        f"Shard: `{shard_path}`  ",
        f"Kyokus scanned: `{seen_kyokus}` (limit {max_kyokus}, skipped {skip_kyokus})  ",
        f"Decisions reported: `{len(collected)}` (limit {target_decisions})",
        "",
    ]

    if not collected:
        header_lines.extend([
            "",
            "No decisions were collected.  Inspect ``--kyokus`` or ``--skip-kyokus``.",
            "",
        ])

    body = "\n".join(header_lines) + "\n" + "\n".join(collected)
    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
        print(
            f"[inspect] wrote {len(collected)} decisions to {out_path}",
            flush=True,
        )
    else:
        print(body)
        print(f"\n[inspect] inspected {len(collected)} decisions", flush=True)


if __name__ == "__main__":
    main()
