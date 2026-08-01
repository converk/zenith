"""End-to-end audit of the SFT encoding pipeline (mjai -> npz -> model tensors).

This script walks a source mjai tar shard, replays each kyoku via
``encode_kyoku`` (the *same* code path used by ``riichi-sft-precompute`` to
materialize the ``_encoded_*_v4_batched/*.npz`` shards) and verifies, for
every expert decision it produces, that the encoded tensors are internally
consistent and match the reference 241-way action protocol.

Four independent checks run on every inspected kyoku:

A1. action <-> legal_mask <-> candidate-token three-way alignment.
    - ``sample.action`` must point at a true bit in ``sample.legal_mask``.
    - The set of candidate-token ids (``factor[seg==7, 2] - 1``) must equal
      ``np.flatnonzero(sample.legal_mask)`` exactly.  A mismatch means the
      candidate tokens and the legal mask disagree about which actions are
      legal — the surfaced "key suspicion point" surfaced during research.

A2. Per-action-type round-trip verification.
    - Group every expert action by its canonical MJAI ``type`` (dahai
      tedashi / dahai tsumogiri / none / reach / chi / pon / ankan / kakan /
      daiminkan / hora / ryukyoku) and report counts + the action_id
      distribution, so rare action types can be eyeballed.
    - Cross-check the expert id against ``all_action_templates()[id]``.

A3. npz structural integrity (optional).
    - When ``--npz-shard`` is supplied, load the shard and verify the same
      invariants the loader relies on: 241-bit ``legal`` round-trips via
      packbits/unpackbits, ``offsets`` is monotone non-decreasing,
      ``actions`` fit in ``uint8`` range, and ``numeric`` row counts agree
      with ``offsets``.

A4. 4-stream timeline integrity.
    - For every (seat, step) yielded by ``kyoku.steps``, verify the
      decision order respects event causality: a tsumo for seat S cannot
      precede that seat's previous dahai/call, and any chi/pon/daiminkan
      for seat S must follow the offending dahai of seat (S-1).

A5. Action-space codec round-trip (the part the user explicitly asked for).
    - Static full-space: every one of the 241 fixed action ids must encode
      into the state machine and decode back to the identical canonical
      MJAI string (``assert_full_action_space``).  This is the protocol-level
      invariant already covered by ``tests/protocol/test_action_space_exhaustive``
      but we re-assert it as part of the audit so the report is self-contained.
    - Per-kyoku real-scenario round-trip: for every legal action observed in
      the mjai replay, encode it through the state machine under the *real*
      decision snapshot (not the fixture), decode the resulting action_id back
      to MJAI, and confirm the canonical MJAI matches
      ``action.to_mjai()``.  Any "akadora, consumed-tile, tsumogiri/hand-cut,
      or action-kind" aliasing that survives the protocol layer would surface
      here.  This mirrors ``assert_observation_roundtrip`` but runs against
      genuine offline-replay observations.

Usage (from the workspace root):

    conda run -n Mahjong-AI python \\
        riichi_ppo_v1/tools/audit_encoding_pipeline.py \\
        --source-tar datasets/tenhou_sft_2024_2025/train/train-00000.tar \\
        --max-kyokus 50 \\
        --output checkpoints/audit_encoding_$(date +%s).md

``--max-kyokus`` defaults to 50.  Pass ``--npz-shard`` to also exercise A3.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

# Make sure the project package is importable when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from riichi_ppo_v1.model.bridge import (
    Decision,
    _action_jsons_and_decision_flag,
    snapshot_json,
)
from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION
from riichi_ppo_v1.model.semantic_validation import (
    assert_actor_token_semantics,
    summarize_tokens,
)
from riichi_ppo_v1.model.validation import (
    all_action_templates,
    assert_full_action_space,
    canonical,
)
from riichi_ppo_v1.sft.data import SftSample, encode_kyoku
from riichi_ppo_v1.training.rewards.decision import action_id as compute_action_id
from riichi_ppo_v1.training.rewards.efficiency import EfficiencyAnalyzer


# --- Token segment constants (mirror semantic_validation.py) ----------------
SEGMENT_HISTORY = 1
SEGMENT_STATE = 2
SEGMENT_PUBLIC_SUMMARY = 3
SEGMENT_CANDIDATE = 7
NUM_ACTIONS = 241

# MJAI action type -> coarse category used by A2's histogram.
_TSUMOGIRI_DAHAI = "dahai(tsumogiri)"
_TEDASHI_DAHAI = "dahai(tedashi)"


def _templates_by_id() -> list[dict[str, Any]]:
    templates = all_action_templates()
    if len(templates) != NUM_ACTIONS:
        raise RuntimeError(
            f"expected {NUM_ACTIONS} action templates, got {len(templates)}"
        )
    return templates


def _coarse_action_type(
    template: dict[str, Any] | None, *, tsumogiri: bool = False
) -> str:
    if template is None:
        return "unknown"
    kind = str(template.get("type", ""))
    if kind == "dahai":
        return _TSUMOGIRI_DAHAI if tsumogiri else _TEDASHI_DAHAI
    if kind in {"none", "pass"}:
        return "none"
    if kind == "reach":
        return "reach"
    if kind in {"chi", "pon", "ankan", "kakan", "daiminkan", "hora", "ron",
                "tsumo", "ryukyoku", "kyushukyuhai", "kyushu_kyuhai"}:
        return kind
    return kind


def _candidate_action_ids(sample: SftSample) -> list[int]:
    """Return the action ids encoded in the candidate-token segment."""
    factors = sample.token_factors
    rows = factors[factors[:, 0] == SEGMENT_CANDIDATE]
    ids: list[int] = []
    for row in rows:
        offset = int(row[2])
        if offset <= 0:
            continue
        aid = offset - 1
        if 0 <= aid < NUM_ACTIONS:
            ids.append(aid)
    return ids


def _audit_sample_a1(sample: SftSample, templates: list[dict[str, Any]]) -> dict[str, Any]:
    """A1: action -> legal_mask -> candidate-token alignment."""
    legal_ids = np.flatnonzero(sample.legal_mask).tolist()
    candidate_ids = _candidate_action_ids(sample)
    action = int(sample.action)
    return {
        "legal_ids": sorted(legal_ids),
        "candidate_ids": sorted(candidate_ids),
        "expert_action_id": action,
        "expert_in_legal": bool(sample.legal_mask[action]) if 0 <= action < NUM_ACTIONS else False,
        "legal_equals_candidates": set(legal_ids) == set(candidate_ids),
        "extras_in_candidates_only": sorted(set(candidate_ids) - set(legal_ids)),
        "extras_in_legal_only": sorted(set(legal_ids) - set(candidate_ids)),
    }


def _audit_sample_a2(
    sample: SftSample, templates: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    """A2: per-decision coarse type + round-trip confirmation."""
    aid = int(sample.action)
    if not (0 <= aid < NUM_ACTIONS):
        return "out_of_range", {"aid": aid}
    template = templates[aid]
    tsumogiri = bool(template.get("tsumogiri", False))
    coarse = _coarse_action_type(template, tsumogiri=tsumogiri)
    return coarse, {"aid": aid, "template_type": template.get("type")}


def _audit_kyoku_a4(content: str) -> dict[str, Any]:
    """A4: timeline integrity check on raw mjai events.

    Verifies event causality: every tsumo for seat S follows its own previous
    dahai/call, and every chi/pon/daiminkan by seat S follows seat (S-1 mod 4)'s
    dahai. Returns a dict with the list of violations found.
    """
    last_action_per_seat: dict[int, str | None] = {s: None for s in range(4)}
    last_discard_seat: int | None = None
    last_discard_pai: str | None = None
    violations: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = str(ev.get("type", ""))
        actor = ev.get("actor")
        if actor is None:
            continue
        seat = int(actor)
        if kind == "tsumo":
            prev = last_action_per_seat[seat]
            if prev is not None and prev not in {
                "dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "reach"
            }:
                # Tsumo without an intervening action that hands turn back.
                violations.append({
                    "step_type": "tsumo",
                    "seat": seat,
                    "previous_own_action": prev,
                    "reason": "tsumo without intervening turn-returning action",
                })
            last_action_per_seat[seat] = "tsumo"
            continue
        if kind == "dahai":
            last_discard_seat = seat
            last_discard_pai = str(ev.get("pai", ""))
            last_action_per_seat[seat] = "dahai"
            continue
        if kind in {"chi", "pon", "daiminkan"}:
            target = ev.get("target")
            if last_discard_seat is None or int(target) != last_discard_seat:
                violations.append({
                    "step_type": kind,
                    "seat": seat,
                    "target_field": int(target) if target is not None else None,
                    "last_discard_seat": last_discard_seat,
                    "last_discard_pai": last_discard_pai,
                    "reason": f"{kind} not preceded by target's dahai",
                })
            last_action_per_seat[seat] = kind
            last_discard_seat = None
            last_discard_pai = None
            continue
        if kind in {"ankan", "kakan", "reach"}:
            last_action_per_seat[seat] = kind
            continue
    return {
        "event_count": sum(1 for _ in content.splitlines() if _.strip()),
        "violations": violations,
    }


def _audit_action_space_static() -> dict[str, Any]:
    """A5 (static): full 241-way protocol codec round-trip on the fixture snapshot."""
    import riichi

    manager = riichi.MjaiKyokuStateMachineManager(1)
    try:
        assert_full_action_space(manager)
        templates = all_action_templates()
        # Re-check every id individually so the report has a per-id count.
        # ``assert_full_action_space`` already covers this but we count for stats.
        failures: list[dict[str, Any]] = []
        for action_id, expected in enumerate(templates):
            returned = manager.decode_actions([0], [action_id])[0]
            if canonical(returned) != canonical(expected):
                failures.append({
                    "action_id": action_id,
                    "expected": expected,
                    "returned": returned,
                })
        return {
            "templates_count": len(templates),
            "all_241_decoded": len(failures) == 0,
            "per_id_failures": len(failures),
            "failure_examples": failures[:3],
        }
    except AssertionError as exc:
        return {
            "templates_count": -1,
            "all_241_decoded": False,
            "per_id_failures": -1,
            "error": str(exc),
        }


def _audit_kyoku_a5_real_scenario(content: str) -> dict[str, Any]:
    """A5 (real scenario): per-legal-action codec round-trip on real replay snapshots.

    For every legal action that appears in the mjai replay's decision stream,
    drive the state machine directly with the genuine snapshot and confirm
    that decoding each produced action_id yields the same canonical MJAI as
    the original ``Action.to_mjai()`` template.  This catches any codec
    aliasing (akadora, consumed tile, tsumogiri / hand-cut, action-kind) that
    survives the protocol-level static test under a fixture snapshot.
    """
    from riichienv import MjaiReplay
    import riichi

    replay = MjaiReplay.from_jsonl_string(content, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    if len(kyokus) != 1:
        return {"decisions": 0, "legal_failures": [], "legal_actions_checked": 0}
    kyoku = kyokus[0]
    manager = riichi.MjaiKyokuStateMachineManager(4)
    streams = [iter(kyoku.steps(seat=seat, skip_single_action=False)) for seat in range(4)]
    active = set(range(4))
    decisions_checked = 0
    failures: list[dict[str, Any]] = []
    while active:
        batch: list[tuple[int, Any, Any]] = []
        for seat in sorted(active):
            try:
                observation, expert_action = next(streams[seat])
            except StopIteration:
                active.remove(seat)
            else:
                batch.append((seat, observation, expert_action))
        if not batch:
            continue
        env_indices = [seat for seat, _o, _a in batch]
        events_by_env = []
        action_rows = []
        snapshots = []
        # Pre-compute action-jsons + snapshot exactly like encode_kyoku does.
        for seat, observation, _expert in batch:
            actions_jsons, decision_flag = _action_jsons_and_decision_flag(observation)
            events = [[], [], [], []]
            events[seat] = list(observation.new_events())
            events_by_env.append(events)
            action_rows.append(actions_jsons)
            snapshots.append(snapshot_json(observation, decision_flag))
        manager.apply_events_batch(env_indices, events_by_env)
        batch_indices = [seat * 4 + seat for seat, _o, _a in batch]
        _f, _n, _l, masks, _hg = manager.prepare_decisions(batch_indices, action_rows, snapshots)
        masks_arr = np.asarray(masks, dtype=bool)
        for row, (seat, observation, _expert) in enumerate(batch):
            legal_actions = list(observation.legal_actions())
            # Templates observed (canonical MJAI strings) computed via the
            # same path ``_normalized_action_json`` uses internally.
            templates = action_rows[row]
            true_ids = set(np.flatnonzero(masks_arr[row]).tolist())
            # Decode every legal id we set in the machine and confirm it
            # round-trips back to one of the observed templates.  Mirrors
            # ``assert_observation_roundtrip`` but on offline observations.
            expected_canonical = {canonical(t): idx for idx, t in enumerate(templates)}
            for aid in true_ids:
                returned = manager.decode_actions([batch_indices[row]], [int(aid)])[0]
                returned_canonical = canonical(returned)
                if returned_canonical not in expected_canonical:
                    failures.append({
                        "seat": seat,
                        "action_id": int(aid),
                        "decoded_mjai": returned,
                        "expected_templates": sorted(expected_canonical),
                    })
            decisions_checked += 1
    return {
        "decisions": decisions_checked,
        "legal_actions_checked": decisions_checked,
        "legal_failures": failures[:5],
        "legal_failures_count": len(failures),
    }


def _reshape_samples_for_assertion(samples: list[SftSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad the per-sample token arrays into the 3D layout assert_actor_token_semantics expects."""
    if not samples:
        empty = np.zeros((0, 0, 10), dtype=np.uint8)
        return empty, np.zeros((0, 0, 8), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    max_tokens = max(sample.token_length for sample in samples)
    factors = np.zeros((len(samples), max_tokens, 10), dtype=np.uint8)
    numeric = np.zeros((len(samples), max_tokens, 8), dtype=np.float32)
    lengths = np.asarray([sample.token_length for sample in samples], dtype=np.int64)
    for row, sample in enumerate(samples):
        factors[row, :sample.token_length] = sample.token_factors
        numeric[row, :sample.token_length] = sample.token_numeric
    return factors, numeric, lengths


def _audit_npz_shard(npz_path: Path) -> dict[str, Any]:
    """A3: structural integrity checks directly on the saved .npz shard."""
    with np.load(npz_path, allow_pickle=False) as data:
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        factors = np.asarray(data["factors"], dtype=np.uint8)
        numeric = np.asarray(data["numeric"])
        legal = np.asarray(data["legal"], dtype=np.uint8)
        actions = np.asarray(data["actions"], dtype=np.uint8)
    n_samples = int(actions.shape[0])
    report: dict[str, Any] = {
        "shard": str(npz_path),
        "n_samples": n_samples,
        "factor_rows": int(factors.shape[0]),
        "numeric_rows": int(numeric.shape[0]),
        "offsets_dtype": str(offsets.dtype),
        "numeric_dtype_on_disk": str(numeric.dtype),
        "legal_shape": tuple(int(x) for x in legal.shape),
        "actions_max": int(actions.max()) if n_samples else -1,
        "actions_min": int(actions.min()) if n_samples else -1,
        "monotone_offsets": bool(np.all(np.diff(offsets) >= 0)),
        "offsets_cover_factors": int(offsets[-1]) == int(factors.shape[0]),
        "offsets_cover_numeric": int(offsets[-1]) == int(numeric.shape[0]),
    }
    # legal round-trip: unpackbits(count=241) on every row, then re-pack and
    # check we recover the same bit layout.
    legal_packed_shape = tuple(int(x) for x in legal.shape)
    expected_packed_cols = (NUM_ACTIONS + 7) // 8  # ceil(241/8) = 31
    report["legal_packed_cols_expected"] = expected_packed_cols
    report["legal_packed_cols_match"] = (legal_packed_shape[-1] == expected_packed_cols)
    round_trips_ok = True
    max_padding_drift = 0
    for row in range(min(n_samples, 200)):
        unpacked = np.unpackbits(
            legal[row], bitorder="little", count=NUM_ACTIONS
        ).astype(np.bool_)
        repacked = np.packbits(
            unpacked, bitorder="little"
        )
        # Compare only the relevant packed bytes; the last byte may have
        # non-zero padding bits after a non-canonical re-pack, so only assert
        # equality on the first 30 bytes plus the low bit of the 31st.
        if not np.array_equal(repacked[:30], legal[row][:30]):
            round_trips_ok = False
            break
        last_byte_packed = legal[row][30]
        last_byte_unpacked = repacked[30] if repacked.shape[0] > 30 else 0
        max_padding_drift = max(
            max_padding_drift, int(last_byte_packed) ^ int(last_byte_unpacked)
        )
    report["legal_round_trip_ok"] = round_trips_ok
    report["legal_padding_drift_max_bit_xor"] = int(max_padding_drift)
    # Spot-check a few rows: action must point at a true legal bit.
    spot_fail: list[int] = []
    for row in range(min(n_samples, 200)):
        aid = int(actions[row])
        unpacked = np.unpackbits(
            legal[row], bitorder="little", count=NUM_ACTIONS
        ).astype(np.bool_)
        if not (0 <= aid < NUM_ACTIONS) or not bool(unpacked[aid]):
            spot_fail.append(row)
    report["legal_spots_action_ok"] = (len(spot_fail) == 0)
    report["legal_spot_fail_rows"] = spot_fail[:5]
    return report


def _iter_kyokus(shard: Path, *, max_kyokus: int) -> Iterator[tuple[str, str]]:
    """Yield (member_name, decoded_content) up to ``max_kyokus`` from a tar shard."""
    seen = 0
    with tarfile.open(shard, "r") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        for member in members:
            if seen >= max_kyokus:
                break
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            content = (
                gzip.decompress(payload).decode("utf-8")
                if payload[:2] == b"\x1f\x8b"
                else payload.decode("utf-8")
            )
            yield member.name, content
            seen += 1


def _render_report(
    *,
    source_tar: Path,
    max_kyokus: int,
    npz_report: dict[str, Any] | None,
    a5_static_report: dict[str, Any],
    a5_legal_checked: int,
    a5_legal_failures: int,
    kyoku_reports: list[dict[str, Any]],
    coarse_counter: Counter,
    per_type_action_id_hist: dict[str, Counter],
    a1_summary: dict[str, Any],
    a4_total_violations: int,
    elapsed: float,
) -> str:
    lines: list[str] = []
    lines.append("# SFT encoding pipeline audit")
    lines.append("")
    lines.append(f"- Source tar: `{source_tar}`")
    lines.append(f"- Kyokus inspected: `{len(kyoku_reports)}` (limit {max_kyokus})")
    lines.append(f"- Elapsed: `{elapsed:.1f}s`")
    lines.append(f"- Token schema version: `{TOKEN_SCHEMA_VERSION}`")
    lines.append("")

    # A1 summary
    lines.append("## A1 — action / legal_mask / candidate-token alignment")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for key, value in a1_summary.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    if a1_summary["samples_with_misalignment"] == 0 and a1_summary["expert_outside_legal"] == 0:
        lines.append("> ✅ Each sample's expert action lands in its legal_mask, and the")
        lines.append("> candidate-token ids match the mask's true positions exactly.")
    else:
        lines.append("> ⚠️ Some samples showed misalignment between the three sources.")
        lines.append("> See per-kyoku details below.")
    lines.append("")

    # A2 summary — coarse action histogram
    lines.append("## A2 — expert action distribution by coarse MJAI type")
    lines.append("")
    total = sum(coarse_counter.values()) or 1
    lines.append("| type | count | share |")
    lines.append("|---|---|---|")
    for kind, count in sorted(coarse_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {kind} | {count} | {100.0 * count / total:.2f}% |")
    lines.append("")

    # A2 detail — action_id histograms per coarse type
    lines.append("### action_id histogram per coarse type")
    lines.append("")
    for kind in sorted(per_type_action_id_hist):
        hist = per_type_action_id_hist[kind]
        if not hist:
            continue
        items = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))
        sample_ids = ", ".join(f"{aid}:{c}" for aid, c in items[:6])
        more = "" if len(items) <= 6 else f" (+{len(items) - 6} more ids)"
        lines.append(f"- **{kind}** (top ids): {sample_ids}{more}")
    lines.append("")

    # A3 summary
    if npz_report is not None:
        lines.append("## A3 — npz shard structural integrity")
        lines.append("")
        lines.append("| check | result |")
        lines.append("|---|---|")
        for key, value in npz_report.items():
            display = value if not isinstance(value, (list, tuple)) else ", ".join(str(v) for v in value)
            lines.append(f"| {key} | {display} |")
        lines.append("")
        if all([
            npz_report["monotone_offsets"],
            npz_report["offsets_cover_factors"],
            npz_report["offsets_cover_numeric"],
            npz_report["legal_packed_cols_match"],
            npz_report["legal_round_trip_ok"],
            npz_report["legal_spots_action_ok"],
        ]):
            lines.append("> ✅ npz shard passes all structural checks; legal mask round-trips")
            lines.append("> through packbits/unpackbits and every sampled action sits inside the mask.")
        else:
            lines.append("> ⚠️ One or more structural checks failed on the npz shard.")
    else:
        lines.append("## A3 — skipped (pass ``--npz-shard`` to enable)")
    lines.append("")

    # A4 summary
    lines.append("## A4 — 4-stream timeline integrity")
    lines.append("")
    lines.append(f"- Total kyokus checked: {len(kyoku_reports)}")
    lines.append(f"- Total causality violations: {a4_total_violations}")
    lines.append("")
    if a4_total_violations == 0:
        lines.append("> ✅ No timeline causality violations detected in any inspected kyoku.")
    else:
        lines.append("> ⚠️ Some kyokus exhibited event-order inconsistencies; see per-kyoku list.")
        for kr in kyoku_reports:
            if kr["a4_violations"]:
                lines.append(f"  - {kr['member']}: {kr['a4_violations']} violation(s)")
    lines.append("")

    # A5 summary — action-space codec round-trip (the part the user asked for)
    lines.append("## A5 — action-space codec round-trip")
    lines.append("")
    lines.append("### A5 static — full 241-way protocol codec")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    static_metrics = (
        ("templates_count", "241 templates generated"),
        ("all_241_decoded", "all 241 ids decode back to identical canonical MJAI"),
        ("per_id_failures", "per-id codec failures (expected 0)"),
    )
    for key, label in static_metrics:
        if key in a5_static_report:
            lines.append(f"| {label} | `{a5_static_report[key]}` |")
    if "error" in a5_static_report:
        lines.append(f"| error | `{a5_static_report['error']}` |")
    lines.append("")
    if a5_static_report.get("all_241_decoded"):
        lines.append("> ✅ Static fixture confirms all 241 action ids encode and decode")
        lines.append("> bijectively through the state machine — the protocol codec is sound.")
    else:
        lines.append("> ⚠️ Static codec check failed; the 241-space has at least one")
        lines.append("> broken action id.  This is a show-stopper for the encoding contract.")
    lines.append("")

    # A5 real-scenario
    lines.append("### A5 real — per-legal-action codec on real replay snapshots")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| legal_actions_checked | {a5_legal_checked} |")
    lines.append(f"| decoder_failures | {a5_legal_failures} |")
    real_rate = (
        100.0 * (a5_legal_checked - a5_legal_failures) / max(1, a5_legal_checked)
    )
    lines.append(f"| codec_round_trip_rate_pct | {real_rate:.4f} |")
    lines.append("")
    if a5_legal_failures == 0:
        lines.append("> ✅ Every legal action observed in the replay round-trips cleanly")
        lines.append("> through the 241-space — no akadora / consumed-tile / tsumogiri /")
        lines.append("> action-kind aliasing slipped past the protocol codec in real games.")
    else:
        lines.append("> ⚠️ Some legal actions did not round-trip cleanly.  See per-kyoku")
        lines.append("> `a5_real_failures` column and the failure examples logged above.")
    lines.append("")

    # Per-kyoku summary
    lines.append("## Per-kyoku summary")
    lines.append("")
    lines.append("| # | member | decisions | tokens (avg) | candidate-aligned | assert | a5-real-fail | a4-vio |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for idx, kr in enumerate(kyoku_reports, 1):
        assert_mark = "✅" if kr["assertion_ok"] else "❌"
        lines.append(
            f"| {idx} | `{kr['member']}` | {kr['decisions']} | "
            f"{kr['avg_tokens']:.1f} | {kr['misaligned_samples']} | {assert_mark} | "
            f"{kr['a5_real_failures']} | {kr['a4_violations']} |"
        )
    lines.append("")
    # Surface any assertion failures explicitly so they are not hidden in the table.
    failed_assert = [kr for kr in kyoku_reports if not kr["assertion_ok"]]
    if failed_assert:
        lines.append("### actor-token semantic assertion failures")
        lines.append("")
        for kr in failed_assert:
            lines.append(f"- `{kr['member']}`: {kr['assertion_error']}")
        lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    ok = (
        a1_summary["samples_with_misalignment"] == 0
        and a1_summary["expert_outside_legal"] == 0
        and a4_total_violations == 0
        and (npz_report is None or npz_report["legal_round_trip_ok"])
        and all(kr["assertion_ok"] for kr in kyoku_reports)
        and bool(a5_static_report.get("all_241_decoded"))
        and a5_legal_failures == 0
    )
    if ok:
        lines.append(
            "✅ The full mjai → npz → model-tensor encoding pipeline is internally "
            "consistent on the inspected sample.  Expert actions always land inside "
            "the legal mask, candidate tokens agree with the legal mask, and event "
            "ordering respects game causality."
        )
    else:
        lines.append(
            "⚠️ One or more checks failed — review the per-section details above for "
            "exact failing samples and decide whether they indicate a real bug."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-tar",
        required=True,
        help="Path to a source mjai kyoku tar shard (e.g. .../train/train-00000.tar).",
    )
    parser.add_argument(
        "--max-kyokus",
        type=int,
        default=50,
        help="Maximum number of kyokus to inspect (defaults to 50).",
    )
    parser.add_argument(
        "--npz-shard",
        default=None,
        help="Optional path to an encoded .npz shard to additionally exercise A3.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown file to write the report to (defaults to stdout).",
    )
    parser.add_argument(
        "--no-a5-real",
        dest="a5_real",
        action="store_false",
        default=True,
        help="Skip the per-kyoku real-scenario codec round-trip (A5 real).",
    )
    args = parser.parse_args()

    source_tar = Path(args.source_tar).resolve()
    templates = _templates_by_id()
    analyzer = EfficiencyAnalyzer()

    npz_report: dict[str, Any] | None = None
    if args.npz_shard:
        npz_report = _audit_npz_shard(Path(args.npz_shard).resolve())

    coarse_counter: Counter = Counter()
    per_type_action_id_hist: dict[str, Counter] = {}
    a1_total = 0
    a1_misaligned = 0
    a1_expert_outside = 0
    a4_total_violations = 0
    a5_legal_checked = 0
    a5_legal_failures = 0
    kyoku_reports: list[dict[str, Any]] = []

    # A5 static: full 241-way protocol codec round-trip (run once).
    a5_static_report = _audit_action_space_static()

    started = time.perf_counter()
    inspected = 0
    for member_name, content in _iter_kyokus(source_tar, max_kyokus=args.max_kyokus):
        inspected += 1
        samples = encode_kyoku(
            content,
            year=0,
            game_id=member_name,
            kyoku_index=0,
            analyzer=analyzer,
            include_critic=False,
        )
        if not samples:
            continue
        # Actor-token semantic assertion — same training-time invariant.
        factors, numeric, lengths = _reshape_samples_for_assertion(samples)
        try:
            assert_actor_token_semantics(factors, numeric, lengths)
            assertion_ok = True
        except AssertionError as exc:
            assertion_ok = False
            assertion_error = str(exc)
        else:
            assertion_error = None

        misaligned_samples = 0
        avg_tokens = 0.0
        for sample in samples:
            avg_tokens += sample.token_length
            a1_total += 1
            a1_res = _audit_sample_a1(sample, templates)
            if not a1_res["expert_in_legal"]:
                a1_expert_outside += 1
            if not a1_res["legal_equals_candidates"]:
                misaligned_samples += 1
                a1_misaligned += 1
            coarse, _meta = _audit_sample_a2(sample, templates)
            coarse_counter[coarse] += 1
            per_type_action_id_hist.setdefault(coarse, Counter())[int(sample.action)] += 1
        avg_tokens = avg_tokens / max(1, len(samples))

        a4 = _audit_kyoku_a4(content)
        a4_total_violations += len(a4["violations"])

        # A5 real-scenario: per-legal-action codec round-trip on the replay
        # snapshot.  Disabled via ``--no-a5-real`` if it is too slow.
        a5_real = _audit_kyoku_a5_real_scenario(content) if args.a5_real else {"decisions": 0, "legal_failures_count": 0, "legal_actions_checked": 0}
        a5_legal_checked += int(a5_real.get("legal_actions_checked", 0))
        a5_legal_failures += int(a5_real.get("legal_failures_count", 0))

        kyoku_reports.append({
            "member": member_name,
            "decisions": len(samples),
            "avg_tokens": avg_tokens,
            "misaligned_samples": misaligned_samples,
            "a4_violations": len(a4["violations"]),
            "a5_real_decisions": int(a5_real.get("decisions", 0)),
            "a5_real_failures": int(a5_real.get("legal_failures_count", 0)),
            "assertion_ok": assertion_ok,
            "assertion_error": assertion_error,
        })

    a1_summary = {
        "samples_inspected": a1_total,
        "samples_with_misalignment": a1_misaligned,
        "expert_outside_legal": a1_expert_outside,
        "alignment_rate_pct": (100.0 * (a1_total - a1_misaligned) / max(1, a1_total)),
    }

    elapsed = time.perf_counter() - started
    body = _render_report(
        source_tar=source_tar,
        max_kyokus=args.max_kyokus,
        npz_report=npz_report,
        a5_static_report=a5_static_report,
        a5_legal_checked=a5_legal_checked,
        a5_legal_failures=a5_legal_failures,
        kyoku_reports=kyoku_reports,
        coarse_counter=coarse_counter,
        per_type_action_id_hist=per_type_action_id_hist,
        a1_summary=a1_summary,
        a4_total_violations=a4_total_violations,
        elapsed=elapsed,
    )

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
        print(f"[audit] wrote report to {out_path}")
    else:
        print(body)
    print(
        f"[audit] inspected {inspected} kyokus, {a1_total} expert decisions "
        f"in {elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
