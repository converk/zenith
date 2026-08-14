"""紧凑 SFT 预处理的确定性端到端完整性校验。"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..model.bridge import action_jsons_and_decision_flag, snapshot_json
from ..model.schema import NUM_ACTIONS
from ..model.critic_features import (
    FIELD_PUBLIC_MELD_TILE,
    FIELD_PUBLIC_RIVER,
    SEGMENT_PUBLIC_SUMMARY,
    collect_replay_table_state,
    encode_public_summary,
)
from ..model.semantic_validation import assert_actor_token_semantics
from ..training.rewards.decision import action_id
from .data import SftSample, _member_metadata, encode_kyoku
from .precompute import _assert_public_history, _decode, _write_chunk, selected

AUDITED_EVENTS = (
    "dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "reach",
    "reach_accepted", "dora",
)
_HISTORY_FIELDS = {
    "start_kyoku": 2, "dahai": 4, "chi": 5, "pon": 6,
    "daiminkan": 7, "ankan": 8, "kakan": 9, "dora": 10,
    "reach": 11, "reach_accepted": 12,
}
_HISTORY_FIELD_NAMES = {field: name for name, field in _HISTORY_FIELDS.items()}
_SEGMENT_CANDIDATE = 7


def _events(content: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in content.splitlines() if line.strip()]


_TYPE_PATTERN = re.compile(rb'"type"\s*:\s*"([^"]+)"')


def _payload_event_types(payload: bytes) -> frozenset[str]:
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return frozenset(
        match.group(1).decode("ascii") for match in _TYPE_PATTERN.finditer(raw)
        if match.group(1).decode("ascii") in AUDITED_EVENTS
    )


def _iter_selected_members(source: Path, denominator: int, remainder: int) -> Iterator[tuple[str, bytes]]:
    for split in ("train", "validation"):
        for shard in sorted((source / split).glob(f"{split}-*.tar")):
            with tarfile.open(shard, "r") as archive:
                for member in archive:
                    if member.isfile() and selected(member.name, denominator, remainder):
                        payload = archive.extractfile(member)
                        if payload is None:
                            raise RuntimeError(f"cannot read {shard}:{member.name}")
                        yield f"{shard}:{member.name}", payload.read()


def _read_identity(identity: str) -> tuple[str, str]:
    shard_name, member_name = identity.rsplit(":", 1)
    with tarfile.open(shard_name, "r") as archive:
        member = archive.getmember(member_name)
        payload = archive.extractfile(member)
        if payload is None:
            raise RuntimeError(f"cannot read {identity}")
        return identity, _decode(payload.read())


def select_coverage_kyokus(
    source: Path, *, denominator: int, remainder: int, sample_size: int,
) -> tuple[list[tuple[str, str]], set[str], int]:
    """Select exactly ``sample_size`` members using a bounded set-cover index."""
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    # There are only 2**9 possible event masks, so a full target-subset scan
    # does not retain raw games in memory while still making coverage claims.
    candidates: dict[frozenset[str], str] = {}
    fallback: list[str] = []
    universe: set[str] = set()
    scanned = 0
    for identity, payload in _iter_selected_members(source, denominator, remainder):
        scanned += 1
        kinds = _payload_event_types(payload)
        universe.update(kinds)
        candidates.setdefault(kinds, identity)
        if len(fallback) < sample_size:
            fallback.append(identity)
        # Once every required event has a witness, additional source members
        # cannot improve the selected set or the uncovered-event verdict.
        if len(fallback) == sample_size and universe == set(AUDITED_EVENTS):
            break
    if scanned < sample_size:
        raise RuntimeError(f"target subset has only {scanned} kyokus, need {sample_size}")

    chosen: list[str] = []
    covered: set[str] = set()
    remaining = dict(candidates)
    while remaining and len(chosen) < sample_size:
        mask, identity = min(
            remaining.items(), key=lambda item: (-len(item[0] - covered), item[1])
        )
        if not mask - covered:
            break
        chosen.append(identity)
        covered.update(mask)
        del remaining[mask]
    used = set(chosen)
    for identity in fallback:
        if len(chosen) == sample_size:
            break
        if identity not in used:
            chosen.append(identity)
            used.add(identity)
    # A selected coverage representative can be fewer than the desired count;
    # fill from the stable full scan candidates before failing.
    for identity in (value for _mask, value in sorted(remaining.items(), key=lambda item: item[1])):
        if len(chosen) == sample_size:
            break
        if identity not in used:
            chosen.append(identity)
            used.add(identity)
    return [_read_identity(identity) for identity in chosen], universe, scanned


def _check_public_summary(observation: Any, seat: int) -> None:
    table = collect_replay_table_state(observation)
    public = encode_public_summary(table, seat).factors
    rivers = public[(public[:, 0] == SEGMENT_PUBLIC_SUMMARY) & (public[:, 2] == FIELD_PUBLIC_RIVER)]
    meld_tiles = public[(public[:, 0] == SEGMENT_PUBLIC_SUMMARY) & (public[:, 2] == FIELD_PUBLIC_MELD_TILE)]
    expected_river_tiles = sum(len(row) for row in observation.discards)
    expected_meld_tiles = sum(len(meld.tiles) for row in observation.melds for meld in row)
    if int(rivers[:, 7].sum()) != expected_river_tiles:
        raise AssertionError(f"public river mismatch: encoded={int(rivers[:, 7].sum())} state={expected_river_tiles}")
    if int(meld_tiles[:, 7].sum()) != expected_meld_tiles:
        raise AssertionError(f"public meld mismatch: encoded={int(meld_tiles[:, 7].sum())} state={expected_meld_tiles}")


def audit_kyoku(content: str, *, identity: str) -> dict[str, Any]:
    """Audit replay, token history, public suffix and actor-only cache rows."""
    import riichi
    from riichienv import MjaiReplay

    replay = MjaiReplay.from_jsonl_string(content, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    if len(kyokus) != 1:
        raise AssertionError(f"{identity}: expected exactly one kyoku, got {len(kyokus)}")
    # Replay bootstrap may deliver an initial start_kyoku/tsumo pair before
    # the round's first real observation. start_kyoku is a protocol reset, so
    # compare the durable public history rather than that implementation detail.
    expected = [str(event["type"]) for event in _events(content) if str(event.get("type")) in AUDITED_EVENTS]
    observed_counts: Counter[str] = Counter()
    decisions = 0
    for seat in range(4):
        manager = riichi.MjaiKyokuStateMachineManager(1)
        cumulative: list[dict[str, Any]] = []
        for decision_index, (observation, expert_action) in enumerate(kyokus[0].steps(seat=seat, skip_single_action=False)):
            fresh = [json.loads(item) for item in observation.new_events()]
            start_indices = [index for index, event in enumerate(fresh) if str(event.get("type")) == "start_kyoku"]
            logical_fresh = fresh
            if start_indices:
                cumulative.clear()
                logical_fresh = fresh[start_indices[-1]:]
            cumulative.extend(logical_fresh)
            for event in fresh:
                kind = str(event.get("type"))
                if kind == "tsumo":
                    actor = int(event["actor"])
                    pai = event.get("pai")
                    if actor == seat and pai == "?":
                        raise AssertionError(f"{identity}: seat {seat} own tsumo was masked")
                    if actor != seat and pai != "?":
                        raise AssertionError(f"{identity}: seat {seat} opponent tsumo leaked")
                if kind in AUDITED_EVENTS:
                    observed_counts[kind] += 1
            events = [[], [], [], []]
            events[seat] = [json.dumps(event, separators=(",", ":")) for event in fresh]
            try:
                manager.apply_events_batch([0], [events])
            except Exception as exc:
                raise AssertionError(
                    f"{identity}: seat={seat} decision={decision_index} rejected replay events: {fresh}"
                ) from exc
            action_jsons, flag = action_jsons_and_decision_flag(observation)
            prepared = manager.prepare_decisions([seat], [action_jsons], [snapshot_json(observation, flag)])
            length = int(prepared[2][0])
            factors = np.asarray(prepared[0], dtype=np.uint8)[0, :length]
            numeric = np.asarray(prepared[1], dtype=np.float32)[0, :length]
            legal = np.asarray(prepared[3], dtype=np.bool_)[0]
            padded_factors = factors[None, :, :]
            padded_numeric = numeric[None, :, :]
            assert_actor_token_semantics(padded_factors, padded_numeric, np.asarray([length], dtype=np.int64))
            event_counts = Counter(str(event.get("type")) for event in cumulative)
            for kind, field in _HISTORY_FIELDS.items():
                token_count = int(np.count_nonzero((factors[:, 0] == 1) & (factors[:, 1] == 1) & (factors[:, 2] == field)))
                if token_count != event_counts[kind]:
                    raise AssertionError(
                        f"{identity}: seat={seat} decision={decision_index} {kind} history "
                        f"tokens={token_count} events={event_counts[kind]}"
                    )
            _check_public_summary(observation, seat)
            target = action_id(expert_action, observation)
            if target is None or not legal[int(target)]:
                raise AssertionError(f"{identity}: seat={seat} decision={decision_index} expert action is illegal")
            decisions += 1
        actual = [str(event.get("type")) for event in cumulative if str(event.get("type")) in AUDITED_EVENTS]
        # A round can terminate immediately after a public event, leaving no
        # subsequent decision at which to expose it. Such a suffix must not be
        # required in actor history; every event that *is* observed is exact.
        if actual != expected[:len(actual)]:
            raise AssertionError(f"{identity}: seat={seat} replay event order differs from source: actual={actual} expected={expected}")

    for kind in AUDITED_EVENTS:
        source_count = expected.count(kind)
        if observed_counts[kind] > source_count * 4:
            raise AssertionError(f"{identity}: {kind} was duplicated by replay")
    return {"identity": identity, "decisions": decisions, "events": {kind: expected.count(kind) for kind in AUDITED_EVENTS}}


def validate_encoded_chunk(path: Path) -> dict[str, int]:
    """Parse a materialized actor-only chunk exactly as SFT training does."""
    with np.load(path, allow_pickle=False) as stored:
        required = {
            "factors", "numeric", "offsets", "legal", "actions", "value_targets", "teacher_masks",
            "years", "game_ids", "kyoku_indices", "seats", "decision_indices",
        }
        if set(stored.files) != required:
            raise AssertionError(f"cache keys differ: {sorted(stored.files)}")
        factors, numeric, offsets, legal, actions = (stored[name] for name in ("factors", "numeric", "offsets", "legal", "actions"))
        if factors.dtype != np.uint8 or factors.ndim != 2 or factors.shape[1] != 10:
            raise AssertionError("cache factors must be uint8[N, 10]")
        if numeric.dtype != np.float16 or numeric.shape != (factors.shape[0], 8):
            raise AssertionError("cache numeric must be float16[N, 8]")
        if offsets.dtype != np.int64 or offsets.ndim != 1 or len(offsets) != len(actions) + 1 or offsets[0] != 0:
            raise AssertionError("cache offsets are malformed")
        if np.any(np.diff(offsets) <= 0) or int(offsets[-1]) != len(factors):
            raise AssertionError("cache offsets must partition non-empty token rows")
        if actions.dtype != np.uint8 or legal.dtype != np.uint8 or legal.shape != (len(actions), 31):
            raise AssertionError("cache actions or packed legal masks are malformed")
        if stored["value_targets"].shape != (len(actions),) or not np.issubdtype(stored["value_targets"].dtype, np.floating):
            raise AssertionError("cache value targets are malformed")
        if stored["teacher_masks"].dtype != np.uint8 or stored["teacher_masks"].shape != (len(actions), 31):
            raise AssertionError("cache packed teacher masks are malformed")
        identity_fields = ("years", "game_ids", "kyoku_indices", "seats", "decision_indices")
        if any(stored[name].shape != (len(actions),) for name in identity_fields):
            raise AssertionError("cache per-sample identity is malformed")
        identities = list(zip(*(stored[name].tolist() for name in identity_fields), strict=True))
        if len(set(identities)) != len(identities):
            raise AssertionError("cache contains duplicate per-sample identities")
        masks = np.unpackbits(legal, axis=1, bitorder="little", count=NUM_ACTIONS).astype(np.bool_)
        totals: Counter[str] = Counter()
        for row, action in enumerate(actions):
            start, end = int(offsets[row]), int(offsets[row + 1])
            token_rows = factors[start:end]
            token_numeric = numeric[start:end].astype(np.float32)
            assert_actor_token_semantics(token_rows[None, :, :], token_numeric[None, :, :], np.asarray([end - start], dtype=np.int64))
            if not masks[row, int(action)]:
                raise AssertionError(f"cache row {row} expert action is outside its legal mask")
            candidates = token_rows[:, 0] == _SEGMENT_CANDIDATE
            public = token_rows[:, 0] == SEGMENT_PUBLIC_SUMMARY
            if int(candidates.sum()) != 2 * int(masks[row].sum()):
                raise AssertionError(f"cache row {row} has unmatched candidate-action tokens and legal mask")
            if candidates.any() and not np.all(candidates[np.flatnonzero(candidates)[0]:np.flatnonzero(candidates)[-1] + 1]):
                raise AssertionError(f"cache row {row} candidate tokens are not contiguous")
            if public.any() and candidates.any() and np.flatnonzero(public)[-1] >= np.flatnonzero(candidates)[0]:
                raise AssertionError(f"cache row {row} public summary must precede candidates")
            totals["history_tokens"] += int(np.count_nonzero(token_rows[:, 0] == 1))
            totals["state_tokens"] += int(np.count_nonzero(token_rows[:, 0] == 2))
            totals["candidate_tokens"] += int(candidates.sum())
            totals["public_tokens"] += int(public.sum())
        return {"rows": len(actions), "token_rows": len(factors), **dict(totals)}


def _describe_sample_near_length(samples: list[SftSample], target_length: int = 100) -> dict[str, Any]:
    """Return a human-readable structural summary for a representative row."""
    sample = min(
        samples,
        key=lambda item: (abs(item.token_length - target_length), item.token_length, item.game_id, item.kyoku_index, item.seat),
    )
    rows = sample.token_factors
    segment_counts = Counter(int(value) for value in rows[:, 0])
    history = Counter(
        _HISTORY_FIELD_NAMES.get(int(value), f"unknown:{int(value)}")
        for value in rows[rows[:, 0] == 1, 2]
    )
    state = Counter(
        f"kind={int(row[1])},field={int(row[2])}" for row in rows[rows[:, 0] == 2]
    )
    public = Counter(
        f"kind={int(row[1])},field={int(row[2])}" for row in rows[rows[:, 0] == SEGMENT_PUBLIC_SUMMARY]
    )
    return {
        "target_token_length": target_length,
        "token_length": sample.token_length,
        "distance_from_target": abs(sample.token_length - target_length),
        "source": {"year": sample.year, "game_id": sample.game_id, "kyoku_index": sample.kyoku_index, "seat": sample.seat},
        "expert_action": sample.action,
        "legal_action_count": int(sample.legal_mask.sum()),
        "legal_action_ids": np.flatnonzero(sample.legal_mask).astype(int).tolist(),
        "segment_counts": {str(key): value for key, value in sorted(segment_counts.items())},
        "history_events": dict(sorted(history.items())),
        "state_tokens": dict(sorted(state.items())),
        "public_summary_tokens": dict(sorted(public.items())),
    }


def _roundtrip_cache(records: list[tuple[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    samples: list[SftSample] = []
    for identity, content in records:
        year, game_id, kyoku_index = _member_metadata(identity.rsplit(":", 1)[1])
        try:
            rows = encode_kyoku(
                content, year=year, game_id=game_id,
                kyoku_index=kyoku_index, include_critic=False,
            )
        except Exception as exc:
            raise RuntimeError(f"cache round-trip failed for {identity}") from exc
        _assert_public_history(rows, identity)
        samples.extend(rows)
    with tempfile.TemporaryDirectory(prefix="riichi-sft-audit-") as directory:
        path = Path(directory) / "sample.npz"
        _write_chunk(path, samples)
        with np.load(path, allow_pickle=False) as stored:
            offsets = stored["offsets"]
            if int(offsets[-1]) != sum(sample.token_length for sample in samples):
                raise AssertionError("cache offsets do not cover encoded factors")
            if not np.array_equal(stored["actions"], np.asarray([sample.action for sample in samples], dtype=np.uint8)):
                raise AssertionError("cache actions changed on round-trip")
            unpacked = np.unpackbits(stored["legal"], axis=1, bitorder="little", count=NUM_ACTIONS).astype(np.bool_)
            if not np.array_equal(unpacked, np.stack([sample.legal_mask for sample in samples])):
                raise AssertionError("cache legal masks changed on round-trip")
        structure = validate_encoded_chunk(path)
    if structure["rows"] != len(samples):
        raise AssertionError("cache parser row count differs from encoded sample count")
    return {
        **structure,
        "average_tokens": structure["token_rows"] / structure["rows"],
    }, _describe_sample_near_length(samples)


def audit_dataset(source: Path, *, denominator: int = 10, remainder: int = 0, sample_size: int = 10) -> dict[str, Any]:
    started = time.perf_counter()
    records, universe, scanned = select_coverage_kyokus(
        source, denominator=denominator, remainder=remainder, sample_size=sample_size,
    )
    results = [audit_kyoku(content, identity=identity) for identity, content in records]
    cache_structure, representative_sample = _roundtrip_cache(records)
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(result["events"])
    return {
        "format": "riichi-sft-audit-v1", "passed": True,
        "source": str(source), "subset_denominator": denominator, "subset_remainder": remainder,
        "scanned_target_kyokus": scanned, "sample_size": sample_size,
        "selected": results, "event_counts": dict(counts),
        "uncovered_events": sorted(set(AUDITED_EVENTS) - universe),
        "cache_rows": cache_structure["rows"], "cache_structure": cache_structure,
        "representative_sample": representative_sample,
        "elapsed_seconds": time.perf_counter() - started,
    }


def print_audit_summary(report: dict[str, Any]) -> None:
    """打印审计报告摘要,不产生任何文件。"""
    print(json.dumps({key: report[key] for key in ("passed", "sample_size", "event_counts", "uncovered_events", "cache_rows", "elapsed_seconds")}, ensure_ascii=False), flush=True)


def write_audit_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_audit_summary(report)


def build_parser() -> argparse.ArgumentParser:
    """构建 sft-audit 命令行解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/tenhou_sft_2024_2025"))
    parser.add_argument("--subset-denominator", type=int, default=10)
    parser.add_argument("--subset-remainder", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "审计报告输出路径(按规范写入 logs/<版本号>/ 或 "
            "audit/reports/<版本号>/eval/);省略时仅打印摘要、不落盘"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_dataset(
        args.source,
        denominator=args.subset_denominator,
        remainder=args.subset_remainder,
        sample_size=args.sample_size,
    )
    if args.report is not None:
        write_audit_report(report, args.report)
    else:
        print_audit_summary(report)


if __name__ == "__main__":
    main()
