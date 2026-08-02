"""Materialize a deterministic compact, actor-only SFT subset for fast training."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import hashlib
import json
from multiprocessing import Manager
import os
from pathlib import Path
from queue import Empty
import random
import tarfile
import time
from typing import Any, Iterator
import zipfile

import numpy as np

from ..model.feature_schema import (
    ENCODED_FORMAT,
)
from ..training.rewards.efficiency import EfficiencyAnalyzer
from .data import SftSample, _member_metadata, encode_kyoku
from .contract import SFT_CONTRACT_VERSION, validate_v13_manifest



def _assert_public_history(samples: list[SftSample], source: str) -> None:
    """Reject replay output that has public state but no event history.

    Public river summaries are a supplement to the ordered MJAI history, not a
    replacement.  This catches a replay extension that advances game state
    without appending its per-player ``new_events`` log.
    """
    for sample in samples:
        factors = sample.token_factors
        river_rows = np.count_nonzero((factors[:, 0] == 3) & (factors[:, 1] == 4) & (factors[:, 2] == 6))
        discard_history = np.count_nonzero((factors[:, 0] == 1) & (factors[:, 1] == 1) & (factors[:, 2] == 4))
        if river_rows and not discard_history:
            raise RuntimeError(
                f"replay history is missing despite {river_rows} public river tokens: {source}; "
                "rebuild the RiichiEnv extension before preprocessing"
            )


def _selection_bucket(game_id: str, namespace: str, denominator: int) -> int:
    """Hash a game independently of the source split and other selectors."""
    payload = f"riichi-sft-{namespace}-v1\0{game_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % denominator


def selected(name: str, denominator: int, remainder: int) -> bool:
    game_id = _member_metadata(name)[1]
    return _selection_bucket(game_id, "subset", denominator) == remainder


def selected_any(
    name: str, denominator: int, remainders: tuple[int, ...],
    game_sample_denominator: int = 1, game_sample_remainder: int = 0,
) -> bool:
    game_id = _member_metadata(name)[1]
    if _selection_bucket(game_id, "subset", denominator) not in remainders:
        return False
    return (
        _selection_bucket(game_id, "canary", game_sample_denominator)
        == game_sample_remainder
    )


def _decode(payload: bytes) -> str:
    return (gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload).decode("utf-8")


def _write_chunk(path: Path, samples: list[SftSample]) -> int:
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([sample.token_length for sample in samples], dtype=np.int64)
    factors = np.concatenate([sample.token_factors for sample in samples], axis=0)
    # Numeric factors are currently bounded semantic values; fp16 halves disk
    # and are restored to fp32 by the loader before model input.
    numeric = np.concatenate([sample.token_numeric for sample in samples], axis=0).astype(np.float16)
    legal = np.packbits(np.stack([sample.legal_mask for sample in samples]), axis=1, bitorder="little")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        factors=factors,
        numeric=numeric,
        offsets=offsets,
        legal=legal,
        actions=np.asarray([sample.action for sample in samples], dtype=np.uint8),
        value_targets=np.asarray([sample.value_target for sample in samples], dtype=np.float16),
        teacher_masks=np.packbits(np.stack([
            sample.teacher_mask if sample.teacher_mask is not None else np.zeros(241, dtype=np.bool_)
            for sample in samples
        ]), axis=1, bitorder="little"),
        years=np.asarray([sample.year for sample in samples], dtype=np.int16),
        game_ids=np.asarray([sample.game_id for sample in samples], dtype=np.str_),
        kyoku_indices=np.asarray([sample.kyoku_index for sample in samples], dtype=np.int16),
        seats=np.asarray([sample.seat for sample in samples], dtype=np.uint8),
        decision_indices=np.asarray([sample.decision_index for sample in samples], dtype=np.int32),
    )
    os.replace(temporary, path)
    return len(samples)


def encoded_identity_digests(dataset: Path, split: str) -> dict[str, object]:
    """Hash every stored sample identity, including its exact chunk layout."""
    sequence = hashlib.sha256()
    sharded_sequence = hashlib.sha256()
    supervision_sequence = hashlib.sha256()
    count = 0
    for path in sorted((dataset / split).glob(f"{split}-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            required = (
                "years", "game_ids", "kyoku_indices", "seats", "decision_indices",
                "actions", "legal",
            )
            if any(name not in data for name in required):
                raise RuntimeError(f"encoded cache lacks per-sample identity: {path}")
            rows = len(data["game_ids"])
            if any(len(data[name]) != rows for name in required):
                raise RuntimeError(f"encoded cache has inconsistent identity arrays: {path}")
            sharded_sequence.update(json.dumps(
                [path.name, rows], ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8") + b"\n")
            for row in range(rows):
                identity = json.dumps([
                    int(data["years"][row]), str(data["game_ids"][row]),
                    int(data["kyoku_indices"][row]), int(data["seats"][row]),
                    int(data["decision_indices"][row]),
                ], ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                sequence.update(identity)
                sharded_sequence.update(identity)
                supervision_sequence.update(identity)
                supervision_sequence.update(int(data["actions"][row]).to_bytes(2, "little"))
                supervision_sequence.update(np.asarray(data["legal"][row], dtype=np.uint8).tobytes())
                count += 1
    return {
        "samples": count,
        "sequence_sha256": sequence.hexdigest(),
        "sharded_sequence_sha256": sharded_sequence.hexdigest(),
        "supervision_sequence_sha256": supervision_sequence.hexdigest(),
    }


def _count_selected_kyokus(
    source: Path, denominator: int, remainders: tuple[int, ...],
    game_sample_denominator: int, game_sample_remainder: int,
) -> tuple[dict[str, int], str]:
    """Count the exact target subset before encoding so ETA is meaningful."""
    totals = {"train": 0, "validation": 0}
    game_ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    record_ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    selection_digest = hashlib.sha256()
    for split in totals:
        shards = sorted((source / split).glob(f"{split}-*.tar"))
        for index, shard in enumerate(shards, start=1):
            with tarfile.open(shard, "r") as archive:
                for member in archive:
                    if not member.isfile() or not selected_any(
                        member.name, denominator, remainders,
                        game_sample_denominator, game_sample_remainder,
                    ):
                        continue
                    totals[split] += 1
                    if member.name in record_ids[split]:
                        raise RuntimeError(
                            f"duplicate selected kyoku record in {split}: {member.name}"
                        )
                    record_ids[split].add(member.name)
                    game_ids[split].add(_member_metadata(member.name)[1])
                    selection_digest.update(f"{split}\0{member.name}\n".encode("utf-8"))
            if index == len(shards) or index % 25 == 0:
                print(
                    f"preflight split={split} scanned_shards={index}/{len(shards)} "
                    f"selected_kyokus={totals[split]}", flush=True,
                )
    overlap = game_ids["train"] & game_ids["validation"]
    if overlap:
        example = min(overlap)
        raise RuntimeError(f"train/validation game-id overlap detected, for example {example}")
    return totals, selection_digest.hexdigest()


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _empty_field_statistics() -> dict[str, np.ndarray]:
    return {
        "categorical_na": np.zeros(10, dtype=np.int64),
        "categorical_total": np.zeros(10, dtype=np.int64),
        "categorical_saturated": np.zeros(10, dtype=np.int64),
        "numeric_saturated": np.zeros(8, dtype=np.int64),
        "numeric_out_of_range": np.zeros(8, dtype=np.int64),
        "numeric_total": np.zeros(8, dtype=np.int64),
        "legal_actions": np.zeros(241, dtype=np.int64),
        "expert_actions": np.zeros(241, dtype=np.int64),
    }


def _accumulate_field_statistics(target: dict[str, np.ndarray], samples: list[SftSample]) -> None:
    for sample in samples:
        selected = np.isin(sample.token_factors[:, 0], (6, 7))
        factors = sample.token_factors[selected]
        numeric = sample.token_numeric[selected]
        categorical_known = np.zeros_like(factors, dtype=np.bool_)
        numeric_known = np.zeros_like(numeric, dtype=np.bool_)
        for row, values in enumerate(factors):
            segment, kind, role = int(values[0]), int(values[1]), int(values[9])
            if segment == 6:
                categorical_known[row, 0:({1: 9, 2: 4, 3: 7}.get(kind, 7))] = True
                if kind == 4 and values[8] != 0:
                    categorical_known[row, 8] = True
                numeric_width = {1: 6, 2: 7, 3: 8}.get(kind, 5)
                numeric_known[row, :numeric_width] = True
                if kind == 4 and values[8] != 0:
                    numeric_known[row, 5] = True
            else:
                categorical_known[row, (0, 1, 2, 9)] = True
                if role == 1 and values[4] != 0:
                    categorical_known[row, 3:9] = True
                    numeric_known[row] = True
                elif role == 2 and (kind == 2 or values[4] != 0):
                    categorical_known[row, 3:8] = True
                    categorical_known[row, 8] = kind == 2
                    numeric_known[row] = True
        target["categorical_na"] += np.count_nonzero(~categorical_known, axis=0)
        target["categorical_total"] += factors.shape[0]
        maxima = np.asarray((7, 31, 255, 7, 7, 15, 3, 15, 255, 3), dtype=np.uint8)
        target["categorical_saturated"] += np.count_nonzero((factors == maxima) & categorical_known, axis=0)
        target["numeric_saturated"] += np.count_nonzero((np.abs(numeric) >= 1.0) & numeric_known, axis=0)
        target["numeric_out_of_range"] += np.count_nonzero((np.abs(numeric) > 1.0) & numeric_known, axis=0)
        target["numeric_total"] += np.count_nonzero(numeric_known, axis=0)
        target["legal_actions"] += sample.legal_mask.astype(np.int64)
        target["expert_actions"][int(sample.action)] += 1


def _action_coverage(values: np.ndarray) -> dict[str, int]:
    groups = {
        "pass": (0, 1), "discard": (1, 75), "reach": (75, 76), "chi": (76, 133),
        "pon": (133, 170), "daiminkan": (170, 171), "ankan": (171, 205),
        "kakan": (205, 239), "hora": (239, 240), "ryukyoku": (240, 241),
    }
    return {name: int(values[start:end].sum()) for name, (start, end) in groups.items()}


def _require_complete_action_coverage(statistics: dict[str, np.ndarray]) -> None:
    missing: list[str] = []
    for label, field in (("legal", "legal_actions"), ("expert", "expert_actions")):
        coverage = _action_coverage(statistics[field])
        missing.extend(f"{label}:{name}" for name, count in coverage.items() if count == 0)
    if missing:
        raise RuntimeError(
            "semantic canary lacks required action-group coverage: " + ", ".join(missing)
        )


def _precompute_source_shard(
    split: str,
    source_shard: str,
    destination: str,
    denominator: int,
    remainders: tuple[int, ...],
    kyokus_per_shard: int,
    game_sample_denominator: int = 1,
    game_sample_remainder: int = 0,
    progress_queue: Any | None = None,
    progress_every_kyokus: int = 32,
) -> tuple[str, int, int, int, dict[str, list[int]]]:
    """Encode one source tar in an isolated process and write unique chunks."""
    shard = Path(source_shard)
    target_dir = Path(destination)
    analyzer = EfficiencyAnalyzer()
    buffered: list[SftSample] = []
    kyokus = decisions = chunk_index = reported_kyokus = reported_decisions = 0
    statistics = _empty_field_statistics()

    def report_progress(*, force: bool = False) -> None:
        nonlocal reported_kyokus, reported_decisions
        if progress_queue is None or (not force and kyokus - reported_kyokus < progress_every_kyokus):
            return
        progress_queue.put((split, kyokus - reported_kyokus, decisions - reported_decisions))
        reported_kyokus, reported_decisions = kyokus, decisions

    with tarfile.open(shard, "r") as archive:
        for member in archive:
            if not member.isfile() or not selected_any(
                member.name, denominator, remainders,
                game_sample_denominator, game_sample_remainder,
            ):
                continue
            file = archive.extractfile(member)
            if file is None:
                raise RuntimeError(f"cannot read {shard}:{member.name}")
            year, game_id, kyoku_index = _member_metadata(member.name)
            samples = encode_kyoku(
                _decode(file.read()), year=year, game_id=game_id, kyoku_index=kyoku_index,
                analyzer=analyzer, include_critic=False,
            )
            _assert_public_history(samples, f"{shard}:{member.name}")
            _accumulate_field_statistics(statistics, samples)
            buffered.extend(samples)
            kyokus += 1
            decisions += len(samples)
            report_progress()
            if kyokus % kyokus_per_shard == 0:
                target = target_dir / f"{shard.stem}-{chunk_index:03d}.npz"
                _write_chunk(target, buffered)
                buffered.clear()
                chunk_index += 1
    if buffered:
        target = target_dir / f"{shard.stem}-{chunk_index:03d}.npz"
        _write_chunk(target, buffered)
        chunk_index += 1
    report_progress(force=True)
    return split, kyokus, decisions, chunk_index, {
        name: values.tolist() for name, values in statistics.items()
    }


def precompute(
    source: Path,
    output: Path,
    *,
    denominator: int = 10,
    remainder: int = 0,
    remainders: tuple[int, ...] | None = None,
    kyokus_per_shard: int = 256,
    workers: int = 8,
    audit_kyokus: int = 10,
    skip_audit: bool = False,
    progress_every_kyokus: int = 32,
    game_sample_denominator: int = 1,
    game_sample_remainder: int = 0,
    require_complete_action_coverage: bool = False,
    require_identity_contract: bool = False,
) -> None:
    remainders = tuple(sorted(set(remainders if remainders is not None else (remainder,))))
    if denominator <= 0 or not remainders or any(not 0 <= value < denominator for value in remainders):
        raise ValueError("subset remainder must be in [0, subset denominator)")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already exists and is non-empty: {output}")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if game_sample_denominator <= 0 or not 0 <= game_sample_remainder < game_sample_denominator:
        raise ValueError("game sample remainder must be in [0, game sample denominator)")
    if progress_every_kyokus <= 0:
        raise ValueError("progress_every_kyokus must be positive")
    audit_reports = [output.with_name(f"{output.name}-audit-r{value}.json") for value in remainders]
    if not skip_audit:
        from .audit import audit_dataset, write_audit_report
        for value, audit_report in zip(remainders, audit_reports, strict=True):
            write_audit_report(
                audit_dataset(source, denominator=denominator, remainder=value, sample_size=audit_kyokus),
                audit_report,
            )
    output.mkdir(parents=True, exist_ok=True)
    print(f"preflight: counting denominator={denominator} remainders={remainders}...", flush=True)
    total_kyokus, selection_manifest_sha256 = _count_selected_kyokus(
        source, denominator, remainders, game_sample_denominator, game_sample_remainder,
    )
    total = sum(total_kyokus.values())
    if total == 0:
        raise RuntimeError("target subset contains no kyokus")
    print(
        f"preflight: target_kyokus={total} train={total_kyokus['train']} "
        f"validation={total_kyokus['validation']}", flush=True,
    )
    counts: dict[str, int] = {"train_kyokus": 0, "validation_kyokus": 0, "train_decisions": 0, "validation_decisions": 0}
    field_statistics = _empty_field_statistics()
    tasks: list[tuple[Any, ...]] = []
    for split in ("train", "validation"):
        destination = output / split
        destination.mkdir()
        for shard in sorted((source / split).glob(f"{split}-*.tar")):
            tasks.append((
                split, str(shard), str(destination), denominator, remainders,
                kyokus_per_shard, game_sample_denominator, game_sample_remainder,
            ))
    completed = processed_kyokus = processed_decisions = 0
    started = time.monotonic()
    with Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {
                executor.submit(
                    _precompute_source_shard, *task, progress_queue, progress_every_kyokus,
                ) for task in tasks
            }
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        _split, kyoku_delta, decision_delta = progress_queue.get_nowait()
                    except Empty:
                        break
                    processed_kyokus += int(kyoku_delta)
                    processed_decisions += int(decision_delta)
                    elapsed = max(time.monotonic() - started, 1e-6)
                    rate = processed_kyokus / elapsed
                    remaining = total - processed_kyokus
                    eta = remaining / rate if rate else float("inf")
                    print(
                        f"progress kyokus={processed_kyokus}/{total} remaining={remaining} "
                        f"decisions={processed_decisions} rate={rate:.2f} kyokus/s eta={_format_eta(eta)}",
                        flush=True,
                    )
                for future in done:
                    split, kyokus, decisions, chunks, shard_statistics = future.result()
                    for name, values in shard_statistics.items():
                        field_statistics[name] += np.asarray(values, dtype=np.int64)
                    counts[f"{split}_kyokus"] += kyokus
                    counts[f"{split}_decisions"] += decisions
                    completed += 1
                    print(
                        f"completed_shards={completed}/{len(tasks)} split={split} "
                        f"kyokus={kyokus} decisions={decisions} output_chunks={chunks}",
                        flush=True,
                    )
    if require_complete_action_coverage:
        _require_complete_action_coverage(field_statistics)
    identity_contract = None
    if require_identity_contract:
        identity_contract = {
            split: encoded_identity_digests(output, split)
            for split in ("train", "validation")
        }
        for split, digest in identity_contract.items():
            if int(digest["samples"]) != int(counts[f"{split}_decisions"]):
                raise RuntimeError(f"encoded {split} identity count differs from decision count")
    manifest = {
        "format": ENCODED_FORMAT,
        "sft_contract_version": SFT_CONTRACT_VERSION,
        "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "subset_denominator": denominator,
        "subset_remainders": list(remainders),
        "game_sample_denominator": game_sample_denominator,
        "game_sample_remainder": game_sample_remainder,
        "actor_only": True,
        "numeric_dtype": "float16",
        "legal_encoding": "packbits-little-241",
        "ordered_public_history_verified": True,
        "complete_action_coverage_required": bool(require_complete_action_coverage),
        "audit_reports": [] if skip_audit else [str(path) for path in audit_reports],
        "audit_skipped": bool(skip_audit),
        "preflight_target_kyokus": total_kyokus,
        "counts": counts,
        "selection_manifest_sha256": selection_manifest_sha256,
        "sample_identity_contract": identity_contract,
        "field_statistics": {
            "categorical_na_by_slot": field_statistics["categorical_na"].tolist(),
            "categorical_total_by_slot": field_statistics["categorical_total"].tolist(),
            "categorical_at_max_by_slot": field_statistics["categorical_saturated"].tolist(),
            "numeric_abs_ge_1_by_slot": field_statistics["numeric_saturated"].tolist(),
            "numeric_abs_gt_1_by_slot": field_statistics["numeric_out_of_range"].tolist(),
            "numeric_total_by_slot": field_statistics["numeric_total"].tolist(),
            "legal_action_type_coverage": _action_coverage(field_statistics["legal_actions"]),
            "expert_action_type_coverage": _action_coverage(field_statistics["expert_actions"]),
            "legal_action_id_counts": field_statistics["legal_actions"].tolist(),
            "expert_action_id_counts": field_statistics["expert_actions"].tolist(),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def iter_precomputed_samples(
    dataset: Path,
    split: str,
    *,
    seed: int,
    shuffle: bool,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[SftSample]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    validate_v13_manifest(manifest)
    paths = sorted((dataset / split).glob(f"{split}-*.npz"))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(paths)
    if not 0 <= int(rank) < int(world_size):
        raise ValueError("rank must be in [0, world_size)")

    # Partition one deterministic global row stream into equally sized,
    # contiguous rank intervals.  File-striding can differ by hundreds of
    # thousands of decisions because encoded chunks have different row counts,
    # leaving one DDP rank to train alone at the end of an epoch.  Contiguous
    # intervals keep almost every NPZ exclusive to one rank (only interval
    # boundary files can be shared) while differing by at most one sample.
    def row_count(path: Path) -> int:
        with zipfile.ZipFile(path) as archive, archive.open("actions.npy") as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(member)
            elif version in {(2, 0), (3, 0)}:
                shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(member)
            else:  # pragma: no cover - NumPy rejects this before normal loading too.
                raise RuntimeError(f"unsupported NPY header version {version} in {path}")
        if len(shape) != 1:
            raise RuntimeError(f"encoded actions must be one-dimensional: {path}")
        return int(shape[0])

    path_rows = [(path, row_count(path)) for path in paths]
    total_rows = sum(rows for _path, rows in path_rows)
    rows_per_rank, extra = divmod(total_rows, int(world_size))
    interval_start = int(rank) * rows_per_rank + min(int(rank), extra)
    interval_end = interval_start + rows_per_rank + int(int(rank) < extra)
    global_start = 0
    for path, rows in path_rows:
        global_end = global_start + rows
        selected_start = max(interval_start, global_start) - global_start
        selected_end = min(interval_end, global_end) - global_start
        global_start = global_end
        if selected_start >= selected_end:
            continue
        with np.load(path, allow_pickle=False) as data:
            offsets, factors, numeric, legal, actions = (data[name] for name in ("offsets", "factors", "numeric", "legal", "actions"))
            values = data["value_targets"] if "value_targets" in data else np.zeros(len(actions), dtype=np.float32)
            teachers = data["teacher_masks"] if "teacher_masks" in data else None
            required_identity = ("years", "game_ids", "kyoku_indices", "seats", "decision_indices")
            if any(name not in data for name in required_identity):
                raise RuntimeError(f"encoded cache lacks per-sample identity: {path}")
            # NpzFile does not cache members: every ``data[name]`` access
            # reopens and decompresses the complete embedded .npy payload.
            # Materialize shard-level identity columns once before iterating
            # rows, just like the factors, targets and teacher masks above.
            years, game_ids, kyoku_indices, seats, decision_indices = (
                data[name] for name in required_identity
            )
            order = list(range(len(actions)))
            if shuffle:
                # Derive file-local order independently so every rank agrees on
                # boundary files without replaying RNG operations for files it
                # does not open.
                random.Random(f"riichi-sft-row-order-v1\0{seed}\0{path.name}").shuffle(order)
            for row in order[selected_start:selected_end]:
                start, end = int(offsets[row]), int(offsets[row + 1])
                yield SftSample(
                    factors[start:end].copy(), numeric[start:end].astype(np.float32),
                    np.unpackbits(legal[row], bitorder="little", count=241).astype(np.bool_), int(actions[row]),
                    float(values[row]), np.zeros((0, 10), dtype=np.uint8),
                    int(years[row]), str(game_ids[row]),
                    int(kyoku_indices[row]), int(seats[row]),
                    int(decision_indices[row]),
                    (np.unpackbits(teachers[row], bitorder="little", count=241).astype(np.bool_)
                     if teachers is not None else np.zeros(241, dtype=np.bool_)),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/tenhou_sft_2024_2025"))
    parser.add_argument("--output", type=Path, default=Path("datasets/tenhou_sft_2024_2025_encoded_10pct_v13"))
    parser.add_argument("--subset-denominator", type=int, default=10)
    parser.add_argument("--subset-remainder", type=int, default=0)
    parser.add_argument(
        "--subset-remainders", type=str, default=None,
        help="comma-separated remainders, e.g. 0,1 for the v13 40%% subset",
    )
    parser.add_argument("--kyokus-per-shard", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8, help="independent tar-shard encoder processes")
    parser.add_argument("--audit-kyokus", type=int, default=10, help="coverage-prioritized kyokus audited before encoding")
    parser.add_argument("--skip-audit", action="store_true", help="local diagnostics only; never use for production caches")
    parser.add_argument("--progress-every-kyokus", type=int, default=32, help="worker progress update interval")
    parser.add_argument("--game-sample-denominator", type=int, default=1)
    parser.add_argument("--game-sample-remainder", type=int, default=0)
    parser.add_argument(
        "--require-complete-action-coverage", action="store_true",
        help="fail unless every legal and expert action group is represented",
    )
    parser.add_argument(
        "--require-identity-contract", action="store_true",
        help="hash the complete identity sequence and chunk layout for auditability",
    )
    args = parser.parse_args()
    selected_remainders = (
        tuple(int(value) for value in args.subset_remainders.split(","))
        if args.subset_remainders else None
    )
    precompute(
        args.source, args.output, denominator=args.subset_denominator,
        remainder=args.subset_remainder, remainders=selected_remainders,
        kyokus_per_shard=args.kyokus_per_shard, workers=args.workers,
        audit_kyokus=args.audit_kyokus, skip_audit=args.skip_audit,
        progress_every_kyokus=args.progress_every_kyokus,
        game_sample_denominator=args.game_sample_denominator,
        game_sample_remainder=args.game_sample_remainder,
        require_complete_action_coverage=args.require_complete_action_coverage,
        require_identity_contract=args.require_identity_contract,
    )


if __name__ == "__main__":
    main()
