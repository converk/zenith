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

import numpy as np

from ..model.schema import TOKEN_SCHEMA_VERSION
from ..training.rewards.efficiency import EfficiencyAnalyzer
from .data import SftSample, _member_metadata, encode_kyoku


ENCODED_FORMAT = "riichi-sft-encoded-v1"


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


def selected(name: str, denominator: int, remainder: int) -> bool:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % denominator == remainder


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
    )
    os.replace(temporary, path)
    return len(samples)


def _count_selected_kyokus(source: Path, denominator: int, remainder: int) -> dict[str, int]:
    """Count the exact target subset before encoding so ETA is meaningful."""
    totals = {"train": 0, "validation": 0}
    for split in totals:
        shards = sorted((source / split).glob(f"{split}-*.tar"))
        for index, shard in enumerate(shards, start=1):
            with tarfile.open(shard, "r") as archive:
                totals[split] += sum(
                    member.isfile() and selected(member.name, denominator, remainder)
                    for member in archive
                )
            if index == len(shards) or index % 25 == 0:
                print(
                    f"preflight split={split} scanned_shards={index}/{len(shards)} "
                    f"selected_kyokus={totals[split]}", flush=True,
                )
    return totals


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _precompute_source_shard(
    split: str,
    source_shard: str,
    destination: str,
    denominator: int,
    remainder: int,
    kyokus_per_shard: int,
    progress_queue: Any | None = None,
    progress_every_kyokus: int = 32,
) -> tuple[str, int, int, int]:
    """Encode one source tar in an isolated process and write unique chunks."""
    shard = Path(source_shard)
    target_dir = Path(destination)
    analyzer = EfficiencyAnalyzer()
    buffered: list[SftSample] = []
    kyokus = decisions = chunk_index = reported_kyokus = reported_decisions = 0

    def report_progress(*, force: bool = False) -> None:
        nonlocal reported_kyokus, reported_decisions
        if progress_queue is None or (not force and kyokus - reported_kyokus < progress_every_kyokus):
            return
        progress_queue.put((split, kyokus - reported_kyokus, decisions - reported_decisions))
        reported_kyokus, reported_decisions = kyokus, decisions

    with tarfile.open(shard, "r") as archive:
        for member in archive:
            if not member.isfile() or not selected(member.name, denominator, remainder):
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
    return split, kyokus, decisions, chunk_index


def precompute(
    source: Path,
    output: Path,
    *,
    denominator: int = 10,
    remainder: int = 0,
    kyokus_per_shard: int = 256,
    workers: int = 8,
    audit_kyokus: int = 10,
    skip_audit: bool = False,
    progress_every_kyokus: int = 32,
) -> None:
    if denominator <= 0 or not 0 <= remainder < denominator:
        raise ValueError("subset remainder must be in [0, subset denominator)")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already exists and is non-empty: {output}")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if progress_every_kyokus <= 0:
        raise ValueError("progress_every_kyokus must be positive")
    audit_report = output.with_name(f"{output.name}-audit.json")
    if not skip_audit:
        from .audit import audit_dataset, write_audit_report
        write_audit_report(
            audit_dataset(source, denominator=denominator, remainder=remainder, sample_size=audit_kyokus),
            audit_report,
        )
    output.mkdir(parents=True, exist_ok=True)
    print("preflight: counting exact target 1/10 subset for progress and ETA...", flush=True)
    total_kyokus = _count_selected_kyokus(source, denominator, remainder)
    total = sum(total_kyokus.values())
    if total == 0:
        raise RuntimeError("target subset contains no kyokus")
    print(
        f"preflight: target_kyokus={total} train={total_kyokus['train']} "
        f"validation={total_kyokus['validation']}", flush=True,
    )
    counts: dict[str, int] = {"train_kyokus": 0, "validation_kyokus": 0, "train_decisions": 0, "validation_decisions": 0}
    tasks: list[tuple[str, str, str, int, int, int]] = []
    for split in ("train", "validation"):
        destination = output / split
        destination.mkdir()
        for shard in sorted((source / split).glob(f"{split}-*.tar")):
            tasks.append((split, str(shard), str(destination), denominator, remainder, kyokus_per_shard))
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
                    split, kyokus, decisions, chunks = future.result()
                    counts[f"{split}_kyokus"] += kyokus
                    counts[f"{split}_decisions"] += decisions
                    completed += 1
                    print(
                        f"completed_shards={completed}/{len(tasks)} split={split} "
                        f"kyokus={kyokus} decisions={decisions} output_chunks={chunks}",
                        flush=True,
                    )
    manifest = {
        "format": ENCODED_FORMAT,
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "subset_denominator": denominator,
        "subset_remainder": remainder,
        "actor_only": True,
        "numeric_dtype": "float16",
        "legal_encoding": "packbits-little-241",
        "ordered_public_history_verified": True,
        "audit_report": str(audit_report),
        "preflight_target_kyokus": total_kyokus,
        "counts": counts,
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
    if manifest.get("format") != ENCODED_FORMAT:
        raise ValueError(f"not an encoded SFT dataset: {dataset}")
    if int(manifest.get("token_schema_version", -1)) != TOKEN_SCHEMA_VERSION:
        raise RuntimeError("encoded SFT token schema is incompatible with this code")
    paths = sorted((dataset / split).glob(f"{split}-*.npz"))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(paths)
    if not 0 <= int(rank) < int(world_size):
        raise ValueError("rank must be in [0, world_size)")
    paths = paths[int(rank)::int(world_size)]
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            offsets, factors, numeric, legal, actions = (data[name] for name in ("offsets", "factors", "numeric", "legal", "actions"))
            order = list(range(len(actions)))
            if shuffle:
                rng.shuffle(order)
            for row in order:
                start, end = int(offsets[row]), int(offsets[row + 1])
                yield SftSample(
                    factors[start:end].copy(), numeric[start:end].astype(np.float32),
                    np.unpackbits(legal[row], bitorder="little", count=241).astype(np.bool_), int(actions[row]),
                    0.0, np.zeros((0, 10), dtype=np.uint8), 0, "", 0, 0,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/tenhou_sft_2024_2025"))
    parser.add_argument("--output", type=Path, default=Path("datasets/tenhou_sft_2024_2025_encoded_10pct_v3"))
    parser.add_argument("--subset-denominator", type=int, default=10)
    parser.add_argument("--subset-remainder", type=int, default=0)
    parser.add_argument("--kyokus-per-shard", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8, help="independent tar-shard encoder processes")
    parser.add_argument("--audit-kyokus", type=int, default=10, help="coverage-prioritized kyokus audited before encoding")
    parser.add_argument("--skip-audit", action="store_true", help="local diagnostics only; never use for production caches")
    parser.add_argument("--progress-every-kyokus", type=int, default=32, help="worker progress update interval")
    args = parser.parse_args()
    precompute(args.source, args.output, denominator=args.subset_denominator, remainder=args.subset_remainder, kyokus_per_shard=args.kyokus_per_shard, workers=args.workers, audit_kyokus=args.audit_kyokus, skip_audit=args.skip_audit, progress_every_kyokus=args.progress_every_kyokus)


if __name__ == "__main__":
    main()
