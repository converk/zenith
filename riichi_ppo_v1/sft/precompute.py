"""物化确定性的 V18 当前局面 actor-only SFT 子集以加速训练。"""

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
import shutil
import tarfile
import time
from typing import Any, Iterator
import zipfile

import numpy as np

from ..model.encoding_protocol import (
    CONTEXT_TOKENS,
    DEFENSE_SLOT_ORDER,
    ENCODED_FORMAT,
    ENCODING_PROTOCOL_VERSION,
    OFFENSE_SLOT_ORDER,
    QUERY_ROW_ANSWER_START,
    SLOT_CARDINALITIES,
    STATE_PROTOCOL_VERSION,
    SEGMENT_CRITIC_FUTURE,
    SEGMENT_CRITIC_PRIVATE,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
)
from ..model.schema import NUM_ACTIONS
from .contract import (
    ACTOR_INPUT_CONTRACT_SHA256,
    validate_manifest,
)
from .data import EncodedSample, _member_metadata, encode_kyoku


def _selection_bucket(game_id: str, namespace: str, denominator: int) -> int:
    """按 game id 与命名空间稳定取模,不受 source split 影响。"""
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


def _count_selected_kyokus(
    source: Path, denominator: int, remainders: tuple[int, ...],
    game_sample_denominator: int, game_sample_remainder: int,
) -> tuple[dict[str, int], str]:
    """编码前统计目标子集,同时验证 train/validation 没有 game-id 重叠。"""
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


def _action_coverage(values: np.ndarray) -> dict[str, int]:
    groups = {
        "pass": (0, 1), "discard": (1, 75), "reach": (75, 76), "chi": (76, 133),
        "pon": (133, 170), "daiminkan": (170, 171), "ankan": (171, 205),
        "kakan": (205, 239), "hora": (239, 240), "ryukyoku": (240, NUM_ACTIONS),
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


def _write_chunk(path: Path, samples: list[EncodedSample]) -> int:
    """物化一批 V18 当前局面样本：actor 序列 + query 元数据 + 身份字段。"""
    count = len(samples)
    actor_offsets = np.zeros(count + 1, dtype=np.int64)
    actor_offsets[1:] = np.cumsum([sample.token_length for sample in samples], dtype=np.int64)
    query_offsets = np.zeros(count + 1, dtype=np.int64)
    query_offsets[1:] = np.cumsum([sample.query_rows.shape[0] for sample in samples], dtype=np.int64)
    action_offsets = np.zeros(count + 1, dtype=np.int64)
    action_offsets[1:] = np.cumsum([sample.query_pair_count for sample in samples], dtype=np.int64)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        actor_offsets=actor_offsets,
        actor_factors=np.concatenate([sample.actor_factors for sample in samples], axis=0),
        actor_numeric=np.concatenate([sample.actor_numeric for sample in samples], axis=0).astype(np.float32),
        query_offsets=query_offsets,
        query_rows=np.concatenate([sample.query_rows for sample in samples], axis=0),
        action_offsets=action_offsets,
        action_ids=np.concatenate([sample.action_ids for sample in samples], axis=0),
        legal=np.packbits(np.stack([sample.legal_mask for sample in samples]), axis=1, bitorder="little"),
        actions=np.asarray([sample.action for sample in samples], dtype=np.uint8),
        years=np.asarray([sample.year for sample in samples], dtype=np.int16),
        game_ids=np.asarray([sample.game_id for sample in samples], dtype=np.str_),
        kyoku_indices=np.asarray([sample.kyoku_index for sample in samples], dtype=np.int16),
        seats=np.asarray([sample.seat for sample in samples], dtype=np.uint8),
        decision_indices=np.asarray([sample.decision_index for sample in samples], dtype=np.int32),
    )
    os.replace(temporary, path)
    return count


def _empty_field_statistics() -> dict[str, np.ndarray]:
    return {
        "query_answer_out_of_range": np.zeros(1, dtype=np.int64),
        "actor_field_out_of_range": np.zeros(1, dtype=np.int64),
        "legal_actions": np.zeros(NUM_ACTIONS, dtype=np.int64),
        "expert_actions": np.zeros(NUM_ACTIONS, dtype=np.int64),
    }


def _accumulate_field_statistics(
    target: dict[str, np.ndarray], samples: list[EncodedSample],
) -> None:
    """累计 query answer 越界、actor 行域越界与动作覆盖率。"""
    from ..model.encoding_protocol import CATEGORY_SCHEMAS, is_separator_kind

    for sample in samples:
        rows = sample.actor_factors
        for token in rows:
            kind = int(token[1])
            if is_separator_kind(kind):
                continue
            schema = CATEGORY_SCHEMAS[kind]
            bad = 0
            for index, field in enumerate(schema.discrete):
                value = int(token[2 + index])
                bad += int(not 0 <= value < field.cardinality)
            target["actor_field_out_of_range"][0] += bad
        pairs = sample.query_pair_count
        query = sample.query_rows[: 2 * pairs]
        offense = query[0::2, QUERY_ROW_ANSWER_START:]
        defense = query[1::2, QUERY_ROW_ANSWER_START:]
        for index, slot in enumerate(OFFENSE_SLOT_ORDER):
            codes = offense[:, index]
            target["query_answer_out_of_range"][0] += np.count_nonzero(
                (codes < 0) | (codes >= SLOT_CARDINALITIES[slot])
            )
        for index, slot in enumerate(DEFENSE_SLOT_ORDER):
            codes = defense[:, index]
            target["query_answer_out_of_range"][0] += np.count_nonzero(
                (codes < 0) | (codes >= SLOT_CARDINALITIES[slot])
            )
        target["legal_actions"] += sample.legal_mask.astype(np.int64)
        target["expert_actions"][int(sample.action)] += 1


def _assert_public_actor(samples: list[EncodedSample], source: str) -> None:
    """拒绝 Actor 行中混入 Critic 私有信息。"""
    for sample in samples:
        segments = sample.actor_factors[:, 0].astype(int)
        if np.any(np.isin(segments, (SEGMENT_CRITIC_PRIVATE, SEGMENT_CRITIC_FUTURE))):
            raise RuntimeError(f"replay actor sequence contains hidden information: {source}")


def _precompute_source_shard(
    split: str,
    source_shard: str,
    destination: str,
    denominator: int,
    remainders: tuple[int, ...],
    kyokus_per_shard: int,
    game_sample_denominator: int,
    game_sample_remainder: int,
    chunk_name_suffix: str | None = None,
    progress_queue: Any | None = None,
    progress_every_kyokus: int = 32,
) -> tuple[str, int, int, int, dict[str, list[int]]]:
    """独立进程内编码一个源 tar 为 V18 chunk。"""
    shard = Path(source_shard)
    target_dir = Path(destination)
    buffered: list[EncodedSample] = []
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
            )
            _assert_public_actor(samples, f"{shard}:{member.name}")
            _accumulate_field_statistics(statistics, samples)
            buffered.extend(samples)
            kyokus += 1
            decisions += len(samples)
            report_progress()
            if kyokus % kyokus_per_shard == 0:
                name = f"{shard.stem}-{chunk_index:03d}.npz"
                if chunk_name_suffix:
                    name = f"{shard.stem}-{chunk_name_suffix}-{chunk_index:03d}.npz"
                _write_chunk(target_dir / name, buffered)
                buffered.clear()
                chunk_index += 1
    if buffered:
        name = f"{shard.stem}-{chunk_index:03d}.npz"
        if chunk_name_suffix:
            name = f"{shard.stem}-{chunk_name_suffix}-{chunk_index:03d}.npz"
        _write_chunk(target_dir / name, buffered)
        chunk_index += 1
    report_progress(force=True)
    return split, kyokus, decisions, chunk_index, {
        name: values.tolist() for name, values in statistics.items()
    }


def precompute(
    source: Path,
    output: Path,
    *,
    denominator: int = 5,
    remainders: tuple[int, ...] = (0, 1, 2),
    kyokus_per_shard: int = 256,
    workers: int = 8,
    progress_every_kyokus: int = 32,
    game_sample_denominator: int = 1,
    game_sample_remainder: int = 0,
    require_complete_action_coverage: bool = False,
    base_encoded: Path | None = None,
) -> None:
    """V18 当前局面子集的确定性重编码,可复用已有 V18 编码缓存。"""
    remainders = tuple(sorted(set(remainders)))
    if denominator <= 0 or not remainders or any(not 0 <= value < denominator for value in remainders):
        raise ValueError("subset remainder must be in [0, subset denominator)")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already exists and is non-empty: {output}")
    if base_encoded is not None and base_encoded.resolve() == output.resolve():
        raise ValueError("base_encoded must not be the same directory as output")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if game_sample_denominator <= 0 or not 0 <= game_sample_remainder < game_sample_denominator:
        raise ValueError("game sample remainder must be in [0, game sample denominator)")
    source_manifest_sha256 = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    reused_manifest: dict[str, Any] | None = None
    if base_encoded is not None:
        manifest_path = base_encoded / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"reuse base encoded manifest does not exist: {manifest_path}")
        reused_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(reused_manifest)
        if int(reused_manifest.get("subset_denominator", -1)) != denominator:
            raise ValueError("base_encoded subset_denominator differs from requested denominator")
        if (
            int(reused_manifest.get("game_sample_denominator", 1)) != game_sample_denominator
            or int(reused_manifest.get("game_sample_remainder", 0)) != game_sample_remainder
        ):
            raise ValueError("base_encoded game sample selection differs from requested selection")
        if reused_manifest.get("source_manifest_sha256") != source_manifest_sha256:
            raise ValueError("base_encoded source manifest differs from --source")
        base_remainders = set(reused_manifest.get("subset_remainders", ()))
        remaining = tuple(sorted(set(remainders) - base_remainders))
        if not remaining:
            raise ValueError("base_encoded already covers all requested remainders")
    else:
        remaining = remainders
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"preflight: counting denominator={denominator} remainders={remainders} "
        f"reuse={base_encoded is not None} remaining={remaining}...", flush=True,
    )
    total_kyokus, selection_manifest_sha256 = _count_selected_kyokus(
        source, denominator, remainders, game_sample_denominator, game_sample_remainder,
    )
    total = sum(total_kyokus.values())
    if total == 0:
        raise RuntimeError("target subset contains no kyokus")
    if base_encoded is not None:
        assert reused_manifest is not None
        base_preflight = reused_manifest.get("preflight_target_kyokus")
        if not isinstance(base_preflight, dict):
            raise RuntimeError("base_encoded manifest lacks preflight_target_kyokus")
        remaining_kyokus = {
            split: int(total_kyokus[split]) - int(base_preflight[split])
            for split in ("train", "validation")
        }
        if any(value < 0 for value in remaining_kyokus.values()):
            raise RuntimeError("base_encoded preflight counts exceed requested subset")
        encode_total = sum(remaining_kyokus.values())
    else:
        remaining_kyokus = total_kyokus
        encode_total = total
    print(
        f"preflight: target_kyokus={total} encode_kyokus={encode_total} "
        f"train={remaining_kyokus['train']} validation={remaining_kyokus['validation']}", flush=True,
    )
    field_statistics = _empty_field_statistics()
    counts: dict[str, int] = {
        "train_kyokus": 0, "validation_kyokus": 0,
        "train_decisions": 0, "validation_decisions": 0,
    }
    if base_encoded is not None:
        assert reused_manifest is not None
        base_counts = reused_manifest.get("counts")
        if not isinstance(base_counts, dict):
            raise RuntimeError("base_encoded manifest lacks counts")
        for key in counts:
            if key not in base_counts:
                raise RuntimeError(f"base_encoded manifest lacks counts.{key}")
            counts[key] = int(base_counts[key])
        base_statistics = reused_manifest.get("field_statistics")
        if not isinstance(base_statistics, dict):
            raise RuntimeError("base_encoded manifest lacks field_statistics")
        manifest_key = {
            "legal_actions": "legal_action_id_counts",
            "expert_actions": "expert_action_id_counts",
        }
        for name in field_statistics:
            source_name = manifest_key.get(name, name)
            if source_name not in base_statistics:
                raise RuntimeError(f"base_encoded manifest lacks field_statistics.{source_name}")
            value = base_statistics[source_name]
            if name in {"query_answer_out_of_range", "actor_field_out_of_range"}:
                field_statistics[name] = np.asarray([value], dtype=np.int64)
            else:
                field_statistics[name] = np.asarray(value, dtype=np.int64)
        for split in ("train", "validation"):
            destination = output / split
            destination.mkdir(parents=True, exist_ok=True)
            for path in sorted((base_encoded / split).glob(f"{split}-*.npz")):
                if path.name.endswith(".tmp.npz"):
                    continue
                shutil.copy2(path, destination / path.name)
    tasks: list[tuple[Any, ...]] = []
    for split in ("train", "validation"):
        destination = output / split
        destination.mkdir(exist_ok=True)
        chunk_suffix = None
        if base_encoded is not None:
            chunk_suffix = f"r{'-'.join(str(value) for value in remaining)}"
        for shard in sorted((source / split).glob(f"{split}-*.tar")):
            tasks.append((
                split, str(shard), str(destination), denominator, remaining,
                kyokus_per_shard, game_sample_denominator, game_sample_remainder,
                chunk_suffix,
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
                    remaining_work = encode_total - processed_kyokus
                    rate = processed_kyokus / elapsed if elapsed else 0.0
                    print(
                        f"progress kyokus={processed_kyokus}/{encode_total} remaining={remaining_work} "
                        f"decisions={processed_decisions} rate={rate:.2f} kyokus/s "
                        f"eta={_format_eta(remaining_work / rate if rate else float('inf'))}",
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
    if int(field_statistics["query_answer_out_of_range"][0]) != 0:
        raise RuntimeError("V18 encoding produced out-of-range query answers")
    if int(field_statistics["actor_field_out_of_range"][0]) != 0:
        raise RuntimeError("V18 encoding produced out-of-range actor fields")
    for split in ("train", "validation"):
        if int(counts[f"{split}_kyokus"]) != int(total_kyokus[split]):
            raise RuntimeError(
                f"{split} kyoku count {counts[f'{split}_kyokus']} != "
                f"preflight {total_kyokus[split]}"
            )
    manifest = {
        "format": ENCODED_FORMAT,
        "encoding_protocol_version": ENCODING_PROTOCOL_VERSION,
        "encoding_contract_sha256": ACTOR_INPUT_CONTRACT_SHA256,
        "state_protocol": STATE_PROTOCOL_VERSION,
        "token_row_width": TOKEN_ROW_WIDTH,
        "token_numeric_width": TOKEN_NUMERIC_WIDTH,
        "context_tokens": CONTEXT_TOKENS,
        "source_manifest_sha256": source_manifest_sha256,
        "subset_denominator": denominator,
        "subset_remainders": list(remainders),
        "game_sample_denominator": game_sample_denominator,
        "game_sample_remainder": game_sample_remainder,
        "actor_only": True,
        "numeric_dtype": "float32",
        "legal_encoding": f"packbits-little-{NUM_ACTIONS}",
        "current_state_snapshot_verified": True,
        "complete_action_coverage_required": bool(require_complete_action_coverage),
        "preflight_target_kyokus": total_kyokus,
        "counts": counts,
        "selection_manifest_sha256": selection_manifest_sha256,
        **({"reused_encoded_cache": str(base_encoded), "reused_counts": dict(reused_manifest["counts"])}
           if base_encoded is not None and reused_manifest is not None else {}),
        "field_statistics": {
            "query_answer_out_of_range": int(field_statistics["query_answer_out_of_range"][0]),
            "actor_field_out_of_range": int(field_statistics["actor_field_out_of_range"][0]),
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
) -> Iterator[EncodedSample]:
    """按确定性全局行流读取 V18 当前局面编码 chunk。"""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    paths = sorted((dataset / split).glob(f"{split}-*.npz"))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(paths)
    if not 0 <= int(rank) < int(world_size):
        raise ValueError("rank must be in [0, world_size)")

    def row_count(path: Path) -> int:
        with zipfile.ZipFile(path) as archive, archive.open("actions.npy") as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(member)
            elif version in {(2, 0), (3, 0)}:
                shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(member)
            else:  # pragma: no cover
                raise RuntimeError(f"unsupported NPY header version {version} in {path}")
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
            (actor_offsets, actor_factors, actor_numeric,
             query_offsets, query_rows,
             action_offsets, action_ids,
             legal, actions) = (
                data[name] for name in (
                    "actor_offsets", "actor_factors", "actor_numeric",
                    "query_offsets", "query_rows",
                    "action_offsets", "action_ids",
                    "legal", "actions",
                )
            )
            # fail-closed：三个 offsets 数组必须自身严格单调且与数组长度一致。
            for offsets, array, label in (
                (actor_offsets, actor_factors, "actor"),
                (query_offsets, query_rows, "query"),
                (action_offsets, action_ids, "action"),
            ):
                offsets = np.asarray(offsets)
                if offsets.ndim != 1 or offsets.size < 1 or int(offsets[0]) != 0:
                    raise RuntimeError(f"{label} offsets malformed in {path}: first offset must be 0")
                if np.any(np.diff(offsets.astype(np.int64)) <= 0):
                    raise RuntimeError(f"{label} offsets must be strictly increasing in {path}")
                if int(offsets[-1]) != int(array.shape[0]):
                    raise RuntimeError(
                        f"{label} offsets[-1] {offsets[-1]} != array length {array.shape[0]} in {path}"
                    )
            if int(actor_offsets[-1]) != actor_factors.shape[0]:
                raise RuntimeError(f"actor offsets length mismatch in {path}")
            if int(query_offsets[-1]) != query_rows.shape[0]:
                raise RuntimeError(f"query offsets length mismatch in {path}")
            if int(action_offsets[-1]) != action_ids.shape[0]:
                raise RuntimeError(f"action offsets length mismatch in {path}")
            years, game_ids, kyoku_indices, seats, decision_indices = (
                data[name] for name in (
                    "years", "game_ids", "kyoku_indices", "seats", "decision_indices",
                )
            )
            order = list(range(len(actions)))
            if shuffle:
                random.Random(f"riichi-sft-v18-row-order\0{seed}\0{path.name}").shuffle(order)
            for row in order[selected_start:selected_end]:
                yield EncodedSample(
                    actor_factors[actor_offsets[row]:actor_offsets[row + 1]].copy(),
                    actor_numeric[actor_offsets[row]:actor_offsets[row + 1]].astype(np.float32),
                    query_rows[query_offsets[row]:query_offsets[row + 1]].copy(),
                    action_ids[action_offsets[row]:action_offsets[row + 1]].copy(),
                    np.unpackbits(legal[row], bitorder="little", count=NUM_ACTIONS).astype(np.bool_),
                    int(actions[row]),
                    int(years[row]), str(game_ids[row]),
                    int(kyoku_indices[row]), int(seats[row]),
                    int(decision_indices[row]),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/tenhou_sft_2024_2025"))
    parser.add_argument("--output", type=Path, required=True, help="V18 编码缓存输出目录(必填)")
    parser.add_argument("--subset-denominator", type=int, default=5)
    parser.add_argument("--subset-remainder", type=int, default=None)
    parser.add_argument(
        "--subset-remainders", type=str, default=None,
        help="comma-separated remainders, e.g. 0,1 for a 40%% subset",
    )
    parser.add_argument(
        "--reuse-encoded", type=Path, default=None,
        help="复用已有 V18 编码缓存,只追加本次 remainders 的新数据",
    )
    parser.add_argument("--kyokus-per-shard", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8, help="independent tar-shard encoder processes")
    parser.add_argument("--progress-every-kyokus", type=int, default=32, help="worker progress update interval")
    parser.add_argument("--game-sample-denominator", type=int, default=1)
    parser.add_argument("--game-sample-remainder", type=int, default=0)
    parser.add_argument(
        "--require-complete-action-coverage", action="store_true",
        help="fail unless every legal and expert action group is represented",
    )
    args = parser.parse_args()
    if args.subset_remainders:
        selected_remainders = tuple(int(value) for value in args.subset_remainders.split(","))
    elif args.subset_remainder is not None:
        selected_remainders = (int(args.subset_remainder),)
    else:
        selected_remainders = (0, 1, 2)
    precompute(
        args.source, args.output, denominator=args.subset_denominator,
        remainders=selected_remainders,
        kyokus_per_shard=args.kyokus_per_shard, workers=args.workers,
        progress_every_kyokus=args.progress_every_kyokus,
        game_sample_denominator=args.game_sample_denominator,
        game_sample_remainder=args.game_sample_remainder,
        require_complete_action_coverage=args.require_complete_action_coverage,
        base_encoded=args.reuse_encoded,
    )


if __name__ == "__main__":
    main()
