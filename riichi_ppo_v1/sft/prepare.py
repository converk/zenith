"""Prepare yearly tenhou-to-mjai archives as replayable kyoku tar shards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import tarfile
from typing import Any, Iterable
from urllib.request import Request, urlopen
import zipfile


DATASET_URLS = {
    2024: "https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/v2.0.0/2024.zip",
    2025: "https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/v2.0.0/2025.zip",
}


def stable_split(game_id: str, validation_percent: int = 1) -> str:
    bucket = int.from_bytes(hashlib.sha256(game_id.encode("utf-8")).digest()[:8], "big") % 100
    return "validation" if bucket < validation_percent else "train"


def _json_lines(events: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n" for event in events)


def split_game_events(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    start_game = next((event for event in events if event.get("type") == "start_game"), {"type": "start_game"})
    result: list[list[dict[str, Any]]] = []
    active: list[dict[str, Any]] | None = None
    for event in events:
        kind = event.get("type")
        if kind == "start_kyoku":
            if active is not None:
                raise ValueError("nested start_kyoku")
            tehais = event.get("tehais")
            if not isinstance(tehais, list) or len(tehais) != 4:
                raise ValueError("SFT dataset requires four-player start_kyoku tehais")
            active = [start_game, event]
        elif active is not None:
            active.append(event)
            if kind == "end_kyoku":
                active.append({"type": "end_game"})
                result.append(active)
                active = None
    if active is not None:
        active.extend(({"type": "end_kyoku"}, {"type": "end_game"}))
        result.append(active)
    if not result:
        raise ValueError("game contains no kyoku")
    return result


def _decode_member(payload: bytes) -> str:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return payload.decode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_directory_is_readable(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            archive.infolist()
        return True
    except (OSError, zipfile.BadZipFile):
        return False


def download_archive(year: int, destination: Path, *, attempts: int = 5) -> Path:
    """Download a release asset resumably and expose it only after ZIP validation."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{year}.zip"
    if target.exists() and _zip_directory_is_readable(target):
        return target
    partial = target.with_suffix(".zip.part")
    if target.exists():
        if not partial.exists() or target.stat().st_size > partial.stat().st_size:
            os.replace(target, partial)
        else:
            target.unlink()
    last_error = "download did not start"
    for _attempt in range(max(1, attempts)):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        request = Request(DATASET_URLS[year], headers=headers)
        try:
            with urlopen(request) as response:
                status = int(getattr(response, "status", response.getcode()))
                append = bool(existing and status == 206)
                mode = "ab" if append else "wb"
                content_range = response.headers.get("Content-Range", "")
                if append and "/" in content_range:
                    expected = int(content_range.rsplit("/", 1)[1])
                else:
                    length = response.headers.get("Content-Length")
                    expected = (existing if append else 0) + int(length) if length else None
                with partial.open(mode) as output:
                    while chunk := response.read(8 << 20):
                        output.write(chunk)
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        actual = partial.stat().st_size
        if expected is not None and actual != expected:
            last_error = f"truncated response: expected {expected} bytes, received {actual}"
            continue
        if not _zip_directory_is_readable(partial):
            last_error = f"downloaded {actual} bytes but ZIP central directory is invalid"
            continue
        os.replace(partial, target)
        return target
    raise RuntimeError(f"failed to download {year}.zip after {attempts} attempts: {last_error}")


@dataclass
class _ShardWriter:
    root: Path
    split: str
    shard_size: int
    shard_index: int = 0
    count: int = 0
    archive: tarfile.TarFile | None = None

    def _ensure_open(self) -> None:
        if self.archive is not None and self.count < self.shard_size:
            return
        if self.archive is not None:
            self.archive.close()
            self.shard_index += 1
            self.count = 0
        directory = self.root / self.split
        directory.mkdir(parents=True, exist_ok=True)
        self.archive = tarfile.open(directory / f"{self.split}-{self.shard_index:05d}.tar", "w")

    def add(self, name: str, payload: bytes) -> str:
        self._ensure_open()
        assert self.archive is not None
        member = f"{name}.mjson"
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        info.mtime = 0
        self.archive.addfile(info, io.BytesIO(payload))
        self.count += 1
        return f"{self.split}/{self.split}-{self.shard_index:05d}.tar:{member}"

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()
            self.archive = None


def _prepare_member(task: tuple[int, str, bytes]) -> dict[str, Any]:
    """CPU-heavy validation/replay work executed in an ordered worker pool."""
    year, member, raw = task
    game_id = Path(member).stem
    try:
        from riichienv import MjaiReplay

        content = _decode_member(raw)
        events = [json.loads(line) for line in content.splitlines() if line.strip()]
        kyoku_events = split_game_events(events)
        replay = MjaiReplay.from_jsonl_string(content, rule="tenhou")
        replay_kyokus = list(replay.take_kyokus())
        if len(replay_kyokus) != len(kyoku_events):
            raise ValueError("event split and RiichiEnv replay kyoku counts differ")
        kyokus: list[dict[str, Any]] = []
        for kyoku_index, (rows, replay_kyoku) in enumerate(
            zip(kyoku_events, replay_kyokus, strict=True)
        ):
            features = replay_kyoku.grp_features()
            deltas = [int(value) for value in features["delta_scores"]]
            if len(deltas) != 4:
                raise ValueError("kyoku does not contain four score deltas")
            decision_count = sum(
                1 for _ in replay_kyoku.steps(seat=None, skip_single_action=False)
            )
            kyokus.append({
                "kyoku_index": kyoku_index,
                "payload": gzip.compress(_json_lines(rows).encode("utf-8"), mtime=0),
                "point_deltas": deltas,
                "decision_count": decision_count,
            })
        return {
            "year": year,
            "member": member,
            "game_id": game_id,
            "kyokus": kyokus,
        }
    except Exception as exc:
        return {
            "year": year,
            "member": member,
            "game_id": game_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


def prepare_archives(
    archives: dict[int, Path],
    output: Path,
    *,
    shard_size: int = 4096,
    validation_percent: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    writers = {
        split: _ShardWriter(output, split, shard_size)
        for split in ("train", "validation")
    }
    counts = {
        "games": 0,
        "kyokus": 0,
        "decisions": 0,
        "train": 0,
        "validation": 0,
        "train_decisions": 0,
        "validation_decisions": 0,
        "errors": 0,
    }
    index_path = output / "index.jsonl.gz"
    errors_path = output / "errors.jsonl"
    archive_meta: dict[str, Any] = {}
    try:
        with gzip.open(index_path, "wt", encoding="utf-8") as index_file, errors_path.open(
            "w", encoding="utf-8"
        ) as error_file:
            for year, archive_path in sorted(archives.items()):
                archive_meta[str(year)] = {
                    "path": str(archive_path),
                    "sha256": _sha256(archive_path),
                    "source": DATASET_URLS.get(year),
                }
                with zipfile.ZipFile(archive_path) as source:
                    members = sorted(
                        name for name in source.namelist()
                        if name.lower().endswith(".mjson") and not name.endswith("/")
                    )
                    tasks = (
                        (year, member, source.read(member))
                        for member in members
                    )
                    if workers > 1:
                        pool = multiprocessing.get_context("fork").Pool(processes=workers)
                        prepared = pool.imap(_prepare_member, tasks, chunksize=4)
                    else:
                        pool = None
                        prepared = map(_prepare_member, tasks)
                    try:
                        for result in prepared:
                            if "error" in result:
                                counts["errors"] += 1
                                error_file.write(json.dumps({
                                    "year": result["year"],
                                    "member": result["member"],
                                    "error": result["error"],
                                }, ensure_ascii=False) + "\n")
                                continue
                            game_id = str(result["game_id"])
                            split = stable_split(game_id, validation_percent)
                            for kyoku in result["kyokus"]:
                                kyoku_index = int(kyoku["kyoku_index"])
                                decision_count = int(kyoku["decision_count"])
                                record_id = f"{year}-{game_id}-{kyoku_index:02d}"
                                location = writers[split].add(record_id, kyoku["payload"])
                                index_file.write(json.dumps({
                                    "id": record_id,
                                    "year": year,
                                    "game_id": game_id,
                                    "kyoku_index": kyoku_index,
                                    "split": split,
                                    "location": location,
                                    "point_deltas": kyoku["point_deltas"],
                                    "decision_count": decision_count,
                                }, separators=(",", ":")) + "\n")
                                counts["kyokus"] += 1
                                counts[split] += 1
                                counts["decisions"] += decision_count
                                counts[f"{split}_decisions"] += decision_count
                            counts["games"] += 1
                    finally:
                        if pool is not None:
                            pool.close()
                            pool.join()
    finally:
        for writer in writers.values():
            writer.close()
    manifest = {
        "format": "riichi-sft-kyoku-v1",
        "years": sorted(archives),
        "validation_percent": validation_percent,
        "shard_size": shard_size,
        "workers": workers,
        "counts": counts,
        "archives": archive_meta,
        "license": "CC-BY-4.0; source: NikkeTryHard/tenhou-to-mjai",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构建 riichi-sft-prepare 命令行解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        required=True,
        help=(
            "天凤年度归档目录(必填);`datasets/tenhou-to-mjai` 已作为废弃中间产物"
            "被决策删除,不得再作为默认路径"
        ),
    )
    parser.add_argument("--archive-2024", type=Path)
    parser.add_argument("--archive-2025", type=Path)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--validation-percent", type=int, default=1)
    parser.add_argument(
        "--workers", type=int, default=min(24, os.cpu_count() or 1),
        help="ordered CPU replay workers (default: up to 24 physical-core-oriented workers)",
    )
    parser.add_argument("--no-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    archives: dict[int, Path] = {}
    for year in (2024, 2025):
        supplied = getattr(args, f"archive_{year}")
        if supplied is not None:
            archives[year] = supplied
        elif args.no_download:
            archives[year] = args.archive_dir / f"{year}.zip"
        else:
            archives[year] = download_archive(year, args.archive_dir)
        if not archives[year].is_file():
            raise FileNotFoundError(archives[year])
    manifest = prepare_archives(
        archives,
        args.output,
        shard_size=args.shard_size,
        validation_percent=args.validation_percent,
        workers=max(1, args.workers),
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
