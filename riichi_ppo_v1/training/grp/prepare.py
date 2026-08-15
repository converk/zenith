"""V16 GRP 数据集构造:4 视角旋转、prefix→最终排名、σ_Score 离线固化。

从 `datasets/tenhou_sft_2024_2025` 的 tar 半庄记录构造
`datasets/tenhou_grp_2024_2025_v16`(40% 划分与 train/validation 比例沿用 SFT),
每个半庄旋转为 4 个 player-relative 视角样本;σ_GRP 在 GRP 训练后由 train.py
写回数据集 JSON。
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...model.grp import GRP_CATEGORIES, GRP_NUMERIC_FEATURES, GRP_UTILITY
from ...sft.data import _member_metadata

SCORE_SCALE = 25_000.0
DELTA_SCALE = 1_000.0
RESULT_CODES = {"ron": 1, "tsumo": 2, "ryukyoku": 3, "abort": 4}


@dataclass(frozen=True)
class KyokuResult:
    result_type: int  # 1=ron, 2=tsumo, 3=ryukyoku, 4=abort
    winner: int | None
    deal_in: int | None
    tenpai_mask: int
    deltas: tuple[int, int, int, int]


@dataclass(frozen=True)
class Boundary:
    round_wind: int  # 0=E, 1=S
    kyoku_index: int  # 0..7(E1..S4)
    dealer: int
    honba: int
    sticks: int
    scores: tuple[int, int, int, int]
    previous: KyokuResult | None  # None = 首局 START


def rank_among(seat: int, scores: tuple[int, int, int, int]) -> int:
    """0-based 顺位:同分按座位号稳定排序。"""
    order = sorted(range(4), key=lambda player: (-scores[player], player))
    return order.index(int(seat))


def encode_view(
    boundaries: list[Boundary],
    viewer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把绝对座位边界序列旋转到统一 viewer 视角。

    返回 categorical [T,9] uint8 与 numeric [T,13] float32;类别字段全部取
    viewer 相对编码,连续字段在编码期完成固定归一化。
    """
    categorical = np.zeros((len(boundaries), len(GRP_CATEGORIES)), dtype=np.uint8)
    numeric = np.zeros((len(boundaries), GRP_NUMERIC_FEATURES), dtype=np.float32)
    for row, boundary in enumerate(boundaries):
        scores = boundary.scores
        relative = [(viewer + offset) % 4 for offset in (0, 1, 2, 3)]
        self_rank = rank_among(viewer, scores)
        pressures = [scores[viewer] - scores[relative[offset]] for offset in (1, 2, 3)]
        if boundary.previous is None:
            result_code, winner_code, deal_code, tenpai_mask, renchan = 0, 0, 0, 0, 0
            deltas = (0, 0, 0, 0)
        else:
            previous = boundary.previous
            result_code = max(0, min(int(previous.result_type), 4))
            winner_code = (
                0 if previous.winner is None else (int(previous.winner) - viewer) % 4 + 1
            )
            deal_code = (
                0 if previous.deal_in is None else (int(previous.deal_in) - viewer) % 4 + 1
            )
            tenpai_mask = int(previous.tenpai_mask) & 0xF
            renchan = int(
                previous.winner is not None and int(previous.winner) == int(boundary.dealer)
            )
            deltas = previous.deltas
        categorical[row] = (
            self_rank,
            min(max(int(boundary.round_wind), 0), 1),
            min(max(int(boundary.kyoku_index), 0), 7),
            (int(boundary.dealer) - viewer) % 4,
            result_code,
            winner_code,
            deal_code,
            tenpai_mask,
            renchan,
        )
        numeric[row, :4] = np.clip(np.asarray(scores, dtype=np.float32) / SCORE_SCALE, -5.0, 5.0)
        numeric[row, 4:7] = np.clip(
            np.asarray(pressures, dtype=np.float32) / SCORE_SCALE, -5.0, 5.0,
        )
        numeric[row, 7] = np.clip(float(boundary.honba) / 10.0, 0.0, 1.0)
        numeric[row, 8] = np.clip(float(boundary.sticks) / 10.0, 0.0, 1.0)
        numeric[row, 9:13] = np.clip(
            np.asarray(deltas, dtype=np.float32) / DELTA_SCALE, -12.0, 12.0,
        )
    return categorical, numeric


def parse_hanchan(content: str) -> list[Boundary]:
    """解析一个 MJAI 半庄 JSONL,得到每个小局开局的边界序列。"""
    events = [json.loads(line) for line in content.splitlines() if line.strip()]
    boundaries: list[Boundary] = []
    scores: tuple[int, int, int, int] = (25000, 25000, 25000, 25000)
    previous: KyokuResult | None = None
    for event in events:
        kind = str(event.get("type", ""))
        if kind == "start_kyoku":
            raw_scores = event.get("scores")
            if isinstance(raw_scores, list) and len(raw_scores) == 4:
                scores = tuple(int(value) for value in raw_scores)
            round_wind = 0 if str(event.get("bakaze", "E")) == "E" else 1
            kyoku_index = min(max(int(event.get("kyoku", 1)) - 1, 0), 7)
            boundaries.append(Boundary(
                round_wind=round_wind,
                kyoku_index=kyoku_index,
                dealer=int(event.get("oya", 0)),
                honba=int(event.get("honba", 0)),
                sticks=int(event.get("kyotaku", 0)),
                scores=scores,
                previous=previous,
            ))
            previous = None
        elif kind == "hora":
            winner = event.get("actor")
            deal_in = event.get("target")
            deltas = _deltas(event)
            result_type = RESULT_CODES["ron"] if deal_in is not None else RESULT_CODES["tsumo"]
            previous = KyokuResult(
                result_type=result_type,
                winner=int(winner) if winner is not None else None,
                deal_in=int(deal_in) if deal_in is not None else None,
                tenpai_mask=0,
                deltas=deltas,
            )
        elif kind == "ryukyoku":
            deltas = _deltas(event)
            tenpais = event.get("tenpais") or []
            mask = 0
            for value in tenpais:
                try:
                    mask |= 1 << int(value)
                except (TypeError, ValueError):
                    pass
            previous = KyokuResult(
                result_type=RESULT_CODES["ryukyoku"],
                winner=None,
                deal_in=None,
                tenpai_mask=mask,
                deltas=deltas,
            )
        elif kind == "end_game":
            raw_scores = event.get("scores")
            if isinstance(raw_scores, list) and len(raw_scores) == 4:
                scores = tuple(int(value) for value in raw_scores)
    if not boundaries:
        raise ValueError("GRP record contains no start_kyoku event")
    return boundaries


def _deltas(event: dict) -> tuple[int, int, int, int]:
    values = event.get("deltas") or event.get("scores_delta") or []
    if len(values) != 4:
        return (0, 0, 0, 0)
    return tuple(int(value) for value in values)


def final_scores(boundaries: list[Boundary]) -> tuple[int, int, int, int]:
    """由最后一局边界分数 + 上一局分差推算半庄终局分数。"""
    last = boundaries[-1]
    if last.previous is not None and last.previous.deltas != (0, 0, 0, 0):
        return tuple(
            int(last.scores[seat]) + int(last.previous.deltas[seat])
            for seat in range(4)
        )
    return last.scores


def _read_member(payload: bytes) -> str:
    return gzip.decompress(payload).decode("utf-8") if payload[:2] == b"\x1f\x8b" else payload.decode("utf-8")


def _selected(member_name: str, denominator: int, remainders: tuple[int, ...]) -> bool:
    from ...sft.precompute import selected_any
    return selected_any(member_name, denominator, remainders)


def prepare_grp_dataset(
    source: Path,
    output: Path,
    *,
    denominator: int = 5,
    remainders: tuple[int, ...] = (0, 1),
    kyokus_per_shard: int = 512,
) -> dict:
    """按 game_id 聚合完整半庄后构造 4 视角 GRP 数据集并固化 σ_Score。"""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    counts = {"train_hanchans": 0, "validation_hanchans": 0, "train_samples": 0, "validation_samples": 0}
    clipped_deltas: list[float] = []
    for split in ("train", "validation"):
        destination = output / split
        destination.mkdir()
        buffer: list[dict] = []
        chunk_index = 0
        game_buffer: dict[str, list[tuple[int, int, str]]] = {}
        current_game_id: str | None = None

        def flush_game(game_id: str, members: list[tuple[int, int, str]]) -> None:
            nonlocal chunk_index
            ordered = sorted(members, key=lambda item: item[1])
            combined = "\n".join(content for _year, _index, content in ordered)
            boundaries = parse_hanchan(combined)
            finals = final_scores(boundaries)
            year = ordered[0][0]
            for viewer in range(4):
                categorical, numeric = encode_view(boundaries, viewer)
                buffer.append({
                    "categorical": categorical,
                    "numeric": numeric,
                    "final_rank": rank_among(viewer, finals),
                    "year": year,
                    "game_id": game_id,
                    "viewer": viewer,
                })
                for boundary in boundaries:
                    if boundary.previous is not None:
                        clipped_deltas.extend(
                            float(np.clip(delta / DELTA_SCALE, -12.0, 12.0))
                            for delta in boundary.previous.deltas
                        )
            counts[f"{split}_hanchans"] += 1
            if len(buffer) >= kyokus_per_shard:
                _write_chunk(destination / f"{split}-{chunk_index:05d}.npz", buffer)
                buffer.clear()
                chunk_index += 1

        for shard in sorted((source / split).glob(f"{split}-*.tar")):
            with tarfile.open(shard, "r") as archive:
                for member in archive:
                    if not member.isfile() or not _selected(member.name, denominator, remainders):
                        continue
                    file = archive.extractfile(member)
                    if file is None:
                        raise RuntimeError(f"cannot read {shard}:{member.name}")
                    year, game_id, kyoku_index = _member_metadata(member.name)
                    if current_game_id is not None and game_id != current_game_id:
                        flush_game(current_game_id, game_buffer.pop(current_game_id, []))
                    current_game_id = game_id
                    game_buffer.setdefault(game_id, []).append((
                        year, kyoku_index, _read_member(file.read()),
                    ))
        if current_game_id is not None and current_game_id in game_buffer:
            flush_game(current_game_id, game_buffer.pop(current_game_id))
        if game_buffer:
            raise RuntimeError(f"{split} 残留未闭合 game 缓冲: {sorted(game_buffer)}")
        if buffer:
            _write_chunk(destination / f"{split}-{chunk_index:05d}.npz", buffer)
        counts[f"{split}_samples"] = counts[f"{split}_hanchans"] * 4
    sigma_score = float(np.std(np.asarray(clipped_deltas, dtype=np.float32))) if clipped_deltas else 1.0
    dataset = {
        "format": "riichi-grp-v16",
        "encoding_protocol_version": 16,
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "subset_denominator": denominator,
        "subset_remainders": list(remainders),
        "utility": list(GRP_UTILITY),
        "counts": counts,
        "normalization": {
            "sigma_score": sigma_score,
            "sigma_grp": None,  # GRP 训练后由 train.py 写回
            "delta_clip": 12.0,
            "reward_clip": 5.0,
        },
        "views": ["SELF", "RIGHT", "ACROSS", "LEFT"],
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dataset, ensure_ascii=False), flush=True)
    return dataset


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_chunk(path: Path, samples: list[dict]) -> None:
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(sample["categorical"]) for sample in samples], dtype=np.int64)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        offsets=offsets,
        categorical=np.concatenate([sample["categorical"] for sample in samples], axis=0),
        numeric=np.concatenate([sample["numeric"] for sample in samples], axis=0).astype(np.float16),
        final_ranks=np.asarray([sample["final_rank"] for sample in samples], dtype=np.uint8),
        years=np.asarray([sample["year"] for sample in samples], dtype=np.int16),
        game_ids=np.asarray([sample["game_id"] for sample in samples], dtype=np.str_),
        viewers=np.asarray([sample["viewer"] for sample in samples], dtype=np.uint8),
    )
    os.replace(temporary, path)


def iter_grp_samples(dataset: Path, split: str):
    """按文件顺序读取 GRP chunk,产出 (categorical, numeric, final_rank)。"""
    for path in sorted((dataset / split).glob(f"{split}-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            offsets = data["offsets"]
            categorical = data["categorical"]
            numeric = data["numeric"].astype(np.float32)
            final_ranks = data["final_ranks"]
            for row in range(len(final_ranks)):
                start, end = int(offsets[row]), int(offsets[row + 1])
                yield (
                    categorical[start:end].copy(),
                    numeric[start:end],
                    int(final_ranks[row]),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/tenhou_sft_2024_2025"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-denominator", type=int, default=5)
    parser.add_argument("--subset-remainders", type=str, default="0,1")
    parser.add_argument("--kyokus-per-shard", type=int, default=512)
    args = parser.parse_args()
    remainders = tuple(int(value) for value in args.subset_remainders.split(","))
    prepare_grp_dataset(
        args.source, args.output,
        denominator=args.subset_denominator, remainders=remainders,
        kyokus_per_shard=args.kyokus_per_shard,
    )


if __name__ == "__main__":
    main()
