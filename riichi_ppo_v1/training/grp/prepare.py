"""V17 GRP 数据集构造(Mortal 方案):7 维 StartKyoku 状态、24 类排列标签。

从 `datasets/tenhou_sft_2024_2025` 的 tar 半庄记录构造
`datasets/tenhou_grp_2024_2025_v17`(40% 划分与 train/validation 比例沿用 SFT);
每个半庄生成 1 条 7 维全局状态序列 ``[grand_kyoku, honba, kyotaku,
s0/1e4, s1/1e4, s2/1e4, s3/1e4]``;``iter_grp_samples`` 把每个半庄的所有
prefix 展开为训练样本(全部监督该半庄的最终排列),不做 4 视角旋转。
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

from ...model.grp import GRP_INPUT_SIZE, GRP_UTILITY
from ...sft.data import _member_metadata

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


def rank_by_player(scores: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """player → 最终顺位(0..3),同分按座位号稳定。"""
    return tuple(rank_among(seat, scores) for seat in range(4))


def grand_kyoku(round_wind: int, kyoku_index: int) -> int:
    """Mortal 的 ``grand_kyoku``:E1=0..E4=3、S1=4..S4=7。"""
    return min(max(int(round_wind), 0), 1) * 4 + min(max(int(kyoku_index), 0), 3)


def features_from_boundaries(boundaries: list[Boundary]) -> np.ndarray:
    """把绝对座位边界序列编码为 [T, 7] float32 全局特征(Mortal 输入契约)。

    每行 ``[grand_kyoku, honba, kyotaku, s0/1e4, s1/1e4, s2/1e4, s3/1e4]``。
    """
    features = np.zeros((len(boundaries), GRP_INPUT_SIZE), dtype=np.float32)
    for row, boundary in enumerate(boundaries):
        features[row, 0] = float(grand_kyoku(boundary.round_wind, boundary.kyoku_index))
        features[row, 1] = float(boundary.honba)
        features[row, 2] = float(boundary.sticks)
        features[row, 3:7] = np.asarray(boundary.scores, dtype=np.float32) / 10_000.0
    return features


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
    max_shards: int | None = None,
) -> dict:
    """按 game_id 聚合完整半庄后构造 v17 GRP 数据集并写 dataset.json。

    每个半庄生成 1 条全局特征序列与 rank_by_player;训练时将 prefix 展开。
    ``max_shards`` 限制每个 split 最多处理的 tar shard 数(默认全部),用于
    控制数据集规模(如 20w prefix 样本)。
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    counts = {"train_hanchans": 0, "validation_hanchans": 0}
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
            features = features_from_boundaries(boundaries)
            buffer.append({
                "features": features,
                "rank_by_player": np.asarray(rank_by_player(finals), dtype=np.uint8),
                "year": ordered[0][0],
                "game_id": game_id,
            })
            counts[f"{split}_hanchans"] += 1
            if len(buffer) >= kyokus_per_shard:
                _write_chunk(destination / f"{split}-{chunk_index:05d}.npz", buffer)
                buffer.clear()
                chunk_index += 1

        shards = sorted((source / split).glob(f"{split}-*.tar"))
        if max_shards is not None and max_shards > 0:
            shards = shards[: max_shards]
        for shard in shards:
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
    dataset = {
        "format": "riichi-grp-v17",
        "input_size": GRP_INPUT_SIZE,
        "num_classes": 24,
        "utility": list(GRP_UTILITY),
        "subsample": {"denominator": denominator, "remainders": list(remainders)},
        "max_shards": max_shards,
        "counts": counts,
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dataset, ensure_ascii=False), flush=True)
    return dataset


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_chunk(path: Path, samples: list[dict]) -> None:
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(sample["features"]) for sample in samples], dtype=np.int64)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        offsets=offsets,
        features=np.concatenate([sample["features"] for sample in samples], axis=0).astype(np.float32),
        rank_by_player=np.stack([sample["rank_by_player"] for sample in samples], axis=0),
        years=np.asarray([sample["year"] for sample in samples], dtype=np.int16),
        game_ids=np.asarray([sample["game_id"] for sample in samples], dtype=np.str_),
    )
    os.replace(temporary, path)


def iter_grp_samples(dataset: Path, split: str):
    """读取 GRP chunk,产出每个半庄每个 prefix 的训练样本。

    ``yield (features[:k], rank_by_player)``:``features[:k]`` 为 (k,7) float32
    前缀,``rank_by_player`` 为该半庄最终 (玩家→顺位) uint8 (4,)。全部 prefix
    监督同一最终排列标签(与 Mortal ``GrpFileDatasetsIter`` 一致)。
    """
    for path in sorted((dataset / split).glob(f"{split}-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            offsets = data["offsets"]
            features = data["features"]
            rank_by_player = data["rank_by_player"]
            for row in range(len(rank_by_player)):
                start, end = int(offsets[row]), int(offsets[row + 1])
                sequence = features[start:end]
                ranks = rank_by_player[row]
                for prefix_len in range(1, len(sequence) + 1):
                    yield sequence[:prefix_len].copy(), ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/tenhou_sft_2024_2025"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-denominator", type=int, default=5)
    parser.add_argument("--subset-remainders", type=str, default="0,1")
    parser.add_argument("--kyokus-per-shard", type=int, default=512)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()
    remainders = tuple(int(value) for value in args.subset_remainders.split(","))
    prepare_grp_dataset(
        args.source, args.output,
        denominator=args.subset_denominator, remainders=remainders,
        kyokus_per_shard=args.kyokus_per_shard,
        max_shards=args.max_shards,
    )


if __name__ == "__main__":
    main()
