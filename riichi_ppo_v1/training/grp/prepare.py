"""GRP 数据集构造(Mortal 方案扩展):21 维边界状态、24 类排列标签。

从 `datasets/tenhou_sft_2024_2025` 的 tar 半庄记录构造
`datasets/tenhou_grp_2024_2025_v18`(沿用 60/40 无重叠惯例:40% 子集,
train/validation 划分沿用原始数据);每个半庄生成 1 条 21 维全局状态序列,
字段布局见 ``model.grp.GRP_INPUT_LAYOUT``(新增:局风类型、上一小局结果类型、
各玩家累计和了/放铳/听牌流局次数,全部只来自公开小局结果);``iter_grp_samples``
把每个半庄的所有 prefix 展开为训练样本(全部监督该半庄的最终排列),不做
4 视角旋转。tar shard 按 ``workers`` 个进程并行解析(默认 6),记录按 shard
顺序拼接,输出与串行处理一致;shard 边界切断半庄时先预扫描分组,组内跨
tar 聚合,保证每场半庄完整且只产出一条记录。

离线与在线(PPO 边界推理)共用 ``feature_row`` 纯函数,累计计数由调用方按
边界链推进,保证两条路径特征逐位一致。
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...model.grp import (
    GAME_TYPE_EAST,
    GAME_TYPE_HALF,
    GAME_TYPE_WEST,
    GRP_INPUT_LAYOUT,
    GRP_INPUT_SIZE,
    GRP_UTILITY,
    PREV_RESULT_NONE,
    PREV_RESULT_RON,
    PREV_RESULT_RYUKYOKU,
    PREV_RESULT_TSUMO,
)
from ...sft.data import _member_metadata

# 局风名称 → 索引(离线 bakaze 与在线 game_mode 共用同一语义)。
_WIND_INDEX = {"E": 0, "S": 1, "W": 2, "N": 3}
_MODE_GAME_TYPE = {
    "single": GAME_TYPE_EAST,
    "east": GAME_TYPE_EAST,
    "half": GAME_TYPE_HALF,
    "west": GAME_TYPE_WEST,
}
_GAME_TYPE_NAMES = ("east", "half", "west")


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


def game_type_from_content(content: str) -> int:
    """由整局 ``start_kyoku`` 的 bakaze 集合推导局风类型(0=东、1=半庄、2=西)。

    东风只经历 E 风、半庄经历 E+S、西风经历 E+S+W;取最大风索引即类型。
    """
    maximum = -1
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if event.get("type") == "start_kyoku":
            maximum = max(maximum, _WIND_INDEX.get(str(event.get("bakaze", "")), -1))
    if maximum < 0:
        raise ValueError("GRP record contains no start_kyoku event")
    return maximum


def game_type_from_mode(game_mode: str) -> int:
    """由运行时 ``game_mode`` 字符串(如 ``4p-red-half``)映射局风类型。

    未知模式 fail-closed;单局(``single``)与东闘同为东风类。
    """
    suffix = str(game_mode).rsplit("-", 1)[-1].lower()
    if suffix not in _MODE_GAME_TYPE:
        raise ValueError(f"unsupported game_mode for GRP game_type: {game_mode!r}")
    return _MODE_GAME_TYPE[suffix]


def result_increment(
    result: KyokuResult | None,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    """一个小局结果对累计计数的一次推进:``(wins, dealins, tenpai)`` 各 4 位。

    荣和:winner 和了 +1、deal_in 放铳 +1;自摸:仅 winner 和了 +1;流局:
    tenpai_mask 中每位玩家听牌流局 +1;无结果(首局/中止)全 0。
    """
    wins = [0, 0, 0, 0]
    dealins = [0, 0, 0, 0]
    tenpai = [0, 0, 0, 0]
    if result is None:
        return tuple(wins), tuple(dealins), tuple(tenpai)
    if result.result_type in (PREV_RESULT_RON, PREV_RESULT_TSUMO):
        if result.winner is not None and 0 <= int(result.winner) < 4:
            wins[int(result.winner)] += 1
        if result.result_type == PREV_RESULT_RON and result.deal_in is not None:
            if 0 <= int(result.deal_in) < 4:
                dealins[int(result.deal_in)] += 1
    elif result.result_type == PREV_RESULT_RYUKYOKU:
        for player in range(4):
            if int(result.tenpai_mask) & (1 << player):
                tenpai[player] += 1
    return tuple(wins), tuple(dealins), tuple(tenpai)


def feature_row(
    boundary: Boundary,
    game_type: int,
    wins: tuple[int, int, int, int],
    dealins: tuple[int, int, int, int],
    tenpai: tuple[int, int, int, int],
) -> np.ndarray:
    """把单个边界编码为 [21] float32 特征行(GRP 输入契约)。

    ``[grand_kyoku, honba, kyotaku, s0..s3/1e4, game_type, prev_result_type,
    wins0..3, dealins0..3, tenpai0..3]``;累计计数为截至本小局开始的值。
    """
    row = np.zeros(GRP_INPUT_SIZE, dtype=np.float32)
    row[0] = float(grand_kyoku(boundary.round_wind, boundary.kyoku_index))
    row[1] = float(boundary.honba)
    row[2] = float(boundary.sticks)
    row[3:7] = np.asarray(boundary.scores, dtype=np.float32) / 10_000.0
    row[7] = float(int(game_type))
    previous = boundary.previous
    row[8] = float(PREV_RESULT_NONE if previous is None else int(previous.result_type))
    row[9:13] = np.asarray(wins, dtype=np.float32)
    row[13:17] = np.asarray(dealins, dtype=np.float32)
    row[17:21] = np.asarray(tenpai, dtype=np.float32)
    return row


def features_from_boundaries(
    boundaries: list[Boundary], game_type: int,
) -> np.ndarray:
    """把绝对座位边界序列编码为 [T, 21] float32 全局特征(GRP 输入契约)。

    累计计数按边界链推进:第 k 行反映截至第 k 小局开始(即前 k 个小局结果)
    的累计值;首局行全 0。
    """
    rows: list[np.ndarray] = []
    wins = [0, 0, 0, 0]
    dealins = [0, 0, 0, 0]
    tenpai = [0, 0, 0, 0]
    for index, boundary in enumerate(boundaries):
        rows.append(feature_row(
            boundary, game_type,
            tuple(wins), tuple(dealins), tuple(tenpai),
        ))
        if index + 1 < len(boundaries):
            w, d, t = result_increment(boundaries[index + 1].previous)
            wins = [wins[seat] + w[seat] for seat in range(4)]
            dealins = [dealins[seat] + d[seat] for seat in range(4)]
            tenpai = [tenpai[seat] + t[seat] for seat in range(4)]
    return np.asarray(rows, dtype=np.float32)


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
            result_type = PREV_RESULT_RON if deal_in is not None else PREV_RESULT_TSUMO
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
                result_type=PREV_RESULT_RYUKYOKU,
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


def _scan_shard(shard_text: str) -> set[str]:
    """只读 tar 头(不解压成员数据),返回 shard 内全部 game_id 集合。

    用于预构建"连续 shard 分组":shard 边界可能切断一场半庄(同 game_id 的
    成员分布在相邻两个 tar 的头尾),分组后每组内跨 tar 聚合。
    """
    game_ids: set[str] = set()
    with tarfile.open(Path(shard_text), "r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            game_ids.add(_member_metadata(member.name)[1])
    return game_ids


def _shard_groups(shard_paths: list[Path], scan_results: list[set[str]]) -> list[list[Path]]:
    """按 game_id 归属把 shard 合并为连续分组(跨 shard 的半庄不被打散)。

    同一 game_id 出现在多个 shard 时对这两个 shard 做并查集合并;由于原始
    数据按顺序分片,越界的半庄只涉及相邻两个 shard,分组即为连续区间。
    """
    parent = list(range(len(shard_paths)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    first_seen: dict[str, int] = {}
    for index, game_ids in enumerate(scan_results):
        for game_id in game_ids:
            previous = first_seen.get(game_id)
            if previous is None:
                first_seen[game_id] = index
            else:
                union(previous, index)
    merged: dict[int, list[Path]] = {}
    for index, path in enumerate(shard_paths):
        merged.setdefault(find(index), []).append(path)
    return [sorted(group) for _, group in sorted(merged.items())]


def _process_group(
    task: tuple[tuple[str, ...], str, int, tuple[int, ...]],
) -> tuple[list[dict], set[str]]:
    """一组连续 tar shard 的解析任务(顶层函数,供进程池 picklable)。

    组内跨 tar 聚合同一 game_id 的全部选中小局(与串行逐 shard 处理一致,
    半庄被 shard 边界切断时也能完整合并),返回 ``(records, game_ids)``。
    """
    shards_text, _split, denominator, remainders = task
    game_buffer: dict[str, list[tuple[int, int, str]]] = {}
    for shard_text in shards_text:
        shard = Path(shard_text)
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if not member.isfile() or not _selected(member.name, denominator, remainders):
                    continue
                file = archive.extractfile(member)
                if file is None:
                    raise RuntimeError(f"cannot read {shard}:{member.name}")
                year, game_id, kyoku_index = _member_metadata(member.name)
                game_buffer.setdefault(game_id, []).append((
                    year, kyoku_index, _read_member(file.read()),
                ))
    records: list[dict] = []
    for game_id, members in game_buffer.items():
        ordered = sorted(members, key=lambda item: item[1])
        combined = "\n".join(content for _year, _index, content in ordered)
        boundaries = parse_hanchan(combined)
        finals = final_scores(boundaries)
        game_type = game_type_from_content(combined)
        features = features_from_boundaries(boundaries, game_type)
        records.append({
            "features": features,
            "rank_by_player": np.asarray(rank_by_player(finals), dtype=np.uint8),
            "year": ordered[0][0],
            "game_id": game_id,
            "game_type": game_type,
        })
    return records, set(game_buffer)


def prepare_grp_dataset(
    source: Path,
    output: Path,
    *,
    denominator: int = 5,
    remainders: tuple[int, ...] = (0, 1),
    kyokus_per_shard: int = 512,
    max_shards: int | None = None,
    workers: int = 6,
) -> dict:
    """按 game_id 聚合完整半庄后构造 GRP 数据集并写 dataset.json。

    每个半庄生成 1 条 21 维全局特征序列与 rank_by_player;训练时将 prefix 展开。
    ``max_shards`` 限制每个 split 最多处理的 tar shard 数(默认全部),
    用于控制数据集规模。``workers`` 个进程(默认 6)并行解析,输出与串行
    处理逐位一致。shard 边界可能切断一场半庄(同 game_id 成员分布在相邻
    tar 的头尾),先做一次只读 tar 头的预扫描,把相邻 shard 合并为分组,
    组内跨 tar 聚合,保证每场半庄完整且只产出一条记录。
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    counts = {"train_hanchans": 0, "validation_hanchans": 0}
    game_types = {"east": 0, "half": 0, "west": 0}
    worker_count = max(1, int(workers))
    for split in ("train", "validation"):
        destination = output / split
        destination.mkdir()
        buffer: list[dict] = []
        chunk_index = 0
        owners: dict[str, str] = {}

        def flush_buffer() -> None:
            nonlocal chunk_index
            if not buffer:
                return
            _write_chunk(destination / f"{split}-{chunk_index:05d}.npz", buffer)
            buffer.clear()
            chunk_index += 1

        shards = sorted((source / split).glob(f"{split}-*.tar"))
        if max_shards is not None and max_shards > 0:
            shards = shards[: max_shards]
        shard_total = len(shards)
        started = time.monotonic()

        def absorb(records: list[dict], game_ids: set[str]) -> None:
            for game_id in game_ids:
                if game_id in owners:
                    raise RuntimeError(
                        f"game {game_id!r} spans multiple groups; "
                        f"shard grouping failed to keep games whole"
                    )
                owners[game_id] = split
            for record in records:
                buffer.append(record)
                counts[f"{split}_hanchans"] += 1
                game_types[_GAME_TYPE_NAMES[record["game_type"]]] += 1
                if len(buffer) >= kyokus_per_shard:
                    flush_buffer()

        processed_groups = 0

        def consume(records: list[dict], game_ids: set[str]) -> None:
            nonlocal processed_groups
            absorb(records, game_ids)
            processed_groups += 1
            if processed_groups % 25 == 0 or processed_groups == group_total:
                print(
                    f"prepare[{split}]: 分组 {processed_groups}/{group_total} "
                    f"({time.monotonic() - started:.0f}s elapsed)",
                    flush=True,
                )

        print(
            f"prepare[{split}]: {shard_total} shards, workers={worker_count}, "
            f"subset={denominator}/{list(remainders)}"
            + (f" (max_shards={max_shards})" if max_shards else ""),
            flush=True,
        )

        def perform_scan() -> list[set[str]]:
            """"只读 tar 头预扫描:返回每个 shard 的 game_id 集合(缩进并行)。"""
            scan_tasks = [str(shard) for shard in shards]
            if worker_count == 1:
                return [_scan_shard(task) for task in scan_tasks]
            # spawn 上下文避免 fork 多线程进程的锁继承风险。
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(processes=worker_count)
            try:
                return list(pool.imap(_scan_shard, scan_tasks))
            finally:
                pool.close()
                pool.join()

        groups = _shard_groups(shards, perform_scan())
        group_total = len(groups)
        processed_groups = 0
        print(
            f"prepare[{split}]: 预扫描完成,{shard_total} shards → "
            f"{group_total} 个分组(半庄跨 shard 时合并)",
            flush=True,
        )
        group_tasks = [
            (tuple(str(path) for path in group), split, int(denominator), remainders)
            for group in groups
        ]
        if worker_count == 1:
            for records, game_ids in map(_process_group, group_tasks):
                consume(records, game_ids)
        else:
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(processes=worker_count)
            try:
                for records, game_ids in pool.imap(_process_group, group_tasks):
                    consume(records, game_ids)
            finally:
                pool.close()
                pool.join()
        flush_buffer()
    dataset = {
        "format": "riichi-grp-v18",
        "input_size": GRP_INPUT_SIZE,
        "feature_layout": list(GRP_INPUT_LAYOUT),
        "num_classes": 24,
        "utility": list(GRP_UTILITY),
        "subsample": {"denominator": denominator, "remainders": list(remainders)},
        "max_shards": max_shards,
        "counts": counts,
        "game_types": game_types,
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dataset, ensure_ascii=False), flush=True)
    return dataset


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

    ``yield (features[:k], rank_by_player)``:``features[:k]`` 为 (k,21) float32
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
    parser.add_argument("--workers", type=int, default=6,
                        help="并行解析进程数(默认 6)")
    args = parser.parse_args()
    remainders = tuple(int(value) for value in args.subset_remainders.split(","))
    prepare_grp_dataset(
        args.source, args.output,
        denominator=args.subset_denominator, remainders=remainders,
        kyokus_per_shard=args.kyokus_per_shard,
        max_shards=args.max_shards,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
