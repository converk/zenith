"""V16 SFT 数据语义审计:全量结构扫描 + 抽样重编码与存量 chunk 比对。

全量扫描只读取 `datasets/tenhou_sft_2024_2025_encoded_40pct_v16`,不重新编码;
抽样部分从原始 `datasets/tenhou_sft_2024_2025` 按 V16 子集规则选取成员,重跑
`encode_kyoku_v16`,再与同 game_id/kyoku 的存量样本逐字段比对。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import random
import tarfile
from pathlib import Path

import numpy as np

from riichi_ppo_v1.model.encoding_protocol import (
    ACTION_TYPE_CARDINALITY,
    DEFENSE_SLOT_ORDER,
    OFFENSE_SLOT_ORDER,
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_ACTION_TYPE,
    QUERY_ROW_ANSWER_START,
    QUERY_ROW_PRIMARY_TILE,
    QUERY_ROW_QUERY_TYPE,
    QUERY_ROW_SOURCE_SEAT,
    SLOT_CARDINALITIES,
)
from riichi_ppo_v1.model.schema import NUM_ACTIONS
from riichi_ppo_v1.sft.contract import validate_v16_manifest
from riichi_ppo_v1.sft.data import _member_metadata, encode_kyoku_v16
from riichi_ppo_v1.sft.precompute import selected_any

ROOT = Path(__file__).resolve().parents[4]
RAW_ROOT = ROOT / "datasets/tenhou_sft_2024_2025"
ENCODED_ROOT = ROOT / "datasets/tenhou_sft_2024_2025_encoded_40pct_v16"
NUM_ACTIONS = 241
CONTEXT_LIMIT = 4096
KYOKUS_PER_CHUNK = 256
SUBSET_DENOMINATOR = 5
SUBSET_REMAINDERS = (0, 1)
AUDIT_SEED = "v16-data-semantics-audit-v1"


def _decode(payload: bytes) -> str:
    """按源数据约定解压成员内容(gzip 或无压缩)。"""
    if payload[:2] == b"\x1f\x8b":
        return gzip.decompress(payload).decode("utf-8")
    return payload.decode("utf-8")


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def scan_chunk(path: Path) -> dict[str, int]:
    """向量化检查单个 V16 NPZ chunk,返回本 chunk 的样本数。"""
    with np.load(path, allow_pickle=False) as data:
        required = (
            "history_offsets", "history_factors", "history_numeric",
            "snapshot_offsets", "snapshot_kinds", "snapshot_cat", "snapshot_num",
            "query_offsets", "action_offsets", "query_rows", "action_ids",
            "legal", "actions", "years", "game_ids", "kyoku_indices",
            "seats", "decision_indices",
        )
        missing = [name for name in required if name not in data]
        _require(not missing, f"{path.name} 缺少数组: {missing}")

        history_offsets = data["history_offsets"]
        snapshot_offsets = data["snapshot_offsets"]
        query_offsets = data["query_offsets"]
        action_offsets = data["action_offsets"]
        actions = data["actions"]
        legal = data["legal"]
        count = int(len(actions))

        for offsets, capacity, label in (
            (history_offsets, len(data["history_factors"]), "history"),
            (snapshot_offsets, len(data["snapshot_kinds"]), "snapshot"),
            (query_offsets, len(data["query_rows"]), "query"),
            (action_offsets, len(data["action_ids"]), "action"),
        ):
            _require(
                len(offsets) == count + 1,
                f"{path.name} {label}_offsets 长度异常: {len(offsets)} != {count + 1}",
            )
            _require(offsets[0] == 0 and offsets[-1] <= capacity, f"{path.name} {label} offset 越界")
            _require(bool(np.all(np.diff(offsets) >= 0)), f"{path.name} {label} offset 非单调")

        identity_arrays = (data["years"], data["game_ids"], data["kyoku_indices"], data["seats"], data["decision_indices"])
        for value in identity_arrays:
            _require(len(value) == count, f"{path.name} 身份数组长度异常")

        years = np.asarray(data["years"])
        game_ids = np.asarray(data["game_ids"])
        kyoku_indices = np.asarray(data["kyoku_indices"])
        seats = np.asarray(data["seats"])
        decision_indices = np.asarray(data["decision_indices"])
        _require(bool(np.all(np.isin(years, (2024, 2025)))), f"{path.name} 存在非法年份")
        _require(bool(np.all(game_ids != "")), f"{path.name} 存在空 game_id")
        _require(bool(np.all(kyoku_indices >= 0)), f"{path.name} 存在负局序号")
        _require(bool(np.all((seats >= 0) & (seats < 4))), f"{path.name} 存在非法座位")
        _require(bool(np.all(decision_indices >= 0)), f"{path.name} 存在负决策序号")

        _require(
            legal.shape == (count, -(-NUM_ACTIONS // 8)),
            f"{path.name} legal 形状异常: {legal.shape}",
        )
        legal_unpacked = np.unpackbits(
            legal.astype(np.uint8), axis=1, bitorder="little", count=NUM_ACTIONS
        ).astype(np.bool_)
        _require(bool(np.all(legal_unpacked.any(axis=1))), f"{path.name} 存在空合法掩码")
        _require(bool(np.all((actions >= 0) & (actions < NUM_ACTIONS))), f"{path.name} 专家动作越界")
        _require(
            bool(np.all(legal_unpacked[np.arange(count), actions])),
            f"{path.name} 存在专家动作不合法",
        )

        history_lengths = np.diff(history_offsets).astype(np.int64)
        snapshot_lengths = np.diff(snapshot_offsets).astype(np.int64)
        query_lengths = np.diff(query_offsets).astype(np.int64)
        action_lengths = np.diff(action_offsets).astype(np.int64)
        _require(bool(np.all(history_lengths >= 0)), f"{path.name} 存在负 history 长度")
        _require(bool(np.all(snapshot_lengths > 0)), f"{path.name} 存在空 snapshot")
        _require(bool(np.all(action_lengths >= 1)), f"{path.name} 存在空动作列表")
        _require(bool(np.all(query_lengths == 2 * action_lengths)), f"{path.name} query 行数不等于 2 倍动作数")
        _require(
            bool(np.all(history_lengths + snapshot_lengths + 2 * action_lengths <= CONTEXT_LIMIT)),
            f"{path.name} 存在超过 {CONTEXT_LIMIT} 的上下文",
        )

        query_rows = data["query_rows"]
        action_ids = data["action_ids"]
        total_queries = int(query_offsets[-1])
        total_actions = int(action_offsets[-1])
        _require(len(query_rows) == total_queries, f"{path.name} query_rows 总行数不一致")
        _require(len(action_ids) == total_actions, f"{path.name} action_ids 总行数不一致")

        row_ids = np.repeat(np.arange(count), action_lengths)
        _require(bool(np.all((action_ids >= 0) & (action_ids < NUM_ACTIONS))), f"{path.name} action_id 越界")
        _require(
            bool(np.all(legal_unpacked[row_ids, action_ids])),
            f"{path.name} action_id 与合法掩码不一致",
        )
        is_first = np.zeros(total_actions, dtype=np.bool_)
        is_first[action_offsets[:-1]] = True
        same_adjacent = np.diff(action_ids) == 0
        duplicate_adjacent = same_adjacent & ~is_first[1:]
        _require(bool(not np.any(duplicate_adjacent)), f"{path.name} 同一样本存在相邻重复 action_id")

        q_row_ids = np.repeat(query_offsets[:-1], query_lengths)
        local_index = np.arange(total_queries) - q_row_ids
        offense = (local_index % 2) == 0
        defense = ~offense
        expected_action_ids = np.repeat(action_ids, 2)
        _require(
            bool(np.all(query_rows[:, QUERY_ROW_QUERY_TYPE][offense] == 1))
            and bool(np.all(query_rows[:, QUERY_ROW_QUERY_TYPE][defense] == 2)),
            f"{path.name} query_type 不是 offense/defense 交替",
        )
        _require(
            bool(np.all(query_rows[:, QUERY_ROW_ACTION_ID] == expected_action_ids)),
            f"{path.name} query 行 action_id 与 action_ids 不一致",
        )
        action_type = query_rows[:, QUERY_ROW_ACTION_TYPE]
        _require(
            bool(np.all((action_type >= 1) & (action_type < ACTION_TYPE_CARDINALITY))),
            f"{path.name} query action_type 越界",
        )
        _require(
            bool(np.all(action_type[offense] == action_type[defense])),
            f"{path.name} 同一动作的两个 query 类型不一致",
        )
        primary_tile = query_rows[:, QUERY_ROW_PRIMARY_TILE]
        source_seat = query_rows[:, QUERY_ROW_SOURCE_SEAT]
        _require(
            bool(np.all((primary_tile >= 0) & (primary_tile <= 34))),
            f"{path.name} primary_tile 编码越界",
        )
        _require(
            bool(np.all((source_seat >= 0) & (source_seat <= 4))),
            f"{path.name} source_seat 编码越界",
        )

        answers = query_rows[:, QUERY_ROW_ANSWER_START:]
        for index, slot in enumerate(OFFENSE_SLOT_ORDER):
            _require(
                bool(np.all(answers[offense, index] < SLOT_CARDINALITIES[slot])),
                f"{path.name} offense slot {slot} 编码越界",
            )
        for index, slot in enumerate(DEFENSE_SLOT_ORDER):
            _require(
                bool(np.all(answers[defense, index] < SLOT_CARDINALITIES[slot])),
                f"{path.name} defense slot {slot} 编码越界",
            )

        kinds = data["snapshot_kinds"]
        snapshot_num = data["snapshot_num"]
        snapshot_cat = data["snapshot_cat"]
        _require(
            bool(np.all((kinds >= 0) & (kinds <= 3))) and bool(np.all(np.isfinite(snapshot_num))),
            f"{path.name} snapshot kind/数值非法",
        )
        base = kinds == 0
        score = kinds == 2
        summary = kinds == 3
        _require(
            bool(np.all((snapshot_num[base, :3] >= 0.0) & (snapshot_num[base, :3] <= 1.0))),
            f"{path.name} base snapshot 数值越界",
        )
        _require(
            bool(np.all(np.abs(snapshot_num[score, :7]) <= 5.0)),
            f"{path.name} score snapshot 数值越界",
        )
        _require(
            bool(np.all((snapshot_num[summary, :5] >= 0.0) & (snapshot_num[summary, :5] <= 1.0))),
            f"{path.name} summary snapshot 数值越界",
        )
        _require(
            bool(np.all(snapshot_cat[base, 0] <= 1))
            and bool(np.all(snapshot_cat[base, 1] <= 7))
            and bool(np.all(snapshot_cat[base, 2] <= 3))
            and bool(np.all(snapshot_cat[base, 3] <= 3)),
            f"{path.name} base snapshot categorical 越界",
        )
        dora = kinds == 1
        _require(
            bool(np.all(snapshot_cat[dora, 0] <= 33)),
            f"{path.name} dora snapshot categorical 越界",
        )
        _require(
            bool(np.all(snapshot_cat[summary, 0] <= 1))
            and bool(np.all(snapshot_cat[summary, 1] <= 1)),
            f"{path.name} summary snapshot categorical 越界",
        )
        return count


def scan_dataset(root: Path, workers: int = 1) -> dict[str, int]:
    """全量扫描所有 train/validation chunk,汇总样本数。"""
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), f"manifest 缺失: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_v16_manifest(manifest)
    expected = {
        "train": int(manifest["counts"]["train_decisions"]),
        "validation": int(manifest["counts"]["validation_decisions"]),
    }
    totals: dict[str, int] = {"train": 0, "validation": 0}
    checked = 0
    for split in ("train", "validation"):
        paths = sorted((root / split).glob(f"{split}-*.npz"))
        _require(bool(paths), f"{split} 无编码 chunk")
        if workers > 1:
            # 每个 chunk 独立,多进程并行只读扫描显著降低全量遍历耗时。
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for count in executor.map(scan_chunk, paths, chunksize=16):
                    totals[split] += count
                    checked += 1
                    if checked % 100 == 0:
                        print(
                            f"scan progress checked_chunks={checked} "
                            f"train_samples={totals['train']} validation_samples={totals['validation']}",
                            flush=True,
                        )
        else:
            for path in paths:
                totals[split] += scan_chunk(path)
                checked += 1
                if checked % 100 == 0:
                    print(
                        f"scan progress checked_chunks={checked} "
                        f"train_samples={totals['train']} validation_samples={totals['validation']}",
                        flush=True,
                    )
    for split in totals:
        _require(
            totals[split] == expected[split],
            f"{split} 样本数与 manifest 不一致: {totals[split]} != {expected[split]}",
        )
    print(
        f"scan summary checked_chunks={checked} train_samples={totals['train']} "
        f"validation_samples={totals['validation']}",
        flush=True,
    )
    return totals


def _selected_members_for_shard(shard: Path) -> list[str]:
    """返回一个 tar 内按 V16 子集规则选中的成员名(保持归档顺序)。"""
    with tarfile.open(shard, "r") as archive:
        return [
            member.name
            for member in archive
            if member.isfile()
            and selected_any(
                member.name,
                SUBSET_DENOMINATOR,
                SUBSET_REMAINDERS,
            )
        ]


def _read_member(shard: Path, name: str) -> str:
    with tarfile.open(shard, "r") as archive:
        member = archive.getmember(name)
        file = archive.extractfile(member)
        if file is None:
            _fail(f"cannot read {shard}:{name}")
        return _decode(file.read())


def _stored_samples_for_member(encoded_root: Path, split: str, shard: Path, member_name: str) -> tuple[np.ndarray, np.ndarray]:
    """定位成员对应的编码 chunk,并返回该 game_id/kyoku 的全部样本数组。"""
    selected = _selected_members_for_shard(shard)
    ordinal = selected.index(member_name)
    chunk_index = ordinal // KYOKUS_PER_CHUNK
    path = encoded_root / split / f"{shard.stem}-{chunk_index:03d}.npz"
    _require(path.is_file(), f"编码 chunk 缺失: {path}")
    data = np.load(path, allow_pickle=False)
    year, game_id, kyoku_index = _member_metadata(member_name)
    game_ids = np.asarray(data["game_ids"])
    kyoku_indices = np.asarray(data["kyoku_indices"])
    rows = np.flatnonzero(
        (game_ids == game_id) & (kyoku_indices == kyoku_index)
    )
    if not len(rows):
        data.close()
        _fail(f"{member_name} 在 {path} 中找不到匹配样本")
    return data, rows


def compare_member(
    encoded_root: Path,
    split: str,
    shard: Path,
    member_name: str,
) -> dict[str, object]:
    """重编码一个原始成员,并与存量 chunk 中同局样本逐字段比对。"""
    content = _read_member(shard, member_name)
    year, game_id, kyoku_index = _member_metadata(member_name)
    fresh = encode_kyoku_v16(
        content, year=year, game_id=game_id, kyoku_index=kyoku_index,
    )
    _require(bool(fresh), f"{member_name} 重编码为空")
    data, rows = _stored_samples_for_member(encoded_root, split, shard, member_name)
    try:
        # 同一个 game_id/kyoku 可能有多个 shard 重复?不存在时按 seat/decision 排序。
        stored_rows = np.asarray(rows)
        order = np.lexsort(
            (
                np.asarray(data["decision_indices"])[stored_rows],
                np.asarray(data["seats"])[stored_rows],
            )
        )
        stored_rows = stored_rows[order]
        _require(
            len(fresh) == len(stored_rows),
            f"{member_name} 重编码样本数 {len(fresh)} 与存量 {len(stored_rows)} 不一致",
        )

        history_offsets = np.asarray(data["history_offsets"])
        snapshot_offsets = np.asarray(data["snapshot_offsets"])
        query_offsets = np.asarray(data["query_offsets"])
        action_offsets = np.asarray(data["action_offsets"])
        legal_unpacked = np.unpackbits(
            np.asarray(data["legal"]).astype(np.uint8),
            axis=1, bitorder="little", count=NUM_ACTIONS,
        ).astype(np.bool_)

        for fresh_row, stored_row in zip(fresh, stored_rows, strict=True):
            stored_row = int(stored_row)
            stored_seat = int(data["seats"][stored_row])
            stored_decision = int(data["decision_indices"][stored_row])
            _require(
                fresh_row.seat == stored_seat and fresh_row.decision_index == stored_decision,
                f"{member_name} 身份错位: fresh=({fresh_row.seat},{fresh_row.decision_index}) "
                f"stored=({stored_seat},{stored_decision})",
            )
            hist_start, hist_end = int(history_offsets[stored_row]), int(history_offsets[stored_row + 1])
            snap_start, snap_end = int(snapshot_offsets[stored_row]), int(snapshot_offsets[stored_row + 1])
            query_start, query_end = int(query_offsets[stored_row]), int(query_offsets[stored_row + 1])
            action_start, action_end = int(action_offsets[stored_row]), int(action_offsets[stored_row + 1])

            _require(
                np.array_equal(data["history_factors"][hist_start:hist_end], fresh_row.history_factors)
                and np.array_equal(data["snapshot_kinds"][snap_start:snap_end], fresh_row.snapshot_kinds)
                and np.array_equal(data["snapshot_cat"][snap_start:snap_end], fresh_row.snapshot_cat)
                and np.array_equal(data["query_rows"][query_start:query_end], fresh_row.query_rows)
                and np.array_equal(data["action_ids"][action_start:action_end], fresh_row.action_ids)
                and np.array_equal(legal_unpacked[stored_row], fresh_row.legal_mask)
                and int(data["actions"][stored_row]) == fresh_row.action,
                f"{member_name} seat={fresh_row.seat} decision={fresh_row.decision_index} 离散字段不一致",
            )
            np.testing.assert_allclose(
                data["history_numeric"][hist_start:hist_end],
                fresh_row.history_numeric,
                rtol=6e-3, atol=6e-3,
                err_msg=f"{member_name} history_numeric fp16 比对失败",
            )
            np.testing.assert_allclose(
                data["snapshot_num"][snap_start:snap_end],
                fresh_row.snapshot_num,
                rtol=6e-3, atol=6e-3,
                err_msg=f"{member_name} snapshot_num fp16 比对失败",
            )
    finally:
        data.close()
    return {
        "member": member_name,
        "split": split,
        "samples": len(fresh),
        "decisions": sum(sample.query_pair_count for sample in fresh),
    }


def run_sample_audit(train_count: int, validation_count: int) -> dict[str, object]:
    """确定性选取原始成员并重编码比对。"""
    results: list[dict[str, object]] = []
    for split, count, per_shard in (("train", train_count, 1), ("validation", validation_count, 2)):
        shards = sorted((RAW_ROOT / split).glob(f"{split}-*.tar"))
        if split == "train":
            chosen_shards = random.Random(AUDIT_SEED).sample(shards, min(count, len(shards)))
            chosen_members = [
                (shard, random.Random(f"{AUDIT_SEED}\0{shard.name}").choice(_selected_members_for_shard(shard)))
                for shard in chosen_shards
            ]
        else:
            chosen_shards = shards
            chosen_members = []
            for shard in chosen_shards:
                members = _selected_members_for_shard(shard)
                rng = random.Random(f"{AUDIT_SEED}\0{shard.name}")
                chosen_members.extend(
                    (shard, rng.choice(members)) for _ in range(min(per_shard, len(members)))
                )
        for shard, member_name in chosen_members:
            results.append(compare_member(ENCODED_ROOT, split, shard, member_name))
            print(f"sample compared {member_name} samples={results[-1]['samples']}", flush=True)
    return {
        "members_compared": len(results),
        "samples_compared": sum(int(item["samples"]) for item in results),
        "decisions_compared": sum(int(item["decisions"]) for item in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-scan", action="store_true", help="跳过全量结构扫描")
    parser.add_argument("--skip-sample", action="store_true", help="跳过抽样重编码比对")
    parser.add_argument("--train-count", type=int, default=120)
    parser.add_argument("--validation-count", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    scan_result = None
    if not args.skip_scan:
        scan_result = scan_dataset(ENCODED_ROOT, workers=max(1, args.workers))
    sample_result = None
    if not args.skip_sample:
        sample_result = run_sample_audit(args.train_count, args.validation_count)
    print(json.dumps({"scan": scan_result, "sample": sample_result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
