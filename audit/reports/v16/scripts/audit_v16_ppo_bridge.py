"""V16 PPO 采样离线一致性审计。

用同一批 Tenhou 原始小局同时走 SFT 的 `encode_kyoku_v16` 与 PPO 的
`BatchedStateBridge.prepare_v16`,逐决策比较 Actor 输入;并验证 action_id 解码、
Critic 特权段与上下文上限。本脚本不启动 Ray/GPU。
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import tarfile
from pathlib import Path

import numpy as np

import riichi
from riichienv import MjaiReplay

from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision, action_jsons
from riichi_ppo_v1.model.critic_features import (
    SEGMENT_CRITIC_FUTURE_WALL,
    SEGMENT_CRITIC_PRIVATE,
)
from riichi_ppo_v1.model.encoding_protocol import (
    QUERY_ROW_ACTION_ID,
    QUERY_ROW_QUERY_TYPE,
    QUERY_DEFENSE,
    QUERY_OFFENSE,
)
from riichi_ppo_v1.sft.data import _member_metadata, encode_kyoku_v16
from riichi_ppo_v1.sft.precompute import selected_any

ROOT = Path(__file__).resolve().parents[4]
RAW_ROOT = ROOT / "datasets/tenhou_sft_2024_2025"
CONTEXT_LIMIT = 4096
SUBSET_DENOMINATOR = 5
SUBSET_REMAINDERS = (0, 1)
AUDIT_SEED = "v16-ppo-bridge-audit-v1"
MEMBER_COUNT = 40


def _decode(payload: bytes) -> str:
    if payload[:2] == b"\x1f\x8b":
        return gzip.decompress(payload).decode("utf-8")
    return payload.decode("utf-8")


def _selected_members(shard: Path) -> list[str]:
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
            raise RuntimeError(f"cannot read {shard}:{name}")
        return _decode(file.read())


def _canonical(mjai: str) -> str:
    return json.dumps(json.loads(mjai), separators=(",", ":"), sort_keys=True)


class _EmptyEventsObservation:
    """离线回放结尾缺失座位时,用空事件代理补齐四座观察。"""

    def __init__(self, base: object) -> None:
        self._base = base

    def new_events(self) -> list[str]:
        return []

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


def _replay_kyoku(content: str) -> object:
    replay = MjaiReplay.from_jsonl_string(content, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    if len(kyokus) != 1:
        raise RuntimeError(f"SFT record must contain exactly one kyoku, got {len(kyokus)}")
    return kyokus[0]


def _compare_segments(sample: object, bridge_row: dict[str, object], context: str) -> None:
    """比较一条决策的 Actor 输入与 SFT 编码。"""
    history_length = int(bridge_row["history_lengths"])
    snapshot_length = int(bridge_row["snapshot_lengths"])
    pair_count = int(bridge_row["query_pair_counts"])
    np.testing.assert_array_equal(
        bridge_row["history_factors"][:history_length],
        sample.history_factors,
        err_msg=f"{context} history_factors 不一致",
    )
    np.testing.assert_array_equal(
        bridge_row["history_numeric"][:history_length],
        sample.history_numeric,
        err_msg=f"{context} history_numeric 不一致",
    )
    np.testing.assert_array_equal(
        bridge_row["snapshot_kinds"][:snapshot_length],
        sample.snapshot_kinds,
        err_msg=f"{context} snapshot_kinds 不一致",
    )
    np.testing.assert_array_equal(
        bridge_row["snapshot_cat"][:snapshot_length],
        sample.snapshot_cat,
        err_msg=f"{context} snapshot_cat 不一致",
    )
    np.testing.assert_array_equal(
        bridge_row["snapshot_num"][:snapshot_length],
        sample.snapshot_num,
        err_msg=f"{context} snapshot_num 不一致",
    )
    np.testing.assert_array_equal(
        bridge_row["query_rows"][: 2 * pair_count],
        sample.query_rows,
        err_msg=f"{context} query_rows 不一致",
    )
    np.testing.assert_array_equal(
        bridge_row["query_action_ids"][:pair_count],
        sample.action_ids,
        err_msg=f"{context} query_action_ids 不一致",
    )
    np.testing.assert_array_equal(
        bridge_row["legal_mask"],
        sample.legal_mask,
        err_msg=f"{context} legal_mask 不一致",
    )
    if history_length + snapshot_length + 2 * pair_count > CONTEXT_LIMIT:
        raise RuntimeError(f"{context} 超过 {CONTEXT_LIMIT} token 上限")


def audit_member(shard: Path, member_name: str) -> dict[str, object]:
    """回放一个小局,比较 SFT 与 PPO 桥接的每个决策。"""
    content = _read_member(shard, member_name)
    year, game_id, kyoku_index = _member_metadata(member_name)
    sft_samples = encode_kyoku_v16(
        content, year=year, game_id=game_id, kyoku_index=kyoku_index,
    )
    by_key = {
        (sample.seat, sample.decision_index): sample
        for sample in sft_samples
    }
    kyoku = _replay_kyoku(content)

    manager = riichi.MjaiKyokuStateMachineManager(1)
    bridge = BatchedStateBridge(manager, 1)
    streams = [iter(kyoku.steps(seat=seat, skip_single_action=False)) for seat in range(4)]
    active = set(range(4))
    decision_counts = [0, 0, 0, 0]
    last_observations: list[object | None] = [None, None, None, None]
    compared = 0
    decoded_actions = 0
    while active:
        batch: list[tuple[int, object, object]] = []
        for seat in sorted(active):
            try:
                observation, expert_action = next(streams[seat])
            except StopIteration:
                active.remove(seat)
            else:
                batch.append((seat, observation, expert_action))
                last_observations[seat] = observation
        if not batch:
            continue
        observations_by_env = [{
            seat: (
                last_observations[seat]
                if seat in {value for value, _obs, _expert in batch}
                else _EmptyEventsObservation(last_observations[seat])
            )
            for seat in range(4)
        }]
        bridge.sync(observations_by_env)
        decisions = [
            Decision(0, seat, observations_by_env[0][seat])
            for seat in range(4)
        ]
        prepared = bridge.prepare_v16(decisions, walls=None)

        for seat, observation, _expert in batch:
            row = seat
            decision_index = decision_counts[seat]
            decision_counts[seat] += 1
            key = (seat, decision_index)
            if key not in by_key:
                raise RuntimeError(f"{member_name} seat={seat} decision={decision_index} SFT 缺少样本")
            context = f"{member_name} seat={seat} decision={decision_index}"
            _compare_segments(
                by_key[key],
                {
                    "history_factors": prepared.history_factors[row],
                    "history_numeric": prepared.history_numeric[row],
                    "history_lengths": prepared.history_lengths[row],
                    "snapshot_kinds": prepared.snapshot_kinds[row],
                    "snapshot_cat": prepared.snapshot_cat[row],
                    "snapshot_num": prepared.snapshot_num[row],
                    "snapshot_lengths": prepared.snapshot_lengths[row],
                    "query_rows": prepared.query_rows[row],
                    "query_action_ids": prepared.query_action_ids[row],
                    "query_pair_counts": prepared.query_pair_counts[row],
                    "legal_mask": prepared.legal_mask[row],
                },
                context,
            )
            ids = np.flatnonzero(prepared.legal_mask[row]).tolist()
            rows = prepared.query_rows[row, : 2 * len(ids)]
            if (
                not np.all(rows[0::2, QUERY_ROW_QUERY_TYPE] == QUERY_OFFENSE)
                or not np.all(rows[1::2, QUERY_ROW_QUERY_TYPE] == QUERY_DEFENSE)
                or not np.all(rows[0::2, QUERY_ROW_ACTION_ID] == ids)
                or not np.all(rows[1::2, QUERY_ROW_ACTION_ID] == ids)
            ):
                raise RuntimeError(f"{context} query 配对/action_id 异常")

            # 每个合法 action_id 都必须能经 Rust 解码回一个合法 MJAI 模板;
            # 离线 MjaiReplay 的 ``select_action_from_mjai`` 对 replay 动作形态
            # 更严格,因此这里按 prepare_v16 的 representative 映射校验。
            representative: dict[str, object] = {}
            for action, template in zip(
                observation.legal_actions(), action_jsons(observation), strict=True
            ):
                representative.setdefault(_canonical(template), action)
            decoded_raw = manager.decode_actions([seat] * len(ids), ids)
            for raw in decoded_raw:
                if _canonical(raw) not in representative:
                    raise RuntimeError(f"{context} 解码动作不在合法模板中")
            decoded_actions += len(decoded_raw)

            critic_length = int(prepared.critic_lengths[row])
            segments = set(
                np.unique(
                    prepared.critic_factors[row, :critic_length, 0]
                ).tolist()
            )
            if not segments <= {SEGMENT_CRITIC_PRIVATE, SEGMENT_CRITIC_FUTURE_WALL}:
                raise RuntimeError(f"{context} Critic 特权段异常: {segments}")
            compared += 1

    if compared != len(sft_samples):
        raise RuntimeError(
            f"{member_name} 比较决策数 {compared} 与 SFT 样本数 {len(sft_samples)} 不一致"
        )
    return {
        "member": member_name,
        "decisions": compared,
        "decoded_actions": decoded_actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=MEMBER_COUNT, help="审计小局数")
    args = parser.parse_args()
    shards = sorted((RAW_ROOT / "train").glob("train-*.tar"))
    chosen_shards = random.Random(AUDIT_SEED).sample(shards, min(max(1, args.count), len(shards)))
    results: list[dict[str, object]] = []
    for shard in chosen_shards:
        members = _selected_members(shard)
        if not members:
            raise RuntimeError(f"{shard.name} 没有 V16 子集成员")
        member = random.Random(f"{AUDIT_SEED}\0{shard.name}").choice(members)
        result = audit_member(shard, member)
        results.append(result)
        print(
            f"bridge compared {member} decisions={result['decisions']} "
            f"decoded_actions={result['decoded_actions']}",
            flush=True,
        )
    summary = {
        "kyokus_compared": len(results),
        "decisions_compared": sum(int(item["decisions"]) for item in results),
        "decoded_actions": sum(int(item["decoded_actions"]) for item in results),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("ppo bridge audit: all checks passed", flush=True)


if __name__ == "__main__":
    main()
