"""V18 端到端编解码往返审查（只读，可复现）。

链路：
    真实 MJAI replay
      → sft.data.encode_kyoku（生产 SFT 预处理，逐决策）
      → current_state.encode_batch（生产 Rust 批编码 + Python 装配）
      → V18 模型 forward（policy_only）
      → policy_logits → 合法域 argmax action id
      → state_machine.decode_actions（Rust）→ 两条实际解码路径：
        B1 SFT 路径：规范 JSON 匹配到合法 Action（与 sft/data.py 一致）
        B2 桥接路径：observation.select_action_from_mjai（与 bridge.decode 一致）
      → action_groups.action_id（Python 独立映射）→ 与原 id 一致

输出：预处理统计、B1/B2 往返失败清单、模型 top-1 可解码率。不写任何文件。

用法（Mahjong-AI 环境）：
    conda run -n Mahjong-AI python audit/reports/v18/scripts/v18_decode_roundtrip.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/disk1/hubowen/zenith")
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "v18_audit_oracle",
    ROOT / "audit/reports/v18/scripts/v18_audit_oracle.py",
)
_oracle = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_oracle)


def _canonical(value: str) -> str:
    return json.dumps(json.loads(value), separators=(",", ":"), sort_keys=True)


def drive_record(record: str):
    """驱动一个 kyoku 的状态机并返回逐决策 (seat, observation, [(action_id, mjai, action), ...])。

    与 sft/data.py 同流程；合法动作 id 映射使用 Rust 直接映射
    ``action_ids_with_source_indices``（与在线 bridge 相同），并校验与 legal mask 一致。
    在每一步登记后立即执行 Rust ``decode_actions``，并同时给出两条解码结果：
    - ``select``：env ``select_action_from_mjai``（bridge.decode 路径）；
    - ``rep``：规范 JSON 匹配（sft/data.py 路径）。
    """
    import riichi
    from riichienv import MjaiReplay

    from riichi_ppo_v1.model.bridge import action_jsons

    replay = MjaiReplay.from_jsonl_string(record, rule="tenhou")
    kyokus = list(replay.take_kyokus())
    assert len(kyokus) == 1, "record must contain exactly one kyoku"
    kyoku = kyokus[0]
    manager = riichi.MjaiKyokuStateMachineManager(4)
    streams = [iter(kyoku.steps(seat=seat, skip_single_action=False)) for seat in range(4)]
    active = set(range(4))
    decisions: list[tuple[int, object, list[tuple[int, str, object, object | None]]]] = []
    while active:
        batch: list[tuple[int, object, object]] = []
        for seat in sorted(active):
            try:
                observation, expert_action = next(streams[seat])
            except StopIteration:
                active.remove(seat)
            else:
                batch.append((seat, observation, expert_action))
        if not batch:
            continue
        env_indices = [seat for seat, _obs, _act in batch]
        events_by_env = []
        action_rows = []
        for seat, observation, _act in batch:
            events = [[], [], [], []]
            events[seat] = list(observation.new_events())
            events_by_env.append(events)
            action_rows.append(action_jsons(observation))
        manager.apply_events_batch(env_indices, events_by_env)
        batch_indices = [seat * 4 + seat for seat, _obs, _act in batch]
        prepared = manager.prepare_decisions(batch_indices, action_rows)
        mask = np.asarray(prepared, dtype=np.bool_)
        index_rows = manager.action_ids_with_source_indices(batch_indices)
        for row, (seat, observation, _expert) in enumerate(batch):
            legal_objects = list(observation.legal_actions())
            templates = action_jsons(observation)
            if len(legal_objects) != len(templates):
                raise RuntimeError("legal/template count mismatch")
            representative: dict[str, object] = {}
            for action, template in zip(legal_objects, templates, strict=True):
                representative.setdefault(_canonical(template), action)
            mappings = index_rows[row]
            expected = np.flatnonzero(mask[row]).tolist()
            if [int(a) for a, _si in mappings] != expected:
                raise RuntimeError("state machine mapping disagrees with legal mask")
            batch_index = seat * 4 + seat
            ids = [int(a) for a, _si in mappings]
            mjai_rows = manager.decode_actions([batch_index] * len(ids), ids)
            decoded: list[tuple[int, str, object, object | None]] = []
            for (aid, _si), mjai in zip(mappings, mjai_rows, strict=True):
                aid = int(aid)
                action = representative.get(_canonical(mjai))
                selected = observation.select_action_from_mjai(mjai)
                decoded.append((aid, mjai, action, selected))
            decisions.append((seat, observation, decoded))
    return decisions


def main() -> int:
    import riichi
    import riichienv

    print("== 环境 ==")
    print("riichi:", riichi.__file__)
    print("riichienv:", riichienv.__file__)
    print("ENCODING_PROTOCOL_VERSION:", getattr(riichi, "ENCODING_PROTOCOL_VERSION", None))
    print("ANALYSIS_VERSION:", getattr(riichi, "ANALYSIS_VERSION", None))
    print("REPLAY_SEMANTICS_VERSION:", getattr(riichienv, "REPLAY_SEMANTICS_VERSION", None))

    records = _oracle.load_kyoku_records()
    print(f"\n== fixture kyoku records: {len(records)} ==")

    # ---- Part A: 生产 SFT 预处理逐决策编码（Q1 数据链） ----
    from riichi_ppo_v1.sft.data import encode_kyoku

    total_samples = 0
    for ridx, record in enumerate(records):
        samples = encode_kyoku(record, year=2024, game_id=f"rec{ridx}", kyoku_index=0)
        total_samples += len(samples)
        for sample in samples:
            assert int(sample.action) in np.flatnonzero(sample.legal_mask).tolist()
            assert sample.actor_factors.shape[1] == 32
            assert sample.token_length <= 256
    print(
        f"production preprocessing (encode_kyoku): {len(records)} kyoku, "
        f"{total_samples} decisions; target∈legal / shape / context≤256 all PASS"
    )

    # ---- Part B: 前 3 局逐决策 decode 往返（Q1+Q2 模型输出解码） ----
    import torch

    from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
    from riichi_ppo_v1.model.action_groups import action_id as remap_action_id
    from riichi_ppo_v1.model.current_state import encode_batch

    torch.manual_seed(0)
    model = KyokuTransformerActorCritic(ModelConfig.preset("v18"))
    model.eval()

    b1_fail: list[str] = []
    b2_fail: list[str] = []
    checked = 0
    top_ok = 0
    part_b_decisions = 0
    examples: list[str] = []
    for ridx, record in enumerate(records):
        decisions = drive_record(record)
        if not decisions:
            continue
        part_b_decisions += len(decisions)
        batch_inputs = [
            (obs, [(action, aid) for aid, _mjai, action, _sel in decoded])
            for _seat, obs, decoded in decisions
        ]
        encoded = encode_batch(batch_inputs)
        batch = len(decisions)
        with torch.no_grad():
            output = model(
                actor_factors=torch.from_numpy(encoded.actor_factors.astype(np.int64)),
                actor_numeric=torch.from_numpy(encoded.actor_numeric.astype(np.float32)),
                actor_lengths=torch.from_numpy(encoded.actor_lengths.astype(np.int64)),
                query_action_ids=torch.from_numpy(encoded.action_ids.astype(np.int64)),
                query_pair_counts=torch.from_numpy(encoded.query_pair_counts.astype(np.int64)),
                legal_mask=torch.from_numpy(encoded.legal_mask.astype(np.bool_)),
                policy_only=True,
            )
        logits = output["policy_logits"]
        assert logits.shape == (batch, 241), logits.shape
        top_ids = torch.argmax(logits, dim=1).tolist()
        for row, (seat, _obs, decoded) in enumerate(decisions):
            for aid, mjai, action, selected in decoded:
                checked += 1
                # B1: SFT 规范 JSON 匹配 + 重映射。
                if action is None:
                    b1_fail.append(f"rec{ridx} seat{seat} id={aid}: no representative for {mjai}")
                else:
                    if remap_action_id(action, _obs) != aid:
                        b1_fail.append(f"rec{ridx} seat{seat} id={aid}: remap mismatch")
                # B2: bridge select 路径。
                if selected is None:
                    b2_fail.append(f"rec{ridx} seat{seat} id={aid}: select failed for {mjai}")
                else:
                    if remap_action_id(selected, _obs) != aid:
                        b2_fail.append(f"rec{ridx} seat{seat} id={aid}: select remap mismatch")
            top = top_ids[row]
            if top in [aid for aid, _mjai, _a, _s in decoded]:
                top_ok += 1
                mjai = next(mjai for aid, mjai, _a, _s in decoded if aid == top)
                examples.append(f"rec{ridx} seat{seat} step{row} top1={top} mjai={mjai}")
    print(f"\ndecode round-trip: legal ids checked={checked}")
    print(f"  B1 SFT canonical-match failures = {len(b1_fail)}")
    for line in b1_fail[:10]:
        print("    ", line)
    print(f"  B2 bridge select failures       = {len(b2_fail)}")
    for line in b2_fail[:10]:
        print("    ", line)
    print(f"  model top-1 ∈ legal & decoded   = {top_ok}/{part_b_decisions}")
    print("示例（模型 top-1 → 解码 MJAI）：")
    for line in examples[:8]:
        print("  ", line)
    print("\n== 结论 ==")
    if total_samples == 0 or checked == 0 or b1_fail:
        print("FAIL（B1 SFT 解码往返失败）")
        return 1
    if b2_fail:
        print("B1 SFT 解码往返 PASS；B2 bridge select 存在失败（见上，需修复）")
        return 0
    print("B1 与 B2 解码往返全部 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
