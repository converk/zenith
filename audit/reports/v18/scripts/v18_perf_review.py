"""V18 Python 侧性能审查（只读，可复现）。

看点（对应用户问题 3）：
 A. SFT 预处理 hot path 的 Python 时间分布（encode_kyoku / encode_batch /
    _accumulate_field_statistics / JSON 匹配）；
 B. encode_batch 中 Python 装配层 vs Rust 批编码的耗时占比；
 C. 训练 collate 的逐批语义校验（assert_actor_input_semantics）开销；
 D. Rust 批编码器是否释放 GIL（源码静态检查结果打印）。

用法（Mahjong-AI 环境）：
    conda run -n Mahjong-AI python audit/reports/v18/scripts/v18_perf_review.py
"""

from __future__ import annotations

import cProfile
import importlib.util
import io
import pstats
import sys
import time
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


def main() -> int:
    records = _oracle.load_kyoku_records()

    # ---- A. encode_kyoku 单局 profile（前 2 局） ----
    from riichi_ppo_v1.sft.data import encode_kyoku

    print("== A. SFT 预处理 encode_kyoku profile（前 2 局，共 4 个观察者） ==")
    profiler = cProfile.Profile()
    profiler.enable()
    for ridx, record in enumerate(records[:2]):
        encode_kyoku(record, year=2024, game_id=f"rec{ridx}", kyoku_index=0)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(18)
    print(stream.getvalue()[:4000])

    # ---- B. encode_batch：Rust 两部分 vs Python 装配 ----
    from riichi_ppo_v1.model.current_state import encode_batch
    from riichi_ppo_v1.model.native_encoding import encode_action_queries_batch_native

    # 单局驱动，构造 (obs, actions)。
    driven = []
    for ridx, record in enumerate(records[:2]):
        from riichienv import MjaiReplay

        from riichi_ppo_v1.model.action_groups import action_id as map_action_id

        replay = MjaiReplay.from_jsonl_string(record, rule="tenhou")
        kyoku = list(replay.take_kyokus())[0]
        for seat in range(4):
            for _step, (obs, _act) in enumerate(kyoku.steps(seat=seat, skip_single_action=False)):
                by_id = {}
                for action in obs.legal_actions():
                    aid = map_action_id(action, obs)
                    if aid is not None:
                        by_id.setdefault(aid, action)
                if by_id:
                    driven.append((obs, [(a, aid) for aid, a in sorted(by_id.items())]))

    print(f"\n== B. encode_batch 耗时分解（{len(driven)} 个决策，单次批量） ==")
    t0 = time.perf_counter()
    encoded = encode_batch(driven)
    t_encode_batch = time.perf_counter() - t0
    native = [getattr(obs, "native_observation", obs) for obs, _a in driven]
    import riichienv

    t0 = time.perf_counter()
    rows = riichienv.prepare_current_state_batch(native)
    t_rust_current = time.perf_counter() - t0
    triples = [(obs, action, aid) for obs, actions in driven for action, aid in actions]
    t0 = time.perf_counter()
    queries = encode_action_queries_batch_native(triples)
    t_rust_query = time.perf_counter() - t0
    print(f"  encode_batch total      = {t_encode_batch * 1000:.1f} ms")
    print(f"    prepare_current_state_batch (Rust) = {t_rust_current * 1000:.1f} ms")
    print(f"    encode_action_queries_batch_native (Rust) = {t_rust_query * 1000:.1f} ms")
    print(f"    Python 装配/其余      = {(t_encode_batch - t_rust_current - t_rust_query) * 1000:.1f} ms")
    lengths = np.asarray(encoded.actor_lengths)
    print(f"  平均 token/决策 = {lengths.mean():.1f}，动作对/决策 = {encoded.query_pair_counts.mean():.2f}")

    # ---- C. 训练 collate 的逐批语义校验开销 ----
    from riichi_ppo_v1.model.semantic_validation import assert_actor_input_semantics

    print("\n== C. collate 前逐批语义校验 assert_actor_input_semantics ==")
    # 用 B 的真实合法批（131 决策 × ~106 token）测量完整校验路径。
    t0 = time.perf_counter()
    for _ in range(5):
        assert_actor_input_semantics(
            encoded.actor_factors.astype(np.int64),
            encoded.actor_numeric.astype(np.float32),
            encoded.actor_lengths.astype(np.int64),
            encoded.query_rows.astype(np.int64),
            encoded.action_ids.astype(np.int64),
            encoded.query_pair_counts.astype(np.int64),
            encoded.legal_mask,
        )
    elapsed = (time.perf_counter() - t0) / 5
    tokens = int(encoded.actor_lengths.sum())
    print(f"  {len(driven)} 决策 × {tokens} token 平均 {elapsed * 1000:.1f} ms "
          f"（≈{elapsed / tokens * 1e6:.1f} µs/token；线性外推 512 决策 ≈ "
          f"{elapsed * (512 / len(driven)) * 1000:.0f} ms）")

    # ---- D. Rust 编码器 GIL 静态检查 ----
    print("\n== D. Rust 批编码函数 GIL 释放（静态检查） ==")
    src = (ROOT / "RiichiEnv/riichienv-python/src/current_state_encoding.rs").read_text(encoding="utf-8")
    facts = (ROOT / "RiichiEnv/riichienv-python/src/encoding_facts.rs").read_text(encoding="utf-8")
    for name, text, marker in (
        ("prepare_current_state_batch", src, "pub fn prepare_current_state_batch"),
        ("prepare_encoding_facts", facts, "pub fn prepare_encoding_facts"),
        ("apply_events_batch(管理器)", (ROOT / "RiichiEnv/riichienv-state-machine/src/MjaiKyokuStateMachine/manager.rs").read_text(encoding="utf-8"), "fn apply_events_batch"),
    ):
        idx = text.find(marker)
        block = text[idx: idx + 3200]
        detached = "detach" in block or "allow_threads" in block
        print(f"  {name}: body uses GIL release = {detached}")
    # ---- A2. precompute 的 per-token 域统计循环（_accumulate_field_statistics） ----
    from riichi_ppo_v1.sft.precompute import _accumulate_field_statistics, _empty_field_statistics

    print("\n== A2. precompute 逐 token 域统计（_accumulate_field_statistics） ==")
    all_samples = []
    for ridx, record in enumerate(records):
        all_samples.extend(
            encode_kyoku(record, year=2024, game_id=f"rec{ridx}", kyoku_index=0)
        )
    t0 = time.perf_counter()
    stats = _empty_field_statistics()
    _accumulate_field_statistics(stats, all_samples)
    elapsed = time.perf_counter() - t0
    tokens = sum(int(sample.token_length) for sample in all_samples)
    print(f"  {len(all_samples)} 决策 × {tokens} token 平均 {elapsed * 1000:.1f} ms "
          f"（≈{elapsed / tokens * 1e6:.1f} µs/token；60%% 数据集 ≈ 100 万决策 "
          f"→ 约 {elapsed / len(all_samples) * 1e6 * 1.0:.0f} s/百万决策）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
