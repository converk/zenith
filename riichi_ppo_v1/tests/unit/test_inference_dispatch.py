"""推理凑批「到齐即 flush」判定的单元测试(Tier 0.2)。"""

from __future__ import annotations

from riichi_ppo_v1.training.inference import (
    MIN_QUORUM_WORKERS,
    dispatch_reason,
)


def test_quorum_flushes_once_window_is_open() -> None:
    """凑批窗口打开(尚有等待预算)且独立 worker ≥ quorum 时立即 flush。"""
    # 12 worker × ~32 行、target_workers 禁用(-1 映射为大数)的典型形态:
    # 旧行为只能睡满窗口由 timeout 触发;新行为在窗口内到齐即 flush。
    assert (
        dispatch_reason(
            [0, 1, 2, 3], 1 << 30, deadline=100.0, now=99.99,
            row_count=128, target_rows=512, batch_wait_s=0.001,
        )
        == "quorum"
    )


def test_quorum_requires_open_window() -> None:
    """窗口未打开(batch_wait_s ≤ 0)时保持旧行为:只有 target/超时触发。"""
    assert (
        dispatch_reason(
            [0, 1, 2, 3], 1 << 30, deadline=100.0, now=99.99,
            row_count=128, target_rows=512, batch_wait_s=0.0,
        )
        is None
    )
    # 窗口未打开但已过截止 → 旧超时路径不受影响。
    assert (
        dispatch_reason(
            [0, 1], 1 << 30, deadline=100.0, now=100.5,
            row_count=64, target_rows=512, batch_wait_s=0.0,
        )
        == "timeout"
    )


def test_quorum_not_triggered_by_single_worker() -> None:
    """单一 worker的多行请求不触发 quorum(大请求不被切碎);行数达到
    阈值时由 rows 优先触发(判定顺序:rows > quorum > target > timeout)。"""
    assert (
        dispatch_reason(
            [0, 0, 0, 0], 1 << 30, deadline=100.0, now=99.99,
            row_count=64, target_rows=512, batch_wait_s=0.001,
        )
        is None
    )
    assert (
        dispatch_reason(
            [0, 0, 0, 0], 1 << 30, deadline=100.0, now=99.99,
            row_count=512, target_rows=512, batch_wait_s=0.001,
        )
        == "rows"
    )


def test_quorum_threshold_is_configurable_and_fail_closed() -> None:
    """quorum 阈值可配置;0/负值显式禁用到齐判定(纯超时凑批)。"""
    assert (
        dispatch_reason(
            [0, 1], 1 << 30, deadline=100.0, now=99.99,
            row_count=64, target_rows=512, batch_wait_s=0.001,
            min_quorum_workers=3,
        )
        is None
    )
    assert (
        dispatch_reason(
            [0, 1, 2], 1 << 30, deadline=100.0, now=99.99,
            row_count=96, target_rows=512, batch_wait_s=0.001,
            min_quorum_workers=3,
        )
        == "quorum"
    )
    assert (
        dispatch_reason(
            [0, 1, 2, 3], 1 << 30, deadline=100.0, now=99.99,
            row_count=128, target_rows=512, batch_wait_s=0.001,
            min_quorum_workers=0,
        )
        is None
    )
    assert MIN_QUORUM_WORKERS == 2


def test_target_workers_path_still_wins_when_reached() -> None:
    """窗口未打开但独立 worker 数达到 target_workers 时走旧 target 路径。"""
    assert (
        dispatch_reason(
            [0, 1, 2], 3, deadline=100.0, now=99.99,
            row_count=96, target_rows=512, batch_wait_s=0.0,
        )
        == "target"
    )
    # 行数阈值优先于一切。
    assert (
        dispatch_reason(
            [0, 1, 2], 3, deadline=100.0, now=99.99,
            row_count=512, target_rows=512, batch_wait_s=0.0,
        )
        == "rows"
    )
