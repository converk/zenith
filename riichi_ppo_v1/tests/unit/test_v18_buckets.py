"""V18 bucket 边界永久化测试（Python 侧；Rust 侧见 current_state_encoding.rs 测试）。"""

from __future__ import annotations

from riichi_ppo_v1.model.encoding_protocol import (
    bucket_d6,
    bucket_d9,
    bucket_o1,
    bucket_o2,
    bucket_o3,
    bucket_o5,
    bucket_o9,
)


def test_bucket_o1_boundaries() -> None:
    # 0..9 精确，10+ 截断到 10。
    assert bucket_o1(0) == 0
    assert bucket_o1(9) == 9
    assert bucket_o1(10) == 10
    assert bucket_o1(33) == 10
    assert bucket_o1(100) == 10


def test_bucket_o2_boundaries() -> None:
    # 0 精确；1-4/5-8/9-12/13-16/17-20/21+。
    assert bucket_o2(0) == 0
    assert bucket_o2(1) == 1
    assert bucket_o2(4) == 1
    assert bucket_o2(5) == 2
    assert bucket_o2(8) == 2
    assert bucket_o2(9) == 3
    assert bucket_o2(12) == 3
    assert bucket_o2(13) == 4
    assert bucket_o2(16) == 4
    assert bucket_o2(17) == 5
    assert bucket_o2(20) == 5
    assert bucket_o2(21) == 6
    assert bucket_o2(1000) == 6


def test_bucket_o3_boundaries() -> None:
    assert bucket_o3(None) == 0
    assert bucket_o3(1) == 1
    assert bucket_o3(13) == 13
    assert bucket_o3(14) == 13


def test_bucket_o5_boundaries() -> None:
    assert bucket_o5(None) == 0
    assert bucket_o5(1) == 1
    assert bucket_o5(4) == 4
    assert bucket_o5(5) == 5
    assert bucket_o5(6) == 5


def test_bucket_o9_boundaries() -> None:
    assert bucket_o9(0) == 0
    assert bucket_o9(4) == 4
    assert bucket_o9(5) == 5
    assert bucket_o9(6) == 5


def test_bucket_d6_boundaries() -> None:
    assert bucket_d6(0) == 0
    assert bucket_d6(3) == 3
    assert bucket_d6(4) == 4
    assert bucket_d6(7) == 4


def test_bucket_d9_boundaries() -> None:
    assert bucket_d9(None) == 5
    assert bucket_d9(0) == 0
    assert bucket_d9(4) == 4
    assert bucket_d9(5) == 4
