"""discounted_empirical_returns 分段向量化与旧 Python 循环的逐位一致测试。"""

from __future__ import annotations

import numpy as np
import pytest

from riichi_ppo_v1.training.learner import discounted_empirical_returns
from riichi_ppo_v1.training.rollout_buffer import RolloutBuffer

from .test_rollout_buffer import _random_transition


def _reference_loop_returns(buffer: RolloutBuffer, gamma: float) -> np.ndarray:
    """旧实现的逐字拷贝(向量化改造前 learner.py 的原循环),作为对照。"""
    returns = np.zeros(len(buffer), dtype=np.float32)
    running = 0.0
    rewards = np.asarray(buffer.rewards, dtype=np.float32)
    done = np.asarray(buffer.done, dtype=np.bool_)
    for index in range(len(buffer) - 1, -1, -1):
        if done[index]:
            running = 0.0
        running = float(rewards[index]) + float(gamma) * running
        returns[index] = np.float32(running)
    return returns


def _random_buffer(rng: np.random.Generator, count: int) -> RolloutBuffer:
    """随机 buffer:reward 用含负数/零/大值的正态分布,done 每行随机。"""
    transitions = []
    for _ in range(count):
        item = _random_transition(rng)
        item.reward = float(np.float32(rng.normal(scale=10.0)))
        item.done = bool(rng.integers(0, 2))
        transitions.append(item)
    return RolloutBuffer(transitions)


def _override(buffer: RolloutBuffer, rewards: list[float], done: list[bool]) -> None:
    buffer.rewards = np.asarray(rewards, dtype=np.float32)
    buffer.done = np.asarray(done, dtype=np.bool_)


def test_gamma1_matches_reference_loop_bitwise() -> None:
    """gamma=1.0(生产值):随机 buffer 上与旧循环 np.array_equal。"""
    buffer = _random_buffer(np.random.default_rng(20260828), 512)
    assert np.array_equal(
        discounted_empirical_returns(buffer, 1.0),
        _reference_loop_returns(buffer, 1.0),
    )


def test_gamma1_int_argument_uses_vectorized_path() -> None:
    """gamma 以 int 传入(如 1)时同样分派到向量化路径且结果一致。"""
    buffer = _random_buffer(np.random.default_rng(7), 128)
    assert np.array_equal(
        discounted_empirical_returns(buffer, 1),
        _reference_loop_returns(buffer, 1.0),
    )


@pytest.mark.parametrize(
    "done",
    [
        np.ones(64, dtype=np.bool_),  # 每行都是局终点(段长 1)。
        np.zeros(64, dtype=np.bool_),  # 全程无终局:整段尾段。
        np.asarray(  # 首行/连续/末行均为边界。
            [i in (0, 10, 11, 12, 63) for i in range(64)], dtype=np.bool_,
        ),
        np.asarray(  # 尾段不终局(rollout 截断形态)。
            [i in (10, 50) for i in range(64)], dtype=np.bool_,
        ),
    ],
)
def test_gamma1_segment_boundaries_bitwise(done: np.ndarray) -> None:
    buffer = _random_buffer(np.random.default_rng(11), len(done))
    buffer.done = done
    result = discounted_empirical_returns(buffer, 1.0)
    assert result.dtype == np.float32
    assert result.shape == (len(done),)
    assert np.array_equal(result, _reference_loop_returns(buffer, 1.0))


def test_gamma1_empty_arrays_return_empty() -> None:
    """gamma1 向量化 helper 对空数组返回空结果(RolloutBuffer 不支持空构造)。"""
    from riichi_ppo_v1.training.learner import (
        _discounted_empirical_returns_gamma1,
    )

    result = _discounted_empirical_returns_gamma1(
        np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.bool_),
    )
    assert result.dtype == np.float32
    assert result.shape == (0,)


def test_hand_computed_values() -> None:
    """手工小例:分段重置与尾段语义按定义核对(gamma=1 与 gamma=0.5)。"""
    buffer = _random_buffer(np.random.default_rng(3), 5)
    _override(buffer, [1.0, 2.0, 3.0, 4.0, 5.0], [False, False, False, True, False])
    # 局终点在 index 3:index0-3 段内后缀和 10/9/7/4;index4 是未终局尾段,只含自身。
    assert np.array_equal(
        discounted_empirical_returns(buffer, 1.0),
        np.asarray([10.0, 9.0, 7.0, 4.0, 5.0], dtype=np.float32),
    )
    # gamma=0.5(旧循环路径):4;3+0.5*4=5;2+0.5*5=4.5;1+0.5*4.5=3.25;尾段 5。
    assert np.array_equal(
        discounted_empirical_returns(buffer, 0.5),
        np.asarray([3.25, 4.5, 5.0, 4.0, 5.0], dtype=np.float32),
    )


def test_gamma_not_one_matches_reference_loop_bitwise() -> None:
    """gamma≠1:保留的旧循环路径与对照实现 np.array_equal。"""
    for gamma in (0.95, 0.999):
        buffer = _random_buffer(np.random.default_rng(23), 256)
        assert np.array_equal(
            discounted_empirical_returns(buffer, gamma),
            _reference_loop_returns(buffer, gamma),
        )
