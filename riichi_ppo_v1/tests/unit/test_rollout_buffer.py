"""RolloutBuffer(SoA 紧凑缓冲)单元测试:与旧 materialize_host_batch 逐元素等价。"""

from __future__ import annotations

import numpy as np
import torch

from riichi_ppo_v1.training.learner import (
    PPOLearner,
    length_bucketed_minibatches,
    materialize_host_batch,
)
from riichi_ppo_v1.training.rollout_buffer import RolloutBuffer
from riichi_ppo_v1.training.trajectory import Transition


def _random_transition(rng: np.random.Generator) -> Transition:
    history_length = int(rng.integers(20, 60))
    snapshot_length = int(rng.integers(2, 8))
    pair_count = int(rng.integers(1, 8))
    critic_length = int(rng.integers(0, 30))
    # 生成满足 forward_v16 形状约束的合法 query 行(与 test_v16_learner.transition 一致:
    # 动作 id 取小集合,query_type 为 1/2,answer 取 0..5 且在卡片基数内)。
    action_ids = rng.integers(0, 10, size=(pair_count,)).astype(np.int32)
    query_rows = np.zeros((2 * pair_count, 15), dtype=np.int32)
    for pair, action_id in enumerate(action_ids):
        query_rows[2 * pair, 0] = 1  # QUERY_OFFENSE
        query_rows[2 * pair + 1, 0] = 2  # QUERY_DEFENSE
        query_rows[2 * pair, 1] = int(action_id)
        query_rows[2 * pair + 1, 1] = int(action_id)
        query_rows[2 * pair, 5:] = rng.integers(0, 6, size=(10,))
        query_rows[2 * pair + 1, 5:] = rng.integers(0, 6, size=(10,))
    legal_mask = np.zeros(241, dtype=np.bool_)
    legal_mask[action_ids] = True
    return Transition(
        history_factors=rng.integers(0, 4, (history_length, 10)).astype(np.uint8),
        history_numeric=rng.random((history_length, 8)).astype(np.float32),
        history_length=history_length,
        snapshot_kinds=rng.integers(0, 4, (snapshot_length,)).astype(np.uint8),
        snapshot_cat=rng.integers(0, 4, (snapshot_length, 4)).astype(np.uint8),
        snapshot_num=rng.random((snapshot_length, 7)).astype(np.float32),
        snapshot_length=snapshot_length,
        query_rows=query_rows,
        query_action_ids=action_ids,
        query_pair_counts=pair_count,
        legal_mask=legal_mask,
        action=int(action_ids[0]),
        logprob=float(rng.random()),
        value=float(rng.random()),
        reward=float(rng.random()),
        done=bool(rng.integers(0, 2)),
        advantage=float(rng.random()),
        critic_factors=(
            rng.integers(0, 4, (critic_length, 10)).astype(np.uint8)
            if critic_length
            else None
        ),
        critic_length=critic_length,
    )


def _transitions(rng: np.random.Generator, count: int) -> list[Transition]:
    return [_random_transition(rng) for _ in range(count)]


def test_collate_matches_materialize_host_batch() -> None:
    rng = np.random.default_rng(0)
    transitions = _transitions(rng, 2000)
    buffer = RolloutBuffer(transitions)
    for trial in range(3):
        indices = np.random.default_rng(trial).permutation(len(transitions))[:512]
        selected = [transitions[int(index)] for index in indices]
        host = materialize_host_batch(selected)
        soa = buffer.collate(indices)
        for key in host:
            assert host[key].dtype == soa[key].dtype, key
            assert host[key].shape == soa[key].shape, key
            if host[key].dtype == torch.float32:
                assert torch.allclose(host[key], soa[key], atol=1e-6, rtol=1e-6), key
            else:
                assert torch.equal(host[key], soa[key]), key


def test_bucketed_minibatches_match_legacy() -> None:
    rng = np.random.default_rng(3)
    transitions = _transitions(rng, 1500)
    buffer = RolloutBuffer(transitions)
    seed = np.random.default_rng(42)
    legacy = length_bucketed_minibatches(transitions, 512, rng=seed)
    soa = buffer.bucketed_minibatches(512, rng=np.random.default_rng(42))
    assert len(legacy) == len(soa)
    for left, right in zip(legacy, soa, strict=True):
        assert np.array_equal(left, right)


def test_soa_update_matches_legacy_losses() -> None:
    """同一批 transition + 同一 shuffle_seed,SoA 与逐样本 collate 应得到一致 loss。"""
    rng = np.random.default_rng(7)
    transitions = _transitions(rng, 9)

    kwargs = {
        "learning_rate": 1e-4,
        "profile_enabled": False,
        "profile_cuda_sync": False,
        "update_epochs": 1,
        "minibatch_size": 3,
        "ppo_clip": 0.2,
        "value_coef": 0.5,
        "entropy_start": 0.01,
        "entropy_end": 0.001,
        "max_grad_norm": 0.5,
        "target_kl": 0.0,
        "total_updates": 50,
        "warmup_fraction": 0.02,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-5,
        "weight_decay": 0.01,
        "value_loss": "huber",
        "value_target_normalization": "batch_std",
        "value_target_std_floor": 1e-2,
        "gamma": 1.0,
        "critic_bootstrap_updates": 0,
    }
    legacy = PPOLearner("v16", "cpu", update_use_soa=False, **kwargs)
    soa = PPOLearner("v16", "cpu", update_use_soa=True, **kwargs)
    # 两个 learner 随机初始化不同;先对齐模型权重,保证同一起点才能比较数值。
    soa.model.load_state_dict(legacy.model.state_dict())
    legacy_metrics = legacy.update(transitions, shuffle_seed=123)
    soa_metrics = soa.update(transitions, shuffle_seed=123)
    for name in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isclose(soa_metrics[name], legacy_metrics[name], atol=1e-6), name
    assert soa_metrics["update/executed_minibatches"] == legacy_metrics["update/executed_minibatches"]
    assert soa_metrics["update/executed_transition_samples"] == legacy_metrics["update/executed_transition_samples"]
