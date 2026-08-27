"""RolloutBuffer 单路径的物化、分片、GAE 与 learner 契约测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from riichi_ppo_v1.tests.v18_fixtures import actor_inputs, critic_inputs
from riichi_ppo_v1.training.learner import (
    PPOLearner,
    accumulation_group_size,
    clip_branch_grad_norms,
    policy_entropy_values,
    rollout_update_targets,
    scheduled_entropy_coefficient,
)
from riichi_ppo_v1.training.rollout_buffer import RolloutBuffer
from riichi_ppo_v1.training.trajectory import Transition


def _random_transition(rng: np.random.Generator) -> Transition:
    pair_count = int(rng.integers(1, 8))
    action_ids = tuple(
        sorted(
            int(value)
            for value in rng.choice(np.arange(1, 80), size=(pair_count,), replace=False)
        )
    )
    actor = actor_inputs(batch=1, action_ids=action_ids)
    critic = critic_inputs(batch=1)
    actor_length = int(actor["actor_lengths"][0])
    critic_length = int(critic["critic_lengths"][0])
    return Transition(
        actor_factors=actor["actor_factors"][0, :actor_length].numpy().astype(np.int32),
        actor_numeric=actor["actor_numeric"][0, :actor_length].numpy().astype(np.float32),
        actor_length=actor_length,
        query_rows=actor["query_rows"][0, : 2 * pair_count].numpy().astype(np.int32),
        query_action_ids=actor["action_ids"][0, :pair_count].numpy().astype(np.int32),
        query_pair_counts=pair_count,
        legal_mask=actor["legal_mask"][0].numpy().astype(np.bool_),
        action=int(action_ids[0]),
        logprob=float(np.float32(rng.random())),
        value=float(np.float32(rng.random())),
        reward=float(np.float32(rng.random())),
        kyoku_reward=float(np.float32(rng.random())),
        done=bool(rng.integers(0, 2)),
        advantage=float(np.float32(rng.random())),
        critic_factors=critic["critic_factors"][0, :critic_length].numpy().astype(np.uint8),
        critic_length=critic_length,
    )


def _transitions(rng: np.random.Generator, count: int) -> list[Transition]:
    return [_random_transition(rng) for _ in range(count)]


def _learner_kwargs(**overrides: object) -> dict[str, object]:
    """基础 PPO learner 超参数(与既有单测保持一致,可覆盖)。"""
    kwargs: dict[str, object] = {
        "learning_rate": 1e-4,
        "profile_enabled": False,
        "profile_cuda_sync": False,
        "update_epochs": 1,
        "minibatch_size": 3,
        "ppo_clip": 0.2,
        "value_coef": 0.5,
        "entropy_start": 0.01,
        "entropy_middle": 0.005,
        "entropy_end": 0.001,
        "entropy_middle_fraction": 0.5,
        "entropy_loss_mode": "normalized",
        "max_grad_norm": 0.5,
        "actor_max_grad_norm": 0.5,
        "shared_max_grad_norm": 0.5,
        "critic_max_grad_norm": 1.0,
        "target_kl": 0.0,
        "target_kl_check_interval": 8,
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
        "critic_private_embedding_grad_scale": 0.25,
        "bucket_window_multiplier": 8,
    }
    kwargs.update(overrides)
    return kwargs


def _assert_buffers_equal(left: RolloutBuffer, right: RolloutBuffer) -> None:
    assert left.size == right.size
    left_arrays = {
        name: value for name, value in vars(left).items()
        if isinstance(value, np.ndarray)
    }
    right_arrays = {
        name: value for name, value in vars(right).items()
        if isinstance(value, np.ndarray)
    }
    assert set(left_arrays) == set(right_arrays)
    for name in left_arrays:
        np.testing.assert_array_equal(left_arrays[name], right_arrays[name], err_msg=name)


def _assert_row_prefix_and_zero_padding(
    actual: torch.Tensor,
    row: int,
    expected: np.ndarray,
) -> None:
    length = expected.shape[0]
    expected_tensor = torch.from_numpy(expected).to(dtype=actual.dtype)
    torch.testing.assert_close(actual[row, :length], expected_tensor)
    if length < actual.shape[1]:
        assert torch.count_nonzero(actual[row, length:]) == 0


def test_collate_preserves_all_segments_and_padding() -> None:
    transitions = _transitions(np.random.default_rng(0), 32)
    buffer = RolloutBuffer(transitions)
    indices = np.random.default_rng(1).permutation(len(transitions))[:16]
    batch = buffer.collate(indices)

    for row, index in enumerate(indices):
        item = transitions[int(index)]
        _assert_row_prefix_and_zero_padding(batch["actor_factors"], row, item.actor_factors)
        _assert_row_prefix_and_zero_padding(batch["actor_numeric"], row, item.actor_numeric)
        _assert_row_prefix_and_zero_padding(batch["query_rows"], row, item.query_rows)
        _assert_row_prefix_and_zero_padding(
            batch["query_action_ids"], row, item.query_action_ids,
        )
        expected_critic = (
            item.critic_factors
            if item.critic_factors is not None
            else np.zeros((0, 10), dtype=np.uint8)
        )
        _assert_row_prefix_and_zero_padding(batch["critic_factors"], row, expected_critic)
        torch.testing.assert_close(batch["legal_mask"][row], torch.from_numpy(item.legal_mask))
        assert batch["actor_lengths"][row] == item.actor_length
        assert batch["query_pair_counts"][row] == item.query_pair_counts
        assert batch["critic_lengths"][row] == item.critic_length
        assert batch["actions"][row] == item.action
        assert batch["old_logprobs"][row] == np.float32(item.logprob)
        assert batch["advantages"][row] == np.float32(item.advantage)


def test_bucketed_minibatches_are_deterministic_and_cover_all_rows() -> None:
    buffer = RolloutBuffer(_transitions(np.random.default_rng(3), 1500))
    left = buffer.bucketed_minibatches(
        512, rng=np.random.default_rng(42), bucket_window_multiplier=8,
    )
    right = buffer.bucketed_minibatches(
        512, rng=np.random.default_rng(42), bucket_window_multiplier=8,
    )
    assert len(left) == len(right) == 3
    assert sorted(int(index) for batch in left for index in batch) == list(range(len(buffer)))
    for first, second in zip(left, right, strict=True):
        np.testing.assert_array_equal(first, second)
        assert len(first) <= 512
        assert len(set(int(value) for value in first)) == len(first)


def test_concatenate_and_select_are_elementwise_exact() -> None:
    transitions = _transitions(np.random.default_rng(19), 37)
    merged = RolloutBuffer.concatenate([
        RolloutBuffer(transitions[:17]), RolloutBuffer(transitions[17:]),
    ])
    _assert_buffers_equal(merged, RolloutBuffer(transitions))
    array_count, array_bytes = merged.payload_stats()
    assert array_count < len(transitions)
    assert array_bytes > 0

    indices = [36, 0, 18, 18, 7, 29]
    _assert_buffers_equal(
        merged.select(indices),
        RolloutBuffer([transitions[index] for index in indices]),
    )


def test_rollout_target_math_matches_frozen_formula() -> None:
    transitions = _transitions(np.random.default_rng(23), 31)
    buffer = RolloutBuffer(transitions)
    advantages, returns, return_mean, return_std = rollout_update_targets(buffer, gamma=0.97)

    raw_advantages = np.asarray([item.advantage for item in transitions], dtype=np.float32)
    expected_advantages = (
        (raw_advantages - raw_advantages.mean(dtype=np.float64))
        / (raw_advantages.std(dtype=np.float64) + 1e-8)
    ).astype(np.float32)
    expected_returns = (
        np.asarray([item.value for item in transitions], dtype=np.float32)
        + raw_advantages
    ).astype(np.float32)
    np.testing.assert_array_equal(advantages, expected_advantages)
    np.testing.assert_array_equal(returns, expected_returns)
    assert return_mean == float(expected_returns.mean(dtype=np.float64))
    assert return_std == float(expected_returns.std(dtype=np.float64))
    assert not np.array_equal(returns, raw_advantages)


def test_entropy_schedule_hits_start_middle_and_end() -> None:
    total = 150
    middle_update = round(total * 0.33)
    assert scheduled_entropy_coefficient(
        0.020, 0.0045, 1, total, middle=0.012, middle_fraction=0.33,
    ) == pytest.approx(0.020)
    assert scheduled_entropy_coefficient(
        0.020, 0.0045, middle_update, total, middle=0.012, middle_fraction=0.33,
    ) == pytest.approx(0.012)
    assert scheduled_entropy_coefficient(
        0.020, 0.0045, total, total, middle=0.012, middle_fraction=0.33,
    ) == pytest.approx(0.0045)


def test_branch_clipping_keeps_independent_scales() -> None:
    actor = torch.nn.Parameter(torch.tensor([0.0]))
    shared = torch.nn.Parameter(torch.tensor([0.0]))
    critic = torch.nn.Parameter(torch.tensor([0.0]))
    actor.grad = torch.tensor([10.0])
    shared.grad = torch.tensor([0.25])
    critic.grad = torch.tensor([2.0])
    metrics = clip_branch_grad_norms(
        {"actor": [actor], "shared": [shared], "critic": [critic]},
        {"actor": 1.0, "shared": 1.0, "critic": 4.0},
    )
    assert metrics["grad_norm_actor_pre_clip"] == pytest.approx(10.0)
    assert metrics["grad_norm_actor_post_clip"] == pytest.approx(1.0)
    assert metrics["grad_norm_shared_post_clip"] == pytest.approx(0.25)
    assert metrics["grad_norm_critic_post_clip"] == pytest.approx(2.0)
    assert actor.grad.item() == pytest.approx(1.0)
    assert shared.grad.item() == pytest.approx(0.25)
    assert critic.grad.item() == pytest.approx(2.0)


def test_accumulation_tail_group_uses_actual_group_size() -> None:
    divisors = [
        accumulation_group_size(10, 4, index)
        for index in range(10)
    ]
    assert divisors == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]
    assert [
        accumulation_group_size(3, 8, index)
        for index in range(3)
    ] == [3, 3, 3]


def test_learner_accepts_only_rollout_buffer() -> None:
    transitions = _transitions(np.random.default_rng(7), 9)
    learner = PPOLearner("v18", "cpu", **_learner_kwargs())
    metrics = learner.update(RolloutBuffer(transitions), shuffle_seed=123)
    for name in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(metrics[name]), name
    for name in (
        "value_explained_variance_lambda",
        "value_explained_variance_mc",
        "lambda_return_mean",
        "mc_return_mean",
        "grad_norm_actor_pre_clip",
        "grad_norm_shared_pre_clip",
        "grad_norm_critic_pre_clip",
    ):
        assert name in metrics
    assert metrics["update/executed_minibatches"] == 3.0
    assert metrics["update/executed_transition_samples"] == 9.0
    with pytest.raises(TypeError, match="RolloutBuffer"):
        learner.update(transitions, shuffle_seed=123)  # type: ignore[arg-type]


def test_target_kl_guardrail_early_stops_update() -> None:
    """target_kl 超标时:第 8 个 optimizer step 后触发提前停止,剩余
    minibatch/epoch 不再执行,update 正常返回汇总指标(不崩溃)。

    随机初始化模型与 rollout old logprob 的差异必然把 KL 推到 0.01 以上;
    DDP 下各 rank 的全局 KL 聚合依赖真实 NCCL 进程组,由双卡 smoke 覆盖。
    """
    transitions = RolloutBuffer(_transitions(np.random.default_rng(11), 40))
    learner = PPOLearner(
        "v18", "cpu",
        **_learner_kwargs(target_kl=0.01, update_epochs=4, minibatch_size=8),
    )
    metrics = learner.update(transitions, shuffle_seed=7)
    assert metrics["update/early_stop"] == 1.0
    assert metrics["update/epochs_completed"] < 4.0
    assert 0.0 < metrics["update/executed_minibatches"] < metrics["update/planned_minibatches"]
    # 检查间隔为 8:第一次检查恰好在第 8 个 optimizer step 之后。
    assert metrics["update/executed_minibatches"] == 8.0
    for name in ("loss", "policy_loss", "value_loss", "approx_kl", "entropy"):
        assert np.isfinite(metrics[name]), name


def test_target_kl_disabled_runs_all_epochs() -> None:
    """target_kl=0 时 guardrail 关闭,完整跑完全部 epoch 不提前停止。"""
    transitions = RolloutBuffer(_transitions(np.random.default_rng(11), 40))
    learner = PPOLearner(
        "v18", "cpu",
        **_learner_kwargs(target_kl=0.0, update_epochs=4, minibatch_size=8),
    )
    metrics = learner.update(transitions, shuffle_seed=7)
    assert metrics["update/early_stop"] == 0.0
    assert metrics["update/epochs_completed"] == 4.0
    assert metrics["update/executed_minibatches"] == metrics["update/planned_minibatches"]


def test_checkpoint_resume_restores_adam_moments_and_schedule(tmp_path) -> None:
    """exact resume:恢复 Adam 一阶/二阶矩与 step 计数、iteration 与
    LR/entropy schedule 位置;恢复后继续 update 与不中断连续 update 一致。

    注意每个 learner 必须使用独立 buffer:update 会原地把
    transitions.advantages 覆盖为归一化值。
    """

    def make_buffer() -> RolloutBuffer:
        return RolloutBuffer(_transitions(np.random.default_rng(5), 24))

    kwargs = _learner_kwargs(update_epochs=2, minibatch_size=8, target_kl=100.0)
    torch.manual_seed(0)
    plain = PPOLearner("v18", "cpu", **kwargs)
    plain.update(make_buffer(), shuffle_seed=7)
    torch.manual_seed(0)
    resumed = PPOLearner("v18", "cpu", **kwargs)
    resumed.update(make_buffer(), shuffle_seed=7)
    path = str(tmp_path / "resume.pt")
    resumed.save(path, {"phase": "test"})

    loaded = PPOLearner("v18", "cpu", **kwargs)
    loaded.load(path)
    assert loaded.iteration == resumed.iteration == 1
    saved_state = resumed.optimizer.state_dict()
    loaded_state = loaded.optimizer.state_dict()
    assert set(saved_state) == set(loaded_state)
    assert saved_state["param_groups"] == loaded_state["param_groups"]
    for group_index in saved_state["state"]:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                loaded_state["state"][group_index][key],
                saved_state["state"][group_index][key],
            )
    # schedule 从恢复的 iteration 继续:恢复后的第 2 次 update 与不中断连续
    # 运行的第 2 次 update 行为完全一致(RNG/optimizer 均精确恢复)。
    metrics_after_resume = loaded.update(make_buffer(), shuffle_seed=9)
    metrics_continuous = plain.update(make_buffer(), shuffle_seed=9)
    for name in (
        "loss", "policy_loss", "value_loss", "entropy", "approx_kl",
        "system/actor_learning_rate", "system/critic_learning_rate",
        "system/entropy_coef", "update/executed_minibatches",
    ):
        assert metrics_after_resume[name] == pytest.approx(
            metrics_continuous[name], rel=1e-6,
        ), name


def test_normalized_entropy_uniform_over_legal_actions() -> None:
    """normalized entropy = H/log(max(n,2)):合法动作上均匀分布时恰为 1.0,
    与合法动作数无关,从而不同局面(2 个 vs 8~10 个合法动作)熵系数尺度一致;
    非法位置 -inf 不污染结果。"""
    num_actions = 241
    logprob_rows: list[torch.Tensor] = []
    prob_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    expected_raw: list[float] = []
    for legal_count in (2, 8, 21):
        mask = torch.zeros(num_actions, dtype=torch.bool)
        mask[:legal_count] = True
        logp = torch.full((num_actions,), float("-inf"))
        logp[:legal_count] = math.log(1.0 / legal_count)
        # 非法位置概率故意非零:验证掩码清零而不是依赖输入为 0。
        prob = torch.full((num_actions,), 0.5)
        prob[:legal_count] = 1.0 / legal_count
        logprob_rows.append(logp)
        prob_rows.append(prob)
        mask_rows.append(mask)
        expected_raw.append(math.log(legal_count))
    raw, normalized = policy_entropy_values(
        torch.stack(logprob_rows), torch.stack(prob_rows), torch.stack(mask_rows),
    )
    torch.testing.assert_close(raw, torch.tensor(expected_raw))
    torch.testing.assert_close(normalized, torch.ones(3))

    # 单合法动作:clamp_min(2) 保证分母不为 log(1)=0。
    one_mask = torch.zeros(num_actions, dtype=torch.bool)
    one_mask[0] = True
    one_logp = torch.full((num_actions,), float("-inf"))
    one_logp[0] = 0.0
    one_prob = torch.zeros(num_actions)
    one_prob[0] = 1.0
    raw_one, normalized_one = policy_entropy_values(
        one_logp[None, :], one_prob[None, :], one_mask[None, :],
    )
    assert raw_one.item() == pytest.approx(0.0)
    assert normalized_one.item() == pytest.approx(0.0)


def test_accumulation_clips_and_steps_only_at_group_boundaries(monkeypatch) -> None:
    """accumulation>1:只在 optimizer step 前裁剪梯度;尾组按实际 minibatch
    数缩放(10 条 → 组划分 [3,3,3,1],只有 2 次 optimizer step)。"""
    import riichi_ppo_v1.training.learner as learner_module

    transitions = RolloutBuffer(_transitions(np.random.default_rng(13), 10))
    group_sizes: list[int] = []
    real_group_size = learner_module.accumulation_group_size
    monkeypatch.setattr(
        learner_module, "accumulation_group_size",
        lambda planned, steps, index: (
            group_sizes.append(int(real_group_size(planned, steps, index)))
            or real_group_size(planned, steps, index)
        ),
    )
    clip_calls: list[tuple[object, object]] = []
    real_clip = learner_module.clip_branch_grad_norms
    monkeypatch.setattr(
        learner_module, "clip_branch_grad_norms",
        lambda parameters, max_norms: (
            clip_calls.append((parameters, max_norms))
            or real_clip(parameters, max_norms)
        ),
    )
    learner = PPOLearner(
        "v18", "cpu",
        **_learner_kwargs(
            update_epochs=1, minibatch_size=3, gradient_accumulation_steps=3,
        ),
    )
    metrics = learner.update(transitions, shuffle_seed=3)
    assert group_sizes == [3, 3, 3, 1]
    # 4 个 minibatch 只有 2 个累积组 → 2 次 optimizer step,clip 只在 step 前发生。
    assert len(clip_calls) == 2
    assert metrics["update/executed_minibatches"] == 4.0
    assert metrics["update/executed_transition_samples"] == 10.0
    for name in ("grad_norm_actor_pre_clip", "grad_norm_shared_pre_clip",
                 "grad_norm_critic_pre_clip", "loss", "value_loss"):
        assert np.isfinite(metrics[name]), name
