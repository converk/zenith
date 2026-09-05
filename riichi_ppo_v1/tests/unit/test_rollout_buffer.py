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
    transfer_batch_to_device,
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
        # V19 合成信念标签:形状与 Rust 标签协议一致,便于 collate/learner
        # 全链路测试;hand 计数 0..4、shanten 0..8、wait/danger 0/1。
        belief_hand=rng.integers(0, 5, size=102, dtype=np.uint8),
        belief_shanten=rng.integers(0, 9, size=3, dtype=np.uint8),
        belief_wait=rng.integers(0, 2, size=105, dtype=np.uint8),
        belief_danger=rng.integers(0, 2, size=102, dtype=np.uint8),
        belief_loss=(rng.random(102) * 24000.0).astype(np.float32),
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
    for name, left_value in left_arrays.items():
        np.testing.assert_array_equal(left_value, right_arrays[name], err_msg=name)


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


def test_collate_belief_fixed_fields_shapes_and_values() -> None:
    """V19 信念五头固定字段随 collate 输出,形状/数值/dtype 与 Transition 一致。"""
    transitions = _transitions(np.random.default_rng(41), 16)
    buffer = RolloutBuffer(transitions)
    indices = np.arange(5)
    batch = buffer.collate(indices)
    assert batch["belief_hand"].shape == (5, 102)
    assert batch["belief_shanten"].shape == (5, 3)
    assert batch["belief_wait"].shape == (5, 105)
    assert batch["belief_danger"].shape == (5, 102)
    assert batch["belief_loss"].shape == (5, 102)
    assert batch["belief_loss"].dtype == torch.float32
    assert batch["belief_present"].all()
    for row, index in enumerate(indices):
        item = transitions[int(index)]
        torch.testing.assert_close(
            batch["belief_hand"][row], torch.from_numpy(item.belief_hand),
        )
        torch.testing.assert_close(
            batch["belief_shanten"][row], torch.from_numpy(item.belief_shanten),
        )
        torch.testing.assert_close(
            batch["belief_wait"][row], torch.from_numpy(item.belief_wait),
        )
        torch.testing.assert_close(
            batch["belief_danger"][row], torch.from_numpy(item.belief_danger),
        )
        torch.testing.assert_close(
            batch["belief_loss"][row], torch.from_numpy(item.belief_loss),
        )


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
    advantages, returns, return_mean, return_std = rollout_update_targets(buffer)

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
    learner = PPOLearner("v19", "cpu", **_learner_kwargs())
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
        "v19", "cpu",
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
        "v19", "cpu",
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
    # torch.compile(△ 级,默认开启)会改变浮点归约顺序;本测试在 CPU 设备
    # 上断言「恢复与连续运行一致」,显式关闭 compile 保持逐位口径。
    kwargs["torch_compile"] = False
    torch.manual_seed(0)
    plain = PPOLearner("v19", "cpu", **kwargs)
    plain.update(make_buffer(), shuffle_seed=7)
    torch.manual_seed(0)
    resumed = PPOLearner("v19", "cpu", **kwargs)
    resumed.update(make_buffer(), shuffle_seed=7)
    path = str(tmp_path / "resume.pt")
    resumed.save(path, {"phase": "test"})

    loaded = PPOLearner("v19", "cpu", **kwargs)
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
        "v19", "cpu",
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


def test_factor_flatten_compacts_to_uint8_and_fail_closed() -> None:
    """V19 token 因子行压成 uint8 存储,collate 以 uint8 输出紧凑索引
    (值域 < 256,传输体积缩 8 倍,由 transfer_batch_to_device 在 GPU 侧
    恢复 long,数值逐位一致);超出 255 的因子列 fail-closed,防止静默回绕。"""
    transitions = _transitions(np.random.default_rng(31), 12)
    buffer = RolloutBuffer(transitions)
    assert buffer.actor_factors_flat.dtype == np.uint8
    assert buffer.query_ids_flat.dtype == np.uint8
    batch = buffer.collate(np.arange(5))
    assert batch["actor_factors"].dtype == torch.uint8
    assert batch["query_action_ids"].dtype == torch.uint8
    assert batch["critic_factors"].dtype == torch.uint8
    # GPU 侧恢复 long:transfer_batch_to_device 按 dtype 分派(数值逐位一致)。
    device_batch = transfer_batch_to_device(batch, torch.device("cpu"))
    assert device_batch["actor_factors"].dtype == torch.int64
    assert device_batch["query_action_ids"].dtype == torch.int64
    assert device_batch["critic_factors"].dtype == torch.int64
    np.testing.assert_array_equal(
        device_batch["actor_factors"].numpy(),
        batch["actor_factors"].numpy().astype(np.int64),
    )

    # 超 255 的因子必须显式报错(绝不能静默回绕)。
    bad = _transitions(np.random.default_rng(33), 3)
    bad[1].actor_factors = bad[1].actor_factors.copy()
    bad[1].actor_factors[0, 0] = 300
    with pytest.raises(ValueError, match="uint8 range"):
        RolloutBuffer(bad)


def test_prefetch_collate_matches_serial_update() -> None:
    """collate 预取线程与串行路径产出完全一致的 minibatch 与训练指标。"""
    kwargs = _learner_kwargs(
        update_epochs=2, minibatch_size=8, target_kl=0.0,
    )
    torch.manual_seed(1234)
    serial = PPOLearner("v19", "cpu", **kwargs)
    torch.manual_seed(1234)
    prefetch = PPOLearner(
        "v19", "cpu", **{**kwargs, "update_collate_prefetch": True},
    )
    serial_metrics = serial.update(
        RolloutBuffer(_transitions(np.random.default_rng(21), 40)),
        shuffle_seed=7,
    )
    prefetch_metrics = prefetch.update(
        RolloutBuffer(_transitions(np.random.default_rng(21), 40)),
        shuffle_seed=7,
    )
    assert prefetch_metrics["update/executed_minibatches"] == (
        serial_metrics["update/executed_minibatches"]
    )
    assert prefetch_metrics["update/executed_transition_samples"] == (
        serial_metrics["update/executed_transition_samples"]
    )
    for name in (
        "loss", "policy_loss", "value_loss", "entropy", "approx_kl",
        "clipfrac", "grad_norm_actor_pre_clip", "grad_norm_shared_pre_clip",
        "grad_norm_critic_pre_clip", "ratio_p95",
    ):
        assert prefetch_metrics[name] == pytest.approx(
            serial_metrics[name], rel=1e-6,
        ), name


def test_prefetch_early_stop_does_not_hang() -> None:
    """预取模式下 target_kl 提前停止:线程被安全停止,不挂起,指标口径不变。"""
    transitions = RolloutBuffer(_transitions(np.random.default_rng(11), 40))
    learner = PPOLearner(
        "v19", "cpu",
        **_learner_kwargs(
            target_kl=0.01, update_epochs=4, minibatch_size=8,
            update_collate_prefetch=True,
        ),
    )
    metrics = learner.update(transitions, shuffle_seed=7)
    assert metrics["update/early_stop"] == 1.0
    assert metrics["update/epochs_completed"] < 4.0
    assert metrics["update/executed_minibatches"] == 8.0
    for name in ("loss", "policy_loss", "value_loss", "approx_kl", "entropy"):
        assert np.isfinite(metrics[name]), name


def test_update_reports_stage_gap_timings() -> None:
    """B1 插计:update 上报未插计段计时(learner_wall / collate_wait /
    collate_put_block),全部为纯计时键,不影响任何训练指标数值。"""
    transitions = RolloutBuffer(_transitions(np.random.default_rng(29), 40))
    learner = PPOLearner(
        "v19", "cpu",
        **_learner_kwargs(
            update_epochs=2, minibatch_size=8, target_kl=0.0,
            update_collate_prefetch=True, profile_enabled=True,
        ),
    )
    metrics = learner.update(transitions, shuffle_seed=7)
    minibatches = metrics["update/executed_minibatches"]
    # learner_wall:整个 rank 侧 update 调用恰有一条记录,总时长为正。
    assert metrics["timing/update/learner_wall/count"] == 1.0
    assert metrics["timing/update/learner_wall/total_s"] > 0.0
    # collate_wait:主线程逐 minibatch 等待预取队列,每批一条。
    assert metrics["timing/update/collate_wait/count"] == minibatches
    assert metrics["timing/update/collate_wait/total_s"] >= 0.0
    # collate_put_block:预取线程 put 反压累计,整个 update 汇总一条。
    assert metrics["timing/update/collate_put_block/count"] == 1.0
    assert metrics["timing/update/collate_put_block/total_s"] >= 0.0


def test_prefetch_propagates_collate_exception(monkeypatch) -> None:
    """预取线程内 collate 抛错:异常经队列安全传播到主线程,不挂起。"""
    transitions = RolloutBuffer(_transitions(np.random.default_rng(17), 30))
    original_collate = transitions.collate
    calls = {"count": 0}

    def failing_collate(indices, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("boom in collate")
        return original_collate(indices, **kwargs)

    monkeypatch.setattr(transitions, "collate", failing_collate)
    learner = PPOLearner(
        "v19", "cpu",
        **_learner_kwargs(
            update_epochs=2, minibatch_size=8, target_kl=0.0,
            update_collate_prefetch=True,
        ),
    )
    with pytest.raises(RuntimeError, match="boom in collate"):
        learner.update(transitions, shuffle_seed=7)


def test_reference_compile_flag_and_build_contract() -> None:
    """C 项旗标与构造契约:torch_compile_reference 默认跟随 torch_compile;
    _build_reference_model 产出冻结、eval 的 reference,state_dict 键与
    eager 完全一致(编译只挂在 forward_actor 方法上,模块本体不包装)。"""
    kwargs = _learner_kwargs(torch_compile=False)
    learner = PPOLearner("v19", "cpu", **kwargs)
    assert learner.torch_compile is False
    assert learner.torch_compile_reference is False  # 默认跟随 torch_compile
    overridden = PPOLearner(
        "v19", "cpu", **{**kwargs, "torch_compile_reference": True},
    )
    assert overridden.torch_compile_reference is True

    weights = {
        name: value.detach().clone()
        for name, value in learner.model.state_dict().items()
    }
    learner._build_reference_model(weights)
    assert learner.reference_model is not None
    assert learner.reference_model.training is False
    assert all(
        not parameter.requires_grad
        for parameter in learner.reference_model.parameters()
    )
    # 键集合与训练模型一致(checkpoint 契约:无 _orig_mod 前缀)。
    assert set(learner.reference_model.state_dict()) == set(weights)


def test_reference_compile_matches_eager_within_bf16_tolerance() -> None:
    """C 项数值对照:编译态 forward_actor 与 eager 的 reference logits 在
    bf16 内核形状级差异内一致(△ 级),-inf 非法位必须逐位一致。"""
    if not torch.cuda.is_available():
        pytest.skip("reference 编译对照需要 CUDA")
    from riichi_ppo_v1.model import KyokuTransformerActorCritic

    buffer = RolloutBuffer(_transitions(np.random.default_rng(97), 24))
    kwargs = _learner_kwargs(profile_enabled=False, torch_compile=False)
    source = PPOLearner("v19", "cuda:0", **kwargs)
    weights = {
        name: value.detach().clone()
        for name, value in source.model.state_dict().items()
    }
    del source
    eager = PPOLearner(
        "v19", "cuda:0", **{**kwargs, "torch_compile_reference": False},
    )
    compiled = PPOLearner(
        "v19", "cuda:0", **{**kwargs, "torch_compile_reference": True},
    )
    eager._build_reference_model(weights)
    compiled._build_reference_model(weights)
    eager_logits = eager._precompute_reference_logits(buffer)
    compiled_logits = compiled._precompute_reference_logits(buffer)
    eager_finite = torch.isfinite(eager_logits)
    assert torch.equal(eager_finite, torch.isfinite(compiled_logits))
    max_diff = float(
        (eager_logits[eager_finite] - compiled_logits[eager_finite]).abs().max()
    )
    # bf16 内核选择随编译图形状变化(既有 B2 预计算记录同量级 ≤2e-3,合成
    # 数据实测 ≤4e-3);容差只用于拦截结构性错误(索引/掩蔽错误会是 ±inf 或
    # 数量级差异),不做逐位断言。
    assert max_diff <= 1e-2, f"compiled reference logits diff {max_diff:.3e}"


def test_reference_precompute_pipeline_matches_serial() -> None:
    """SFT reference 预计算的 collate 后台线程化与串行版逐位一致。

    线程只做 host collate(纯 CPU numpy),chunk 输入与 GPU 前向顺序不变;
    以同一 RolloutBuffer、同一冻结模型分别走两条路径,断言 logits 逐位相等。
    """
    if not torch.cuda.is_available():
        pytest.skip("reference precompute pipeline 对比需要 CUDA")
    from riichi_ppo_v1.model import KyokuTransformerActorCritic

    transitions = _transitions(np.random.default_rng(77), 24)
    buffer = RolloutBuffer(transitions)
    model = KyokuTransformerActorCritic().cuda().eval()
    model.requires_grad_(False)
    kwargs = _learner_kwargs()
    kwargs["update_reference_precompute_batch_size"] = 8
    kwargs["total_updates"] = 50
    learner_a = PPOLearner("v19", "cuda:0", **{**kwargs, "profile_enabled": False})
    learner_a.reference_model = model
    learner_b = PPOLearner("v19", "cuda:0", **{**kwargs, "profile_enabled": False})
    learner_b.reference_model = model

    with torch.inference_mode():
        pipelined = learner_a._precompute_reference_logits(buffer)
    # 串行基线:直接内联复刻旧循环(collate+前向同序,无后台线程)。
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(buffer), 8):
            host_batch = buffer.collate(np.arange(start, min(start + 8, len(buffer))))
            host_batch.pop("critic_total_capacity", None)
            shared_capacity = host_batch.pop("shared_capacity", None)
            kind_row_plan = host_batch.pop("kind_row_plan", None)
            critic_kind_row_plan = host_batch.pop("critic_kind_row_plan", None)
            batch = transfer_batch_to_device(host_batch, learner_b.device, None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    batch["actor_factors"], batch["actor_numeric"], batch["actor_lengths"],
                    batch["query_action_ids"], batch["query_pair_counts"], batch["legal_mask"],
                    policy_only=True, validate_structure=learner_b.update_validate_structure,
                    shared_capacity=shared_capacity,
                    kind_row_plan=kind_row_plan, critic_kind_row_plan=critic_kind_row_plan,
                )["policy_logits"]
            chunks.append(logits)
    serial = torch.cat(chunks, dim=0)
    assert pipelined.shape == serial.shape
    assert torch.equal(pipelined, serial)
