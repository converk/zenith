from __future__ import annotations

import torch
import pytest
from torch.nn import functional as F

from riichi_ppo_v1.training.inference import inference_autocast_config


def test_inference_dtype_fp32_disables_autocast() -> None:
    dtype, enabled = inference_autocast_config("fp32", device_type="cuda")
    assert dtype is torch.float32
    assert enabled is False


def test_inference_dtype_bf16_uses_hardware_gate() -> None:
    dtype, enabled = inference_autocast_config("bf16", device_type="cuda")
    assert dtype is torch.bfloat16
    assert enabled is bool(torch.cuda.is_bf16_supported())


def test_inference_dtype_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="inference_dtype"):
        inference_autocast_config("fp16", device_type="cuda")


def test_same_weights_double_forward_ratio_is_one() -> None:
    """参数未更新时:同一 batch 二次 forward 的 logits/logprob 应完全一致,
    ratio 非常接近 1(数值误差范围),作为 rollout/update dtype 一致性的地基。"""
    from riichi_ppo_v1.model import KyokuTransformerActorCritic
    from riichi_ppo_v1.tests.v18_fixtures import actor_inputs

    torch.manual_seed(0)
    model = KyokuTransformerActorCritic().eval()
    inputs = actor_inputs(batch=2, action_ids=(1, 7, 12))

    def forward() -> torch.Tensor:
        return model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            policy_only=True,
        )["policy_logits"]

    with torch.inference_mode():
        logits_a = forward()
        logits_b = forward()
    torch.testing.assert_close(logits_a, logits_b)
    logprob_a = F.log_softmax(logits_a, dim=-1)
    logprob_b = F.log_softmax(logits_b, dim=-1)
    # 非法动作 logits 为 -inf:ratio 只需在合法动作上接近 1。
    legal_diff = logprob_b - logprob_a
    assert torch.allclose(
        legal_diff[inputs["legal_mask"]],
        torch.zeros_like(legal_diff[inputs["legal_mask"]]),
        atol=1e-6,
    )
    legal_ratio = legal_diff[inputs["legal_mask"]].exp()
    assert torch.allclose(legal_ratio, torch.ones_like(legal_ratio), atol=1e-6)


def test_learner_and_rollout_share_dtype_gate() -> None:
    """rollout 与 learner update 的 autocast 开关由同一 inference_dtype
    配置驱动,避免 rollout=BF16 / update=FP32 的不一致组合。"""
    from riichi_ppo_v1.training.learner import PPOLearner

    kwargs = {
        "learning_rate": 1e-4,
        "actor_max_grad_norm": 0.5,
        "shared_max_grad_norm": 0.5,
        "critic_max_grad_norm": 1.0,
        "entropy_loss_mode": "normalized",
    }
    for dtype_name in ("bf16", "fp32"):
        learner = PPOLearner("v18", "cpu", inference_dtype=dtype_name, **kwargs)
        _dtype, inference_enabled = inference_autocast_config(dtype_name, device_type="cpu")
        assert learner.use_bf16 == inference_enabled
    # CUDA 下两条路径同样以硬件 BF16 支持为门。
    _dtype, cuda_enabled = inference_autocast_config("bf16", device_type="cuda")
    assert cuda_enabled == bool(torch.cuda.is_bf16_supported())
