"""V18 PPO 自包含配置断言:2048 半庄/update、稳定性策略与评测节奏。"""

from __future__ import annotations

from pathlib import Path

import pytest

from riichi_ppo_v1.training.train import load_config


def _v18_config() -> dict:
    path = Path(__file__).resolve().parents[2] / "configs" / "v18_ppo.yaml"
    return load_config(str(path))


def test_v18_ppo_config_matches_stability_plan() -> None:
    config = _v18_config()
    assert config["model_size"] == "v18"
    assert config["policy_head_type"] == "current_state_snapshot"
    assert config["context_tokens"] == 256
    assert config["learner_gpus"] == 2
    assert config["iterations"] == 150
    assert config["total_updates"] == 150
    assert config["games_per_update"] == 2048
    assert config["update_epochs"] == 4
    assert config["minibatch_size"] == 512
    assert config["gradient_accumulation_steps"] == 1
    assert config["gamma"] == 1.0
    assert config["gae_lambda"] == 0.95
    assert config["ppo_clip"] == 0.20
    assert config["critic_bootstrap_updates"] == 2
    assert config["critic_bootstrap_learning_rate"] == pytest.approx(2e-5)
    assert config["actor_learning_rate"] == pytest.approx(4e-5)
    assert config["actor_learning_rate_min"] == pytest.approx(1.5e-5)
    assert config["shared_learning_rate"] == pytest.approx(5e-6)
    assert config["shared_learning_rate_min"] == pytest.approx(2.5e-6)
    assert config["critic_learning_rate"] == pytest.approx(4e-5)
    assert config["critic_learning_rate_min"] == pytest.approx(1.5e-5)
    assert config["adam_beta1"] == 0.95
    assert config["weight_decay"] == 0.0
    assert config["actor_max_grad_norm"] == 0.5
    assert config["shared_max_grad_norm"] == 0.5
    assert config["critic_max_grad_norm"] == 1.0
    assert config["critic_public_grad_scale"] == 0.25
    assert config["critic_private_embedding_grad_scale"] == 0.25
    assert config["entropy_loss_mode"] == "normalized"
    assert config["entropy_start"] == pytest.approx(0.020)
    assert config["entropy_middle"] == pytest.approx(0.012)
    assert config["entropy_end"] == pytest.approx(0.0045)
    assert config["entropy_middle_fraction"] == pytest.approx(0.33)
    assert config["target_kl"] == 0.01
    assert config["target_kl_check_interval"] == 8
    assert config["bucket_window_multiplier"] == 8
    assert config["checkpoint_interval_updates"] == 5
    assert config["eval1v3_interval_updates"] == 5
    assert config["init_model"].endswith("train_riichi_v18/sft/best.pt")
    assert config["grp_checkpoint"].endswith("train_riichi_v18/grp/best.pt")
    for key in ("q_coef", "q_boost_coef", "q_boost_lambda", "q_temperature", "qboost_lambda"):
        assert key not in config


def test_v18_global_effective_minibatch_is_1024() -> None:
    config = _v18_config()
    per_gpu = int(config["minibatch_size"])
    learner_gpus = int(config["learner_gpus"])
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    # global effective = per_gpu × gpus × accumulation。
    assert per_gpu * learner_gpus * accumulation == 1024
