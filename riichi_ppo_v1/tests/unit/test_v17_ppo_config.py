"""V17 PPO 自包含配置断言:2048 半庄/update、3072 global batch、评测节奏等。"""

from __future__ import annotations

from pathlib import Path

import pytest

from riichi_ppo_v1.training.train import load_config


def _v17_config() -> dict:
    path = Path(__file__).resolve().parents[2] / "configs" / "v17_ppo.yaml"
    return load_config(str(path))


def test_v17_config_is_self_contained() -> None:
    config = _v17_config()
    # 完整拓扑。
    assert config["learner_gpus"] == 2
    assert config["num_workers"] == 12
    assert config["envs_per_worker"] == 32
    assert config["model_size"] == "v16"
    assert config["inference_dtype"] == "bf16"
    # 2048 半庄/update(rollout 停止条件为完整半庄数)。
    assert config["games_per_update"] == 2048
    # 2 GPU DDP,每 GPU 1536 → global 3072,update_epochs=2。
    assert config["minibatch_size"] == 1536
    assert config["update_epochs"] == 2
    assert config.get("gradient_accumulation_steps", 1) >= 1
    # PPO 超参。
    assert config["actor_learning_rate"] == pytest.approx(3e-5)
    assert config["actor_learning_rate_min"] == pytest.approx(5e-6)
    assert config["shared_learning_rate"] == pytest.approx(5e-6)
    assert config["shared_learning_rate_min"] == pytest.approx(1.25e-6)
    assert config["critic_learning_rate"] == pytest.approx(4e-5)
    assert config["critic_learning_rate_min"] == pytest.approx(1e-5)
    assert config["ppo_clip"] == 0.2
    assert config["target_kl"] == 0.01
    assert config["max_grad_norm"] == 0.75
    assert config["entropy_start"] == 0.01
    assert config["entropy_end"] == 0.003
    assert config["critic_bootstrap_updates"] == 2
    assert config["iterations"] == 150
    assert config["total_updates"] == 150
    # 完全移除 Q:GAE value-based advantage。
    assert config["gae_lambda"] == 0.95
    for key in ("q_coef", "q_boost_coef", "q_boost_lambda", "q_temperature", "qboost_lambda"):
        assert key not in config
    # SFT KL。
    assert config["sft_kl_coef_start"] == 0.0025
    assert config["sft_kl_coef_middle"] == 0.001
    assert config["sft_kl_coef_end"] == 0.002
    # 对手纯 current self-play。
    assert config["opponent_mix"]["enabled"] is False
    # 评测节奏:每 5 updates,4000 半庄(10 进程 × 400,同卡串行)。
    assert config["checkpoint_interval_updates"] == 5
    assert config["eval1v3_enabled"] is True
    assert config["eval1v3_model_b"].endswith("train_riichi_v16/sft/best.pt")
    assert config["eval1v3_output_dir"] == "audit/reports/v17/eval"
    assert config["eval1v3_devices"] == ["0", "1"]
    assert config["eval1v3_parallel_hanchans"] == 200


def test_v17_config_grp_checkpoint_points_at_v17() -> None:
    config = _v17_config()
    assert config["grp_checkpoint"].startswith("checkpoints/train_riichi_v17/grp/")


def test_global_effective_minibatch_is_3072() -> None:
    config = _v17_config()
    per_gpu = int(config["minibatch_size"])
    learner_gpus = int(config["learner_gpus"])
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    # global effective = per_gpu × gpus × accumulation。
    assert per_gpu * learner_gpus * accumulation == 3072
