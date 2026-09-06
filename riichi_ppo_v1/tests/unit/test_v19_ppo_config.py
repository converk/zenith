"""V19 PPO 自包含配置断言:信念键 + 当前拓扑 + 有效批量不变量。"""

from __future__ import annotations

from pathlib import Path

from riichi_ppo_v1.training.train import load_config


def _v19_config() -> dict:
    path = Path(__file__).resolve().parents[2] / "configs" / "v19_ppo.yaml"
    return load_config(str(path))


def test_v19_config_contains_belief_keys() -> None:
    """v19_ppo.yaml 必须携带训练分册 §6 的全部信念键。"""
    config = _v19_config()
    expected = {
        "belief_public_grad_scale": 0.25,
        "belief_head_weight_hand": 0.6,
        "belief_head_weight_shanten": 1.0,
        "belief_head_weight_wait": 1.5,
        "belief_head_weight_danger": 5.0,
        "belief_head_weight_loss": 5.0,
        "belief_wait_danger_weight": 0.05,
        "belief_readout_enabled": True,
        "belief_readout_detach": True,
        "belief_wait_tenpai_weight": 1.0,
        "belief_wait_tile_weight": 0.0,
        "belief_danger_pos_weight": 5.0,
        "belief_loss_positive_weight": 20.0,
    }
    for name, value in expected.items():
        assert name in config, f"v19_ppo.yaml 缺少信念键 {name}"
        assert float(config[name]) == value, name


def test_v19_config_init_model_points_to_standard_sft() -> None:
    """PPO init_model 必须指向 V19 标准 SFT 产物（不启动）。"""
    config = _v19_config()
    assert config["init_model"] == "checkpoints/train_riichi_v19/sft/best.pt"


def test_v19_sft_config_initial_belief_head_weights() -> None:
    """v19_sft.yaml 五头权重与 wait_tile 关闭必须与现行决策一致。"""
    path = Path(__file__).resolve().parents[2] / "configs" / "v19_sft.yaml"
    config = load_config(str(path))
    expected = {
        "belief_head_weight_hand": 0.6,
        "belief_head_weight_shanten": 1.0,
        "belief_head_weight_wait": 1.5,
        "belief_head_weight_danger": 5.0,
        "belief_head_weight_loss": 5.0,
        "belief_wait_tenpai_weight": 1.0,
        "belief_wait_tile_weight": 0.0,
    }
    for name, value in expected.items():
        assert name in config, f"v19_sft.yaml 缺少信念权重键 {name}"
        assert float(config[name]) == value, name


def test_v19_sft_resume_config_is_self_contained() -> None:
    """resume 配置必须自包含：resume 指向稳定快照，其余关键项与 v19_sft 一致。"""
    root = Path(__file__).resolve().parents[2] / "configs"
    resume_config = load_config(str(root / "v19_sft_resume.yaml"))
    base_config = load_config(str(root / "v19_sft.yaml"))
    assert resume_config["resume"] == "checkpoints/train_riichi_v19/sft/resume_84000.pt"
    assert resume_config["init_model"] is None
    # resume 配置里不得缺失任何 v19_sft 顶层键（自包含，非 overlay）。
    assert set(base_config) <= set(resume_config)
    for key in (
        "dataset", "checkpoint_dir", "learner_gpus", "batch_size", "epochs",
        "seed", "belief_sft_coef", "belief_head_weight_hand",
        "belief_head_weight_shanten", "belief_head_weight_wait",
        "belief_head_weight_danger", "belief_head_weight_loss",
        "belief_wait_tile_weight", "belief_wait_tenpai_weight",
        "belief_public_grad_scale", "belief_readout_detach",
    ):
        assert resume_config[key] == base_config[key], key
    assert resume_config["tensorboard_dirname"] == "tensorboard_resume"


def test_v19_config_topology() -> None:
    """V19 拓扑:5 层 shared/actor 重排 + 1 层 critic + context 320。"""
    config = _v19_config()
    assert config["model_size"] == "v19"
    assert config["layers"] == 5
    assert config["shared_layers"] == 3
    assert config["critic_layers"] == 1
    assert config["context_tokens"] == 320
    assert config["policy_head_type"] == "current_state_snapshot"


def test_v19_global_effective_minibatch_is_40960() -> None:
    """有效批 = per_gpu × learner_gpus × 梯度累积(既定基线,配置自包含)。"""
    config = _v19_config()
    per_gpu = int(config["minibatch_size"])
    learner_gpus = int(config["learner_gpus"])
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    assert per_gpu * learner_gpus * accumulation == 40960
