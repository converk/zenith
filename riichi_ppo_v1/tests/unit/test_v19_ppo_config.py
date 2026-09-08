"""V19 PPO 自包含配置断言:信念键 + 当前拓扑 + 有效批量不变量。"""

from __future__ import annotations

from pathlib import Path

from riichi_ppo_v1.training.train import load_config


def _v19_config() -> dict:
    path = Path(__file__).resolve().parents[2] / "configs" / "v19_ppo.yaml"
    return load_config(str(path))


def test_v19_config_contains_belief_keys() -> None:
    """v19_ppo.yaml 必须携带训练分册 §6 的全部信念键（2026-09-08 四头精简版）。

    信念头精简为四头（hand/wait/danger/loss_bucket）：shanten 头已删除；
    belief_public_grad_scale=0（残差式隔离——信念监督不更新公共权重）。
    """
    config = _v19_config()
    expected = {
        "belief_public_grad_scale": 0.0,
        "belief_head_weight_hand": 1.0,
        "belief_head_weight_wait": 1.0,
        "belief_head_weight_danger": 1.0,
        "belief_head_weight_loss_bucket": 1.0,
        "belief_readout_enabled": True,
        "belief_readout_detach": True,
        "belief_danger_pos_weight": 5.0,
        "belief_loss_positive_weight": 20.0,
    }
    for name, value in expected.items():
        assert name in config, f"v19_ppo.yaml 缺少信念键 {name}"
        assert float(config[name]) == value, name
    # shanten 头已删除：旧五头权重键不得再出现。
    assert "belief_head_weight_shanten" not in config
    assert "belief_head_weight_loss" not in config
    assert "belief_wait_danger_weight" not in config
    assert "belief_wait_tile_weight" not in config


def test_v19_config_init_model_points_to_standard_sft() -> None:
    """PPO init_model 必须指向 V19 标准 SFT 产物（不启动）。"""
    config = _v19_config()
    assert config["init_model"] == "checkpoints/train_riichi_v19/sft/best.pt"


def test_v19_sft_config_initial_belief_head_weights() -> None:
    """v19_sft.yaml 四头权重必须为归一化均衡值（λ=1.0，2026-09-08 四头精简）。"""
    path = Path(__file__).resolve().parents[2] / "configs" / "v19_sft.yaml"
    config = load_config(str(path))
    expected = {
        "belief_head_weight_hand": 1.0,
        "belief_head_weight_wait": 1.0,
        "belief_head_weight_danger": 1.0,
        "belief_head_weight_loss_bucket": 1.0,
    }
    for name, value in expected.items():
        assert name in config, f"v19_sft.yaml 缺少信念权重键 {name}"
        assert float(config[name]) == value, name
    # shanten 头已删除：旧五头权重键不得再出现。
    assert "belief_head_weight_shanten" not in config
    assert "belief_head_weight_loss" not in config
    assert "belief_wait_tenpai_weight" not in config
    assert "belief_wait_tile_weight" not in config


def test_v19_sft_config_epochs_one_and_log_interval() -> None:
    """v19_sft.yaml 必须使用 1 epoch、batch=512（2026-09-08 沿用 V18 正式配置），日志每 100 步。"""
    path = Path(__file__).resolve().parents[2] / "configs" / "v19_sft.yaml"
    config = load_config(str(path))
    assert int(config["epochs"]) == 1
    assert int(config["batch_size"]) == 512
    assert int(config["log_interval_steps"]) == 100



def test_v19_config_topology() -> None:
    """V19 拓扑:5 层 shared/actor 重排 + 1 层 critic + context 320。"""
    config = _v19_config()
    assert config["model_size"] == "v19"
    assert config["layers"] == 5
    assert config["shared_layers"] == 3
    assert config["critic_layers"] == 1
    assert config["context_tokens"] == 320
    assert config["policy_head_type"] == "current_state_snapshot"


def test_v19_config_exploration_plan_and_diagnostics() -> None:
    """V19 从头重训方案（2026-09-08）:熵三锚沿用 u45 resume 配置
    （0.016/0.008/0.003 @0.4，衰减快于旧 V19、慢于 V18）+ 删除熵地板
    + 梯度归因 + 200 updates。"""
    config = _v19_config()
    assert int(config["total_updates"]) == 200
    assert int(config["iterations"]) == 200
    assert float(config["entropy_start"]) == 0.016
    assert float(config["entropy_middle"]) == 0.008
    assert float(config["entropy_end"]) == 0.003
    assert float(config["entropy_middle_fraction"]) == 0.4
    # 熵地板屏障已删除（用户 2026-09-08 决策）。
    assert "entropy_floor" not in config
    assert "entropy_floor_coef" not in config
    assert int(config["grad_term_diagnostics_interval_updates"]) == 10
    # belief_sft_coef 在 SFT/PPO 两侧同构生效。
    assert float(config["belief_sft_coef"]) == 1.0


def test_v19_global_effective_minibatch_is_40960() -> None:
    """有效批 = per_gpu × learner_gpus × 梯度累积(既定基线,配置自包含)。"""
    config = _v19_config()
    per_gpu = int(config["minibatch_size"])
    learner_gpus = int(config["learner_gpus"])
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    assert per_gpu * learner_gpus * accumulation == 40960
