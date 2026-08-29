"""V18 PPO 自包含配置断言:有效批量不变量。

(原整份配置快照断言 test_v18_ppo_config_matches_stability_plan 已删除:
逐键快照随训练策略迭代持续漂移,2026-08-29 维护者决定移除。)
"""

from __future__ import annotations

from pathlib import Path

from riichi_ppo_v1.training.train import load_config


def _v18_config() -> dict:
    path = Path(__file__).resolve().parents[2] / "configs" / "v18_ppo.yaml"
    return load_config(str(path))


def test_v18_global_effective_minibatch_is_30720() -> None:
    config = _v18_config()
    per_gpu = int(config["minibatch_size"])
    learner_gpus = int(config["learner_gpus"])
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    # global effective = per_gpu × gpus × accumulation。
    assert per_gpu * learner_gpus * accumulation == 30720
