"""Configuration-layer loading and precedence tests."""

from __future__ import annotations

from pathlib import Path
import unittest

from riichi_ppo_v1.training.train import load_config
from riichi_ppo_v1.training.train import partition_worker_indices


class ConfigLoadingTest(unittest.TestCase):
    def test_packaged_groups_provide_the_complete_default_configuration(self) -> None:
        config = load_config()
        self.assertEqual(config["model_size"], "mid")
        self.assertEqual(config["critic_layers"], 2)
        self.assertEqual(config["game_mode"], "4p-red-half")
        self.assertEqual(config["envs_per_worker"], 32)
        self.assertEqual(config["update_epochs"], 4)
        self.assertEqual(config["minibatch_size"], 512)
        self.assertEqual(config["kyokus_per_worker"], 1)
        self.assertEqual(config["kyoku_reward_clip_points"], 32000)
        self.assertEqual(config["total_updates"], 5000)
        self.assertEqual(config["checkpoint_dir"], "checkpoints/train_riichi_current")
        self.assertIsNone(config["resume"])
        self.assertEqual(config["update_batch_mode"], "streaming")
        self.assertEqual(config["gae_lambda"], 0.95)
        self.assertEqual(config["critic_head_type"], "action_value")
        self.assertEqual(config["target_kl"], 0.02)
        self.assertEqual(config["learning_rate"], 0.00002)
        self.assertEqual(config["actor_learning_rate"], 2e-5)
        self.assertEqual(config["shared_learning_rate"], 5e-6)
        self.assertEqual(config["critic_learning_rate"], 4e-5)
        self.assertEqual(config["total_updates"], 5000)
        self.assertEqual(config["warmup_fraction"], 0.02)
        self.assertEqual(config["entropy_start"], 0.01)
        self.assertEqual(config["entropy_end"], 0.001)
        self.assertEqual(config["adam_epsilon"], 1e-5)
        self.assertEqual(config["weight_decay"], 0.01)
        self.assertEqual(config["value_loss"], "huber")
        self.assertEqual(config["value_target_normalization"], "batch_std")
        self.assertEqual(config["value_target_std_floor"], 0.01)
        self.assertTrue(config["zero_q_head_on_sft_init"])
        self.assertEqual(config["critic_bootstrap_updates"], 40)
        self.assertEqual(config["critic_bootstrap_learning_rate"], 2e-5)
        self.assertEqual(config["critic_public_grad_scale"], 0.25)
        self.assertEqual(config["sft_kl_coef_start"], 0.02)
        self.assertEqual(config["sft_kl_coef_end"], 0.002)
        self.assertEqual(config["sft_kl_anneal_updates"], 5000)
        self.assertEqual(config["learner_gpus"], 2)
        self.assertEqual(config["checkpoint_dir"], "checkpoints/train_riichi_current")
        self.assertEqual(config["gamma"], 1.0)
        self.assertIsNone(config["resume"])
        self.assertNotIn("evaluation_enabled", config)
        self.assertNotIn("evaluation_interval_updates", config)
        self.assertNotIn("evaluation_hanchan_count", config)
        self.assertNotIn("evaluation_parallel_hanchan_count", config)
        self.assertNotIn("initial_discard_weight", config)
        self.assertNotIn("initial_call_weight", config)
        self.assertNotIn("evaluation_seed_count", config)
        self.assertNotIn("opponent_pool_capacity", config)
        self.assertNotIn("split_policy_inference", config)

    def test_self_contained_version_config_does_not_merge_packaged_defaults(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "v15_ppo.yaml"
        )
        config = load_config(str(config_path))

        # 自包含:不叠加打包默认,监控组键不得混入。
        self.assertNotIn("profile_enabled", config)
        self.assertNotIn("semantic_metrics_jsonl", config)

    def test_v15_config_is_self_contained(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "v15_ppo.yaml"
        )
        config = load_config(str(config_path))

        # 生产拓扑完整写在版本配置内。
        self.assertEqual(config["learner_gpus"], 2)
        self.assertEqual(config["num_workers"], 12)
        self.assertEqual(config["envs_per_worker"], 32)
        self.assertEqual(config["env_step_threads"], 4)
        self.assertEqual(config["inference_max_batch_size"], 512)
        self.assertEqual(config["inference_batch_wait_ms"], 5.0)
        self.assertEqual(config["minibatch_size"], 512)
        self.assertEqual(config["update_batch_mode"], "streaming")
        self.assertEqual(config["model_size"], "mid")
        self.assertEqual(config["context_tokens"], 4096)
        self.assertEqual(config["critic_layers"], 2)
        self.assertEqual(config["inference_dtype"], "bf16")

        # V15 正式运行参数。
        self.assertEqual(config["kyokus_per_worker"], 16)
        self.assertEqual(config["iterations"], 1200)
        self.assertEqual(config["total_updates"], 1200)
        self.assertTrue(config["offense_fusion"])
        self.assertEqual(config["critic_head_type"], "action_value")
        self.assertEqual(config["critic_bootstrap_updates"], 40)
        self.assertEqual(config["sft_kl_coef_start"], 0.005)
        self.assertEqual(config["sft_kl_coef_middle"], 0.0005)
        self.assertEqual(config["sft_kl_coef_end"], 0.002)
        self.assertEqual(config["eval1v3_processes"], 10)
        self.assertEqual(config["eval1v3_hanchans_per_process"], 160)
        self.assertEqual(config["eval1v3_devices"], ["0", "1"])

    def test_v14_resume_config_is_flattened_and_self_contained(self) -> None:
        configs = Path(__file__).resolve().parents[2] / "configs"
        base = load_config(str(configs / "v14_ppo.yaml"))
        resumed = load_config(str(configs / "v14_ppo_resume.yaml"))

        # 展平:除 resume/init_model 外与 v14_ppo.yaml 完全一致。
        self.assertNotEqual(base["resume"], resumed["resume"])
        for name, value in base.items():
            if name in ("resume", "init_model"):
                continue
            self.assertEqual(resumed[name], value, name)
        self.assertEqual(
            resumed["resume"],
            "checkpoints/train_riichi_v14/checkpoint_00600.pt",
        )
        self.assertIsNone(resumed["init_model"])
        self.assertEqual(resumed["num_workers"], 12)

    def test_worker_partitioning_balances_workers_across_learners(self) -> None:
        self.assertEqual(partition_worker_indices(6, 2), [[0, 2, 4], [1, 3, 5]])
        self.assertEqual(partition_worker_indices(5, 2), [[0, 2, 4], [1, 3]])
