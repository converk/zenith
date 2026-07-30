"""Configuration-layer loading and precedence tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
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
        self.assertEqual(config["kyokus_per_worker"], 16)
        self.assertEqual(config["total_updates"], 5000)
        self.assertEqual(config["checkpoint_dir"], "checkpoints/train_riichi_ppo")
        self.assertIsNone(config["resume"])
        self.assertEqual(config["update_batch_mode"], "streaming")
        self.assertEqual(config["gae_lambda"], 0.95)
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
        self.assertTrue(config["zero_value_head_on_sft_init"])
        self.assertEqual(config["critic_bootstrap_updates"], 30)
        self.assertEqual(config["critic_bootstrap_learning_rate"], 2e-5)
        self.assertEqual(config["critic_public_grad_scale"], 0.25)
        self.assertEqual(config["sft_kl_coef_start"], 0.02)
        self.assertEqual(config["sft_kl_coef_end"], 0.002)
        self.assertEqual(config["sft_kl_anneal_updates"], 5000)
        self.assertEqual(config["learner_gpus"], 2)
        self.assertEqual(config["checkpoint_dir"], "checkpoints/train_riichi_ppo")
        self.assertEqual(config["gamma"], 0.99)
        self.assertIsNone(config["resume"])
        self.assertTrue(config["evaluation_enabled"])
        self.assertEqual(config["evaluation_interval_updates"], 15)
        self.assertEqual(config["evaluation_hanchan_count"], 96)
        self.assertEqual(config["evaluation_parallel_hanchan_count"], 48)
        self.assertNotIn("initial_discard_weight", config)
        self.assertNotIn("initial_call_weight", config)
        self.assertNotIn("evaluation_seed_count", config)
        self.assertNotIn("opponent_pool_capacity", config)
        self.assertNotIn("split_policy_inference", config)

    def test_group_overrides_precede_the_legacy_full_config_overlay(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.yaml"
            overlay = root / "overlay.yaml"
            training.write_text("context_tokens: 512\nmodel_size: small\nminibatch_size: 512\nupdate_batch_mode: auto\n", encoding="utf-8")
            overlay.write_text("context_tokens: 1024\nnum_workers: 2\n", encoding="utf-8")
            config = load_config(str(overlay), training_path=str(training))

        self.assertEqual(config["model_size"], "small")
        self.assertEqual(config["context_tokens"], 1024)
        self.assertEqual(config["num_workers"], 2)
        self.assertEqual(config["minibatch_size"], 512)
        self.assertEqual(config["update_batch_mode"], "auto")
        self.assertEqual(config["game_mode"], "4p-red-half")

    def test_worker_partitioning_balances_workers_across_learners(self) -> None:
        self.assertEqual(partition_worker_indices(6, 2), [[0, 2, 4], [1, 3, 5]])
        self.assertEqual(partition_worker_indices(5, 2), [[0, 2, 4], [1, 3]])
