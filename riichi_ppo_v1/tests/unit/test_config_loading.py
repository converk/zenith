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
        self.assertEqual(config["game_mode"], "4p-red-half")
        self.assertEqual(config["envs_per_worker"], 32)
        self.assertEqual(config["update_epochs"], 4)
        self.assertEqual(config["minibatch_size"], 512)
        self.assertEqual(config["kyokus_per_worker"], 1)
        self.assertEqual(config["update_batch_mode"], "streaming")
        self.assertEqual(config["gae_lambda"], 0.97)
        self.assertEqual(config["target_kl"], 0.02)
        self.assertEqual(config["learning_rate"], 0.00002)
        self.assertEqual(config["total_updates"], 7000)
        self.assertEqual(config["warmup_fraction"], 0.02)
        self.assertEqual(config["entropy_start"], 0.01)
        self.assertEqual(config["entropy_end"], 0.001)
        self.assertEqual(config["adam_epsilon"], 1e-5)
        self.assertEqual(config["weight_decay"], 0.01)
        self.assertEqual(config["value_loss"], "huber")
        self.assertEqual(config["learner_gpus"], 2)
        self.assertEqual(config["checkpoint_dir"], "checkpoints/train_riichi_v4")
        self.assertEqual(config["gamma"], 1)
        self.assertIsNone(config["resume"])
        self.assertEqual(config["initial_efficiency_weight"], 0.10)
        self.assertEqual(config["efficiency_decay_fraction"], 0.10)
        self.assertNotIn("history_pool_size", config)
        self.assertNotIn("resident_historical_models", config)
        self.assertEqual(config["evaluation_hanchan_count"], 100)
        self.assertNotIn("evaluation_seed_count", config)
        self.assertFalse(config["split_policy_inference"])

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
