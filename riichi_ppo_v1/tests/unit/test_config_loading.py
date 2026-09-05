"""Configuration-layer loading and precedence tests."""

from __future__ import annotations

import unittest

from riichi_ppo_v1.training.train import (
    load_config,
    partition_worker_indices,
    rollout_worker_options,
    summarize_worker_rollout,
)


class ConfigLoadingTest(unittest.TestCase):
    def test_packaged_groups_provide_v19_neutral_defaults(self) -> None:
        config = load_config()
        self.assertEqual(config["model_size"], "v19")
        self.assertEqual(config["policy_head_type"], "current_state_snapshot")
        self.assertEqual(config["game_mode"], "4p-red-half")
        self.assertEqual(config["envs_per_worker"], 32)
        self.assertEqual(config["update_epochs"], 4)
        self.assertEqual(config["target_kl"], 0.0)
        self.assertEqual(config["games_per_update"], 512)
        self.assertEqual(config["checkpoint_dir"], "checkpoints/train_riichi_v19/ppo")
        self.assertIsNone(config["resume"])
        self.assertIsNone(config["init_model"])
        self.assertNotIn("offense_" + "fusion", config)
        self.assertNotIn("critic_" + "head_type", config)

    def test_worker_partitioning_balances_workers_across_learners(self) -> None:
        self.assertEqual(partition_worker_indices(6, 2), [[0, 2, 4], [1, 3, 5]])
        self.assertEqual(partition_worker_indices(5, 2), [[0, 2, 4], [1, 3]])

    def test_rollout_worker_options_scope_thread_limits_to_actor(self) -> None:
        options = rollout_worker_options({
            "rollout_worker_num_cpus": 2,
            "rollout_worker_cpu_threads": 1,
        })
        self.assertEqual(options["num_cpus"], 2.0)
        self.assertEqual(options["runtime_env"]["env_vars"], {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        self.assertEqual(rollout_worker_options({}), {"num_cpus": 1.0})

    def test_worker_summary_reports_rollout_and_timing_percentiles(self) -> None:
        results = [
            ([], {
                "rollout_s": float(index),
                "games": float(index + 40),
                "grp_calls": 5.0,
                "timing/env/step_batch_native/total_s": float(index) / 10.0,
                "timing/env/step_batch_native/count": 10.0,
            })
            for index in range(1, 13)
        ]
        metrics, timing = summarize_worker_rollout(results)
        self.assertEqual(metrics["rollout/worker/rollout_s/min"], 1.0)
        self.assertEqual(metrics["rollout/worker/rollout_s/max"], 12.0)
        self.assertEqual(metrics["rollout/worker/rollout_s/p50"], 6.5)
        self.assertAlmostEqual(metrics["rollout/worker/rollout_s/p90"], 10.9)
        self.assertEqual(metrics["rollout/grp_calls"], 60.0)
        self.assertAlmostEqual(
            timing["env/step_batch_native"]["worker_p50_total_s"], 0.65,
        )
