"""Configuration-layer loading and precedence tests."""

from __future__ import annotations

from pathlib import Path
import unittest

from riichi_ppo_v1.training.train import (
    load_config,
    partition_worker_indices,
    rollout_worker_options,
    summarize_worker_rollout,
)


class ConfigLoadingTest(unittest.TestCase):
    def test_packaged_groups_provide_v16_neutral_defaults(self) -> None:
        config = load_config()
        self.assertEqual(config["model_size"], "v16")
        self.assertEqual(config["policy_head_type"], "symmetric_action_query")
        self.assertEqual(config["game_mode"], "4p-red-half")
        self.assertEqual(config["envs_per_worker"], 32)
        self.assertEqual(config["update_epochs"], 4)
        self.assertEqual(config["target_kl"], 0.0)
        self.assertEqual(config["games_per_update"], 512)
        self.assertEqual(config["checkpoint_dir"], "checkpoints/train_riichi_current")
        self.assertIsNone(config["resume"])
        self.assertIsNone(config["init_model"])
        self.assertNotIn("isolated_" + "action_query", str(config))
        self.assertNotIn("offense_" + "fusion", config)
        self.assertNotIn("critic_" + "head_type", config)

    def test_v16_and_v17_configs_are_self_contained(self) -> None:
        configs = Path(__file__).resolve().parents[2] / "configs"
        for name in ("v16_ppo.yaml", "v17_ppo.yaml"):
            config = load_config(str(configs / name))
            self.assertNotIn("profile_enabled", config)
            self.assertNotIn("semantic_metrics_jsonl", config)
            self.assertEqual(config["model_size"], "v16")
            self.assertEqual(config["policy_head_type"], "symmetric_action_query")
            self.assertEqual(config["learner_gpus"], 2)
            self.assertEqual(config["eval1v3_output_dir"].split("/")[:2], ["audit", "reports"])
            self.assertNotIn("offense_" + "fusion", config)
            self.assertNotIn("critic_" + "head_type", config)

    def test_v17_resume_config_is_self_contained_and_exact_resume(self) -> None:
        configs = Path(__file__).resolve().parents[2] / "configs"
        resumed = load_config(str(configs / "v17_ppo_resume.yaml"))

        self.assertEqual(resumed["model_size"], "v16")
        self.assertEqual(resumed["policy_head_type"], "symmetric_action_query")
        self.assertEqual(
            resumed["resume"],
            "checkpoints/train_riichi_v17/ppo/checkpoint_00060.pt",
        )
        self.assertIsNone(resumed["init_model"])
        self.assertEqual(resumed["games_per_update"], 512)
        self.assertEqual(resumed["eval1v3_output_dir"], "audit/reports/v17/eval")

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
        self.assertAlmostEqual(
            timing["env/step_batch_native"]["worker_p50_total_s"], 0.65,
        )
