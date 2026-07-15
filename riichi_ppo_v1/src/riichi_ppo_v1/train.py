"""Command-line entry points for synchronous Ray PPO training."""

from __future__ import annotations

import argparse
from importlib import resources
import os
from pathlib import Path
import random
import time
from typing import Any

# Keep the user-facing project convention while using CUDA's standard
# visibility variable.  This must run before importing torch (and before Ray
# launches workers) so ``CUDA_DEVICE=3`` makes physical GPU 3 appear as
# ``cuda:0`` to the learner.
if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .profiling import GpuSampler, append_jsonl


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("training config must be a mapping")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summarize_worker_rollout(
    results: list[tuple[list[Any], dict[str, float]]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Aggregate worker stats while retaining the slowest worker per stage."""
    stats_list = [stats for _transitions, stats in results]
    metrics: dict[str, float] = {}
    for name in {name for stats in stats_list for name in stats}:
        values = [float(stats[name]) for stats in stats_list if name in stats and np.isfinite(stats[name])]
        if values:
            metrics[f"rollout/{name}"] = float(np.mean(values))

    rows: dict[str, dict[str, float]] = {}
    for total_name in sorted(name for stats in stats_list for name in stats if name.startswith("timing/") and name.endswith("/total_s")):
        stage = total_name.removeprefix("timing/").removesuffix("/total_s")
        totals = [float(stats[total_name]) for stats in stats_list if total_name in stats]
        count_name = total_name.removesuffix("/total_s") + "/count"
        counts = [float(stats.get(count_name, 0.0)) for stats in stats_list]
        row = {
            "worker_mean_total_s": float(np.mean(totals)),
            "worker_max_total_s": float(np.max(totals)),
            "worker_min_total_s": float(np.min(totals)),
            "worker_sum_count": float(np.sum(counts)),
        }
        rows[stage] = row
        for field, value in row.items():
            metrics[f"rollout/timing/{stage}/{field}"] = value
    return metrics, rows


def timing_rows(stats: dict[str, float], label: str) -> dict[str, dict[str, float]]:
    """Convert StageProfiler's flat metric schema into printable rows."""
    rows: dict[str, dict[str, float]] = {}
    for total_name in sorted(name for name in stats if name.startswith("timing/") and name.endswith("/total_s")):
        stage = total_name.removeprefix("timing/").removesuffix("/total_s")
        base = total_name.removesuffix("/total_s")
        name = f"{label}/{stage}" if label else stage
        rows[name] = {
            "total_s": float(stats[total_name]),
            "mean_ms": float(stats.get(f"{base}/mean_ms", 0.0)),
            "max_ms": float(stats.get(f"{base}/max_ms", 0.0)),
            "count": float(stats.get(f"{base}/count", 0.0)),
        }
    return rows


def print_timing_table(title: str, rows: dict[str, dict[str, float]]) -> None:
    print(f"{title} timing:", flush=True)
    for name, row in sorted(rows.items()):
        if "worker_mean_total_s" in row:
            print(
                f"  {name:<48} worker_total_s mean={row['worker_mean_total_s']:.4f} "
                f"max={row['worker_max_total_s']:.4f} min={row['worker_min_total_s']:.4f} "
                f"count_sum={row['worker_sum_count']:.0f}",
                flush=True,
            )
        else:
            print(
                f"  {name:<48} total_s={row['total_s']:.4f} mean_ms={row['mean_ms']:.3f} "
                f"max_ms={row['max_ms']:.3f} count={row['count']:.0f}",
                flush=True,
            )


def print_worker_details(results: list[tuple[list[Any], dict[str, float]]]) -> None:
    """Print non-aggregated rollout throughput and timing for every worker."""
    print("rollout worker detail:", flush=True)
    for worker_id, (transitions, stats) in enumerate(results):
        elapsed = float(stats.get("rollout_s", 0.0))
        model_decisions = float(stats.get("model_decisions", 0.0))
        recorded_decisions = float(stats.get("recorded_decisions", len(transitions)))
        print(
            f"  worker={worker_id} kyokus={stats.get('kyokus', 0.0):.0f} "
            f"transitions={len(transitions)} model_decisions={model_decisions:.0f} "
            f"rollout_s={elapsed:.3f} transition_sps={len(transitions) / max(elapsed, 1e-9):.2f} "
            f"model_decision_sps={model_decisions / max(elapsed, 1e-9):.2f} "
            f"recorded_decision_sps={recorded_decisions / max(elapsed, 1e-9):.2f}",
            flush=True,
        )
        for total_name in sorted(name for name in stats if name.startswith("timing/") and name.endswith("/total_s")):
            stage = total_name.removeprefix("timing/").removesuffix("/total_s")
            total = float(stats[total_name])
            count = float(stats.get(total_name.removesuffix("/total_s") + "/count", 0.0))
            print(
                f"    {stage:<46} total_s={total:.4f} mean_ms={total * 1_000.0 / max(count, 1.0):.3f} count={count:.0f}",
                flush=True,
            )


def run(config: dict[str, Any]) -> None:
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError("Ray is required: pip install -e riichi_ppo_v1") from exc
    from .worker import RolloutWorker
    from .inference import RolloutInferenceActor
    if RolloutWorker is None:
        raise RuntimeError("Ray rollout worker could not be defined")
    if RolloutInferenceActor is None:
        raise RuntimeError("Ray rollout inference actor could not be defined")
    if config.get("opponent_checkpoint"):
        raise ValueError("opponent_checkpoint is not supported: rollout self-play uses the live model for all four seats")
    seed_everything(int(config["seed"]))
    device = str(config["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")
    ray.init(ignore_reinit_error=True)
    inference = RolloutInferenceActor.remote(config)
    workers = [RolloutWorker.remote(index, config, inference) for index in range(int(config["num_workers"]))]
    writer = SummaryWriter(str(Path(config["checkpoint_dir"]) / "tensorboard"))
    gpu_sampler = GpuSampler(bool(config.get("gpu_monitor", True)) and device.startswith("cuda"), float(config.get("gpu_sample_interval_s", 0.25)))
    gpu_sampler.start()
    try:
        start_iteration = ray.get(inference.iteration.remote())
        for iteration in range(start_iteration, int(config["iterations"])):
            gpu_cursor = gpu_sampler.checkpoint()
            algorithm_started = time.perf_counter()
            begin_rollout_started = time.perf_counter()
            ray.get(inference.begin_rollout.remote())
            begin_rollout_s = time.perf_counter() - begin_rollout_started
            rollout_started = time.perf_counter()
            results = ray.get([worker.collect.remote() for worker in workers])
            rollout_wall_s = time.perf_counter() - rollout_started
            profile_summary_started = time.perf_counter()
            actor_profile = ray.get(inference.profile_summary.remote())
            profile_summary_s = time.perf_counter() - profile_summary_started
            transition_assembly_started = time.perf_counter()
            transitions = [transition for worker_transitions, _ in results for transition in worker_transitions]
            transition_assembly_s = time.perf_counter() - transition_assembly_started
            update_started = time.perf_counter()
            metrics = ray.get(inference.update.remote(transitions))
            update_wall_s = time.perf_counter() - update_started
            algorithm_wall_s = time.perf_counter() - algorithm_started
            gpu_metrics = gpu_sampler.summary(gpu_cursor)
            rollout_reward = float(np.nanmean([stats["reward_mean"] for _, stats in results]))
            rollout_kyokus = sum(stats["kyokus"] for _, stats in results)
            rollout_tps = float(np.nanmean([stats.get("transitions_per_s", float("nan")) for _, stats in results]))
            model_decisions = sum(stats.get("model_decisions", 0.0) for _, stats in results)
            recorded_decisions = sum(stats.get("recorded_decisions", 0.0) for _, stats in results)
            rollout_metrics, worker_timing = summarize_worker_rollout(results)
            actor_metrics = {f"rollout/inference_actor/{name}": float(value) for name, value in actor_profile.items()}
            rollout_metrics.update(actor_metrics)
            rollout_metrics.update({
                "rollout/begin_rollout_s": begin_rollout_s,
                "rollout/profile_summary_s": profile_summary_s,
                "rollout/transition_assembly_s": transition_assembly_s,
                "rollout/wall_s": rollout_wall_s,
                "update/wall_s": update_wall_s,
                "iteration/algorithm_wall_s": algorithm_wall_s,
                "iteration/sps": float(recorded_decisions / max(algorithm_wall_s, 1e-9)),
                "iteration/model_forward_sps": float(model_decisions / max(algorithm_wall_s, 1e-9)),
                "iteration/model_decisions_per_s": float(model_decisions / max(algorithm_wall_s, 1e-9)),
                "iteration/sampled_decisions_per_s": float(recorded_decisions / max(algorithm_wall_s, 1e-9)),
                "iteration/effective_transitions_per_s": float(len(transitions) / max(algorithm_wall_s, 1e-9)),
            })
            actor_timing = timing_rows(actor_profile, "inference_actor")
            update_timing = timing_rows(metrics, "")
            env_step_s = float(worker_timing.get("env/step_batch_native", {}).get("worker_mean_total_s", 0.0))
            update_forward_s = float(metrics.get("timing/update/model_forward/total_s", 0.0))
            inference_rows = float(actor_profile.get("inference/full_forward_rows_mean", 0.0))
            inference_dispatch_rows = float(actor_profile.get("inference/dispatch_rows_mean", 0.0))
            print(f"iteration={iteration + 1} transitions={len(transitions)} model_decisions={model_decisions:.0f} recorded_decisions={recorded_decisions:.0f} kyokus={rollout_kyokus:.0f} reward={rollout_reward:.4f} worker_transitions_per_s={rollout_tps:.2f} algorithm_wall_s={algorithm_wall_s:.3f} sps={recorded_decisions / max(algorithm_wall_s, 1e-9):.2f} model_forward_sps={model_decisions / max(algorithm_wall_s, 1e-9):.2f} rollout_wall_s={rollout_wall_s:.3f} begin_rollout_s={begin_rollout_s:.3f} profile_summary_s={profile_summary_s:.3f} transition_assembly_s={transition_assembly_s:.3f} env_step_s={env_step_s:.3f} inference_rows_per_forward={inference_rows:.2f} inference_rows_per_dispatch={inference_dispatch_rows:.2f} update_wall_s={update_wall_s:.3f} update_forward_s={update_forward_s:.3f} epochs={metrics['update/epochs_completed']:.0f}/{metrics['update/configured_epochs']:.0f} minibatches={metrics['update/executed_minibatches']:.0f}/{metrics['update/planned_minibatches']:.0f} early_stop={bool(metrics['update/early_stop'])} executed_samples={metrics['update/executed_transition_samples']:.0f} event_blocks_mean={metrics['update/executed_transition_event_blocks_mean']:.2f} input_tokens_mean={metrics['update/executed_transition_input_tokens_mean']:.2f} " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items() if key in {"loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "grad_norm"}), flush=True)
            print_timing_table("rollout worker", worker_timing)
            print_worker_details(results)
            print_timing_table("rollout inference actor", actor_timing)
            print_timing_table("PPO update", update_timing)
            writer.add_scalar("rollout/kyokus", rollout_kyokus, iteration + 1)
            writer.add_scalar("rollout/reward_mean", rollout_reward, iteration + 1)
            for name, value in rollout_metrics.items():
                writer.add_scalar(name, value, iteration + 1)
            for name, value in metrics.items():
                writer.add_scalar(f"ppo/{name}", value, iteration + 1)
            for name, value in gpu_metrics.items():
                writer.add_scalar(name, value, iteration + 1)
            append_jsonl(Path(config["checkpoint_dir"]) / "performance.jsonl", {
                "iteration": iteration + 1,
                "timestamp": time.time(),
                "transitions": len(transitions),
                "kyokus": int(rollout_kyokus),
                "rollout_reward": rollout_reward,
                **rollout_metrics,
                **{f"ppo/{name}": float(value) for name, value in metrics.items()},
                **gpu_metrics,
            })
            if (iteration + 1) % int(config["checkpoint_interval"]) == 0:
                path = Path(config["checkpoint_dir"]) / f"iteration_{iteration + 1}.pt"
                ray.get(inference.save.remote(str(path), config))
                print("checkpoint=" + str(path), flush=True)
        ray.get(inference.save.remote(str(Path(config["checkpoint_dir"]) / "latest.pt"), config))
    finally:
        gpu_sampler.stop()
        writer.close()
        ray.shutdown()


def _parser(smoke: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(resources.files("riichi_ppo_v1").joinpath("default.yaml")))
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    if not smoke:
        parser.add_argument("--num-workers", type=int, default=None)
        parser.add_argument("--envs-per-worker", type=int, default=None)
        parser.add_argument("--kyokus-per-worker", type=int, default=None)
        parser.add_argument("--update-epochs", type=int, default=None)
        parser.add_argument("--minibatch-size", type=int, default=None)
        parser.add_argument("--target-kl", type=float, default=None)
        parser.add_argument("--profile-cuda-sync", action=argparse.BooleanOptionalAction, default=None)
    if smoke:
        parser.add_argument("--kyokus", type=int, default=1)
    return parser


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        "num_workers": getattr(args, "num_workers", None),
        "envs_per_worker": getattr(args, "envs_per_worker", None),
        "kyokus_per_worker": getattr(args, "kyokus_per_worker", None),
        "update_epochs": getattr(args, "update_epochs", None),
        "minibatch_size": getattr(args, "minibatch_size", None),
        "target_kl": getattr(args, "target_kl", None),
        "profile_cuda_sync": getattr(args, "profile_cuda_sync", None),
    }
    config.update({name: value for name, value in overrides.items() if value is not None})


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    apply_cli_overrides(config, args)
    run(config)


def smoke_main() -> None:
    args = _parser(smoke=True).parse_args()
    config = load_config(args.config)
    config.update({"device": args.device or "cpu", "num_workers": 1, "envs_per_worker": 1, "kyokus_per_worker": args.kyokus, "iterations": 1, "update_epochs": 1, "minibatch_size": 32, "checkpoint_interval": 1, "checkpoint_dir": "checkpoints/riichi_ppo_v1_smoke"})
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    apply_cli_overrides(config, args)
    run(config)


if __name__ == "__main__":
    main()
