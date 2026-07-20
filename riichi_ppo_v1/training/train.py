"""Command-line entry points for synchronous Ray PPO training."""

from __future__ import annotations

import argparse
from importlib import resources
import os
from pathlib import Path
import random
import socket
import time
from typing import Any

# Keep the user-facing project convention while using CUDA's standard
# visibility variable.  This must run before importing torch (and before Ray
# launches workers) so ``CUDA_DEVICE=0,3`` exposes those physical GPUs while
# each Ray GPU actor still sees its assigned device as ``cuda:0``.
if os.environ.get("CUDA_DEVICE") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .profiling import GpuSampler, append_jsonl
from .metrics import RollingKyokuMetrics, append_metric_jsonl, metric_counters
from .tensorboard import learner_peak_allocated_mb, write_curated_scalars


_CONFIG_GROUPS = ("training", "monitoring")


def _load_config_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("training config must be a mapping")
    return config


def _packaged_config_path(group: str) -> str:
    if group not in _CONFIG_GROUPS:
        raise ValueError(f"unknown config group {group!r}")
    return str(resources.files("riichi_ppo_v1").joinpath("configs", f"{group}.yaml"))


def load_config(
    path: str | None = None,
    *,
    model_path: str | None = None,
    environment_path: str | None = None,
    training_path: str | None = None,
) -> dict[str, Any]:
    """Merge packaged defaults, optional group overrides, then an overlay YAML.

    The packaged defaults are split into training/runtime parameters and
    monitoring/profiling parameters while preserving the old ``--config``
    entry point as a final full-config override.
    """
    config: dict[str, Any] = {}
    for group in _CONFIG_GROUPS:
        config.update(_load_config_file(_packaged_config_path(group)))
    for override in (model_path, environment_path, training_path):
        if override:
            config.update(_load_config_file(override))
    if path:
        config.update(_load_config_file(path))
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def partition_worker_indices(num_workers: int, num_learners: int) -> list[list[int]]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if num_learners <= 0:
        raise ValueError("num_learners must be positive")
    if num_workers < num_learners:
        raise ValueError("num_workers must be at least learner_gpus")
    return [list(range(rank, num_workers, num_learners)) for rank in range(num_learners)]


def local_distributed_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def summarize_worker_rollout(
    results: list[tuple[list[Any], dict[str, float]]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Aggregate worker stats while retaining the slowest worker per stage."""
    stats_list = [stats for _transitions, stats in results]
    metrics: dict[str, float] = {}
    for name in {name for stats in stats_list for name in stats}:
        values = [float(stats[name]) for stats in stats_list if name in stats and np.isfinite(stats[name])]
        if values:
            # Semantic metric tags are already public TensorBoard paths.  Keep
            # them out of the legacy ``rollout/`` namespace and sum explicit
            # event counts across workers.
            target = name if name.startswith(("train/", "reward_schedule/")) else f"rollout/{name}"
            if name.endswith("/count") or name.endswith("_count"):
                metrics[target] = float(np.sum(values))
            elif name.startswith("train/kyoku/"):
                # Per-kyoku rates, points and lengths must be weighted by the
                # number of settled kyokus each worker actually produced.
                weighted = [
                    (float(stats[name]), float(stats.get("train/kyoku/count", 0.0)))
                    for stats in stats_list if name in stats and np.isfinite(stats[name])
                ]
                total_weight = sum(weight for _value, weight in weighted)
                metrics[target] = (
                    float(sum(value * weight for value, weight in weighted) / total_weight)
                    if total_weight else float(np.mean(values))
                )
            else:
                metrics[target] = float(np.mean(values))

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


def aggregate_actor_profiles(profiles: list[dict[str, float]]) -> dict[str, float]:
    """Sum per-rank inference/update counters for a single iteration summary."""
    result: dict[str, float] = {}
    for profile in profiles:
        for name, value in profile.items():
            result[name] = result.get(name, 0.0) + float(value)
    return result


def is_better_kyoku_selection(
    candidate: dict[str, float], best: dict[str, float] | None, *, score_epsilon: float = 1e-3,
) -> bool:
    """Compare evaluation windows without reintroducing half-match objectives."""
    if best is None:
        return True
    score_delta = candidate["eval/kyoku/point_delta_mean"] - best["eval/kyoku/point_delta_mean"]
    if score_delta > score_epsilon:
        return True
    if abs(score_delta) > score_epsilon:
        return False
    deal_in_delta = candidate["eval/kyoku/deal_in_rate"] - best["eval/kyoku/deal_in_rate"]
    if deal_in_delta < -1e-12:
        return True
    if abs(deal_in_delta) > 1e-12:
        return False
    return (
        candidate["eval/efficiency/optimal_shanten_rate"]
        > best["eval/efficiency/optimal_shanten_rate"]
    )


def run(config: dict[str, Any]) -> None:
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError("Ray is required: pip install -e riichi_ppo_v1") from exc
    from .worker import RolloutWorker
    from .inference import RolloutInferenceActor
    from .evaluation import EvaluationWorker, evaluation_cases, merge_evaluation_summaries
    if RolloutWorker is None:
        raise RuntimeError("Ray rollout worker could not be defined")
    if RolloutInferenceActor is None:
        raise RuntimeError("Ray rollout inference actor could not be defined")
    seed_everything(int(config["seed"]))
    device = str(config["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")
    learner_gpus = int(config.get("learner_gpus", 1))
    if learner_gpus > 1 and not device.startswith("cuda"):
        raise ValueError("learner_gpus > 1 requires --device cuda")
    partitions = partition_worker_indices(int(config["num_workers"]), learner_gpus)
    init_method = local_distributed_init_method() if learner_gpus > 1 else None
    ray.init(ignore_reinit_error=True)
    inference_actors = []
    for rank, worker_ids in enumerate(partitions):
        actor_config = dict(config)
        actor_config["inference_actor_num_workers"] = len(worker_ids)
        inference_actors.append(RolloutInferenceActor.remote(actor_config, rank, learner_gpus, init_method))
    inference = inference_actors[0]
    worker_to_rank = {
        worker_id: rank
        for rank, worker_ids in enumerate(partitions)
        for worker_id in worker_ids
    }
    print(
        "learner_gpus="
        + str(learner_gpus)
        + " worker_partitions="
        + ",".join(f"rank{rank}:{worker_ids}" for rank, worker_ids in enumerate(partitions)),
        flush=True,
    )
    workers = [
        RolloutWorker.remote(
            index,
            config,
            inference_actors[worker_to_rank[index]],
        )
        for index in range(int(config["num_workers"]))
    ]
    evaluation_workers = []
    if bool(config.get("evaluation_enabled", True)):
        if EvaluationWorker is None:
            raise RuntimeError("Ray evaluation worker could not be defined")
        evaluation_workers = [
            EvaluationWorker.remote(index, config, inference_actors[index % len(inference_actors)])
            for index in range(max(1, int(config.get("evaluation_workers", 4))))
        ]
    writer = SummaryWriter(str(Path(config["checkpoint_dir"]) / "tensorboard"))
    semantic_path = Path(config["checkpoint_dir"]) / str(config.get("semantic_metrics_jsonl", "metrics.jsonl"))
    global_decisions, global_kyokus = metric_counters(semantic_path)
    rolling_kyokus = RollingKyokuMetrics(int(config.get("metrics_rolling_kyokus", 1000)))
    gpu_sampler = GpuSampler(bool(config.get("gpu_monitor", True)) and device.startswith("cuda"), float(config.get("gpu_sample_interval_s", 0.25)))
    gpu_sampler.start()
    last_evaluated = -1
    evaluation_history: list[dict[str, float]] = []
    best_selection: dict[str, float] | None = None

    def run_evaluation(update: int) -> None:
        """Run the fixed public baseline at a safe rollout/update boundary."""
        nonlocal last_evaluated, evaluation_history, best_selection
        if not evaluation_workers:
            return
        evaluation_interval = max(1, int(config.get("evaluation_interval_updates", 15)))
        cases = evaluation_cases(
            int(config.get("evaluation_seed_base", 20260717)),
            int(config.get("evaluation_hanchan_count", 10)),
            cycle=int(update) // evaluation_interval,
        )
        futures = [
            evaluation_workers[index % len(evaluation_workers)].evaluate.remote(seed, seat, recipe)
            for index, (seed, seat, recipe) in enumerate(cases)
        ]
        results = ray.get(futures)
        values = merge_evaluation_summaries([result["metrics"] for result in results])
        write_curated_scalars(writer, values, update)
        append_metric_jsonl(semantic_path, update=update, global_decisions=global_decisions,
                            global_kyokus=global_kyokus, source="evaluation", metrics=values,
                            metadata={"seed_base": int(config.get("evaluation_seed_base", 20260717)),
                                      "cases": [{key: result[key] for key in ("seed", "candidate_seat", "opponents")} for result in results]})
        print(
            f"evaluation update={update} kyokus={values.get('eval/kyoku/count', 0):.0f} "
            f"point_delta_mean={values.get('eval/kyoku/point_delta_mean', 0.0):.4f} "
            f"deal_in_rate={values.get('eval/kyoku/deal_in_rate', 0.0):.4f} "
            f"optimal_shanten_rate={values.get('eval/efficiency/optimal_shanten_rate', 0.0):.4f}",
            flush=True,
        )
        evaluation_history = (evaluation_history + [{
            name: float(values[name])
            for name in (
                "eval/kyoku/point_delta_mean", "eval/kyoku/deal_in_rate",
                "eval/efficiency/optimal_shanten_rate",
            )
            if name in values
        }])[-3:]
        if len(evaluation_history) == 3:
            window = {
                name: float(np.mean([row[name] for row in evaluation_history]))
                for name in evaluation_history[0]
                if all(name in row for row in evaluation_history)
            }
            required = {
                "eval/kyoku/point_delta_mean", "eval/kyoku/deal_in_rate",
                "eval/efficiency/optimal_shanten_rate",
            }
            if required.issubset(window):
                is_best = is_better_kyoku_selection(window, best_selection)
                if is_best:
                    best_selection = window
                    path = Path(config["checkpoint_dir"]) / "best_kyoku.pt"
                    ray.get(inference.save.remote(str(path), config))
                    append_metric_jsonl(
                        semantic_path, update=update, global_decisions=global_decisions,
                        global_kyokus=global_kyokus, source="selection",
                        metrics={f"selection/{name.removeprefix('eval/')}": value for name, value in window.items()},
                        metadata={"window_updates": [update - 2 * int(config.get("evaluation_interval_updates", 50)),
                                                      update - int(config.get("evaluation_interval_updates", 50)), update],
                                  "checkpoint": str(path)},
                    )
                    print(
                        f"best_kyoku_checkpoint={path} "
                        f"point_delta_mean_3eval={window['eval/kyoku/point_delta_mean']:.4f}",
                        flush=True,
                    )
        last_evaluated = update

    try:
        start_iteration = ray.get(inference.iteration.remote())
        run_evaluation(start_iteration)
        for iteration in range(start_iteration, int(config["iterations"])):
            gpu_cursor = gpu_sampler.checkpoint()
            algorithm_started = time.perf_counter()
            begin_rollout_started = time.perf_counter()
            ray.get([actor.begin_rollout.remote(
                iteration, split_policy_inference=False,
            )
                     for actor in inference_actors])
            begin_rollout_s = time.perf_counter() - begin_rollout_started
            rollout_started = time.perf_counter()
            results = ray.get([
                worker.collect.remote(iteration)
                for worker in workers
            ])
            rollout_wall_s = time.perf_counter() - rollout_started
            profile_summary_started = time.perf_counter()
            actor_profiles = ray.get([actor.profile_summary.remote() for actor in inference_actors])
            actor_profile = aggregate_actor_profiles(actor_profiles)
            profile_summary_s = time.perf_counter() - profile_summary_started
            transition_assembly_started = time.perf_counter()
            transitions = [transition for worker_transitions, _ in results for transition in worker_transitions]
            transition_assembly_s = time.perf_counter() - transition_assembly_started
            update_started = time.perf_counter()
            update_seed = int(config["seed"]) + (iteration + 1) * 1_000_003
            update_results = ray.get([actor.update.remote(transitions, update_seed) for actor in inference_actors])
            metrics = update_results[0]
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
            for rank, profile in enumerate(actor_profiles):
                rollout_metrics.update({
                    f"rollout/inference_actor/rank{rank}/{name}": float(value)
                    for name, value in profile.items()
                })
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
            global_decisions += int(recorded_decisions)
            global_kyokus += int(rollout_kyokus)
            if "train/kyoku/point_delta_mean" in rollout_metrics:
                point_count = int(rollout_metrics.get("train/kyoku/count", 0.0))
                rollout_metrics.update(rolling_kyokus.update(
                    [float(rollout_metrics["train/kyoku/point_delta_mean"])] * point_count
                ))
            env_step_s = float(worker_timing.get("env/step_batch_native", {}).get("worker_mean_total_s", 0.0))
            update_forward_s = float(metrics.get("timing/update/model_forward/total_s", 0.0))
            inference_rows = float(actor_profile.get("inference/full_forward_rows_mean", 0.0))
            inference_dispatch_rows = float(actor_profile.get("inference/dispatch_rows_mean", 0.0))
            print(f"iteration={iteration + 1} transitions={len(transitions)} model_decisions={model_decisions:.0f} recorded_decisions={recorded_decisions:.0f} kyokus={rollout_kyokus:.0f} reward={rollout_reward:.4f} worker_transitions_per_s={rollout_tps:.2f} algorithm_wall_s={algorithm_wall_s:.3f} sps={recorded_decisions / max(algorithm_wall_s, 1e-9):.2f} model_forward_sps={model_decisions / max(algorithm_wall_s, 1e-9):.2f} rollout_wall_s={rollout_wall_s:.3f} begin_rollout_s={begin_rollout_s:.3f} profile_summary_s={profile_summary_s:.3f} transition_assembly_s={transition_assembly_s:.3f} env_step_s={env_step_s:.3f} inference_rows_per_forward={inference_rows:.2f} inference_rows_per_dispatch={inference_dispatch_rows:.2f} update_wall_s={update_wall_s:.3f} update_forward_s={update_forward_s:.3f} epochs={metrics['update/epochs_completed']:.0f}/{metrics['update/configured_epochs']:.0f} minibatches={metrics['update/executed_minibatches']:.0f}/{metrics['update/planned_minibatches']:.0f} early_stop={bool(metrics['update/early_stop'])} executed_samples={metrics['update/executed_transition_samples']:.0f} tokens_mean={metrics['update/executed_transition_tokens_mean']:.2f} input_tokens_mean={metrics['update/executed_transition_input_tokens_mean']:.2f} " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items() if key in {"loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "grad_norm"}), flush=True)
            tensorboard_metrics = {
                **rollout_metrics,
                **{f"ppo/{name}": float(value) for name, value in metrics.items()},
            }
            learner_peak_mb = learner_peak_allocated_mb(update_results)
            if learner_peak_mb is not None:
                tensorboard_metrics["system/learner_gpu_peak_allocated_mb"] = learner_peak_mb
            write_curated_scalars(writer, tensorboard_metrics, iteration + 1)
            histogram_interval = int(config.get("metrics_histogram_interval", 25))
            if histogram_interval > 0 and (iteration + 1) % histogram_interval == 0 and transitions:
                writer.add_histogram("diagnostics/return", np.asarray([item.return_ for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/advantage", np.asarray([item.advantage for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/value_prediction", np.asarray([item.value for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/legal_action_count", np.asarray([item.legal_mask.sum() for item in transitions], dtype=np.int16), iteration + 1)
                writer.add_histogram("diagnostics/token_length", np.asarray([item.token_length for item in transitions], dtype=np.int16), iteration + 1)
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
            if bool(config.get("semantic_metrics_enabled", True)):
                semantic_values = {**{name: value for name, value in rollout_metrics.items() if name.startswith(("train/", "reward_schedule/"))},
                                   **{f"ppo/{name}": value for name, value in metrics.items()}, **gpu_metrics}
                append_metric_jsonl(semantic_path, update=iteration + 1, global_decisions=global_decisions,
                                    global_kyokus=global_kyokus, source="train", metrics=semantic_values,
                                    metadata={
                                        "game_mode": config["game_mode"],
                                        "opponent_mix": "current_self_play_all_seats",
                                    })
            if (iteration + 1) % max(1, int(config.get("evaluation_interval_updates", 50))) == 0:
                run_evaluation(iteration + 1)
        final_update = ray.get(inference.iteration.remote())
        if final_update != last_evaluated:
            run_evaluation(final_update)
        ray.get(inference.save.remote(str(Path(config["checkpoint_dir"]) / "latest.pt"), config))
    finally:
        if "inference_actors" in locals():
            try:
                ray.get([actor.shutdown.remote() for actor in inference_actors])
            except Exception:
                pass
        gpu_sampler.stop()
        writer.close()
        ray.shutdown()


def _parser(smoke: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML overlay applied after all grouped defaults")
    parser.add_argument("--model-config", default=None, help="YAML overlay for model settings")
    parser.add_argument("--environment-config", default=None, help="YAML overlay for environment settings")
    parser.add_argument("--training-config", default=None, help="YAML overlay for training/runtime settings")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    if not smoke:
        parser.add_argument("--num-workers", type=int, default=None)
        parser.add_argument("--learner-gpus", type=int, default=None)
        parser.add_argument("--envs-per-worker", type=int, default=None)
        parser.add_argument("--kyokus-per-worker", type=int, default=None)
        parser.add_argument("--update-epochs", type=int, default=None)
        parser.add_argument("--minibatch-size", type=int, default=None)
        parser.add_argument(
            "--update-batch-mode",
            choices=("streaming", "prefetch", "gpu_cache", "auto"),
            default=None,
        )
        parser.add_argument("--inference-batch-wait-ms", type=float, default=None)
        parser.add_argument("--target-kl", type=float, default=None)
        parser.add_argument("--profile-cuda-sync", action=argparse.BooleanOptionalAction, default=None)
    if smoke:
        parser.add_argument("--kyokus", type=int, default=1)
    return parser


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        "num_workers": getattr(args, "num_workers", None),
        "learner_gpus": getattr(args, "learner_gpus", None),
        "envs_per_worker": getattr(args, "envs_per_worker", None),
        "kyokus_per_worker": getattr(args, "kyokus_per_worker", None),
        "update_epochs": getattr(args, "update_epochs", None),
        "minibatch_size": getattr(args, "minibatch_size", None),
        "update_batch_mode": getattr(args, "update_batch_mode", None),
        "inference_batch_wait_ms": getattr(args, "inference_batch_wait_ms", None),
        "target_kl": getattr(args, "target_kl", None),
        "profile_cuda_sync": getattr(args, "profile_cuda_sync", None),
    }
    config.update({name: value for name, value in overrides.items() if value is not None})


def main() -> None:
    args = _parser().parse_args()
    config = load_config(
        args.config,
        model_path=args.model_config,
        environment_path=args.environment_config,
        training_path=args.training_config,
    )
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
    config = load_config(
        args.config,
        model_path=args.model_config,
        environment_path=args.environment_config,
        training_path=args.training_config,
    )
    config.update({"device": args.device or "cpu", "num_workers": 1, "learner_gpus": 1, "envs_per_worker": 1, "kyokus_per_worker": args.kyokus, "iterations": 1, "update_epochs": 4, "target_kl": 0.0, "minibatch_size": 32, "update_batch_mode": "auto", "checkpoint_dir": "checkpoints/riichi_ppo_v1_smoke", "evaluation_enabled": False})
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    apply_cli_overrides(config, args)
    run(config)


if __name__ == "__main__":
    main()
