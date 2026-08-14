"""Command-line entry points for synchronous Ray PPO training."""

from __future__ import annotations

import argparse
import datetime
from importlib import resources
import json
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
from ..model.schema import TOKEN_SCHEMA_VERSION
from .learner import validate_fresh_model_checkpoint_contract
from ..evaluation.head_to_head_1v3_shards import (
    REQUIRED_1V3_PROCESSES,
    run_sharded_1v3,
    validate_1v3_shard_plan,
)


_CONFIG_GROUPS = ("training", "monitoring")


def _progress_md_path(config: dict[str, Any]) -> Path:
    return (
        Path(config.get("eval1v3_output_dir", "audit/reports/v14_ppo_20260812/eval"))
        .parent
        / "PROGRESS.md"
    )


def _update_progress_md(
    config: dict[str, Any],
    update: int,
    rollout_metrics: dict[str, float],
    ppo_metrics: dict[str, float],
    eval_summary: dict[str, Any] | None,
) -> None:
    """Append the required per-60-update progress entry to PROGRESS.md."""
    progress = _progress_md_path(config)
    if not progress.is_file():
        return
    interval = max(1, int(config.get("progress_update_interval_updates", 60)))
    if int(update) <= 0 or int(update) % interval != 0:
        return
    combined = {
        **rollout_metrics,
        **{f"ppo/{name}": float(value) for name, value in ppo_metrics.items()},
    }

    def value(name: str) -> str:
        if name in combined:
            return f"{float(combined[name]):.5g}"
        return "n/a"

    lines = [
        "",
        f"## {datetime.date.today().isoformat()} update={update}",
        "",
        f"- reward_mean={value('rollout/reward_mean')} "
        f"q_loss={value('ppo/q_loss')} "
        f"sft_reference_kl={value('ppo/sft_reference_kl')} "
        f"actor_grad_norm={value('ppo/grad_norm_actor')} "
        f"critic_grad_norm={value('ppo/grad_norm_critic')} "
        f"shared_grad_norm={value('ppo/grad_norm_shared')}",
        f"- rollout_wall_s={value('rollout/wall_s')} "
        f"update_wall_s={value('update/wall_s')} "
        f"sps={value('iteration/sps')} "
        f"history_seats={value('rollout/opponent_mix/history_seats')} "
        f"history_pool_size={value('rollout/opponent_mix/history_pool_size')}",
    ]
    if eval_summary is not None:
        model_a = eval_summary["model_a"]
        lines.append(
            f"- 1v3 vs SFT: first_place_rate={model_a['first_place_rate']:.4f} "
            f"top2_rate={model_a['top2_rate']:.4f} "
            f"mean_rank={model_a['mean_rank']:.3f} "
            f"point_diff_mean={model_a['point_diff_mean']:+.1f} "
            f"ci95={model_a['point_diff_bootstrap_ci95']}"
        )
    with progress.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


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
) -> dict[str, Any]:
    """加载自包含版本配置或打包的当前默认配置。

    传入 ``path`` 时该文件必须是自包含的完整版本配置,直接加载、不叠加打包
    默认;未传时合并打包的 ``training`` 与 ``monitoring`` 两组当前默认。
    """
    if path is not None:
        return _load_config_file(path)
    config: dict[str, Any] = {}
    for group in _CONFIG_GROUPS:
        config.update(_load_config_file(_packaged_config_path(group)))
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
            elif name == "train/action/riichi_opportunity_accept_rate":
                weighted = [
                    (float(stats[name]), float(stats.get("train/action/riichi_opportunity_count", 0.0)))
                    for stats in stats_list if name in stats and np.isfinite(stats[name])
                ]
                total_weight = sum(weight for _value, weight in weighted)
                metrics[target] = (
                    float(sum(value * weight for value, weight in weighted) / total_weight)
                    if total_weight else float(np.mean(values))
                )
            elif name == "train/action/call_opportunity_accept_rate":
                weighted = [
                    (float(stats[name]), float(stats.get("train/action/call_opportunity_count", 0.0)))
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


def run(config: dict[str, Any]) -> None:
    if bool(config.get("eval1v3_enabled", False)):
        configured_processes = int(
            config.get("eval1v3_processes", REQUIRED_1V3_PROCESSES)
        )
        if configured_processes != REQUIRED_1V3_PROCESSES:
            raise ValueError(
                f"all 1v3 evaluations require exactly {REQUIRED_1V3_PROCESSES} processes"
            )
        if int(config.get("eval1v3_hanchans_per_process", 160)) <= 0:
            raise ValueError("eval1v3_hanchans_per_process must be positive")
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
    seed_everything(int(config["seed"]))
    resume_payload: dict[str, Any] = {}
    init_payload: dict[str, Any] = {}
    if config.get("resume") and config.get("init_model"):
        raise ValueError("resume and init_model are mutually exclusive")
    if config.get("resume"):
        resume_payload = torch.load(config["resume"], map_location="cpu", weights_only=False)
        schema = int(resume_payload.get("token_schema_version", 0))
        if schema != TOKEN_SCHEMA_VERSION:
            raise RuntimeError(
                f"cannot resume token schema {schema}; required schema is {TOKEN_SCHEMA_VERSION}"
            )
    elif config.get("init_model"):
        init_payload = torch.load(config["init_model"], map_location="cpu", weights_only=False)
        try:
            validate_fresh_model_checkpoint_contract(init_payload)
        except RuntimeError as exc:
            raise RuntimeError(f"cannot initialize from incompatible checkpoint: {exc}") from exc
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
    saved_actor_rng = resume_payload.get("extra_state", {}).get("actor_rng_states", [])
    if saved_actor_rng:
        if len(saved_actor_rng) != len(inference_actors):
            raise RuntimeError("checkpoint learner rank count differs from current learner_gpus")
        ray.get([
            actor.load_rng_state.remote(state)
            for actor, state in zip(inference_actors, saved_actor_rng, strict=True)
        ])
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
    writer = SummaryWriter(str(Path(config["checkpoint_dir"]) / "tensorboard"))
    semantic_path = Path(config["checkpoint_dir"]) / str(config.get("semantic_metrics_jsonl", "metrics.jsonl"))
    global_decisions, global_kyokus = metric_counters(semantic_path)
    rolling_kyokus = RollingKyokuMetrics(int(config.get("metrics_rolling_kyokus", 1000)))
    gpu_sampler = GpuSampler(bool(config.get("gpu_monitor", True)) and device.startswith("cuda"), float(config.get("gpu_sample_interval_s", 0.25)))
    gpu_sampler.start()

    def checkpoint_extra_state() -> dict[str, Any]:
        return {
            "schema_version": TOKEN_SCHEMA_VERSION,
            "actor_rng_states": ray.get([
                actor.rng_state.remote() for actor in inference_actors
            ]),
        }

    def run_1v3_evaluation(update: int) -> dict[str, Any] | None:
        """Block on the fixed 1600-hanchan 1v3 vs V13 SFT evaluation."""
        if not bool(config.get("eval1v3_enabled", False)):
            return None
        interval = max(1, int(config.get("eval1v3_interval_updates", 30)))
        if update <= 0 or int(update) % interval != 0:
            return None
        checkpoint_path = (
            Path(config["checkpoint_dir"]) / f"checkpoint_{int(update):05d}.pt"
        )
        if not checkpoint_path.is_file():
            raise RuntimeError(
                "checkpoint required by the 1v3 evaluation does not exist: "
                f"{checkpoint_path}"
            )
        output_dir = Path(config["eval1v3_output_dir"])
        processes = int(config.get("eval1v3_processes", REQUIRED_1V3_PROCESSES))
        if processes != REQUIRED_1V3_PROCESSES:
            raise RuntimeError(
                f"all 1v3 evaluations require exactly {REQUIRED_1V3_PROCESSES} processes"
            )
        hanchans_per_process = int(
            config.get("eval1v3_hanchans_per_process", 160)
        )
        seed_base = int(config.get("eval1v3_seed_base", 20260812))
        summary_path = output_dir / f"vs_sft_u{int(update):03d}.json"
        if summary_path.is_file():
            with open(summary_path, encoding="utf-8") as file:
                summary = json.load(file)
            validate_1v3_shard_plan(
                list(summary.get("shards", [])),
                seed_base=seed_base,
                hanchans_per_process=hanchans_per_process,
            )
        else:
            summary = run_sharded_1v3(
                checkpoint_path,
                config["eval1v3_model_b"],
                update=int(update),
                processes=processes,
                hanchans_per_process=hanchans_per_process,
                parallel_hanchans=int(
                    config.get("eval1v3_parallel_hanchans", 160)
                ),
                devices=tuple(
                    str(device)
                    for device in config.get("eval1v3_devices", ("0", "2"))
                ),
                seed_base=seed_base,
                output_dir=output_dir,
            )
        model_a = summary["model_a"]
        print(
            f"1v3_eval update={update} "
            f"first_place_rate={model_a['first_place_rate']:.4f} "
            f"top2_rate={model_a['top2_rate']:.4f} "
            f"mean_rank={model_a['mean_rank']:.3f} "
            f"point_diff_mean={model_a['point_diff_mean']:.2f} "
            f"ci95={model_a['point_diff_bootstrap_ci95']}",
            flush=True,
        )
        append_jsonl(
            Path(config["checkpoint_dir"]) / "eval1v3.jsonl",
            {"update": int(update), "timestamp": time.time(), **summary},
        )
        return summary

    try:
        start_iteration = ray.get(inference.iteration.remote())
        eval1v3_summary: dict[str, Any] | None = None
        for iteration in range(start_iteration, int(config["iterations"])):
            update_number = iteration + 1
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
                worker.collect.remote(
                    update_number,
                )
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
                "iteration/policy_update": float(update_number),
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
            print(f"iteration={iteration + 1} transitions={len(transitions)} model_decisions={model_decisions:.0f} recorded_decisions={recorded_decisions:.0f} kyokus={rollout_kyokus:.0f} reward={rollout_reward:.4f} worker_transitions_per_s={rollout_tps:.2f} algorithm_wall_s={algorithm_wall_s:.3f} sps={recorded_decisions / max(algorithm_wall_s, 1e-9):.2f} model_forward_sps={model_decisions / max(algorithm_wall_s, 1e-9):.2f} rollout_wall_s={rollout_wall_s:.3f} begin_rollout_s={begin_rollout_s:.3f} profile_summary_s={profile_summary_s:.3f} transition_assembly_s={transition_assembly_s:.3f} env_step_s={env_step_s:.3f} inference_rows_per_forward={inference_rows:.2f} inference_rows_per_dispatch={inference_dispatch_rows:.2f} update_wall_s={update_wall_s:.3f} update_forward_s={update_forward_s:.3f} epochs={metrics['update/epochs_completed']:.0f}/{metrics['update/configured_epochs']:.0f} minibatches={metrics['update/executed_minibatches']:.0f}/{metrics['update/planned_minibatches']:.0f} early_stop={bool(metrics['update/early_stop'])} executed_samples={metrics['update/executed_transition_samples']:.0f} tokens_mean={metrics['update/executed_transition_tokens_mean']:.2f} input_tokens_mean={metrics['update/executed_transition_input_tokens_mean']:.2f} " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items() if key in {"loss", "policy_loss", "q_loss", "entropy", "approx_kl", "clipfrac", "grad_norm"}), flush=True)
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
                writer.add_histogram("diagnostics/q_target", np.asarray([item.q_target for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/qboost_advantage", np.asarray([item.advantage for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/q_taken", np.asarray([item.q_taken for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/expected_q", np.asarray([item.expected_q for item in transitions], dtype=np.float32), iteration + 1)
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
            checkpoint_interval = max(1, int(config.get("checkpoint_interval_updates", 50)))
            if update_number % checkpoint_interval == 0:
                checkpoint_path = Path(config["checkpoint_dir"]) / f"checkpoint_{update_number:05d}.pt"
                ray.get(inference.save.remote(
                    str(checkpoint_path), config, checkpoint_extra_state(),
                ))
                eval1v3_summary = run_1v3_evaluation(update_number)
            _update_progress_md(
                config, update_number, rollout_metrics, metrics, eval1v3_summary,
            )
        final_update = ray.get(inference.iteration.remote())
        ray.get(inference.save.remote(
            str(Path(config["checkpoint_dir"]) / "latest.pt"), config, checkpoint_extra_state(),
        ))
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
    parser.add_argument("--config", default=None, help="自包含版本配置 YAML(不与打包默认叠加)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--init-model", default=None, help="SFT/model checkpoint used for a fresh PPO run")
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
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.init_model:
        config["init_model"] = args.init_model
    apply_cli_overrides(config, args)
    run(config)


def smoke_main() -> None:
    args = _parser(smoke=True).parse_args()
    config = load_config(args.config)
    config.update({"device": args.device or "cpu", "num_workers": 1, "learner_gpus": 1, "envs_per_worker": 1, "kyokus_per_worker": args.kyokus, "iterations": 1, "update_epochs": 4, "target_kl": 0.0, "minibatch_size": 32, "update_batch_mode": "auto", "checkpoint_dir": "checkpoints/riichi_ppo_v1_smoke"})
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    apply_cli_overrides(config, args)
    run(config)


if __name__ == "__main__":
    main()
