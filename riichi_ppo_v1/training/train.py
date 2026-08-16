"""同步 Ray PPO 训练的命令行入口(V16)。

Ray 闭环:worker.collect → 驱动侧 learner.update → inference.update_weights;
每 30 updates 按 ``evaluation/mechanism.py`` 的固定 1v3 机制(10 进程 × 160 =
1600 半庄)评测一次。learner 持有模型 + optimizer(``cuda:0``),推理 actor 只
持有 eval 模型并在每个 update 后接收最新权重。
"""

from __future__ import annotations

import argparse
import datetime
from importlib import resources
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

# 保持项目自定义 CUDA_DEVICE 约定,同时在启动 torch(以及 Ray 拉起子进程)前
# 映射为 CUDA 标准的 CUDA_VISIBLE_DEVICES,使每个 Ray GPU actor 仍看到 cuda:0。
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
from .learner import PPOLearner, validate_fresh_model_checkpoint_contract
from ..evaluation.head_to_head_1v3_shards import (
    run_sharded_1v3,
    validate_1v3_shard_plan,
)
from ..evaluation.mechanism import (
    DEFAULT_1V3_HANCHANS_PER_PROCESS,
    DEFAULT_1V3_INTERVAL_UPDATES,
    REQUIRED_1V3_PROCESSES,
    progress_md_path,
)


_CONFIG_GROUPS = ("training", "monitoring")


def _progress_md_path(config: dict[str, Any]) -> Path | None:
    """进度报告路径:`audit/reports/<版本号>/report/PROGRESS.md`。"""
    output_value = config.get("eval1v3_output_dir")
    if not output_value:
        return None
    return progress_md_path(output_value)


def _update_progress_md(
    config: dict[str, Any],
    update: int,
    rollout_metrics: dict[str, float],
    ppo_metrics: dict[str, float],
    eval_summary: dict[str, Any] | None,
) -> None:
    """Append the required per-60-update progress entry to PROGRESS.md."""
    progress = _progress_md_path(config)
    if progress is None or not progress.is_file():
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
        f"value_loss={value('ppo/value_loss')} "
        f"q_loss={value('ppo/q_loss')} "
        f"entropy={value('ppo/entropy')} "
        f"actor_grad_norm={value('ppo/grad_norm_actor')} "
        f"critic_grad_norm={value('ppo/grad_norm_critic')} "
        f"shared_grad_norm={value('ppo/grad_norm_shared')}",
        f"- rollout_wall_s={value('rollout/wall_s')} "
        f"update_wall_s={value('update/wall_s')} "
        f"sps={value('iteration/sps')} "
        f"grp_calls={value('rollout/grp_calls')} "
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
    """加载自包含版本配置或打包的当前默认配置(不叠加)。"""
    if path is not None:
        return _load_config_file(path)
    config: dict[str, Any] = {}
    for group in _CONFIG_GROUPS:
        config.update(_load_config_file(_packaged_config_path(group)))
    return config


def configure_ray_stderr_logging() -> None:
    """让 Ray 与子进程日志流向标准错误,由脚本统一收敛到 ``logs/<版本号>/``。"""
    os.environ.setdefault("RAY_LOG_TO_STDERR", "1")


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


def summarize_worker_rollout(
    results: list[tuple[list[Any], dict[str, float]]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Aggregate worker stats while retaining the slowest worker per stage."""
    stats_list = [stats for _transitions, stats in results]
    metrics: dict[str, float] = {}
    for name in {name for stats in stats_list for name in stats}:
        values = [
            float(stats[name])
            for stats in stats_list
            if name in stats and np.isfinite(stats[name])
        ]
        if not values:
            continue
        target = name if name.startswith(("train/", "reward_schedule/")) else f"rollout/{name}"
        if name.endswith("/count") or name.endswith("_count"):
            metrics[target] = float(np.sum(values))
        elif name.startswith("train/kyoku/"):
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
    for total_name in sorted(
        name
        for stats in stats_list
        for name in stats
        if name.startswith("timing/") and name.endswith("/total_s")
    ):
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
    """Sum per-rank inference counters for a single iteration summary."""
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
    else:
        raise ValueError("V16 PPO requires --init-model or a resume checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("V16 PPO requires CUDA")
    learner_gpus = int(config.get("learner_gpus", 1))
    if learner_gpus < 1:
        raise ValueError("learner_gpus must be positive")
    partitions = partition_worker_indices(int(config["num_workers"]), learner_gpus)
    configure_ray_stderr_logging()
    ray.init(ignore_reinit_error=True)

    learner_hp = {
        key: value
        for key, value in config.items()
        if key not in {"model_size", "device"}
    }
    learner = PPOLearner("v16", "cuda:0", **learner_hp)
    if config.get("resume"):
        learner.load(config["resume"])
    elif config.get("init_model"):
        learner.load_model_weights(config["init_model"])

    inference_actors = []
    for rank, worker_ids in enumerate(partitions):
        actor_config = dict(config)
        actor_config["inference_actor_num_workers"] = len(worker_ids)
        inference_actors.append(
            RolloutInferenceActor.options(num_gpus=1).remote(actor_config)
        )
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
    # 先推送初始权重再创建 worker,保证第一个 rollout 从 SFT/resume 权重采样。
    ray.get([actor.update_weights.remote(learner.weights()) for actor in inference_actors])
    workers = [
        RolloutWorker.remote(
            index,
            config,
            inference_actors[worker_to_rank[index]],
        )
        for index in range(int(config["num_workers"]))
    ]
    writer = SummaryWriter(str(Path(config["checkpoint_dir"]) / "tensorboard"))
    semantic_path = Path(config["checkpoint_dir"]) / str(
        config.get("semantic_metrics_jsonl", "metrics.jsonl")
    )
    global_decisions, global_kyokus = metric_counters(semantic_path)
    rolling_kyokus = RollingKyokuMetrics(int(config.get("metrics_rolling_kyokus", 1000)))
    gpu_sampler = GpuSampler(
        bool(config.get("gpu_monitor", True)),
        float(config.get("gpu_sample_interval_s", 0.25)),
    )
    gpu_sampler.start()

    def checkpoint_extra_state() -> dict[str, Any]:
        return {
            "schema_version": TOKEN_SCHEMA_VERSION,
            "actor_rng_states": ray.get([
                actor.rng_state.remote() for actor in inference_actors
            ]),
        }

    saved_actor_rng = resume_payload.get("extra_state", {}).get("actor_rng_states", [])
    if saved_actor_rng:
        if len(saved_actor_rng) != len(inference_actors):
            raise RuntimeError("checkpoint inference actor count differs from learner_gpus")
        ray.get([
            actor.load_rng_state.remote(state)
            for actor, state in zip(inference_actors, saved_actor_rng, strict=True)
        ])

    def run_1v3_evaluation(update: int) -> dict[str, Any] | None:
        """阻塞执行固定 1600 半庄的 1v3 对抗评测。"""
        if not bool(config.get("eval1v3_enabled", False)):
            return None
        interval = max(
            1,
            int(config.get("eval1v3_interval_updates", DEFAULT_1V3_INTERVAL_UPDATES)),
        )
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
        output_value = config.get("eval1v3_output_dir")
        if not output_value:
            raise RuntimeError("eval1v3_output_dir is required when 1v3 evaluation is enabled")
        output_dir = Path(output_value)
        model_b = config.get("eval1v3_model_b")
        if not model_b:
            raise RuntimeError("eval1v3_model_b is required when 1v3 evaluation is enabled")
        processes = int(config.get("eval1v3_processes", REQUIRED_1V3_PROCESSES))
        if processes != REQUIRED_1V3_PROCESSES:
            raise RuntimeError(
                f"all 1v3 evaluations require exactly {REQUIRED_1V3_PROCESSES} processes"
            )
        hanchans_per_process = int(
            config.get("eval1v3_hanchans_per_process", DEFAULT_1V3_HANCHANS_PER_PROCESS)
        )
        seed_base = int(config["eval1v3_seed_base"])
        parallel_hanchans = int(
            config.get("eval1v3_parallel_hanchans", hanchans_per_process)
        )
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
                model_b,
                update=int(update),
                processes=processes,
                hanchans_per_process=hanchans_per_process,
                parallel_hanchans=parallel_hanchans,
                devices=tuple(
                    str(device) for device in config.get("eval1v3_devices", ("0", "1"))
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
            output_dir / "eval1v3.jsonl",
            {"update": int(update), "timestamp": time.time(), **summary},
        )
        return summary

    try:
        eval1v3_summary: dict[str, Any] | None = None
        for iteration in range(learner.iteration, int(config["iterations"])):
            update_number = iteration + 1
            gpu_cursor = gpu_sampler.checkpoint()
            algorithm_started = time.perf_counter()
            begin_rollout_started = time.perf_counter()
            ray.get([actor.begin_rollout.remote(update_number) for actor in inference_actors])
            begin_rollout_s = time.perf_counter() - begin_rollout_started
            rollout_started = time.perf_counter()
            results = ray.get([
                worker.collect.remote(update_number)
                for worker in workers
            ])
            rollout_wall_s = time.perf_counter() - rollout_started
            profile_summary_started = time.perf_counter()
            actor_profiles = ray.get([
                actor.profile_summary.remote() for actor in inference_actors
            ])
            actor_profile = aggregate_actor_profiles(actor_profiles)
            profile_summary_s = time.perf_counter() - profile_summary_started
            transition_assembly_started = time.perf_counter()
            transitions = [
                transition
                for worker_transitions, _stats in results
                for transition in worker_transitions
            ]
            transition_assembly_s = time.perf_counter() - transition_assembly_started
            update_started = time.perf_counter()
            update_seed = int(config["seed"]) + (iteration + 1) * 1_000_003
            metrics = learner.update(transitions, shuffle_seed=update_seed)
            ray.get([
                actor.update_weights.remote(learner.weights())
                for actor in inference_actors
            ])
            update_wall_s = time.perf_counter() - update_started
            algorithm_wall_s = time.perf_counter() - algorithm_started
            gpu_metrics = gpu_sampler.summary(gpu_cursor)
            rollout_reward = float(np.nanmean([stats["reward_mean"] for _t, stats in results]))
            rollout_kyokus = sum(stats["kyokus"] for _t, stats in results)
            rollout_tps = float(np.nanmean([
                stats.get("transitions_per_s", float("nan"))
                for _t, stats in results
            ]))
            model_decisions = sum(stats.get("model_decisions", 0.0) for _t, stats in results)
            recorded_decisions = sum(stats.get("recorded_decisions", 0.0) for _t, stats in results)
            rollout_metrics, worker_timing = summarize_worker_rollout(results)
            actor_metrics = {
                f"rollout/inference_actor/{name}": float(value)
                for name, value in actor_profile.items()
            }
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
            env_step_s = float(
                worker_timing.get("env/step_batch_native", {}).get("worker_mean_total_s", 0.0)
            )
            update_forward_s = float(metrics.get("timing/update/model_forward/total_s", 0.0))
            inference_rows = float(actor_profile.get("inference/full_forward_rows_mean", 0.0))
            inference_dispatch_rows = float(actor_profile.get("inference/dispatch_rows_mean", 0.0))
            print(
                f"iteration={iteration + 1} transitions={len(transitions)} "
                f"model_decisions={model_decisions:.0f} recorded_decisions={recorded_decisions:.0f} "
                f"kyokus={rollout_kyokus:.0f} reward={rollout_reward:.4f} "
                f"worker_transitions_per_s={rollout_tps:.2f} "
                f"algorithm_wall_s={algorithm_wall_s:.3f} "
                f"sps={recorded_decisions / max(algorithm_wall_s, 1e-9):.2f} "
                f"rollout_wall_s={rollout_wall_s:.3f} "
                f"env_step_s={env_step_s:.3f} "
                f"inference_rows_per_forward={inference_rows:.2f} "
                f"inference_rows_per_dispatch={inference_dispatch_rows:.2f} "
                f"update_wall_s={update_wall_s:.3f} update_forward_s={update_forward_s:.3f} "
                f"epochs={metrics['update/epochs_completed']:.0f}/{metrics['update/configured_epochs']:.0f} "
                f"minibatches={metrics['update/executed_minibatches']:.0f}/{metrics['update/planned_minibatches']:.0f} "
                f"early_stop={bool(metrics['update/early_stop'])} "
                f"executed_samples={metrics['update/executed_transition_samples']:.0f} "
                f"tokens_mean={metrics['update/executed_transition_tokens_mean']:.2f} "
                + " ".join(
                    f"{key}={value:.5f}"
                    for key, value in metrics.items()
                    if key in {"loss", "policy_loss", "value_loss", "q_loss", "entropy", "approx_kl", "clipfrac", "grad_norm"}
                ),
                flush=True,
            )
            tensorboard_metrics = {
                **rollout_metrics,
                **{f"ppo/{name}": float(value) for name, value in metrics.items()},
            }
            learner_peak_mb = learner_peak_allocated_mb([metrics])
            if learner_peak_mb is not None:
                tensorboard_metrics["system/learner_gpu_peak_allocated_mb"] = learner_peak_mb
            write_curated_scalars(writer, tensorboard_metrics, iteration + 1)
            histogram_interval = int(config.get("metrics_histogram_interval", 25))
            if histogram_interval > 0 and (iteration + 1) % histogram_interval == 0 and transitions:
                writer.add_histogram("diagnostics/q_target", np.asarray([item.q_target for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/qboost_advantage", np.asarray([item.advantage for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/q_taken", np.asarray([item.q_taken for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/expected_q", np.asarray([item.expected_q for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/value", np.asarray([item.value for item in transitions], dtype=np.float32), iteration + 1)
                writer.add_histogram("diagnostics/legal_action_count", np.asarray([item.legal_mask.sum() for item in transitions], dtype=np.int16), iteration + 1)
                writer.add_histogram("diagnostics/sequence_length", np.asarray([item.history_length + item.snapshot_length + 2 * item.query_pair_counts for item in transitions], dtype=np.int16), iteration + 1)
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
                semantic_values = {
                    **{
                        name: value
                        for name, value in rollout_metrics.items()
                        if name.startswith(("train/", "reward_schedule/"))
                    },
                    **{f"ppo/{name}": value for name, value in metrics.items()},
                    **gpu_metrics,
                }
                append_metric_jsonl(
                    semantic_path,
                    update=iteration + 1,
                    global_decisions=global_decisions,
                    global_kyokus=global_kyokus,
                    source="train",
                    metrics=semantic_values,
                    metadata={
                        "game_mode": config["game_mode"],
                        "opponent_mix": "current_self_play_all_seats",
                    },
                )
            checkpoint_interval = max(1, int(config.get("checkpoint_interval_updates", 30)))
            if update_number % checkpoint_interval == 0:
                checkpoint_path = Path(config["checkpoint_dir"]) / f"checkpoint_{update_number:05d}.pt"
                learner.save(str(checkpoint_path), config, checkpoint_extra_state())
                eval1v3_summary = run_1v3_evaluation(update_number)
            _update_progress_md(
                config, update_number, rollout_metrics, metrics, eval1v3_summary,
            )
        learner.save(
            str(Path(config["checkpoint_dir"]) / "latest.pt"),
            config,
            checkpoint_extra_state(),
        )
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
            choices=("streaming", "auto"),
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


def cleanup_smoke_artifacts(path: str | Path) -> None:
    """删除冒烟测试产生的 checkpoint/日志/结果目录(仅限冒烟自身产物)。"""
    shutil.rmtree(path, ignore_errors=True)


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
    config.update({
        "device": args.device or "cuda",
        "num_workers": 1,
        "learner_gpus": 1,
        "envs_per_worker": 1,
        "kyokus_per_worker": args.kyokus,
        "iterations": 1,
        "update_epochs": 4,
        "target_kl": 0.0,
        "minibatch_size": 32,
        "update_batch_mode": "streaming",
        "checkpoint_dir": "checkpoints/riichi_ppo_v1_smoke",
    })
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    apply_cli_overrides(config, args)
    smoke_dir = Path(config["checkpoint_dir"])
    preexisting = smoke_dir.exists()
    try:
        run(config)
    finally:
        # 冒烟结束时必须删除其产生的日志与结果文件;只在目录由本次冒烟创建时
        # 清理,避免误删用户显式指定的既有目录。
        if not preexisting:
            cleanup_smoke_artifacts(smoke_dir)


if __name__ == "__main__":
    main()
