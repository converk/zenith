#!/usr/bin/env python
"""CPU 对照 V16 Python batch oracle 与 Rust 融合 query 编码阶段。"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--game-mode", default="4p-red-half")
    args = parser.parse_args()

    import riichi
    from riichienv import BatchedRiichiEnv
    from riichi_ppo_v1.model.bridge import BatchedStateBridge, Decision
    from riichi_ppo_v1.training.profiling import StageProfiler

    envs = BatchedRiichiEnv(
        args.num_envs,
        seed=args.seed,
        step_threads=1,
        game_mode=args.game_mode,
    )
    python_profiler = StageProfiler(enabled=True)
    rust_profiler = StageProfiler(enabled=True)
    python_bridge = BatchedStateBridge(
        riichi.MjaiKyokuStateMachineManager(args.num_envs),
        args.num_envs,
        python_profiler,
        batch_query=True,
    )
    rust_bridge = BatchedStateBridge(
        riichi.MjaiKyokuStateMachineManager(args.num_envs),
        args.num_envs,
        rust_profiler,
        batch_query=True,
        rust_encoding=True,
    )
    observations = list(envs.reset())
    python_bridge.sync(observations)
    rust_bridge.sync(observations)
    action_rows = 0
    unique_offense_rows = 0
    unique_shanten_rows = 0
    decision_batches = 0

    for step in range(args.steps):
        decisions = [
            Decision(env_index, seat, observation)
            for env_index, table in enumerate(observations)
            for seat, observation in table.items()
            if observation.legal_actions()
        ]
        if decisions:
            walls = list(envs.walls())
            oracle = python_bridge.prepare_v16(decisions, walls=walls)
            fused = rust_bridge.prepare_v16(decisions, walls=walls)
            np.testing.assert_array_equal(fused.query_rows, oracle.query_rows)
            stats = rust_bridge.last_v16_rust_stats
            action_rows += stats["actions"]
            unique_offense_rows += stats["unique_offense_rows"]
            unique_shanten_rows += stats["unique_shanten_rows"]
            decision_batches += 1

        actions = [{} for _ in range(args.num_envs)]
        for env_index, table in enumerate(observations):
            for seat, observation in table.items():
                legal = list(observation.legal_actions())
                if legal:
                    actions[env_index][seat] = legal[(step + env_index + seat) % len(legal)]
        observations = list(envs.step_batch(actions))
        python_bridge.sync(observations)
        rust_bridge.sync(observations)

    python_timing = python_profiler.checkpoint()["state/v16_query_assembly"]
    rust_timing = rust_profiler.checkpoint()["state/v16_query_assembly"]
    print(f"decision_batches={decision_batches} action_rows={action_rows}")
    print(
        "python_query_assembly_s="
        f"{python_timing.total_s:.6f} mean_ms={python_timing.total_s * 1000 / python_timing.count:.3f}"
    )
    print(
        "rust_query_assembly_s="
        f"{rust_timing.total_s:.6f} mean_ms={rust_timing.total_s * 1000 / rust_timing.count:.3f}"
    )
    print(f"speedup={python_timing.total_s / rust_timing.total_s:.3f}x")
    print(
        f"unique_offense_rows={unique_offense_rows} "
        f"unique_shanten_rows={unique_shanten_rows}"
    )


if __name__ == "__main__":
    main()
