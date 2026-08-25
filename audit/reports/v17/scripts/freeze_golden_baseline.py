#!/usr/bin/env python
"""冻结 V16 PPO 编码 golden baseline(纯 CPU, 无 GPU/无 Ray)。

目的:确定性重放当前 Rust 单路径的 V16 编码输出(history / snapshot / query /
critic / legal_mask)与状态机事件、transition 字段,并与已经冻结的优化前 oracle
NPZ 做逐元素正确性回归。

用法(Mahjong-AI 环境):
  python audit/reports/v17/scripts/freeze_golden_baseline.py \
      --config riichi_ppo_v1/configs/v17_ppo_perf_512g4e.yaml \
      --num-envs 4 --max-steps 40 --seed 20260819 \
      --out audit/reports/v17/eval/golden_baseline_v17.npz

脚本只走 CPU:创建 BatchedRiichiEnv + MjaiKyokuStateMachineManager +
BatchedStateBridge,用「每个合法动作取第一个」做确定性 self-play,在若干决策点
调用 bridge.prepare_v16 并把编码快照与decode结果保存。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

# 保持项目自定义 CUDA_DEVICE 约定,但本脚本不需要 GPU。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def get_observation(env_index: int, seat: int, observations: list[dict[int, object]]) -> object:
    return observations[env_index][seat]


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--out", default="audit/reports/v17/eval/golden_baseline_v17.npz")
    args = parser.parse_args()

    config = _load_config(args.config)
    config.setdefault("seed", args.seed)

    import riichi
    from riichienv import BatchedRiichiEnv

    from riichi_ppo_v1.model.bridge import BatchedStateBridge
    from riichi_ppo_v1.training.profiling import StageProfiler

    profiler = StageProfiler(enabled=True)
    num_envs = int(args.num_envs)
    envs = BatchedRiichiEnv(
        num_envs,
        seed=int(config.get("seed", args.seed)) % 1_000_000,
        step_threads=1,
        game_mode=config["game_mode"],
    )
    state_machine = riichi.MjaiKyokuStateMachineManager(num_envs)
    bridge = BatchedStateBridge(state_machine, num_envs, profiler)

    observations = list(envs.reset())
    walls = list(envs.walls())
    bridge.sync(observations)

    snapshots: dict[str, list[np.ndarray]] = {key: [] for key in (
        "history_factors", "history_numeric", "history_lengths",
        "snapshot_kinds", "snapshot_cat", "snapshot_num", "snapshot_lengths",
        "query_rows", "query_action_ids", "query_pair_counts", "legal_mask",
        "critic_factors", "critic_lengths", "batch_indices",
    )}
    decode_log: list[dict] = []
    transition_log: list[dict] = []

    def snapshot_if_decisions(step_index: int) -> None:
        from riichi_ppo_v1.model.bridge import Decision

        decisions = []
        for env_index, obs_by_env in enumerate(observations):
            for seat, observation in obs_by_env.items():
                if observation.legal_actions():
                    decisions.append(Decision(env_index, seat, observation))
        if not decisions:
            return
        prepared = bridge.prepare_v16(decisions, walls=walls)
        for key in (
            "history_factors", "history_numeric", "history_lengths",
            "snapshot_kinds", "snapshot_cat", "snapshot_num", "snapshot_lengths",
            "query_rows", "query_action_ids", "query_pair_counts", "legal_mask",
            "critic_factors", "critic_lengths",
        ):
            snapshots[key].append(np.asarray(getattr(prepared, key)))
        snapshots["batch_indices"].append(np.asarray([d.batch_index for d in decisions]))
        # decode 全部合法 action id, 冻结 id->MJAI 映射。
        ids = []
        for row, decision in enumerate(decisions):
            for action_id in np.flatnonzero(np.asarray(prepared.legal_mask)[row]):
                ids.append((int(decision.batch_index), int(action_id)))
        if ids:
            mjai = state_machine.decode_actions(
                [pair[0] for pair in ids], [pair[1] for pair in ids]
            )
            for (batch_index, action_id), raw in zip(ids, mjai, strict=True):
                decode_log.append({
                    "step": step_index, "batch_index": int(batch_index),
                    "action_id": int(action_id), "mjai": raw,
                })

    def choose_actions(observations: list[dict[int, object]]) -> list[dict[int, object]]:
        actions_by_env = [{} for _ in range(num_envs)]
        for env_index, obs_by_env in enumerate(observations):
            for seat, observation in obs_by_env.items():
                legal = list(observation.legal_actions())
                if legal:
                    actions_by_env[env_index][seat] = legal[0]  # 确定性: 第一个合法动作
        return actions_by_env

    for step_index in range(args.max_steps):
        snapshot_if_decisions(step_index)
        actions = choose_actions(observations)
        observations = list(envs.step_batch(actions))
        walls = list(envs.walls())
        end_kyoku, end_game = bridge.sync(observations)
        for env_index, kyoku in enumerate(end_kyoku):
            if kyoku:
                transition_log.append({
                    "step": step_index, "env": env_index,
                    "end_kyoku": bool(kyoku), "end_game": bool(end_game[env_index]),
                    "scores": [int(value) for value in envs.scores()[env_index]],
                })
        if all(end_game):
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 每个决策 batch 的长度(history/snapshot/query)不同,不能简单拼接;
    # 用 object 数组保留「batch -> array」的 ragged 结构,供逐元素对比。
    arrays: dict[str, np.ndarray] = {}
    for key, values in snapshots.items():
        if values:
            holder = np.empty(len(values), dtype=object)
            for index, value in enumerate(values):
                holder[index] = np.asarray(value)
            arrays[key] = holder
    np.savez_compressed(out, allow_pickle=True, **arrays)
    # 保存解码与边界日志(便于人工审查)。
    import json
    with out.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump({"decode": decode_log, "transitions": transition_log},
                  file, ensure_ascii=False, indent=2)

    print(f"golden baseline saved to {out}")
    if arrays:
        print(f"  decision batches: {len(arrays.get('history_factors', []))}")
        print(f"  first batch shapes: "
              f"history_factors {arrays['history_factors'][0].shape} "
              f"snapshot_kinds {arrays['snapshot_kinds'][0].shape} "
              f"query_rows {arrays['query_rows'][0].shape} "
              f"legal_mask {arrays['legal_mask'][0].shape}")
    print(f"  decode rows {len(decode_log)}; kyoku boundaries {len(transition_log)}")


if __name__ == "__main__":
    main()
