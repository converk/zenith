# PPO 优化 Goal 运行目录（2026-08-08）

本目录存放本次 Codex `/goal` 运行的**运行日志与相关结果**（模型 checkpoint 仍在 `checkpoints/train_riichi_ppo_goal_*`）。

## 文件约定

- `PROGRESS.md`：每次实验前后更新的进度日志（checkpoint、改动、指标表、结论、阻塞）。
- `EXPERIMENT_RESULTS.md`：最终结论报告（逐实验：假设/改动/指标表/判定/结论，附最优 checkpoint 路径与 2v2 结果）。
- `e0_baseline/`、`e1_ev_gate/`、`e2_opponent_mix/`、`e3_grp/`、`e4_dense_reward/`、`e5_stretch/`：各实验的评测 JSON、config diff、指标汇总。

执行依据：`../ppo_rl_optimization_20260807/EXPERIMENT_PLAN.md` 与 `GOAL_PROMPT.md`。
