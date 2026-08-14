# V14 PPO（2026-08-12）

本目录存放 V14 PPO 的 goal 提示词、进度日志与评测结果。

- `GOAL_PROMPT.md`：可直接复制给 `/goal` 的完整提示词。
- `PROGRESS.md`：训练/实现进度日志（每 60 update 更新）。
- `EXPERIMENT_RESULTS.md`：最终结论（goal 结束后生成）。
- `eval/`：每次 30 update 的 1v3 1600 半庄评测汇总与 shard 原文。

模型 checkpoint 放 `checkpoints/train_riichi_ppo_v14/`，不入本目录。
