# Codex /goal 模式提示词（可直接复制）

## 使用方法

1. 在仓库根目录 `/mnt/disk1/hubowen/zenith` 新开一个 Codex 会话（确保 `/goal` 可用；不可用先运行 `codex features enable goals`）。
2. 复制下方「推荐提示词」整段（以 `/goal` 开头），粘贴发送。
3. Codex 会先读计划与调研文件，然后按 E0→E1→E2→E3→E4 连续执行；期间用 `/goal` 查看状态，`/goal pause` / `/goal resume` / `/goal clear` 控制运行。
4. 中途看 `audit/reports/ppo_rl_goal_run_20260808/PROGRESS.md`；结束后看 `EXPERIMENT_RESULTS.md`。

> 提示：如果不想依赖文件引用，可以把 `EXPERIMENT_PLAN.md` 的正文直接拼在 `/goal` 之后作为完整目标文本。

---

## 推荐提示词（复制这一段）

```text
/goal 按 audit/reports/ppo_rl_optimization_20260807/EXPERIMENT_PLAN.md 执行 PPO 优化实验，直到达到可验证停止条件：对 v13 SFT（checkpoints/train_riichi_v13_sft/best_heuristic.pt）的 2v2 240 半庄胜率 >55% 且 paired bootstrap 95% CI 不跨 50%，且对固定启发式对手不再出现 u100 后一位率持续退化。

开始前先完整阅读：audit/reports/ppo_rl_optimization_20260807/DEEP_RESEARCH_PROMPT.md、PPO_RL_OPTIMIZATION_REPORT.md、REFERENCES.md、EXPERIMENT_PLAN.md 与 results/*.json。先复现 E0 基线，再按 E1（EV 门控/四席组内基线）→ E2（80/20 对手混合，可与 E1 并行）→ E3（先离线 GRP 验证，通过后再做 point-delta/GRP/0.5 混合三组 PPO A/B）→ E4（小权重密集效率奖励）的顺序执行。每次只改一个变量，每个实验独立 checkpoint 目录（checkpoints/train_riichi_ppo_goal_*）与独立 config overlay。

验证方式：每 15 update 跑现有固定启发式评测；每个关键 checkpoint（u50/u100/u200/最终，即实验训练 200 update）用 python -m riichi_ppo_v1.sft.head_to_head --model-a <ppo_ckpt> --model-b checkpoints/train_riichi_v13_sft/best_heuristic.pt --hanchans 240 --parallel-hanchans 24 跑 2v2 240 半庄，读取输出 JSON 的 model_a.team_win_rate 与 model_a.team_point_diff_paired_bootstrap_ci95；训练 reward 不作为成功依据，训练 reward 上升但 2v2 不升按 reward hacking 处理。

约束：不改模型 backbone；一律从 best_heuristic.pt 初始化；固定 seed=1；所有 Python/训练/评测/冒烟测试命令必须通过 conda run -n Mahjong-AI 执行（或显式激活 Mahjong-AI 环境），不允许使用其他 Python 环境。GPU 优先级：优先单卡 CUDA_DEVICE=2（物理 GPU3）、learner_gpus=1；只有必须双卡时才用 CUDA_DEVICE=1,2（物理 GPU1+GPU3）、learner_gpus=2（CUDA=1 是补充卡）。冒烟/性能测试用 target_kl=0.0、update_epochs=4、kyokus_per_worker=1、跑 3 次（第 1 次热身，报告后两次性能与耗时），长期训练 kyokus_per_worker=16；不删除或覆盖现有 checkpoints/datasets/评测结果；输出中文，结论区分实证与推测。

边界：只允许修改 riichi_ppo_v1/ 下的训练代码、新增 riichi_ppo_v1/configs/goal/ 配置与 checkpoints/train_riichi_ppo_goal_* 目录；运行日志与相关结果统一放 audit/reports/ppo_rl_goal_run_20260808/（PROGRESS.md 每次实验前后更新、EXPERIMENT_RESULTS.md 为最终结论、各实验评测 JSON/config diff 放对应子目录），模型 checkpoint 不放该目录。如需使用子 agent，必须按 EXPERIMENT_PLAN.md 的「子代理实验方法」执行（复制全部上下文启动、每个子 agent 只负责一个实验臂、同一单卡不同时跑两个训练实验、子 agent 产出须回写主线程并由主 agent 审校）。

迭代策略：每完成一个实验，先按 EXPERIMENT_PLAN.md 的判定标准给出结论（成功/部分成功/失败）并更新 PROGRESS.md，再决定进入下一个实验或做一次针对性调整；E3 必须在 E1/E2 之后。

受阻停止条件：如果 GPU/环境不可用、评测工具跑不通、GRP 数据无法读取、基线无法复现，或同一实验连续 3 次针对性调整无进展，停止并报告：已尝试路径、证据、阻塞原因、下一步需要的输入。
```

## 一句话版本（先建立目标，再逐步细化）

```text
/goal 按 audit/reports/ppo_rl_optimization_20260807/EXPERIMENT_PLAN.md 完成 PPO 优化实验，使对 v13 SFT 的 2v2 240 半庄胜率 >55% 且 95% CI 不跨 50%，否则在 E0–E4 完成后输出结论与阻塞分析。
```
