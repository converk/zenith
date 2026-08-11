# PPO RL 阶段优化调研包

创建日期：2026-08-07（调研完成：2026-08-08）

这个目录是围绕「SFT 之后如何继续优化 PPO」的深度调研包，包含提示词、证据与最终研究报告。

## 文件

- `DEEP_RESEARCH_PROMPT.md`：可直接作为提示词交给 Deep Research 的主文档。
- `REFERENCES.md`：本地证据汇总（SFT/PPO 指标、对战结果、reward 实现、数据清单）。
- `RIICHIENV_ML_GRP_REVIEW.md`：对 `smly/RiichiEnv/riichienv-ml` 中 GRP 奖励模型实现的调研与可行性判断。
- `PPO_RL_OPTIMIZATION_REPORT.md`：最终中文深度研究报告（失败原因诊断、GRP 重点评估、奖励设计、替代算法、自博弈与对手设计、5 个 next experiments）。
- `EXPERIMENT_PLAN.md`：可执行的实验计划（E0 基线 → E1 EV 门控 → E2 对手混合 → E3 GRP A/B → E4 密集奖励 → 可选 E5），含每个实验的改动点、判定标准与评测命令。
- `GOAL_PROMPT.md`：可直接复制给 Codex `/goal` 模式的提示词（推荐版 + 一句话版）。
- `outline.yaml` / `fields.yaml`：研究项与字段定义。
- `results/`：7 个结构化调研结果 JSON（均已通过字段覆盖校验）。

## 使用方式

1. 把 `DEEP_RESEARCH_PROMPT.md` 整体交给 Deep Research。
2. 如果工具支持上传文件，同时上传 `REFERENCES.md`、`RIICHIENV_ML_GRP_REVIEW.md`，以及提示词「参考文件」一节列出的 JSONL/JSON/YAML/Python 文件。
3. 要实际跑实验：阅读 `EXPERIMENT_PLAN.md`，然后把 `GOAL_PROMPT.md` 里的 `/goal` 提示词粘贴给 Codex（`/goal` 模式），中途看 `../ppo_rl_goal_run_20260808/PROGRESS.md`，结束后看该目录下的 `EXPERIMENT_RESULTS.md`。GPU 默认优先单卡 `CUDA_DEVICE=2`（物理 GPU3），必须双卡时才用 `CUDA_DEVICE=1,2`（`CUDA=1` 为补充卡）。
