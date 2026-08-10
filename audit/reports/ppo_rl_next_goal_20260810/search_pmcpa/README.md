# 树搜索离线可行性验证（pMCPA 根搜索）

## 目标

验证用轻量根搜索增强 E3-b u200 是否显著优于纯贪心基线（2v2 240 半庄 A/B，
paired walls + 座位交换，判定为 paired bootstrap 95% CI 不跨 0）。

## 实现

- 代码：`riichi_ppo_v1/sft/search_head_to_head.py`（仅改 `riichi_ppo_v1/`）。
- 主循环与 `head_to_head` 一致：24 桌并行 `BatchedRiichiEnv`，每桌维护一个
  `RiichiEnv` 镜像用于克隆。
- 搜索范围：搜索队（E3-b u200）的立直/切牌决策窗口（`include_responses=false`）。
- 候选：policy logits 的 **top-3**（用户指示：SFT 初始化模型 top-3 已覆盖有利决策）。
- rollout：每个候选应用后走 3 个决策轮（`search_depth=3`），所有席位按 base policy
  温度 1.0 采样；每候选 2 次 rollout。
- 计分：rollout 内小局结束时用 GRP 终局奖励（与训练 reward 同公式）；否则用搜索席
  最近一次决策的 critic value，无则回退根值。
- 选择：argmax 平均 rollout 值，平局按 policy logits 先验。

## 运行记录

- 冒烟（8 半庄，width=4/depth=1/rollouts=1）：`smoke_ab_u200.json`。
- 试点 A（24 半庄，width=8/depth=3/rollouts=2，seat）：`pilot_ab_u200.json`。
- 试点 B（24 半庄，width=8/depth=3/rollouts=2，team）：`pilot_ab_team_u200.json`。
- 试点 C（24 半庄，width=3/depth=3/rollouts=2，seat）：`pilot_ab_w3_u200.json`。
- 正式 A/B（240 半庄，width=3/depth=3/rollouts=2，seat）：`ab_w3_vs_greedy_u200.json`。

## 判定标准

搜索增强 2v2 显著优于纯贪心基线（`model_a.team_point_diff_paired_bootstrap_ci95` 不跨 0）
→ 进入搜索蒸馏；否则记录结果并跳到信号组合实验（E5-a/E5-b）。
