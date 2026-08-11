# PIMC 消融实验计划书（给新会话的任务提示词）

> 本文件即新会话的输入提示词。新会话请完整阅读本文件并按其执行；
> 实验过程与结果写入 `audit/reports/ppo_rl_next_goal_20260810/pimc_ablation/`，
> 并同步更新 `audit/reports/ppo_rl_next_goal_20260810/PROGRESS.md`。

## 0. 背景（前序结论）

现有树搜索（`riichi_ppo_v1/sft/search_head_to_head.py`）是**完全信息搜索**：
在 2v2 评测主循环里维护每桌 `RiichiEnv` 镜像，搜索时 `mirror.clone()` 直接拿到
含三家手牌/牌山的真实隐藏状态，再做 top-3 根搜索（depth=3、rollouts=2，
GRP 终局奖励 + critic 计分）。240 半庄 A/B 已验证该搜索显著优于贪心基线
（CI [1103.5, 8207.7] 不跨 0）。

真实对局中拿不到对手手牌与牌山，需要 **PIMC（Perfect-Information Monte Carlo）**：
按公开信息采样 N 个“可能世界”（determinization），每个世界做完全信息搜索，
按世界平均候选价值后选动作。已实测单桌单决策完全信息搜索时延约 73.5ms
（`search_pmcpa/single_table_latency_bench.json`），N 世界外推约为 N/2 倍。

## 1. 本任务目标

实现 PIMC 模式并做 **N 消融**：

- N ∈ {16, 24, 32, 40}；
- 每配置只跑 **20 半庄**（控制总时长）；
- 核心对比：**PIMC-N vs 完全信息搜索**（同一批牌墙/种子，paired 2v2）；
- 附带对比：PIMC-N vs 贪心基线（可选，若时间允许）；
- 输出：每个 N 的 2v2 胜率/分差/CI、**与完全信息搜索的动作一致率**、
  每决策时延、以及 PIMC 相对完全信息的性能损失结论。

## 2. 硬约束

- 所有 Python/评测必须 `conda run -n Mahjong-AI`（或显式激活该环境）执行；
- 启动评测前先 `nvidia-smi` 确认空闲 GPU；优先 CUDA_DEVICE=0 / CUDA_DEVICE=2
  （物理 GPU0 / GPU3）；同一张卡不要同时跑两个训练，评测进程可多路并行但注意 CPU 负载；
- 代码只允许修改/新增 `riichi_ppo_v1/`；新配置与报告放
  `audit/reports/ppo_rl_next_goal_20260810/pimc_ablation/`；
- 不删除、不覆盖已有 checkpoints/评测结果；复用
  `checkpoints/train_riichi_ppo_goal_e3_grp_reward/checkpoint_00200.pt`（被搜索策略）与
  `checkpoints/train_riichi_v13_sft/best_heuristic.pt`（SFT）；
- 输出与结论用中文；区分「实证支持」与「推测」。

## 3. 实现规格（PIMC 世界采样）

在 `riichi_ppo_v1/sft/search_head_to_head.py` 增加 PIMC 模式（不要破坏现有
完全信息模式；建议通过 `--pimc-worlds N`（N>0 时启用）切换）：

1. **世界构造（关键）**：对搜索决策，不再直接用 `mirror.clone()`，而是：
   - `world = mirror.clone()`（克隆拿到真实隐藏状态）；
   - 收集“未知牌集合” = 三家对手手牌 + 牌山（克隆环境里可读）——
     用 numpy `Generator`（以确定性种子派生）随机打乱后，按原手牌张数
     重新分配给三家，剩余作为牌山；**自己的手牌、所有公开牌（河、副露、
     宝牌、立直、分数）保持不变**。
   - 这样每个 `world` 都与公开信息一致，且严格满足每种牌 ≤4。
2. **搜索流程**：每个候选动作在每个世界上跑 1 次 rollout（depth=3），
   价值取“候选 → 世界均值”；最后选均值最大的候选。rollout 内部仍用
   base policy 采样、GRP 终局奖励/critic 计分（与现有逻辑一致）。
3. **动作一致率**：对每个搜索决策记录「PIMC-N 选择」与「完全信息搜索选择」，
   统计一致比例（只在双方都做搜索的决策上统计）。
4. **时延**：沿用 `search_stats.search_wall_s / searched_decisions` 输出
   每决策毫秒数，并记录每配置总耗时。
5. 单元测试：世界采样需验证——自己手牌不变、公开牌不变、每种牌总数 ≤4、
   三家手牌张数正确、未知牌总数守恒。测试放 `riichi_ppo_v1/tests/unit/`。

## 4. 实验协议

被搜索策略：E3-b u200；对手：同一策略贪心（A/B 用现有 2v2 paired 语义）。
每配置（N=16/24/32/40）跑 20 半庄（`--hanchans 20 --parallel-hanchans 20`，
同一种子基，例如 20270000），三个对比：

| 对比 | model_a | model_b | 半庄 |
|---|---|---|---|
| PIMC-N vs 完全信息 | PIMC-N 搜索（E3-b u200） | 完全信息搜索（E3-b u200） | 20 |
| PIMC-N vs 贪心基线 | PIMC-N 搜索 | 贪心 E3-b u200 | 20 |
| （可选）完全信息 vs 贪心 | 完全信息搜索 | 贪心 E3-b u200 | 20 |

输出 JSON 放 `pimc_ablation/`：`pimc_N16_vs_full.json` 等；JSON 需包含
`model_a/model_b` 的 `team_win_rate`、`team_point_diff_mean`、
`team_point_diff_paired_bootstrap_ci95`、`search_stats`（含 action 一致率）。

## 5. 验收/停止条件

1. PIMC 模式实现完成、单测通过、20 半庄全配置跑完；
2. 给出四张 N 的对比表：胜率/分差/CI、动作一致率、每决策时延；
3. 明确结论：N 取多少时 PIMC 与完全信息搜索差距可接受（例如 2v2 胜率不低于
   完全信息若干个百分点、一致率趋势），以及 N=40 的时延是否仍实时可用；
4. 更新 `PROGRESS.md`，写小结到 `pimc_ablation/RESULTS.md`。

## 6. 建议执行步骤

1. 读 `riichi_ppo_v1/sft/search_head_to_head.py`、`training/rewards/decision.py`
   相关部分，确认世界构造点与 rollout 计分路径；
2. 实现 `--pimc-worlds` + 世界采样 + 动作一致率统计 + 单测；
3. `nvidia-smi` 后先跑 2–4 半庄冒烟（N=16），确认能出结果、无崩溃；
4. 4 个 N 配置可并行（每卡 2 个进程）跑 20 半庄；
5. 汇总并写结论。

## 7. 参考数据

- 完全信息搜索 A/B：`search_pmcpa/ab_w3_vs_greedy_u200.json`
  （54.58%，CI [1103.5, 8207.7]）；
- 单桌时延：`search_pmcpa/single_table_latency_bench.json`（≈73.5ms/决策）；
- 主报告：`EXPERIMENT_RESULTS.md`、`PROGRESS.md`。
