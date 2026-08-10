# 新 goal 进度日志

> 每个实验前后更新；以 2v2 240 半庄 + 启发式 96 半庄为唯一真相。
> 判定标准见 `GOAL_PROMPT.md` §1（合格线 A/B，放宽版）。

## 进度总览

| 日期 | 实验 | 状态 | 关键结论 | 下一步 |
|---|---|---|---|---|
| 2026-08-10 | 树搜索离线验证（pMCPA 根搜索 harness） | **验证通过** | 240 半庄 A/B（width=3/depth=3/rollouts=2，seat）：搜索队 **54.58%**（131/109），分差 +4407.5，**CI [1103.5, 8207.7] 不跨 0**，override 15.3%（10715 切牌/168 立直/40 杠）；71,579 次搜索 | **进入搜索蒸馏**（数据生成+训练） |
| 2026-08-10 | E5-a arm1：GRP+SFT-KL 0.05 | **训练+2v2 评测完成** | 200 update 完成；启发式 u30–u180 一位率 37.5%–44.8%、明显比 E3-b 稳；**2v2 u50=53.13%（CI 跨 0）、u100=51.88%（CI 跨 0）、u200=47.92%（CI 跨 0）**——KL 锚定稳启发式但损伤 u200，未达合格线 A/B（u200<50% 且 u150=37.5%<40%） | 记录结论，等待蒸馏臂 |
| 2026-08-10 | E5-a arm2：GRP+SFT-KL 0.1 | **训练完成** | 200 update 完成（20:40）；启发式 u30–u180 一位率 35.4%–52.1%（u90 峰值 52.1%），u150 低点 35.4% 后 u180 回升 42.7%，整体优于 E3-b | u50/u100/u200 2v2 评测 |
| 2026-08-10 | 搜索蒸馏实现+数据生成 | **完成** | 单元测试 4/4；**run1 29,408 条 + run2 29,119 条 = 合并 58,527 条**（`search_distill_data_combined/`）；蒸馏 KL hook 冒烟通过：coef=0.5、`search_distill_kl≈0.54-0.56` 稳定进入 loss | 接 GPU 空闲启动两个蒸馏变体 |
| 2026-08-10 | 蒸馏变体设计 | 确定 | **变体 A（E3-b 基座）**：init=E3-b u200 + GRP + 搜索 KL；**变体 B（SFT 基座）**：init=best_heuristic + GRP + 搜索 KL（用户确认两种都跑）。两者唯一差异为 init_model | 接 GPU 空闲依次启动 |
| 2026-08-10 | E5-a arm2 / E5-b 配置 | 配置就绪 | 见 `configs/next_goal/`（kl010、GRP→dense、dense→GRP） | 等搜索 A/B 结束后启动 |
| 2026-08-10 | 蒸馏 KL 实现 bug 修复 | **已修复并重启** | 首版用 reverse-KL 代理（稀疏目标下可为负，-0.31），导致 u60 启发式崩溃（一位率 1%）；改为 **forward-KL（交叉熵）**并修复 `0×-inf=NaN`；单元测试 7/7、冒烟 CE=3.05→2.87 正常。**两个蒸馏训练于 21:32 重启（v2 checkpoint 目录）** | 训练 200 update → u50/u100/u200 评测 |
| 2026-08-10 | 全模型 300 半庄 2v2 评测（用户新增要求） | 编排就绪 | 评测脚本 `riichi_ppo_v1/tools/final_2v2_suite.py`（双卡并行、可续跑）：7 个候选 vs SFT@300 + 蒸馏 vs E3-b@300 ×2 + u50/u100@240 ×6 = 15 场，输出 `final_2v2_300/` | 蒸馏训练完成后启动 |
| 2026-08-11 | 蒸馏变体 A/B 训练 | **完成（200/200）** | A（E3-b 基座）03:44 完成；B（SFT 基座）03:42 完成；CE 降至 ~1.0 附近。**启发式一位率 A：u30–u180=37.5/31.2/29.2/29.2/32.3/21.9%；B：42.7/32.3/28.1/27.1/31.2/16.7%**——CE 蒸馏后固定启发式表现明显下滑（u100–u200 多次 <40%），2v2 待 300 半庄套件判定 | **15 场 2v2 套件运行中（03:50 启动，双卡并行）** |

> 训练进度快照：E5-a arm1 完成（2v2 见上表）；E5-a arm2 完成（启发式
> 42.7/44.8/52.1/40.6/35.4/42.7%，2v2 评测待跑）。
> **蒸馏 KL 修复说明**：首版实现把 `categorical_kl_values(policy, log(target))` 当蒸馏
> 目标，稀疏 top-3 目标下该代理可算成负值（实测 -0.31），梯度方向错误导致策略崩溃。
> 修复为 forward-KL/交叉熵 `-Σ target·log policy`（`search_distill_cross_entropy`），
> 并对非法动作 -inf 乘 0 做掩码。旧 checkpoint（`*_search_distill/`、`*_search_distill_sft/`）
> 已作废，正式跑使用 `*_v2/`。
> 蒸馏变体 A/B 已于 21:32 在 CUDA=2/CUDA=0 重启（各 200 update，评测间隔 30）。
> 进度快照（23:15）：A/B 均约 update 73–74/200，~130s/update（含 CE 辅助 forward），
> CE 从 ~2.8 降至 ~1.4；u60 启发式一位率 A=31.2%/B=32.3%（修复后未再崩溃，
> 属 KL 重塑期波动）。预计 03:45–04:10 完成训练；随后 u50/u100/u200 2v2 评测
> （两个蒸馏臂 + E5-a arm2）+ 最终报告，总完成预计 06:00–07:00。
> 进度快照（00:13）：A/B 均 update 100/200；u90 启发式一位率 A=29.2%/B=28.1%
> （CE 蒸馏重塑期低位，待 u120/u150 观察）。新增用户要求：训练全部完成后做
> **300 半庄 2v2 全模型评测**（≥6 模型，双卡并行）——选定 7 个候选：
> 本 goal 4 个（E5-a kl005/kl010、蒸馏 A/B u200）+ 上一 goal 3 个
> （E3-b u200、E2 u200、E4 u200），全部 vs SFT；另加蒸馏 A/B vs E3-b 的方法归因对照。
> 进度快照（03:50）：两个蒸馏训练完成（200/200）；u180 一位率 A=21.9%、B=16.7%
> （蒸馏策略在固定启发式上明显不稳）。**15 场 2v2 评测套件已启动**
> （`final_2v2_300/`，CUDA=0/2 并行，预计 ~2h）。

## 实验明细

### 2026-08-10：树搜索离线验证 harness（seat 模式）

- 代码：`riichi_ppo_v1/sft/search_head_to_head.py`（仅修改 `riichi_ppo_v1/`）。
- 设计：2v2 主循环与 `head_to_head` 一致（paired walls、座位交换），每桌维护 `RiichiEnv`
  镜像；搜索队（E3-b u200）在立直/切牌窗口做根搜索：top-W 候选 × R 个 rollout，
  每个 rollout 应用候选动作后走 D 个决策轮，全部席位按 base policy 采样；小局结束时用
  GRP 终局奖励计分，否则用搜索席 critic value 计分。
- 冒烟（8 半庄，depth=1/width=4/rollouts=1）：完成，66s，2308 次搜索，override 5.8%。
- 试点 A（24 半庄，depth=3/width=8/rollouts=2，seat 模式）：完成，15.2min；
  搜索队 10/24（41.7%），分差 -6875，CI [-18958, 6010] 跨 0；override 23.5%。
- 试点 B（24 半庄，width=8，team 模式）：完成，18.6min；搜索队 1/24（4.2%），
  分差 -56500，CI [-74935, -42808] 显著为负；override 77.7%——team 值近似（critic 求和）
  导致过度偏离基线，弃用该模式。
- 试点 C（24 半庄，depth=3/width=3/rollouts=2，seat 模式，按用户指示 top-3）：完成，6.9min；
  搜索队 13/24（54.2%），分差 -792，CI [-16117, 13958] 跨 0；override 15.7%，
  mean chosen−greedy=+0.156（搜索选择在 rollout 值上更优的候选）。样本仍不足以判定。
- 2v2 JSON：`search_pmcpa/pilot_ab_u200.json`（width=8 seat）、`pilot_ab_team_u200.json`（team）、
  `pilot_ab_w3_u200.json`（width=3 seat）。
- 正式 A/B（240 半庄，width=3/depth=3/rollouts=2，seat）：**完成**，
  `search_pmcpa/ab_w3_vs_greedy_u200.json`。搜索队 131/109（54.58%），
  team_point_diff_mean=+4407.5，paired bootstrap 95% CI **[1103.5, 8207.7]**（不跨 0）。
  search_stats：searched=71,579，override=10,924（15.26%），search_wall_s=3872，
  mean_chosen−greedy=+0.163（rollout 值更优）。
- **判定：验证通过**（显著优于基线）→ 进入搜索蒸馏（E5-search-distill）。

### 2026-08-10：E5-a arm1（GRP + SFT-KL 0.05）

- config：`riichi_ppo_v1/configs/next_goal/e5a_grp_sftkl_005.yaml`；checkpoint：
  `checkpoints/train_riichi_ppo_next_e5a_kl005/`。
- 设计：相对 E3-b 只改 `sft_kl_coef_start=0.0`、`sft_kl_coef_end=0.05`、
  `sft_kl_anneal_updates=200`（KL 在 200 update 内升到 0.05），GRP λ=1.0、从头初始化
  `best_heuristic.pt`。
- 冒烟（3 轮，单卡，target_kl=0.0/epochs=4/kyokus_per_worker=1）：第 1 轮热身 34.4s
  （sps 740.7）；第 2/3 轮 27.96/27.52s，sps 899.4/898.4。
- 状态：200 update 训练运行中（CUDA=2，物理 GPU3）；评测间隔 30 update。

### 2026-08-10：E5-a arm2（GRP + SFT-KL 0.1）

- config：`riichi_ppo_v1/configs/next_goal/e5a_grp_sftkl_010.yaml`；checkpoint：
  `checkpoints/train_riichi_ppo_next_e5a_kl010/`。与 arm1 唯一差异为 `sft_kl_coef_end=0.1`。
- 冒烟（3 轮，单卡）：第 1 轮热身 33.6s（sps 741）；第 2/3 轮 28.69/28.14s，sps 862.9/899.6。
- 状态：200 update 训练运行中（CUDA=0，物理 GPU0）；评测间隔 30 update。

### 2026-08-10：搜索蒸馏（E5-search-distill）实现

- 代码（均在 `riichi_ppo_v1/`）：
  - `sft/search_head_to_head.py`：新增 `--record-distill-dir`（保存每条搜索决策的
    policy 输入张量 + top-3 目标分布，softmax(rollout 均值/tau)）与 `--distill-tau`。
  - `training/search_distill.py`：`SearchDistillDataset` 加载 `part_*.npz` 并采样批次。
  - `training/learner.py`：每个 PPO minibatch 额外采样 `search_distill_batch_size` 条
    搜索状态，加入 `λ·KL(policy ‖ search_target)`（与 sft_kl 同方向）；critic bootstrap
    期关闭；配置默认关闭，不影响既有训练。
  - 单元测试：`tests/unit/test_search_distill.py`（3 通过）。
- config：`riichi_ppo_v1/configs/next_goal/e5_search_distill.yaml`（init=E3-b u200，
  GRP λ=1.0，sft_kl=0，search_distill_coef=0.5，200 update，评测间隔 30）。
- 数据：`audit/reports/ppo_rl_next_goal_20260810/search_distill_data/`（96 半庄生成中）。
- 数据 run1 完成：`search_distill_data/` 共 **29,408 条**（8 个 part）。
- 数据 run2（新 seed，补充覆盖度）：`search_distill_data_run2/` 生成中。
- 合并数据集：`search_distill_data_combined/`（run1 + run2 全部 part）。
- 变体 A config：`riichi_ppo_v1/configs/next_goal/e5_search_distill.yaml`（E3-b 基座）。
- 变体 B config：`riichi_ppo_v1/configs/next_goal/e5_search_distill_sft.yaml`（SFT 基座）。
