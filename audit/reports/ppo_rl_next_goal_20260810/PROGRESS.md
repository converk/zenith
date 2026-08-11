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
| 2026-08-11 | 300 半庄 2v2 全模型套件 | **完成（15 场）** | 见 `final_2v2_300/summary.json`：**E5-a arm2 最优（u200@300=54.0%、u100@240=61.7% CI 不跨 0）但未达合格线；蒸馏 A/B 显著失败（vs SFT 28.7%/26.3%，CI 显著为负）；上一 goal E3-b/E2=54.7%（CI 跨 0）** | **写最终报告（已完成）** |
| 2026-08-11 | 最终报告与提交 | **完成** | `EXPERIMENT_RESULTS.md` 最终结论：**未达合格线 A/B**，默认收尾条款满足（树搜索验证 + 至多 2–3 个训练臂）；git 提交 `3c2b8d6`（中文消息） | goal 完成 |

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

## 最终收尾（2026-08-11 05:30）

> 追加评测（08-11）：E5-a arm2 u100 vs SFT **1000 半庄**（双卡并行 2×500，
> seed 20260730/20260780 互补）：段A 55.2%（+3341，CI 不跨 0）、段B 49.5%（+1406，
> CI 跨 0）；**合并 52.35%（523/476/1），分差 +2374，近似 95% CI [-483, 5231] 跨 0**。
> 文件：`e5a_grp_sftkl_010/vs_sft_u100_1000.json`（含两段明细）。
>
> 追加评测（08-11）：E5-a arm2 u100 vs **E3-b u200** 1000 半庄（2×500，同 seed 段）：
> 段A 46.8%（-1221）、段B 52.0%（+667）；**合并 49.4%（494/506），分差 -277，
> 近似 95% CI [-3162, 2608] 跨 0**——两者基本打平。文件 `vs_e3b_u200_1000.json`。
> 随后 E3-b u200 vs SFT 1000 半庄运行中（seed 20260830/20260880）。
>
> 追加评测（08-11，修正版，seed 差 +1000 完全独立）：三个对比各 1000 半庄合并结果
> （文件 `*_1000_v2.json`）：
> - E5-a arm2 u100 vs SFT：**53.8%**（536/460/4），分差 **+2817**，近似 CI [79, 5555]；
> - E5-a arm2 u100 vs E3-b u200：**49.2%**（489/506/5），分差 **-549**，CI [-3333, 2235]；
> - E3-b u200 vs SFT：**50.05%**（500/499/1），分差 **+62**，CI [-2875, 2999]。
> 另按用户要求，使用全新种子（20270000/20271000，此前从未使用）对三个对比各再跑
> 1000 半庄（6×500，双卡并行，~1h）作为独立复现。
>
> 全新种子复现（文件 `*_1000_fresh.json`，6×500 全并行完成）：
> - E5-a arm2 u100 vs SFT：**50.1%**（499/497/4），分差 **+702**，CI [-2251, 3655]；
> - E5-a arm2 u100 vs E3-b u200：**51.5%**（513/484/3），分差 **+1253**，CI [-1854, 4360]；
> - E3-b u200 vs SFT：**49.1%**（490/509/1），分差 **+781**，CI [-2154, 3716]。
> 结论：全新种子上三个对比全部 CI 跨 0，无显著差异；arm2-u100 相对 SFT 的
> 早前 1000 半庄（53.8%，CI 下界 79）未在独立复现中保持。
>
> 追加（08-11，1500 个全新独立半庄，seed 20280000/20281000/20282000，9×500 全并行）：
> - E5-a arm2 u100 vs SFT：**53.17%**（796/701/3），分差 **+2444**，近似 CI [783, 4106]
>   （**不跨 0**）；
> - E5-a arm2 u100 vs E3-b u200：**53.23%**（796/699/5），分差 **+2426**，CI [762, 4090]
>   （**不跨 0**）；
> - E3-b u200 vs SFT：**48.57%**（728/771/1），分差 **-843**，CI [-2566, 880]（跨 0）。
> 结论：在 1500 个全新独立半庄上，arm2-u100 显著优于 SFT 与 E3-b u200；E3-b 相对
> SFT 无显著差异。文件：`*_1500_ind.json`（含三段时间明细）。
>
> 追加（08-11，第二批 1500 个全新独立半庄，seed 20290000/20291000/20292000，
> 9×500 全并行；文件 `*_1500_more.json`）：
> - E5-a arm2 u100 vs SFT：**50.57%**（756/739/5），分差 **+1594**，CI [-118, 3305]（跨 0）；
> - E5-a arm2 u100 vs E3-b u200：**48.67%**（727/767/6），分差 **+557**，CI [-1280, 2395]（跨 0）；
> - E3-b u200 vs SFT：**51.97%**（778/719/3），分差 **+1438**，CI [-267, 3142]（跨 0）。
> 结论：第二批 1500 上三个对比均不显著；与第一批 1500（arm2 显著优于 SFT/E3-b）
> 方向大体一致但幅度收窄，累计 3000 个全新半庄的证据支持 arm2-u100 略优于
> SFT/E3-b，但效应量不稳定。
>
> **全量聚合（不包含 seed 差 50 的重叠段，每对比 5000 半庄 = 10×500 独立段；
> 文件 `*_ALL.json`）**：
> - E5-a arm2 u100 vs SFT：**51.90%**（2587/2397/16），分差 **+1915**，
>   近似 CI [1001, 2830]（**不跨 0**）；
> - E5-a arm2 u100 vs E3-b u200：**50.69%**（2525/2456/19），分差 **+1036**，
>   CI [87, 1985]（**不跨 0**，下界接近 0）；
> - E3-b u200 vs SFT：**49.98%**（2496/2498/6），分差 **+347**，CI [-587, 1282]（跨 0）。
> 最终结论：5000 半庄口径下 arm2-u100 对 SFT 与 E3-b u200 均显著为正（效应较小且
> 逐段波动），E3-b 与 SFT 无显著差异。
>
> 追加（08-11，单桌时延基准）：`search_pmcpa/single_table_latency_bench.json`——
> `--parallel-hanchans 1`、4 半庄、width=3/depth=3/rollouts=2：
> **单桌单决策搜索时延 ≈73.5ms**（74.921s / 1019 次搜索）。确定性采样 N 世界外推：
> N=8 ≈0.29s、N=16 ≈0.59s、N=32 ≈1.18s（按克隆数 3N/6 线性外推，不含采样开销）。
>
> 追加（08-11，1v3 评测：候选 1 席 vs 对手 3 席，每对比 1500 半庄 = 3×500 独立段，
> seed 20290000/20291000/20292000，9 进程并行；文件 `head_to_head_1v3/*_1v3_ALL.json`）：
> - E5-a arm2 u100 vs SFT×3：一位率 **28.2%**、top2 **53.6%**、四位 **24.1%**、
>   均顺位 **2.42**、分差 **+1732**（近似 CI [692, 2772]，**不跨 0**）；
> - E5-a arm2 u100 vs E3-b×3：一位 **25.3%**、top2 **50.7%**、四位 **23.5%**、
>   均顺位 **2.48**、分差 **+565**（CI [-468, 1597]，跨 0）；
> - E3-b u200 vs SFT×3：一位 **26.1%**、top2 **50.7%**、四位 **24.8%**、
>   均顺位 **2.48**、分差 **+803**（CI [-219, 1825]，跨 0）。
> 结论（1v3 口径，随机基线一位 25%/均顺位 2.5）：arm2 u100 显著优于 SFT；
> E3-b 相对 SFT 略正但不显著；arm2 相对 E3-b 不显著（E3-b 在 1v3 下比 2v2 更抗打）。
>
> 追加（08-11）：搜索深度语义修正——新增 `--depth-mode round|own`（默认 round 不变）：
> `own` 模式按「搜索玩家自己的决策次数」计深（根候选动作算第 1 次），
> 修正原 round 模式下 depth=3 只含全桌 3 轮、搜索玩家平均仅轮到 0–1 次的问题。
> 冒烟（2 半庄，own-depth=2）：每决策 ~74ms，rollout 平均 4.8 步，小局内结束占比
> 19.3%，与贪心不一致率 55%（round-depth3 为 16%）——更长自身前瞻显著改变决策。

**未达合格线 A/B，按「默认收尾」条款完成本 goal。**

- 树搜索离线验证：**通过**（240 半庄，搜索增强 E3-b 54.58%，CI [1103.5, 8207.7] 不跨 0）。
- 搜索蒸馏（A：E3-b 基座；B：SFT 基座）：**显著失败**——vs SFT 300 半庄
  28.67%/26.33%（CI 均显著为负），vs E3-b 33.67%/28.00%（显著为负）；
  启发式一位率 u100–u200 多次 <40%。推测：top-3 稀疏目标 + 0.5 系数导致熵坍缩。
- E5-a arm1（KL 0.05）：u200@300=50.33%（CI 跨 0），启发式最稳但未达标。
- E5-a arm2（KL 0.1）：**本 goal 最优**——u100@240=61.67%（CI [3456,11960] 不跨 0），
  u200@300=54.00%（CI [-1694,6135] 跨 0）；u150=35.4% 单点 <40%，未达 A/B。
- 上一 goal 对照（300 半庄复测）：E3-b=54.67%、E2=54.67%（均 CI 跨 0）、E4=47.50%。
- 交付物齐备：PROGRESS.md / EXPERIMENT_RESULTS.md / `final_2v2_300/`（summary + 15 JSON）/
  各实验子目录（config、train.log、evaluation.jsonl）；git 提交 `3c2b8d6`。
- 下一步（未在本 goal 执行）：E5-b 两阶段奖励课程（配置已就绪）、蒸馏消融
  （低系数/温度软化/仅蒸馏不一致决策）、自博弈数据重训 GRP。

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
