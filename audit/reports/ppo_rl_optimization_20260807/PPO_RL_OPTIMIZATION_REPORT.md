# Riichi PPO 强化学习阶段优化：深度研究报告

日期：2026-08-08

范围：基于 `DEEP_RESEARCH_PROMPT.md` 与本地证据包（`REFERENCES.md`、`RIICHIENV_ML_GRP_REVIEW.md`、PPO/SFT 训练与评测数据）对「SFT 之后如何继续优化 PPO」做深度调研。所有结论尽量区分 **[实证]**（论文/开源项目/本地数据支持）与 **[推测]**（机制推断、尚未验证）。

方法：本报告由 3 个并行深度调研代理基于同一提示词与本地证据分别研究「失败诊断」「GRP/奖励设计」「搜索/算法/对手/实验计划」，再由根代理统一审校合并；关键本地数字（EV 曲线、2v2 胜率与 CI、立直率等）已逐一与 `metrics.jsonl`/`evaluation.jsonl` 核对，文献与开源实现（Suphx、riichienv-ml GRP、Mortal、Territory Paint Wars、EVPO、AsyPPO 等）均经联网核实。

---

## 0. TL;DR

1. **最可能的失败原因（按证据强度排序）**：critic/value 学习失效（本地强证据：value explained variance 从 +0.23 一路掉到负值，意味着 critic 比“用批次均值做基线”更差）→ reward 目标错配且稀疏（用小局点差代替最终排名收益，Suphx 论文明确反对；本地表现为立直过度激进 72.6% vs SFT 31%–33%）→ 自博弈 competitive overfitting（文献强证据：自博弈胜率恒为 50% 时对固定对手泛化能力可能崩溃；本地正是“对启发式先升后降、对 SFT 只打平”）。
2. **GRP 值得作为低成本第一候选**，但它只解决“点差 ≠ 最终排名收益”的错配，不解决小局内信用分配，也不能单独修复 critic 失效与自博弈退化；应先离线验证，再与对手混合、EV 门控等改造组合。
3. **推荐实验顺序**：E1 EV 门控优势/四席组内基线（RLOO/GRPO 式，天然适配 4 席自博弈）→ E2 对手混合（80% current + 20% 固定 SFT/随机）→ E3 离线 GRP 验证 + PPO A/B → E4 密集效率奖励 → E5（若仍无效）搜索蒸馏/pMCPA 式 rollout 适应。
4. **不建议**：直接上 MCTS/AlphaZero 风格在线搜索、CFR/league 级 population、DPO/SPIN——在本项目规模下性价比低，证据也不足。

---

## 1. 失败原因诊断

### 1.1 本地证据回顾

从 `checkpoints/train_riichi_ppo/metrics.jsonl` 提取的关键指标（训练配置为 `target_kl=0.02`、`update_epochs=4`、SFT KL anchor 0.02→0.002、value loss 回传 shared backbone 梯度 ×0.25）：

| update | value EV | approx_kl | entropy | policy_loss | sft_kl | return_std | advantage_std | value_pred_std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 | 4.7e-6 | 0.596 | -1.0e-5 | 0.000 | 2.78 | 2.78 | 0.00 |
| 50 | 0.225 | 9.2e-5 | 0.610 | -0.0010 | 0.001 | 3.40 | 2.99 | 1.50 |
| 100 | 0.125 | 1.1e-3 | 0.563 | -0.0049 | 0.040 | 3.33 | 3.12 | 2.17 |
| 200 | 0.096 | 1.8e-3 | 0.600 | -0.0081 | 0.128 | 3.26 | 3.10 | 2.61 |
| 300 | 0.079 | 2.3e-3 | 0.581 | -0.0081 | 0.169 | 3.30 | 3.17 | 2.55 |
| 500 | -0.019 | 2.2e-3 | 0.589 | -0.0082 | 0.183 | 3.35 | 3.38 | 2.86 |
| 700 | -0.086 | 2.4e-3 | 0.540 | -0.0080 | 0.206 | 3.27 | 3.41 | 2.94 |
| 950 | -0.070 | 2.5e-3 | 0.594 | -0.0081 | 0.226 | 3.49 | 3.61 | 3.11 |

对战证据（`REFERENCES.md`）：PPO vs SFT 2v2（320 半庄）u50 胜率 47.34%（分差 -1392.5，CI 跨 0）、u100 胜率 49.69%（分差 -436.9，CI 跨 0）；PPO vs 固定启发式，u75–105 最佳（一位率 44%–52%、四位率 5%–10%、分差 +8k~+10.8k），随后退化（u540 一位率 32.3%、四位率 26.0%；u945 一位率 32.3%、四位率 13.5%）。u100 立直机会利用率 72.6%，SFT 对手只有 31.4%–33%。

### 1.2 候选原因排序

| 排序 | 候选原因 | 证据等级 | 关键证据与机制 |
|---|---|---|---|
| 1 | **critic/value 学习失效** | **[实证]**（本地 + 文献） | 本地：EV 在 u500 后转负，value_pred_std 从 1.5 涨到 3.1（≈return_std），说明 critic 已退化成“预测噪声”；u100 后 policy_loss 基本停滞（-0.0081）而 sft_kl 继续涨。文献：EVPO（arXiv:2604.19485）证明稀疏奖励下“学出来的 critic 注入的估计噪声可能超过其捕获的状态信号”，并给出 EV=0 为切换边界的理论；AsyPPO（arXiv:2510.01656）用轻量 mini-critics 缓解 value 噪声。机制上，GAE 通过一个烂 critic 做 bootstrap，优势≈带噪声的 MC 收益，PPO 更新变成随机漂移。 |
| 2 | **reward 与最终目标错配且稀疏** | **[实证]**（Suphx + 本地） | Suphx（arXiv:2003.13590）明确说“小局得分不能直接作为学习反馈”（领先者最后一局应弃和保位），并因此引入 GRP；其消融 RL-1（GRP）> RL-basic（小局分差）。本地：立直率 72.6% 说明策略在追逐小局点差/和牌而不是最终排名收益；rollout reward 恒约 0（4 席零和），单小局一个 terminal reward，对 critic 极不友好。 |
| 3 | **自博弈 competitive overfitting / 非平稳性** | **[实证]**（文献 + 本地） | Territory Paint Wars（arXiv:2604.04983）在受控环境中复现：自博弈胜率恒为 50%，但对固定对手的泛化胜率从 73.5% 崩到 21.6%，20% 随机对手混合即可恢复（77.1%）。OpenAI Five 用 80% 最新 + 20% 历史对手，Unity ml-agents 官方文档建议固定历史版本池。本地现象（自博弈恒 50%、对启发式先升后降、对 SFT 打平）与该病态高度吻合。 |
| 4 | **SFT KL anchor 累积漂移** | **[混合]**（本地 + 文献） | 本地：sft_kl 0.04→0.226，但 per-update approx_kl 只有 0.002——说明不是单步失控，而是上千步小步漂移累积；熵 0.54–0.60 未塌缩，说明不是熵崩。文献：Catastrophic Goodhart（arXiv:2407.14503）指出 KL 正则无法根治重尾的 reward misspecification。判定：KL 是放大器/帮凶，不是根因。 |
| 5 | **超参与规模** | **[推测]** | 单次 rollout 1.5k–2.5 万 transition/update、12 workers × 32 envs、4 epochs/minibatch 512，对 241 动作的 transformer 偏小；actor LR 2e-5 也保守。但无本地对照实验，不能确证；应先修 1–3 再看。 |
| 6 | **对手分布问题** | 并入 3 | 4 席全 current 使对手分布完全跟随策略，是 competitive overfitting 的温床。 |

### 1.3 一句话结论

**[实证]** 最合理的解释是：稀疏且错配的 reward + 始终学不好的 critic → PPO 的优势信号噪声过大，策略只能靠 SFT KL 锚缓慢漂移；同时 4 席 current 自博弈让策略对“自己”过拟合，所以对固定对手先升后降。三者互相放大：烂 critic 让 RL 学不到真信号，GRP/奖励错配让信号本身是错的，自博弈让评测看不到退化。

---

## 2. GRP（Global Reward Predictor）重点评估

### 2.1 Suphx 原版与 riichienv-ml 实现的定位

[**实证**] Suphx 的 GRP 是一个 2 层 GRU + 2 层 FC 的序列模型：以“当前及之前所有小局的信息”预测整局最终奖励（MSE 训练，数据为天凤顶级玩家日志）；训练完成后，第 k 个小局的 reward = Φ(x^k) − Φ(x^(k−1))，即“新增信息对最终奖励的边际贡献”。消融显示 RL-1（GRP）> RL-basic（小局分差），且论文给出典型案例：领先者最后一局不追和、打安全牌保一位。

[**实证**] riichienv-ml 的实现是简化版：单小局 MLP（20 维 → 128 → 64 → 4 类 CE 预测最终排名），推理时 4 玩家 softmax × `pts_weight`（默认 [10,4,-4,-10] 或 [6,4,2,0]）再减均值，作为该小局 terminal reward（[仓库](https://github.com/smly/RiichiEnv/tree/main/riichienv-ml)、[issue #75](https://github.com/smly/RiichiEnv/issues/75)）。特征：四家初始分/结束分/小局分差 + 场风/局数/本场/立直棒 + player one-hot。注意该实现的 `chang/ju` 编码与常见 `kyoku` 序号不同，接入要核对。

[**实证**] Mortal（开源立直麻将 AI）走的是 Suphx 原版的序列路线：GRP 输入按小局序列展开，`RewardCalculator.calc_delta_pt` 用 `exp_pts[1:] − exp_pts[:-1]` 把「相邻小局预测排名收益的差」作为奖励，`pts` 默认 [3,1,-1,-3]（[reward_calculator.py](https://github.com/Equim-chan/Mortal/blob/main/mortal/reward_calculator.py)）；其作者在 [讨论 #82](https://github.com/Equim-chan/Mortal/discussions/82) 中记录过 online 自博弈阶段因奖励/探索造成的不稳定。这为「先 MLP 后 GRU」的两步路线提供了同类项目参照。

### 2.2 能解决什么、不能解决什么

| 维度 | 结论 | 依据 |
|---|---|---|
| 终局排名 vs 点差错配 | **能解决**：把小局结果换算成“最终排名收益”，终盘保位/弃和行为可以被正确奖励 | Suphx 消融（RL-1 > RL-basic）+ 论文案例 |
| reward 方差/噪声 | **可能改善**：GRP reward 数值更贴近“距最终胜利还有多远”，比 raw 点差更平稳，理论上降低 critic 学习难度 | [推测]（无麻将领域的直接消融，EVPO 的机制结论可类比） |
| 小局内逐动作信用分配 | **不能解决**：仍是每小局一个 terminal reward；GAE/critic 必须自己往回传 | riichienv-ml 实现本身 |
| 跨小局序列（终盘策略） | **原版能、MLP 简化版弱**：原版 GRU 显式建模；MLP 只靠初始/结束分数隐式携带累计分 | Suphx 论文 vs issue #75 说明 |
| 自博弈 competitive overfitting | **不能解决**：它只换奖励，不换对手分布 | Territory Paint Wars 结论 |
| critic 失效 | **间接可能改善**（信号更平滑 → EV 可能回升），但不会自动修好 | [推测] |

### 2.3 对「只打平 SFT、后期退化」的可能贡献

**[实证]** 退化在时间上对应 EV 转负（u500 后）与 sft_kl 累积（u500 后 0.18+），这两者都指向“更新噪声主导、信号微弱”。GRP 把“点差”换成“排名收益”能同时改善信号质量与目标正确性，因此**可能缓解退化**；但如果 competitive overfitting 是主因，GRP 单独做不够，必须叠加对手混合。

### 2.4 直接照搬 riichienv-ml 的可行性

总体判断：**可行，改动量小，值得先做离线验证**（与 `RIICHIENV_ML_GRP_REVIEW.md` 一致）。需要改动的清单：

1. **数据读取**：仓库 GRP dataset 直接 glob `.jsonl.gz`；本地 raw 日志在 `datasets/tenhou-to-mjai/2024.zip`/`2025.zip` 且扩展名 `.mjson`。用 `zipfile` 流式读取或先解压，再用 `MjaiReplay.from_jsonl_string` 解析；每条小局构造 4 行特征，标签为最终排名。3.8M 小局足够，先跑 10%–20% 子集即可验证。
2. **特征与编码**：核对 `chang`（bakaze：东=0 南=1 西=2 北=3）与 `ju`（亲家局数，0 起）语义；本场/立直棒按仓库归一化（÷3、÷4 等）。
3. **模型结构**：先用仓库原样 MLP（训练成本极低，CPU 都行）。验证集要报告：rank 分类准确率（随机基线 25%）、校准曲线（ECE），以及 **GRP reward 与 point-delta reward 的 Spearman 相关**——若相关 >0.9，收益会很小，需要改用 GRU 或加手牌/牌河特征。
4. **worker 集成**：在 `worker.py` 小局结束时（现在写 `terminal_kyoku_reward` 的位置）构造 GRP 特征（四家分数已有，场风/局数/本场/立直棒可从 env/bridge 拿到），把 reward 替换为 `λ·r_grp + (1−λ)·clip(delta/1000, ±24)`（先 λ=0.5），仍只写最后 transition；GAE/value head 不动。
5. **是否需要 GRU**：**先不要**。先用 MLP 版 A/B；若终盘保位行为仍异常（如最后几局立直率不降），再升级 GRU 版（输入按半庄内小局序列）。
6. **pts_weight**：默认 [10,4,-4,-10] 更强调排名离散度；[6,4,2,0] 减均值后等价 [3,1,-1,-3]，更温和。建议先默认，做 2–3 档超参扫描；softmax 可加温度。
7. **分布偏移**：GRP 训练自人类牌谱，自博弈策略偏离人类分布后会外推失真；训练中期应定期用自博弈数据重训 GRP 或做小权重微调。

### 2.5 风险

- **Reward hacking**：GRP 输入含分数，策略可能靠激进追分“刷”GRP 输出；必须用「GRP 训练 reward 提升是否转化为 SFT 2v2 胜率提升」作为最终判定，而不是看训练 reward。
- **MLP 独立性假设**：跨小局依赖（保位/放铳权衡）可能学不到；若 A/B 无效，优先试 GRU 而不是换别的奖励。

---

## 3. 奖励设计：GRP vs 动作级奖励模型 vs 密集规则奖励

### 3.1 三类信号对比

| 信号 | 粒度 | 解决什么 | 主要证据 | 主要风险 |
|---|---|---|---|---|
| 当前 point-delta terminal（现状） | 每小局 1 个 | 零 | 已证不够 | 目标错配（排名≠点差）、稀疏 |
| GRP terminal | 每小局 1 个 | 排名收益错配 | Suphx 消融；Mortal 也使用 Suphx 式 GRP（[Mortal 讨论 #82](https://github.com/Equim-chan/Mortal/discussions/82)） | 分布偏移、可被刷分 |
| 密集规则奖励（shanten/ukeire） | 每决策 | 小局内效率信号 | 单机麻将 reward shaping（arXiv:2305.04145）证明 shanten 类 shaping 有效；本仓库 `efficiency.py` 已实现且有界（shanten 退步 -1，ukeire 相对损失 -0.25×，最优 0） | 牺牲防守换 ukeire（reward hacking）；改变最优策略（非 potential 形式） |
| 动作级奖励模型（RLHF 式） | 每决策 | 逐动作打分 | 麻将领域**缺乏成熟实证**；LLM 领域 reward model 有 overoptimization 问题（Gao et al. 2023；Catastrophic Goodhart） | 需要偏好/标签来源，训练与校准成本高，最容易被 hack |

### 3.2 推荐组合与顺序

**[实证/推测结合]** 按性价比：

1. 先保留 terminal（point-delta 或 GRP），**不要去掉**——它是唯一直接对应胜负目标的信号。
2. 再叠加小权重 dense 效率奖励：只对 discard 决策加（`efficiency_reward` 已有），权重 0.01–0.05 起步；作用不是“教正确”，而是降低探索方差、让 critic 早点看到结构性信号。注意效率奖励不是 potential-based shaping（Ng et al. 1999），理论上可能改变最优策略，所以权重必须小且以 SFT 2v2 为准。
3. 动作级 RM：**初期不建议**。没有天然的偏好对；若要做，标签应来自“同局面 MC 终局收益排序”（RUDDER 式 return decomposition 的实证思路，[arXiv:1806.07857](https://arxiv.org/abs/1806.07857)），而不是人类动作概率。复用现有 backbone 做价值/优势头比独立 RM 更省，但当前 critic 已学不好，先解决 critic 再谈。
4. 如果做奖励模型输入：GRP 用分数+场况即可；动作级模型必须加手牌/牌河/攻防特征，否则学不到“为什么这手牌该弃和”。

### 3.3 防 reward hacking

- 训练 reward 提升 ≠ 真实变强：所有奖励实验的**唯一成功判据**是 SFT 2v2 240 半庄胜率与 95% CI（执行标准已按用户要求由 320 改为 240）。
- KL anchor 不能根治 hacking（[Catastrophic Goodhart，arXiv:2407.14503](https://ui.adsabs.harvard.edu/abs/2024arXiv240714503K/abstract)），所以还要监控行为指标：立直率、放铳率、和牌率、副露率、被飞率是否漂移出 SFT 合理区间。
- dense 奖励加“防守正则”再上线：suji/危险度惩罚只在效率奖励验证有效后作为第二阶段考虑。

---

## 4. 树搜索评估

### 4.1 各方案结论

| 方案 | 可行性 | 与 PPO 结合 | 主要证据/注意点 |
|---|---|---|---|
| top3 受限 MCTS | 中低 | 需 PIMC（采样对手手牌+牌山），且麻将回合顺序不规则 | Suphx 明确说麻将不规则博弈树使 MCTS/CFR 难以直接应用，改用 pMCPA（rollout + 策略微调）（[Suphx, arXiv:2003.13590](https://arxiv.org/abs/2003.13590)）；top3≈98% 专家动作只缩小了动作面，没解决隐藏信息与随机牌山 |
| AlphaZero 风格搜索（含 Gumbel） | 低 | 需要环境模型/重规划；DeltaDou 在斗地主做贝叶斯推断+搜索，训练 2 个月，仍被无搜索的 DouZero 10 天反超（[arXiv:2106.06135](https://arxiv.org/abs/2106.06135)） | 本项目 2 GPU、12 workers，推理预算不允许在线搜索 |
| SPG / 搜索蒸馏 / expert iteration | 中（离线） | 先离线用弱搜索生成“改进动作”，再蒸馏回策略（AlphaGo Zero 的 SPG 思路；expert iteration，Anthony et al. NeurIPS 2017） | 搜索教师本身要先可靠；在本场景只能作为后续实验 |
| 在线搜索 rollout（pMCPA 式） | 中 | 每局开始按当前手牌采样对手手牌/牌山 rollout，微调策略后再打（Suphx 3.4） | 有 Suphx 实证；推理成本可控（只在小局开始时做一次，K 条 rollout） |
| ReBeL 式 RL+搜索 | 低 | 框架在 2 人零和可证明收敛（[arXiv:2007.13544](http://xxx.itp.ac.cn/abs/2007.13544v2)）；4 人非零和麻将无直接支撑 | 不适合当前阶段 |

### 4.2 结论

**[实证]** 搜索不是当前瓶颈：Suphx 在更强算力下都选择 rollout 适应而不是 MCTS；DeltaDou vs DouZero 说明搜索收益会被成本和随机性吃掉。**建议**：训练期不做在线 MCTS；若一定要探索搜索，先做 pMCPA 式“局首 rollout 适应”或离线搜索蒸馏，且用 2v2 胜率判定是否值得。

---

## 5. 替代 RL 算法与机制（可落地性排序）

| 排序 | 方案 | 证据 | 本项目落地性 |
|---|---:|---|---|
| 1 | **EV 门控优势 / critic-free 组内基线（EVPO / RLOO / GRPO 式）** | EVPO（[arXiv:2604.19485](https://arxiv.org/abs/2604.19485)）：稀疏奖励下 critic 噪声超信号，EV<0 时切到批次均值基线；RLOO/GRPO 用组内比较避免 critic（DeepSeekMath，arXiv:2402.03300；RLOO，arXiv:2402.14740） | 高。4 席自博弈天然提供“同一小局 4 条轨迹”的组，可做零和组内基线；改动集中在 learner 的 advantage 计算 |
| 2 | **对手混合/历史池** | Territory Paint Wars（20% 随机）；OpenAI Five（80% 最新/20% 历史） | 高。worker lineup 已有 per-seat policy 结构，加配置调度即可 |
| 3 | **GRP + 密集效率奖励** | Suphx；arXiv:2305.04145 | 中高。见 §2/§3 |
| 4 | **critic 预训练/多 critic** | AsyPPO mini-critics（[arXiv:2510.01656](https://arxiv.org/abs/2510.01656)）；用人类日志离线 MC target 预热 value head | 中。能直接打 EV<0 的痛点；工程量大一点 |
| 5 | **IMPALA/V-trace** | [arXiv:1802.01561](https://arxiv.org/abs/1802.01561) | 中低。主要解决异步 off-policy，不是当前根因 |
| 6 | **best-of-n / rejection sampling / expert iteration** | BoN 与 RLHF 等价性（NeurIPS 2024 BoNBoN）；DouZero 的 MC 采样思路 | 中低。可做“rollout 里按终局收益筛选轨迹再 imitate”，作为离线补充而非主循环 |
| 7 | **DPO/SPIN** | SPIN 把自博弈做成两玩家游戏（Chen et al. 2024），等价于迭代 DPO | 低。需要构造偏好对；4 人随机麻将里“胜者轨迹优于败者轨迹”的偏好噪声极大，直接应用风险高 |
| 8 | **league / population 训练** | AlphaStar（Vinyals et al. 2019） | 低。基础设施太重；轻量对手池已覆盖大部分收益 |
| 9 | **CFR/regret 混合（ACH）** | ACH（ICLR 2022）在 1v1 麻将击败人类冠军，理论针对 2 人零和 | 低。本项目是 4 人非零和排名博弈，NE 理论支撑弱 |
| 10 | **Gumbel AlphaZero / Sampled MuZero** | ICLR 2022 / ICML 2021 | 低。需环境模型与搜索，见 §4 |

---

## 6. 自博弈与对手设计

### 6.1 问题证据

**[实证]** 4 席 current self-play 是本项目最大的结构风险：

- Territory Paint Wars 证明“自博弈 50% + 泛化崩盘”可以在受控环境稳定复现，且 20% 随机对手混合是单行修复（[arXiv:2604.04983](https://arxiv.org/abs/2604.04983)）。
- OpenAI Five 用 80% 最新 + 20% 历史版本池（Berner et al. 2019；社区整理细节 [知乎](https://zhuanlan.zhihu.com/p/336608429)）；Unity ml-agents 官方自博弈文档也建议对固定历史版本池训练（[文档](https://github.com/Unity-Technologies/ml-agents/blob/422bbcd3d4e82dae6acc3b12e189f257d160eaa7/docs/Training-Self-Play.md)）。
- Suphx 的 RL 只更新 discard 模型、其余模型保持 SL 固定，客观上让对手分布接近“半固定”混合；Mortal 采用 offline CQL → online 自博弈，其作者在 [讨论 #82](https://github.com/Equim-chan/Mortal/discussions/82) 里明确记录 online 阶段因奖励/探索造成的不稳定。
- 本地：自博弈恒 ~50%（4 席同策略），对固定启发式先升后降、对 SFT 打平——与 competitive overfitting 的“评测外看不到”特征吻合。

### 6.2 推荐比例与实现

建议从轻到重：

1. **80/20 对手混合（首选）**：每环境 4 席中，20% 的对局把 1–3 席换成固定对手。固定对手先选 **冻结的 v13 SFT**（与最终评测同分布），再加少量随机策略（Territory Paint Wars 证明随机有效）。`worker.py` 的 `lineups` 已是 per-seat policy 字符串，reward 也只发给 `current` 席，改动点：config 驱动 lineup 生成 + 让非 current 席用固定 checkpoint/随机策略推理。
2. **历史 checkpoint 池（次选）**：u50/u100/u200 等 checkpoint 放进池子，按质量分 softmax 采样（OpenAI Five 式）；若 80/20 已解决退化，这步可跳过。
3. **评测纪律**：自博弈指标（win rate≈50%、rollout reward≈0）**不透明**，必须固定节拍做外部评测：每 15 update 对启发式（已有）、每 50 update 对 SFT 2v2 240 半庄（已有工具，按用户要求由 320 改为 240），并把 u100 后的“先升后降”作为失败信号立刻回滚 checkpoint。

---

## 7. 推荐实验计划（按性价比排序）

统一成功判据：对 v13 SFT 的 2v2 240 半庄胜率 >55% 且 95% CI 不跨 50%（执行标准，原提示词为 320 半庄；历史证据仍为 320 半庄）；次要指标：启发式评测一位率不再随 update 退化、sft_reference_kl 增速放缓、EV 转正。

| # | 实验 | 假设 | 改动点 | 期望观察 | 最短时长 | 成功标准 |
|---|---|---|---|---|---|---|
| E1 | **EV 门控优势 / 四席组内基线** | 稀疏奖励下 critic 噪声主导 → 优势信号是垃圾 | learner 里计算 batch EV；EV<0 时 advantage 用“MC 终局收益 − 组内均值”（同一小局 4 席天然成组，RLOO/GRPO 式）；EV≥0 维持 GAE | EV 转正或维持 ≥0；policy_loss 重新开始下降；sft_kl 增速放缓 | 1–2 天（300–500 update） | u300 时 2v2 >55% 且 CI 不跨 50% |
| E2 | **对手混合 80/20** | 自博弈 overfitting → 对固定对手退化 | config 驱动 lineup：20% 对局混入冻结 SFT/随机席 | 对启发式不再先升后降；sft_kl 更稳 | 1–2 天 | u200–u400 启发式一位率不显著低于 u100 峰值；2v2 不差于基线 |
| E3 | **离线 GRP 验证 + PPO A/B** | GRP 修复“点差≠排名”错配 | 先用 10%–20% 日志训 MLP GRP，报告 rank acc/ECE/与 point-delta 相关性；再跑 (a) point-delta、(b) GRP、(c) 0.5 混合 三组各 300–500 update | GRP acc 明显 >25%；相关性不高；GRP 组 EV 更好 | GRP 训练数小时；PPO 每组 1–3 天 | 混合组 2v2 >55% 且优于 point-delta 基线 |
| E4 | **密集效率奖励（小权重）** | discard 效率信号降低探索方差 | 在 E1/E2 之上给 discard 决策加 `efficiency_reward`×0.01–0.05 | 训练 EV 回升；立直率回落到合理区间（<60%）；放铳率不升 | 1–2 天 | 2v2 胜率不下降且行为指标改善；若胜率下降则回滚 |
| E5 | **（后续）critic 预热 / pMCPA / 搜索蒸馏** | E1–E4 不够时再打 critic 或搜索 | 用人类日志 MC target 预热 value head，或局首 rollout 适应，或离线蒸馏搜索策略 | 视 E1–E4 结果再定 | — | 同上 |

**并行建议**：E1 与 E2 是独立代码路径（learner vs worker），GPU 有空闲时可并行；E3 的离线 GRP 训练不占 GPU 也可提前启动。E3 必须放在 E1/E2 之后对照，否则三处同时改无法归因。

---

## 8. 监控与风险清单

- 每 update 记录：`value_explained_variance`（EV<0 即报警）、`value_prediction_std / return_std`、`sft_reference_kl`、`entropy`、`clipfrac`、行为指标（立直/放铳/和牌/副露率）。
- 每 15 update：对固定启发式评测；每 50 update：对 SFT 2v2 240 半庄评测（paired bootstrap CI）。**以外部评测为唯一真相**。
- Reward hacking 防护：任何新奖励都要回答“训练 reward 涨了，2v2 胜率涨了吗”；GRP 定期用自博弈数据重训；KL anchor 保留但不要依赖它防 hacking。
- 所有实验从 `best_heuristic.pt` 初始化、固定 seed 1、同一评测脚本，保证可比。

---

## 9. 参考资料

### 论文
- Suphx: Mastering Mahjong with Deep Reinforcement Learning，arXiv:2003.13590（GRP、oracle guiding、pMCPA、熵敏感性；RL-1>RL-basic）
- Territory Paint Wars: PPO Failure Modes，arXiv:2604.04983（自博弈 competitive overfitting、20% 随机对手混合）
- EVPO: Explained Variance Policy Optimization，arXiv:2604.19485（稀疏奖励下 critic 噪声、EV=0 门控）
- Asymmetric PPO（mini-critics），arXiv:2510.01656
- DouZero: Mastering DouDizhu with Self-Play Deep RL，ICML 2021 / arXiv:2106.06135（Deep Monte-Carlo、无搜索击败 DeltaDou）
- Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game（ACH），ICLR 2022（1v1 麻将，regret 混合）
- Combining Deep RL and Search for Imperfect-Information Games（ReBeL），arXiv:2007.13544
- RUDDER: Return Decomposition for Delayed Rewards，NeurIPS 2019 / arXiv:1806.07857
- A Novel Reward Shaping Function for Single-Player Mahjong，arXiv:2305.04145
- Variational Oracle Guiding for RL（VLOG），ICLR 2022
- Catastrophic Goodhart，arXiv:2407.14503
- DeepSeekMath（GRPO），arXiv:2402.03300；Back to Basics（RLOO），arXiv:2402.14740
- BoNBoN（Best-of-N 与 RLHF 等价），NeurIPS 2024
- Policy Improvement by Planning with Gumbel（Gumbel AlphaZero），ICLR 2022；Sampled MuZero，ICML 2021
- AlphaGo Zero（SPG 思路），Nature 2017；Expert Iteration，Anthony et al. NeurIPS 2017
- AlphaStar，Vinyals et al. Nature 2019；OpenAI Five，Berner et al. 2019
- Ng, Harada, Russell 1999：potential-based reward shaping（理论）

### 项目/文档
- riichienv-ml GRP：<https://github.com/smly/RiichiEnv/tree/main/riichienv-ml>；issue #75
- Mortal：<https://github.com/Equim-chan/Mortal>；online 不稳定讨论 #82；Mortal-Policy（offline→online）<https://github.com/Nitasurin/Mortal-Policy>
- Unity ml-agents self-play 文档
- Suphx 论文（“无法直接用 MCTS/CFR” 的出处）：<https://arxiv.org/abs/2003.13590>

### 本地证据
- `checkpoints/train_riichi_v13_sft/metrics.json`、`v11_vs_v13_2v2_detailed/sft_v11_vs_v13.json`
- `checkpoints/train_riichi_ppo/metrics.jsonl`、`evaluation.jsonl`、`ppo_vs_sft_2v2_detailed/checkpoint_00050/00100_vs_sft.json`
- `riichi_ppo_v1/configs/training.yaml`、`training/worker.py`、`training/learner.py`、`training/rewards/*`
- 本目录 `REFERENCES.md`、`RIICHIENV_ML_GRP_REVIEW.md`
