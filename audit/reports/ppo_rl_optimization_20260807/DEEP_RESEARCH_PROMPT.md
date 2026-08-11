# Deep Research 提示词：Riichi PPO 强化学习阶段优化

> 使用方式：把本文件整体交给 Deep Research 作为提示词。如果工具支持上传附件，请同时上传「参考文件」一节列出的本地文件；如果不支持，本提示词已经包含关键数字和代码事实，可以直接使用。

## 提示词正文

你是一名深度调研专家。请基于真实论文、开源项目和公开评测做研究，不要给泛泛的 RL 科普。以下是我正在训练一个日麻 AI，请围绕我的数据和约束给出可落地的优化方案。

### 任务目标

我已经完成 SFT，接下来要优化强化学习（RL）阶段，让最终策略明显超过现有 SFT 模型。当前 PPO 从 SFT checkpoint 初始化，训练到约 100 个 update 就达到最好水平，之后没有稳定提升，甚至对启发式对手出现退化。请帮我找到最可能的失败原因，并推荐新的训练机制、奖励设计或算法改进。

### 现有模型与数据

- 小型 transformer：4 层、192 宽、4096 context，输出固定 241 个合法动作的离散策略，另有一个 2 层 critic。
- SFT v13 验证集：top1 = 81.7%，top3 = 97.9%；分动作看 discard top1 = 78.0%、top3 = 97.3%，chi/pon top3 ≈ 100%，reach top3 = 99.5%。
- SFT 训练数据：约 363k 场、384 万小局、2.37 亿决策；raw mjai 日志在 `datasets/tenhou-to-mjai/2024.zip` 与 `2025.zip` 中。
- 可以离线计算 shanten、ukeire、suji、放铳风险等规则特征。
- v13 SFT vs v11 SFT 的 2v2 对战（320 半庄）：胜率 60.16% vs 39.84%，平均分差 +8426，95% CI 不跨 0，说明 v13 SFT 明显更强。
- 新 PPO 将固定从 `checkpoints/train_riichi_v13_sft/best_heuristic.pt` 初始化。

### 当前 PPO 实现

- 四席 current-policy 自博弈：一张桌子上四个座位全部由同一个当前策略控制；12 workers × 32 envs，长期 `kyokus_per_worker = 16`。
- reward 只有每个小局终局分差：`clip(delta_score / 1000, ±24)`（`kyoku_reward_clip_points = 24000`）。代码里已有 shanten/ukeire 效率分析器，但只统计 `reward=0.0`，没有进入训练 reward。
- 超参：PPO clip 0.2，4 epochs，minibatch 512，GAE gamma=0.99、lambda=0.95，target_kl=0.02，entropy 0.01→0.001，SFT KL anchor 0.02→0.002（冻结 SFT reference），前 40 步只训练 critic，actor/shared/critic LR 分别为 2e-5 / 5e-6 / 4e-5，value loss 进 shared backbone 的梯度乘以 0.25。
- 训练规模：2 张 GPU，单次 rollout 约 1.5k–2.5 万 transition/update。

### 已观测到的问题

- PPO vs v13 SFT（2v2，320 半庄）：update 50 胜率 47.34%；update 100 胜率 49.69%，平均分差 -436.9，95% CI 跨 0，基本打平且略负。
- PPO vs 固定启发式对手（每 15 update 评测 96 半庄）：约 update 75–105 最好（一位率 44%–52%、四位率 5%–10%、平均分差 +8k 到 +10k），之后波动甚至退化（update 540 一位率 32%、四位率 26%、分差 +1.9k；update 945 一位率 32%、四位率 13%）。
- 训练曲线：update 100 后 policy loss / approx_kl 增长明显变慢；update 200–950 的 rollout reward 围绕 0 波动，没有稳定提升；sft_reference_kl 从约 0.04 升到 0.22；熵在 0.53–0.60 波动。
- value explained variance 早期只有约 0.04，value loss 明显高于 policy loss，说明 critic 对稀疏终局奖励拟合较差。
- update 100 时 PPO 立直机会利用率高达 72.6%，而 SFT 只有 31%–33%，中后期仍偏高，疑似过度激进或风格偏移。

### 我已调研的一个候选方法：GRP（Global Reward Predictor）

我在 `smly/RiichiEnv` 的 `riichienv-ml` 里找到一个奖励模型实现，来源是 Suphx（Li et al., 2020）的 Global Reward Prediction：

- 仓库：`https://github.com/smly/RiichiEnv/tree/main/riichienv-ml/src/riichienv_ml`，相关 issue：`https://github.com/smly/RiichiEnv/issues/75`。
- 它训练一个很小的 MLP（20 维输入 → 128 → 64 → 4 类），输入是当前小局开始/结束时四家分数、本小局分数差、场风/局数/本场/立直棒、玩家 one-hot；标签是该玩家最终半庄排名；损失是 CrossEntropy。
- 推理时对 4 个玩家批量做 softmax，乘上 `pts_weight`（默认 [10, 4, -4, -10] 或配置里的 [6, 4, 2, 0]），再减去均值，作为这个小局的 reward。
- 它在 PPO worker 中把小局结束时的 GRP reward 当作该小局轨迹的 terminal reward；也可以离线给 CQL/BC 生成带折扣的 MC target。
- 与 Suphx 原版不同：原版是 GRU 序列模型、MSE 回归最终奖励；这个实现是单小局 MLP 分类，没有显式跨小局序列建模（但初始/结束分数隐含了累计分状态）。

请把这个 GRP 实现作为重点候选之一来评估，而不是只做一般性讨论。具体请回答：

1. GRP 相比当前小局分差 reward，理论上能解决什么、不能解决什么？它是否可能改善我观察到的「PPO 对 SFT 只打平、后期退化」问题？
2. 直接照搬 `riichienv-ml` 的 GRP 是否可行？需要哪些改动（数据读取、模型结构、与现有 PPO worker/GAE/value head 的集成、是否需要 GRU、pts_weight 如何选）？
3. GRP 与 dense 效率奖励、动作级奖励模型、SFT KL anchor、自博弈对手池之间应该以什么顺序实验？

### 请重点调研以下问题

1. 基于以上现象，最可能的失败原因是什么？请区分：稀疏/高方差 reward、critic/value learning 不足、self-play 非平稳性、KL/entropy 约束过强或过弱、超参问题、自博弈对手分布问题，并用证据排序。
2. 奖励设计：是否应该引入 GRP、动作级奖励模型、密集规则奖励，或三者结合？如果做奖励模型，输入是什么、训练标签用什么（专家动作、小局结果、半庄排名、点差？）、架构怎么设计（复用现有 backbone 还是独立小模型？）、如何防止 reward hacking？请与当前 sparse terminal reward 做对比，并给出具体实验建议。
3. 其他 RL 算法/机制：从当前 PPO 出发，哪些替代或改进最有希望？例如 IMPALA、regret/CFR 混合、opponent shaping、population/league self-play、best-of-n / rejection sampling、DPO/SPIN 等。请结合我的规模（小模型、2 GPU、12 workers）给出可落地性排序。
4. 自博弈与对手设计：current self-play 是否本身就是问题？是否应该混入固定 SFT 对手、历史 checkpoint 池、启发式对手或 population 训练？给出具体比例和实现方式。
5. 实验计划：请给出 3–5 个按性价比排序的 next experiments。每个写明：假设、改动点、期望观察指标、最少实验时长、判定成功标准（例如对 SFT 的 2v2 320 半庄胜率 >55%，且 95% CI 不跨 50%）。

### 硬约束

- 不改模型 backbone 结构；policy/value head 的输入输出用途可以调整。
- 从现有 v13 SFT checkpoint 出发。
- 我有 SFT 数据集、自博弈环境、规则分析器，可以离线算 shanten/ukeire/suji/风险等特征。
- 输出用中文；结论必须区分「有实证支持」和「推测」；每条建议给出关键论文/项目出处和实现注意点；不要推荐「增加数据量」「加大模型」这类无法实施的建议。

### 参考文件

如果 Deep Research 支持上传附件，请阅读以下文件：

- `checkpoints/train_riichi_v13_sft/metrics.json`：SFT 验证指标。
- `checkpoints/train_riichi_v13_sft/v11_vs_v13_2v2_detailed/sft_v11_vs_v13.json`：v11 vs v13 2v2 详细结果。
- `checkpoints/train_riichi_ppo/ppo_vs_sft_2v2_detailed/checkpoint_00050_vs_sft.json` 与 `checkpoint_00100_vs_sft.json`：PPO vs SFT 2v2 详细结果。
- `checkpoints/train_riichi_ppo/metrics.jsonl` 与 `evaluation.jsonl`：PPO 训练/评测曲线。
- `riichi_ppo_v1/configs/training.yaml`：PPO 配置。
- `riichi_ppo_v1/training/worker.py`、`riichi_ppo_v1/training/learner.py`：rollout 与 PPO 更新实现。
- `riichi_ppo_v1/training/rewards/`：现有 reward 组件（terminal/efficiency/decision/public_state）。
- 同目录 `REFERENCES.md` 与 `RIICHIENV_ML_GRP_REVIEW.md`：我已整理的项目证据与 GRP 调研摘要。
