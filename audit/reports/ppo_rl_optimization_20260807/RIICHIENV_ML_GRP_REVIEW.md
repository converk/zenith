# RiichiEnv riichienv-ml 奖励模型调研：GRP 是否可行

## 1. 结论摘要

`smly/RiichiEnv` 的 `riichienv-ml` 里确实有训练奖励模型的代码，但它不是 RLHF 那种「给动作打分的奖励模型」，而是 Suphx 论文中的 **Global Reward Predictor（GRP）**：根据当前小局结束时四家累计分数/小局分数差等特征，预测玩家最终半庄排名，再换算成小局奖励。

**可行性判断：可行，且是一个低成本、值得优先试的奖励塑形实验；但它不是万能药。** 它主要解决「小局分差 ≠ 最终排名收益」的问题，不能直接解决「小局内逐动作信用分配」的问题。当前 PPO 已经在用每小局终局分差，所以 GRP 的收益主要来自把「点差」换成「预测最终排名的期望收益」；这与 dense 效率奖励是不同维度，两者可以组合。

## 2. 来源

- 仓库：`https://github.com/smly/RiichiEnv/tree/main/riichienv-ml/src/riichienv_ml`
- 调研时参考 commit：`b1d08b3615a710f929679fefb50d1c384f2070b9`（2026-05-09）
- 相关 issue：`https://github.com/smly/RiichiEnv/issues/75`（Add Code Example and Interface for Training a Reward Model）
- 原论文：Suphx: Mastering Mahjong with Deep Reinforcement Learning，arXiv:2003.13590。

## 3. 仓库实现细节

### 3.1 训练脚本与数据

- 入口：`riichienv-ml/scripts/train_grp.py`
- 训练器：`riichienv-ml/src/riichienv_ml/trainers/grp.py`
- 数据集：`riichienv-ml/src/riichienv_ml/datasets/grp_dataset.py`
- 模型：`riichienv-ml/src/riichienv_ml/models/grp_model.py`
- 配置：`riichienv-ml/src/riichienv_ml/configs/4p/grp.yml`

训练数据来自 MJAI 格式 replay。每个小局构造一条特征，标签是该玩家在半庄结束时的最终排名 one-hot，损失为 CrossEntropy。默认配置 1 epoch、batch 512、lr 5e-4、CosineAnnealing。

### 3.2 特征

对 4 人日麻，每个玩家输入 20 维：

- 四家小局初始分 `/25000`
- 四家小局结束分 `/25000`
- 四家本小局分差 `/12000`
- 场风 `/3`、局数 `/3`、本场 `/4`、立直棒 `/4`
- 玩家 one-hot

注意：`grp_features` 中的 `chang` 是 `bakaze` 编码（东=0 南=1 西=2 北=3），`ju` 是亲家局数（0 开始），不是我们项目里常见的 `kyoku` 序号，接入时需要核对。

### 3.3 模型与推理

- `RankPredictor`：`Linear(20,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,4)`，带 0.1 dropout。
- `RewardPredictor.calc_all_player_rewards`：对 4 个玩家各构造一条输入，softmax 出 4 个排名的概率，点乘 `pts_weight`（默认 [10, 4, -4, -10]），再减去权重均值，得到每个玩家的该小局 reward。
- `calc_pts_rewards` 还会输出相邻小局预测收益的差值，可用于把「全局预测收益」分摊到小局序列。

### 3.4 在训练中的用法

- 在线 PPO：`riichienv-ml/src/riichienv_ml/trainers/_ppo_worker.py` 在每个小局结束时用环境分数和场况构造 `grp_features`，调用 GRP 得到该小局 reward，作为该小局轨迹的 terminal reward。
- 离线 CQL/BC：`riichienv-ml/src/riichienv_ml/datasets/mjai_logs.py` 用 GRP 对小局打分，再按 `gamma^(T-t-1)` 衰减作为 Q 学习 target；`trainers/bc_logs.py` 也用它给离线数据标 reward。
- 配置里通过 `grp_model: <path>` 和 `pts_weight: [...]` 控制是否启用。

## 4. 与 Suphx 原版的差异

| 维度 | Suphx 原版 | riichienv-ml 实现 |
|---|---|---|
| 模型 | 2 层 GRU + 2 层 FC | 单小局 MLP |
| 输入 | 当前及之前所有小局信息，含手牌/牌河特征 | 仅四家初始/结束/本小局分差 + 场况 + player one-hot |
| 标签/损失 | MSE 回归最终奖励 | 4 类 CrossEntropy 预测最终排名 |
| 输出 | 直接预测最终 reward | softmax 排名概率 × pts_weight |
| 适用 | RL 中间 reward | 在线 PPO 小局 reward + 离线 CQL/BC target |

该实现基于一个简化假设：每个小局结果对最终排名的预测是独立的，不显式建模跨小局依赖。初始分/结束分已经携带累计分信息，因此不是完全无状态，但丢失了「本场手牌好坏」「本局过程」等更细的信息。

## 5. 对我们项目的可行性分析

### 5.1 数据可用性

我们有：

- `datasets/tenhou-to-mjai/2024.zip` 与 `2025.zip`：约 36.3 万场 MJAI 格式日志（JSON Lines，扩展名 `.mjson`）。
- 本地 `RiichiEnv` 已有 `MjaiReplay` 和 `Kyoku.take_grp_features()`，可以直接拿到 GRP 所需特征。
- `datasets/tenhou_sft_2024_2025/manifest.json`：已验证 363,312 场 / 3,846,384 小局 / 2.37 亿决策。

需要做的数据适配很小：仓库 GRP dataset 直接 glob `.jsonl.gz`，我们的原始日志在 zip 里且扩展名是 `.mjson`；可以用 `zipfile` 流式读取，或先解压一部分，再用 `MjaiReplay.from_jsonl_string` 解析。

### 5.2 与现有 PPO 的集成成本

当前项目 `riichi_ppo_v1/training/worker.py` 在小局结束时已经拿到四家分数，只需要再拿到场风/局数/本场/立直棒，就能构造 GRP 输入。接入点非常清晰：

1. 先离线训练 GRP（小模型，训练成本很低）。
2. 在 worker 小局结束时把 `terminal_kyoku_reward` 替换为 GRP reward，或做 `λ * GRP + (1-λ) * point_delta`。
3. 不需要改模型 backbone，不改变 SFT checkpoint 初始化方式。

### 5.3 它能解决什么

- 小局分差大不代表最终排名收益大，例如已经大幅领先时继续进攻的边际收益低、放铳惩罚高。GRP 能把「最终排名收益」分摊到每个小局，理论上能改善终盘策略。
- 相比纯终局 reward，GRP 提供的 reward 在数值上更接近「这个局面距离最终胜利还有多远」，可能降低 value learning 的难度。
- 如果 reward 的方差和噪声下降，critic 的 explained variance 可能改善，PPO 更新会更稳定。

### 5.4 它不能解决什么

- 仍然是每小局一个 terminal reward，没有把小局内每一步动作的信用分配变密。若 PPO 平台期的根因是「小局内信号太稀疏」，GRP 帮助有限。
- 该实现只用了分数和场况，没有用到手牌/牌河/攻防信息；它不是动作级 reward model，不能回答「这个动作比另一个动作好多少」。
- 训练数据是人类牌谱，reward 分布来自人类玩家；自博弈策略可能偏离人类分布，GRP 可能需要定期用自博弈数据重训，否则会外推失真。
- 简化版 MLP 假设小局独立；如果发现终盘策略需要跨小局上下文（如「为了保一位故意不放铳」「弃和保排名」），可能需要升级为 GRU/Transformer 序列模型。

### 5.5 风险与注意点

- 防止 reward hacking：GRP 输入包含分数，输出是排名的软概率；理论上策略可能通过激进追分「刷分」来操纵预测，需要在评测中确认 GRP reward 的提升确实转化为 SFT 2v2 胜率提升，而不是只提高训练 reward。
- `pts_weight` 的选择会影响风险偏好；[6,4,2,0] 与 [10,4,-4,-10] 差别很大，建议作为超参。
- 需要单独验证 GRP 的验证集准确率/校准，以及 GRP reward 与 point-delta reward 的相关性；如果 GRP 只是 point-delta 的单调函数，收益可能有限。
- 当前 PPO 已有 SFT KL anchor；GRP 改变 reward 后，KL 系数和熵系数可能需要重新调。

## 6. 建议的实验顺序

1. **离线 GRP 验证**：用现有 mjson 日志训练 MLP GRP，验证集上报告 rank 分类准确率和校准；顺便对比「GRP reward」与「point-delta reward」在相同小局上的相关性。
2. **PPO 替换实验**：从 v13 SFT 出发，分别跑 (a) 当前 point-delta reward、(b) GRP reward、(c) GRP + 小权重 point-delta，各训练到至少 300–500 update，用同一套 2v2 SFT 320 半庄评测。
3. **与 dense 效率奖励组合**：如果 GRP 有效，再尝试在 discard/副露决策上加入 `efficiency_reward`（代码已存在，目前只统计不训练），对比消融。
4. **如果平台期仍在**：再考虑 GRU 版 GRP、自博弈对手池，而不是继续堆 reward。

## 7. 结论

可行，建议作为「低成本第一个实验」而不是最终方案。它解决的是「终局排名收益的分摊」，不解决「小局内信用分配」。如果要引入，优先在现有 PPO worker 上做 GRP reward vs point-delta reward 的 A/B 实验，并用 320 半庄 2v2 对 SFT 的胜率与 95% CI 作为最终判定标准。
