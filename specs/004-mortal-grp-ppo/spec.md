# Feature Specification: Mortal 式 GRP 纯奖励 PPO(V17 PPO Rework)

**Feature Branch**: `004-mortal-grp-ppo`

**Created**: 2026-08-19

**Status**: Draft

**Input**: 基于 V16 实现下一版 PPO 训练方案。尽量保持现有工程结构,不做无关重构。
新方案要点:①GRP 改为 Mortal 方案(7 维输入、24 类排列预测、GRU、离线训练、
冻结);②PPO 使用纯 GRP reward(基于排名 utility [1, 1/3, -1/3, -1]),删除小局
点差 reward;③每 update 收集 512 个完整半庄,rollout 停止条件改为完整半庄数,
2 GPU DDP,effective global minibatch 3072(1536/GPU),`update_epochs=2`;④PPO
超参:`actor_lr=2e-5`、`shared_lr=5e-6`、`critic_lr=4e-5`、`ppo_clip=0.2`、
`target_kl=0.01`、`max_grad_norm=0.5`、`entropy_start=0.01`、`entropy_end=0.005`、
`critic_bootstrap_updates=2`、`total_updates=100`;⑤Q-Boosting 保留但适度减弱
(`q_boost_coef=0.05`、`q_boost_lambda=1.0`、`q_temperature=1.5`、`top_k=3`);
⑥SFT KL 作为中等偏弱长期 anchor(`sft_kl_coef_start=0.002`、middle=0.0005、
end=0.001),并继续记录 SFT reference KL;⑦对手只使用 current self-play;⑧中途
评测每 5 updates 保存 checkpoint 并对 V16 SFT 1v3 评测 4000 个完整半庄
(u005..u100 全部评测),最终选择 1V3 SFT 表现最佳的 checkpoint;⑨TensorBoard
重点监控 normalized entropy、PPO approx KL/clipfrac、SFT reference KL、actor/
critic/Q loss、Q explained variance、GRP reward、advantage/return、grad norm。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Mortal 式 GRP 离线训练与冻结 (Priority: P1)

作为模型训练维护者,我希望 GRP 采用 Mortal 方案重新实现:输入仅 7 维
`[grand_kyoku, honba, kyotaku, score0/10000, score1/10000, score2/10000,
score3/10000]`,对单个 StartKyoku 生成全半庄 prefix 并预测最终四人排序的 24 类
排列;模型 `7 → 2-layer GRU(hidden=64) → concat hidden → Linear(128,128) → ReLU
→ Linear(128,24)`;训练 `batch_size=512`、AdamW、`lr=1e-5`,保存 validation loss
最低的 checkpoint;实现 `calc_matrix()` 把 24 类联合概率转换为 `[4,4]` 玩家排名
概率矩阵。GRP 独立离线训练,训练完成后完全冻结供 PPO 只读。

**Why this priority**: GRP 是 PPO 的唯一奖励信号来源,输入契约、模型结构与
calc_matrix 语义的正确性直接决定后续所有训练结论,必须先落地并经过验证。

**Independent Test**: 构造少量半庄样本做 GRP 正向/反向与 calc_matrix 单测;单机
小数据集跑通训练并产出冻结 checkpoint;验证 prefix 版式结构与原 V16 样本格式
的差异不影响 rewrite 完整性。

**Acceptance Scenarios**:

1. **Given** 任意半庄的边界序列, **When** 构造 7 维输入并训练, **Then** 模型
   输出 24 类 logits,`calc_matrix` 输出 `[4,4]` 概率且行和为 1。
2. **Given** 训练后的 GRP, **When** 执行 `freeze()`, **Then** 全部参数
   `requires_grad=False` 且 PPO 阶段只读不更新。

---

### User Story 2 - 纯 GRP Reward 与每小局 Credit Assignment (Priority: P1)

作为模型训练维护者,我希望 PPO 使用纯 GRP reward(删除小局点差 reward):排名
utility 为 `[1, 1/3, -1/3, -1]`;每小局 reward = 下一小局开始时 expected rank
utility − 本小局开始时 expected rank utility;最后一局 = 真实最终排名 utility −
最后一局开始时 GRP expected utility。保持 Mortal/Suphx 风格的每小局 credit
assignment。

**Why this priority**: 奖励信号是策略学习的直接目标,删除点差分量后 GRP delta
成为唯一 reward,行为与 Mortal 一致,直接改变 PPO 学到的目标。

**Independent Test**: 单元测试验证 reward 公式(δ = V_{k+1} − V_k)、终局使用真实
排名 utility、边界调用次数 = 边界数 × 4、(PPO 侧)不再出现 score delta 分量。

**Acceptance Scenarios**:

1. **Given** 一个完整半庄的边界序列, **When** 计算纯 GRP reward, **Then**
   每个非终局边界的 reward 恰为 `V(boundary_{k+1}) − V(boundary_k)`,终局为
   `U(真实排名) − V(终局前)`。
2. **Given** 奖励装配合约, **When** 统计 GRP 调用, **Then** 调用数 = 边界数
   × 4(首局 4 + 每小局边界 4),与动作数量无关。

---

### User Story 3 - 512 半庄/Update 与 Global Batch 3072 (Priority: P2)

作为模型训练维护者,我希望每个 update 收集 512 个完整半庄(rollout 停止条件由
`kyokus_per_worker` 改为完整半庄数),2 GPU DDP 训练,`minibatch_size=1536/GPU`,
effective global minibatch 3072,`update_epochs=2`。若 1536/GPU 因最长序列导致
显存不足,允许使用 gradient accumulation 保持 global effective batch ≈ 3072,
不得直接退回小 batch。

**Why this priority**: 更大的半庄数降低每 update 方差,更大的 global batch 提高
梯度质量;这是本方案的核心训练特征,与 batch 相关的显存/padding 优化需要在此
故事内完成。

**Independent Test**: 冒烟/单元验证:worker collect 返回的统计含完整半庄计数;
DDP 分片后每 rank 本地 minibatch 为 1536;两 rank 合计 global effective batch
3072。

**Acceptance Scenarios**:

1. **Given** 双卡 DDP 配置, **When** 每 GPU `minibatch_size=1536`, **Then**
   每次 optimizer step 的有效样本量 = 3072(两 rank 各 1536,collective 对齐)。
2. **Given** 显存不足的边界条件, **When** 启用 gradient accumulation, **Then**
   单卡仍按 1536 累积多步 backward 后做一次 step,global effective batch ≈ 3072。

---

### User Story 4 - 4000 半庄 1V3 中途评测与 Best-Checkpoint 选择 (Priority: P2)

作为模型训练维护者,我希望每 5 updates 保存 checkpoint 并执行「当前 PPO 1 座位
vs V16 SFT 3 座位」的 1v3 评测,每次 4000 个完整半庄(u005、u010、…、u100 全部
评测),保存平均分差、一位率、四位率、Top2 率、平均顺位、和牌率、放铳率等现有
指标;训练结束后选择 1V3 SFT 表现最佳的 checkpoint,而非默认最后一个。

**Why this priority**: 每 5 updates 的高频评测能捕捉早期能力的快速变化,是
u100 短程训练选择最佳策略的判据;评测量 4000 半庄保证小样本噪音可控。

**Independent Test**: 对少量半庄跑通 1v3 评测脚本;验证 shard 计划(如 10 进程
× 400)与 4000 总半庄;验证 best-checkpoint 选择逻辑读取评测 JSON 汇总。

**Acceptance Scenarios**:

1. **Given** 100 updates 训练完成, **When** 检查 checkpoint 目录, **Then**
   存在 u005..u100 共 20 个 checkpoint 及对应 1v3 评测 JSON。
2. **Given** 全部评测汇总, **When** 按 1v3 SFT 表现排序, **Then** 输出表现最佳
   的 checkpoint 路径作为最终产物。

---

### User Story 5 - 现状与 TensorBoard 监控 (Priority: P3)

作为模型训练维护者,我希望 TensorBoard 继续记录 normalized entropy、PPO approx
KL / clipfrac、SFT reference KL、actor / critic / Q loss、Q explained variance、
GRP reward、advantage / return、grad norm,以判断 ① entropy 不再次快速下降到
~0.03;② SFT reference KL 是否持续无上限增长;③ Q explained variance 是否足够
支持 Q-Boosting;④ 4000 半庄 1V3 是否继续提升。

**Why this priority**: 上轮 V16 的熵坍缩(0.229→0.110→0.050→0.025)与 KL 漂移
(0.203→0.31)是主要失败模式,继续记录这些指标是判读训练健康度的前提。

**Independent Test**: 冒烟训练后检查 TensorBoard event 文件与 metrics jsonl 是否
包含上述指标键。

**Acceptance Scenarios**:

1. **Given** 一次冒烟训练, **When** 检查运行时指标, **Then** 上述全部监控键
   存在且数值有限。
2. **Given** 训练历史, **When** 观察 normalized entropy 曲线, **Then** 能明确
   判断其是否再次快速下降到 ~0.03。

---

### Edge Cases

- 半庄数统计与对齐:环境并行推进各桌时可能因在途结算小幅超额,worker 的完整
  半庄计数必须精确(exact game count),并在统计中报告实际半庄数,防止 DDP
  分片下两 rank 处理量失衡。
- 显存不足:1536/GPU 因最长序列导致 OOM 时,启用 gradient accumulation 保持
  global effective batch ≈ 3072,禁止退回小 batch;需在配置中显式声明。
- validation loss 最低 checkpoint:GRP 训练必须保存 validation loss 最低的
  checkpoint(而非最后一轮),并在训练脚本内记录该 loss 以便追溯。
- 24 类排列标签:tie(同分)时四人排名按座位号稳定排序,permutation 映射需与
  该排序一致。
- 4000 半庄评测的设备分摊:10 进程 × 400 = 4000,进程数必须仍与机制常量一致
  (宪法原则 IV 修订后允许 400 半庄/进程)。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: GRP 模型输入必须为 7 维 `[grand_kyoku, honba, kyotaku, s0/10000,
  s1/10000, s2/10000, s3/10000]`;grand_kyoku 编码 E1=0、S4=7(与 Mortal 一致)。
- **FR-002**: GRP 模型结构必须为 `7 → 2-layer GRU(hidden=64) → concat hidden →
  Linear(128,128) → ReLU → Linear(128,24)`,24 类对应 4! 排列。
- **FR-003**: GRP 训练必须使用 `batch_size=512`、AdamW、`lr=1e-5`(可经配置
  覆盖),并保存 validation loss 最低的 checkpoint。
- **FR-004**: GRP 必须实现 `calc_matrix()`,把 24 类联合概率转换为 `[4,4]`
  玩家排名概率矩阵(`matrix[player, rank]`)。
- **FR-005**: GRP 训练完成后必须完全冻结;PPO 训练期只读,不更新 GRP 参数。
- **FR-006**: PPO reward 必须为纯 GRP delta reward,排名 utility 为
  `[1, 1/3, -1/3, -1]`;删除小局点差 reward 分量。
- **FR-007**: 每小局 reward = 下一小局开始时 expected rank utility − 本小局
  开始时 expected rank utility;最后一局 = 真实最终排名 utility − 最后一局开始时
  GRP expected utility。
- **FR-008**: 每个 update 必须收集 512 个完整半庄;rollout 停止条件为完整半庄数
  而非 `kyokus_per_worker`。
- **FR-009**: 2 GPU DDP;`minibatch_size = 1536 / GPU`;effective global minibatch
  = 3072;`update_epochs = 2`。
- **FR-010**: 若 1536/GPU 因最长序列导致显存不足,必须使用 gradient accumulation
  保持 global effective batch ≈ 3072,不得直接退回小 batch。
- **FR-011**: PPO 超参:`actor_lr=2e-5`、`shared_lr=5e-6`、`critic_lr=4e-5`、
  `ppo_clip=0.2`、`target_kl=0.01`、`max_grad_norm=0.5`、`entropy_start=0.01`、
  `entropy_end=0.005`、`critic_bootstrap_updates=2`、`total_updates=100`;不得因
  batch 增大而线性放大学习率。
- **FR-012**: Q-Boosting 保留:`q_boost_coef=0.05`、`q_boost_lambda=1.0`、
  `q_temperature=1.5`、`top_k=3`。
- **FR-013**: SFT KL anchor 为中等偏弱长期 anchor:`sft_kl_coef_start=0.002`、
  `middle=0.0005`、`end=0.001`;继续记录「SFT reference KL」指标。
- **FR-014**: 对手只能使用 current self-play;不使用 SFT opponent、historical
  opponent、random/heuristic opponent。
- **FR-015**: 每 5 updates 保存 checkpoint 并执行 1v3(PPO 1 座位 vs V16 SFT 3
  座位)评测 4000 个完整半庄(u005..u100 全部评测);保存平均分差、一位率、四位
  率、Top2 率、平均顺位、和牌率、放铳率等现有指标。
- **FR-016**: 训练结束后选择 1V3 SFT 表现最佳的 checkpoint,而非默认最后一个。
- **FR-017**: TensorBoard 继续记录 normalized entropy、PPO approx KL/clipfrac、
  SFT reference KL、actor/critic/Q loss、Q explained variance、GRP reward、
  advantage/return、grad norm。

### Key Entities

- **GRP 模型**:7 维输入前缀序列 → 24 类排列 logits → `calc_matrix` → `[4,4]`
  排名概率;离线训练、冻结、PPO 只读。
- **GRP 排名 Utility**:`[1, 1/3, -1/3, -1]`(由 24 类概率经 calc_matrix 聚合)。
- **PPO Transition**:现有 V16 Transition 结构不变,reward 字段改为纯 GRP delta。
- **1v3 评测快照**:每 5 updates 一个 `vs_sft_uNNN.json` 汇总,含 first_place_rate、
  top2_rate、fourth_place_rate、mean_rank、point_diff、kyoku_metrics。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: GRP 单测通过:24 类输出、calc_matrix 行和=1、4 视角输入等价
  (同分稳定排序下的排列标签一致)。
- **SC-002**: GRP 训练全程可复现:`python -m riichi_ppo_v1.training.grp.train
  --config <v17 grp 配置>` 产出 `best.pt`(validation loss 最低)与
  `config_snapshot.json`。
- **SC-003**: 纯 GRP reward 单测通过:δ = V_{k+1} − V_k、终局真实排名、无点差
  分量,GRP 调用数 = 边界数 × 4。
- **SC-004**: 512 半庄/update 冒烟通过:worker 统计含精确完整半庄计数;每 GPU
  minibatch 1536;global effective batch 3072(或 gradient accumulation 等价)。
- **SC-005**: 100 updates 训练完成后存在 u005..u100 共 20 个 checkpoint 与对应
  4000 半庄 1v3 评测 JSON;best-checkpoint 选择逻辑输出 1V3 SFT 表现最佳者。
- **SC-006**: TensorBoard/metrics jsonl 包含 FR-017 全部监控键且数值有限。
- **SC-007**: 对手配置只含 current self-play(opponent_mix 关闭或 current_frac=1)。

## Assumptions

- 版本命名:新方案作为 V16 的下一迭代,按项目惯例将 checkpoint、日志、评测产物
  落到 `train_riichi_v17`/`logs/v17`/`audit/reports/v17`,但信息编码协议保持 v16
  (不引入新输入编码),token schema 仍为 13。
- 1v3 评测机制常量(进程数 10、间隔、总半庄数)属于宪法原则 IV;本方案要求
  4000 半庄/次、每 5 updates,需经 `$speckit-constitution` 修订后落地。
- GRP 采用 Mortal 的「一个半庄所有全 prefix」训练形式;现有 V16 GRP 数据集
  (4 视角 × chunk npz)可复用编码路径,但输入契约与 24 类排列标签需要新增,
  新数据集目录为 `datasets/tenhou_grp_2024_2025_v17`(保持 40% 子集与
  train/validation 划分规则)。
- 最终最佳 checkpoint 由「1V3 vs SFT 平均分差(point_diff_vs_mean_opponent_mean)」
  排序选出;若指标缺失则回退到 mean_rank。