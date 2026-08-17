# Feature Specification: V16 模型重构与训练(V16 Model Rework)

**Feature Branch**: `V16`

**Created**: 2026-08-15

**Status**: Draft

**Input**: 在 V16 分支上完成 V16 版本开发。唯一业务/技术权威来源为
`audit/reports/v16/design/V16 网络结构与训练方案.md`(以下简称「设计文档」),
涵盖网络结构与参数量、Actor/Critic 输入、统一 Action Query Schema、策略头融合、
Top-3 Q-boosting、GRP 模型与奖励、训练流程等全部具体设计。训练数据来自
tenhou-to-mjai 的 2024/2025 Tenhou 对局,现行原始数据目录
`datasets/tenhou_sft_2024_2025`、编码数据目录
`datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16`;V16 的 SFT 重编码与 GRP
数据集均从该来源构造并按命名规范版本化。开发流程为 spec-kit 全链
(specify→clarify(按需)→plan→tasks→implement→analyze/converge),三件套存
`specs/<NNN-name>/`。约束:①旧代码可删(先全仓 rg 零引用+测试通过,每主题一个
commit,checkpoint 与数据集不删,v11 权重/v14 资产冷存储);②新增模型输入协议文档
并同步宪法 Principle II 版本契约(经 `$speckit-constitution` 修订);③新局况分析
函数按 state-machine/core/模型输入侧归属并写明理由;④全部 action query slot 必须
与独立 oracle 逐项比对的语义正确性硬门槛;⑤工程治理(目录职责、领域常量单一来源、
自包含配置、产物路径、评测机制不变、性能基线)。完成判定:语义测试全通过;协议与
实现一致且版本经宪法登记;旧代码零引用清理+全仓测试通过;README/docs 同步、评测
机制未被悄悄改动。

## Clarifications

### Session 2026-08-16

- Q: 新的信息编码协议用哪个单一版本号? → A: v16(与 V16 实验代对齐;输入编码收敛
  为单一协议版本,废弃 `v14_v16` 之类的组合命名与多版本拆分)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 统一 Action Query 输入契约与语义落地 (Priority: P1)

作为模型训练维护者,我希望 V16 的 Actor 输入按「Objective Facts + Compact
Snapshot + 每个合法动作一对 Offense/Defense Query」重新编码,删除全部 Derived
Features,并把 20 个 query slot 的语义、N/A 规则与 bucket 边界逐项实现、验证,使
进入模型的输入与局面上实际发生/计算的事实完全一致。

**Why this priority**: 这是 V16 模型、SFT、PPO 全部后续工作的输入地基;任一 slot
语义错误都会污染整个训练结论,且语义正确性是本任务的硬性验收门槛,必须最先落地。

**Independent Test**: 在抽样局面与全部边界局面清单上,对每个合法动作用独立 oracle
重算 20 个 slot 并与编码器输出逐项比对,100% 一致;验证不依赖模型训练即可完成。

**Acceptance Scenarios**:

1. **Given** 任意局面与任一合法动作, **When** 编码 Offense/Defense 两个 Query,
**Then** 20 个 slot 的取值、N/A 规则与独立 oracle 重算结果逐项一致。
2. **Given** 删除 Derived Features 后的编码器, **When** 检查 Actor 输入,
**Then** 不含任何隐藏信息(三家手牌、牌山),特权信息只出现在 Critic 输入。
3. **Given** 已声明的全部 bucket 基数, **When** 全量语义测试, **Then** 所有
categorical 因子取值均在 cardinality 范围内,越界样本数为 0。
4. **Given** 环境局面与编码 tensor, **When** 执行回放/桥接测试, **Then** 局面上
实际发生的事实与 tensor 内容一致。

---

### User Story 2 - V16 网络结构与对称策略头 + SFT 从头训练 (Priority: P1)

作为模型训练维护者,我希望 Actor-Critic 扩容到设计目标(d_model=256、16/4 GQA、
FFN=1088、4 Shared + 1 Actor、4 Shared + 2 Critic,总参数约 7.5–7.8M、Actor 推理
约 5.3M),策略头由 Offense/Defense 对称融合输出且无 zero-init,并能以重编码数据集
从头完成 SFT。

**Why this priority**: 容量提升与攻守对称是 V16 的核心设计意图;输入 Schema、
hidden 大小、Transformer 深度、查询融合全部改变后,旧 SFT 参数不可继承,必须从头
训练,是 PPO 的前置条件。

**Independent Test**: 构造模型后统计参数量并在容差内;检查策略头融合结构;以新编码
数据集跑通 SFT 冒烟并产出带配置快照的 checkpoint;记录验证集 Recall@3。

**Acceptance Scenarios**:

1. **Given** V16 网络配置, **When** 统计参数, **Then** 总参数在 7.5–7.8M 容差
范围、Actor 推理参数约 5.3M,不采用 288/320 等更大 hidden。
2. **Given** 任一合法动作的 Offense/Defense 表示, **When** 前向传播, **Then**
二者经对称融合(concat→Linear 512→256→SiLU→Policy MLP)输出唯一 logit,且不存在
zero-init 分支。
3. **Given** 新编码数据集, **When** 从头启动 SFT, **Then** 训练可进行、按 3000
steps 节奏保存 checkpoint(内含配置快照),验证集 Recall@3 实际值记录到
`audit/reports/v16/report/PROGRESS.md`。

---

### User Story 3 - GRP 模型、数据集与离线训练 (Priority: P1)

作为模型训练维护者,我希望从 Tenhou 数据构造 GRP 数据集(每半庄 4 个
player-relative 旋转视角、每个 prefix 监督该视角最终排名),训练轻量 GRP 模型
(约 50–70K 参数),训练完成后完全冻结供 PPO 使用。

**Why this priority**: GRP 是 V16 奖励的主体(70%),必须先于 PPO 离线训练并冻结;
其质量直接决定 PPO 的学习目标,且与 Actor 输入契约相互独立,可并行落地。

**Independent Test**: 构造数据集并校验旋转视角与 prefix 标签;训练 GRP 至损失下降
且排名预测显著优于均匀随机;验证只在半庄序列的小局边界执行一次;权重冻结后 PPO
不更新。

**Acceptance Scenarios**:

1. **Given** 一个完整半庄, **When** 构造 GRP 样本, **Then** 生成 4 个视角样本,
每个 prefix 的监督标签为该视角玩家的最终排名,首局的 previous-result 为 START。
2. **Given** GRP 模型, **When** 离线训练, **Then** 总参数 50–70K,验证集排名预测
显著优于均匀随机,checkpoint 含配置快照。
3. **Given** PPO 运行, **When** 更新网络参数, **Then** GRP 权重保持不变(冻结)。

---

### User Story 4 - PPO 集成:Top-3 Q-boosting 与 GRP+Score 奖励 (Priority: P2)

作为模型训练维护者,我希望 PPO 以 70% 归一化 GRP delta + 30% 归一化小局分差为
奖励,Critic 使用特权信息并通过 Top-3 Q-boosting 辅助 Actor,训练可稳定跑通。

**Why this priority**: 这是 V16 训练方案的最终闭环;依赖前面三项(输入契约、SFT、
GRP),但其机制(候选集、奖励归一化、detach)必须独立验证正确。

**Independent Test**: 以性能基线参数(target_kl=0.0、update_epochs=4、
kyokus_per_worker=16)跑通短训练;检查 Q scorer 训练候选 = Top-3 ∪ 实际行为动作
(最多 4 个)、boost 候选 = Top-3、动作表示 detach;奖励按离线固定的归一化统计量
计算。

**Acceptance Scenarios**:

1. **Given** Actor 概率分布, **When** 执行 Q-boosting, **Then** 只对 Top-3(训练时
并集实际行为动作,最多 4 个)打 Q,动作表示 detach,Q loss 不直接更新 Actor。
2. **Given** 一局轨迹, **When** 计算奖励, **Then** R = 0.7·clip(R_GRP/σ_GRP,
±5) + 0.3·clip(clip(Δscore/1000,±12)/σ_Score,±5),σ 为离线固定值,终局使用真实
排名 utility [12,4,-6,-10]。
3. **Given** 性能基线配置, **When** 跑 3 轮, **Then** 首轮视为预热、后两轮单独
报告,打印耗时监控与全部相关性能指标,冒烟结束后删除其日志与结果文件。

---

### User Story 5 - 治理闭环:协议、版本、清理与文档 (Priority: P2)

作为维护者,我希望 V16 的模型输入协议文档、schema/编码版本、宪法契约、旧代码
清理、常量收敛、自包含配置、产物路径与评测机制全部合规,交付可回滚、可追溯。

**Why this priority**: 任务完成判定与宪法原则的硬性要求;治理缺口会导致契约漂移或
产物散落,削弱版本可追溯性。

**Independent Test**: 删除目标 rg 零引用且全仓测试通过;协议文档与实现一致;宪法
修订完成并记录;评测机制常量无 diff;README/docs 路径同步。

**Acceptance Scenarios**:

1. **Given** 删除清单, **When** 执行清理, **Then** 删除前 rg 零引用、测试通过、
每主题一个 commit、checkpoint 与数据集未改动,v11 权重/v14 资产仅冷存储保留。
2. **Given** 新输入协议, **When** 对照实现, **Then** 协议文档、契约常量(输入编码
协议 v16)与代码一致,宪法 Principle II 已修订并记录 Sync Impact Report。
3. **Given** 评测机制常量, **When** 检查差异, **Then** PPO 1v3(10 进程 ×160 =
1600 hanchan、每 30 updates)与 SFT(每 3000 steps 验证/保存、最终 96 hanchan)
仍为单点定义,未被实验配置复制或悄悄改动。

---

### Edge Cases

- 终局动作(自摸/荣和):进攻与防守 slot 按 Assumptions 的终局约定取值,不得编码
  「动作后的下一决策状态」。
- 流局动作(九种九牌等):Offense 按动作后手牌现状计算,听牌相关 slot 在非听牌时
  为 N/A,O8=N/A,D0–D5=N/A。
- 无实际打牌动作(吃/碰/杠):D0–D5=N/A;D6–D8 按动作后手牌现物库存计算;D9 为主牌
  的公开出现数。
- 立直宣告本身是打牌:现物/筋按宣告牌对三家计算;O8=N/A(已立直),听牌相关 slot
  按立直后手牌计算。
- 非听牌状态:O3/O4/O5/O6=N/A。
- 振听三分支(无/永久/临时)与部分等待无役(O4=PARTIAL_YAKU)。
- 有效牌全部可见耗尽:O1=0、O2=0。
- 宝牌/赤牌数大于 5 → 5+;安全牌库存大于 4 → 4+;公开出现数第 4 张与 N/A 情形。
- 立直巡目、本场、立直棒、剩余牌数的边界值(0 本场、立直棒归零、牌山耗尽)。
- 局末 all-last 与供托、流局听牌(tenpai mask)、庄家连庄在 GRP 输入中的正确编码;
  流局时 winner/deal-in 为 N/A。
- GRP 首局 START 标记、超长连庄序列的 padding 一致性。
- 每个合法动作恰好一对 query(不重不漏),query 与 action_id 一一对应。
- 自身与三家相对分差的符号一致性,以及 Actor 视角与 GRP 旋转视角的一致性。

## Requirements *(mandatory)*

### Functional Requirements

#### 网络结构与容量

- **FR-001**: V16 Actor-Critic MUST 沿用 GQA + gated FFN,配置为 d_model=256、
  Q heads=16、KV heads=4、head dim=16、FFN=1088、Shared Transformer=4、
  Actor-only=1、Critic-only=2;Actor 路径 5 层、Critic 路径 6 层。
- **FR-002**: 模型总参数量 MUST 约为 7.5–7.8M,Actor 推理参数约为 5.3M;MUST NOT
  继续把 hidden 拉到 288/320。

#### Actor 输入结构

- **FR-003**: Actor 输入 MUST 统一划分为 Objective Facts + Compact Snapshot +
  每个合法动作的 Offense/Defense Query。
- **FR-004**: Objective Facts MUST 保留现有原始事实输入(自己当前手牌、历史事件、
  摸牌、打牌、chi/pon/kan、reach/reach accepted、和牌/流局等基础客观事件);历史
  事件继续承担完整公共信息记录职责。
- **FR-005**: 现有全部 Derived Features MUST 被删除;Snapshot MUST NOT 再包含三家
  完整牌河与三家完整副露序列的重复表示(该信息已在 Historical Events 中)。
- **FR-006**: Snapshot 的基础场况 MUST 包含场风、局数、庄家、本场、立直棒、剩余
  牌数、宝牌指示、四家当前点数、当前顺位。
- **FR-007**: Snapshot MUST 提供自身排名压力信息:自身点数与其余三家点数的三个
  相对分差(或等价表示)。
- **FR-008**: Snapshot MUST 为三个对手各提供固定 7 个 Summary Token(是否立直、
  立直巡目、副露数、是否门清、舍牌数、手切次数、摸切次数),共 3×7=21 个;详细牌河
  与副露仍从 Historical Events 读取。

#### 统一 Action Query Schema

- **FR-009**: 每个合法动作 MUST 固定生成一对 Offense Query + Defense Query,两个
  Query 使用完全相同的结构:query_type、action_id、action_type、primary_tile、
  source_seat、answer_0…answer_9,共 10 个 Question Answer Slot。
- **FR-010**: 每个 slot MUST 使用独立的 categorical/bucket embedding;Query
  embedding MUST 聚合为单个 Transformer token(E_action + E_queryType +
  ΣE_{type,i}(answer_i),经 LayerNorm/Projection 到 d_model=256),不得因问题增多而
  增加 sequence length。
- **FR-011**: Offense Query MUST 回答「执行该动作后我的手牌进攻状态」,固定 10 问:
  O0 动作后向听数;O1 动作后有效牌种类数;O2 动作后有效牌剩余总枚数;O3 听牌时合法
  等待牌种类数;O4 当前等待是否有役;O5 当前等待基础番数范围;O6 动作后是否振听;
  O7 动作后是否门清;O8 动作后是否满足立直条件;O9 动作后手中保留的宝牌/赤牌数。
- **FR-012**: Defense Query MUST 回答「执行该动作后面对三个对手的防守性质」,固定
  10 问:D0/D1/D2 候选打牌是否分别为对手 1/2/3 的现物;D3/D4/D5 候选打牌是否分别对
  对手 1/2/3 构成筋关系;D6/D7/D8 动作后手中对手 1/2/3 现物的剩余张数;D9 当前候选
  牌已公开出现的张数。
- **FR-013**: Query 只允许包含能从当前真实局面确定计算出的客观答案(向听、有效牌、
  等待、有无役、振听、现物、筋、公开牌数、安全牌库存);MUST NOT 输入预计和率、
  预计放铳率、预计 EV、预计最终得点、人工危险度、手牌综合评分等预测性结论。
- **FR-014**: 每个 slot 的取值域、N/A 规则与 bucket 边界 MUST 唯一确定并收敛为
  单一命名常量、单一来源;设计文档给定值域如下,未写死的部分按 Assumptions 拍板:
  O0={AGARI,0,1,2,3,4,5+}、O4={N/A,NO_YAKU,PARTIAL_YAKU,ALL_YAKU}、
  O5={N/A,1,2,3,4,5+}、O6={N/A,NO_FURITEN,PERMANENT_FURITEN,TEMPORARY_FURITEN}、
  O7={YES,NO}、O8={YES,NO,N/A}、O9={0,1,2,3,4,5+}、D0–D5={GENBUTSU/SUJI,
  NOT_GENBUTSU/NOT_SUJI,N/A}、D6–D8={0,1,2,3,4+}、D9={0,1,2,3,4,N/A}。

#### 策略头与 Critic

- **FR-015**: 策略头 MUST 让 Offense/Defense 完全对称地参与输出:
  h_a=f([h^offense_a;h^defense_a]),经 concat→Linear 512→256→SiLU→Policy MLP→
  logit(a);MUST NOT 对 Offense 分支做零初始化;普通初始化并从头参加 SFT。
- **FR-016**: Critic MUST NOT 增加 Action Query Token;输入保持 Encoded Objective
  Facts + Snapshot + Score Pressure + Opponent Summary + 三家对手手牌 + 后续 5 张
  牌山 + Value Query,经 4 Shared → 2 Critic → Value Query Hidden。

#### Top-3 Q-boosting

- **FR-017**: Actor MUST 先生成全部合法动作概率 π(a|s) 并取 Top-1/2/3;Critic Q
  scorer 只评估候选,输入为 [z_critic; h_a],其中动作表示 h_a MUST detach,避免
  Q loss 经动作表示直接更新 Actor;scorer 结构为 256+256→512→Linear
  512→256→SiLU→Linear 256→1,输出原始优势评分 u_i。
- **FR-018**: Dueling 约束 MUST 为:候选集合内把 Actor 概率重新归一化为 p_i,
  `A_i = u_i - Σ p_j·u_j`,`Q_i = V(s) + A_i`,由构造恒有 `Σ p_i·Q_i = V(s)`;
  Value 只负责绝对局面价值,Top-3 Q 只编码候选间的相对差异。
- **FR-019**: Critic 训练候选 MUST 为 Top-3 ∪ 实际 rollout 行为动作(最多 4 个
  Q);只有行为动作的 Q(s,a_t) 回归到与 Value 相同的 rollout return 目标,未执行
  的候选 MUST NOT 构造虚假 Q target。真正对 Actor boosting 的候选 MUST 为 Top-3:
  `p_boost ∝ p_i·exp(λ_q·A_i/T)` 且保持 Top-3 原始总概率质量不变,`p_boost.detach()`
  作为 Top-3 交叉熵辅助蒸馏目标(权重 q_boost_coef),MUST NOT 替代 PPO policy loss。

#### GRP 奖励与归一化

- **FR-019**: V16 PPO 奖励 MUST 为 R = 0.7·R̂_GRP + 0.3·R̂_Score,且 MUST 在归一化
  后组合,不得对原始数值直接按 70/30 加权。
- **FR-020**: GRP 排名 utility MUST 为 [12,4,-6,-10](和为 0,加大争一、抬高三位
  代价、四位最差);GRP 输出 P(rank=1..4),V_GRP=12P1+4P2-6P3-10P4;小局 GRP
  reward R^GRP_k=V_{k+1}-V_k;最后一局结束后 MUST 使用真实最终排名的 utility,而非
  GRP 预测。
- **FR-021**: 归一化统计量 σ_GRP 与 σ_Score MUST 在训练数据上离线统计一次后固定:
  R̂_GRP=clip(R_GRP/σ_GRP,-5,5),R_Score=clip(Δscore/1000,-12,12),
  R̂_Score=clip(R_Score/σ_Score,-5,5);训练过程中 MUST NOT 动态修改这两个标准差。

#### GRP 模型与训练

- **FR-022**: GRP MUST 保持轻量:输入特征→Linear 64→2 层 GRU(hidden=64)→Linear
  64→32→SiLU→Linear 32→4→Rank Softmax,总参数约 50–70K。
- **FR-023**: GRP MUST 只在每个小局边界执行一次,不得每个 action 执行一次。
- **FR-024**: GRP 输入 MUST 按 SELF/RIGHT/ACROSS/LEFT 把每个玩家旋转到统一视角;
  一条 GRP sequence 对应完整一个半庄;每个小局 boundary 输入当前比赛状态(四家点数、
  自身与其他三家分差、自身排名、场风/局数、庄家相对位置、honba、立直棒)与上一小局
  结果(四家 score delta、结果类型、winner seat、deal-in seat、流局 tenpai mask、
  庄家是否连庄),第一小局使用 START;类别字段用小型 embedding、点数与分差用连续值
  归一化,拼接约 40–60 维后投影到 64;MUST NOT 加入完整手牌、牌河、牌山等高维信息。
- **FR-025**: GRP 训练集 MUST 从 Tenhou 2024+2025 数据(与 SFT 相同的 40% 采样
  划分)构造;每个完整半庄旋转出 4 个 player-relative samples;每个 prefix 监督该
  视角的最终排名,损失为 prefix 平均 CE(P_φ(rank|s_{0:k}),rank_final);训练完成后
  GRP MUST 完全冻结,PPO 不得更新 GRP。

#### 训练流程与数据

- **FR-026**: V16 训练流程 MUST 为:Tenhou 2024/2025 原始数据 → SFT 重编码数据集
  与 GRP 数据集 → V16 Actor SFT(从头)+ GRP 训练并冻结 → PPO(GRP+Score 奖励、
  特权 Critic、Top-3 Q-boost)。
- **FR-027**: 因 hidden 大小、Transformer 深度、输入 Schema、Derived Features、
  Summary Token、Query Schema、Query Fusion 全部改变,V16 MUST 从头重新训练 SFT,
  不得继承旧 SFT 参数。
- **FR-028**: 新 SFT 编码数据集与 GRP 数据集 MUST 从
  `datasets/tenhou_sft_2024_2025` 构造并按仓库命名规范版本化存放(命名见
  Assumptions);现有数据集只保留不删除。
- **FR-029**: V16 checkpoint MUST 保存到 `checkpoints/train_riichi_v16/` 的阶段
  子目录(如 `sft`、`grp`、`ppo`),每个 checkpoint 内含配置快照;运行日志 MUST 写入
  `logs/v16/`;训练启动与 Ray 运维沿用现行规范(前台、任意目录、日志落盘、
  `RAY_LOG_TO_STDERR=0`,结束后 `ray stop --force`)。

#### 语义与业务正确性(硬性门槛)

- **FR-030**: 每个 action query 的每个 slot MUST 与独立 oracle 重算比对,编码器
  不得自证;任一 slot 与独立 oracle 不一致即视为失败。
- **FR-031**: 测试 MUST 覆盖设计文档定义的全部 slot 语义、N/A 规则、bucket 边界
  与边界局面;MUST 用回放/桥接测试验证环境局面与编码 tensor 一致。
- **FR-032**: 验证 MUST 证明 Actor 输入不含隐藏信息,特权信息只出现在 Critic;
  MUST 验证所有 categorical 因子取值在 cardinality 范围内。
- **FR-033**: 新局况分析函数 MUST 按归属规则落位:仅由公开 MJAI 状态与自身信息可
  确定的事实放 `RiichiEnv/riichienv-state-machine`(公开模块名保持 `riichi`,不得
  依赖 `riichienv`);需要规则/手牌结构评价的事实放 `RiichiEnv/riichienv-core`,并
  优先复用既有 shanten/手牌评价/yaku 分析;仅训练侧可组合的事实放模型输入转换侧;
  每项函数在 plan 写明归属与理由,不得引入反向依赖。

#### 协议、版本与治理

- **FR-034**: MUST 新增或更新模型输入协议文档,明确 V16 模型输入协议;契约/协议
  文档 MUST 与代码实现同步(如 `KyokuEventTupleProtocol.md` 等若受影响必须同步)。
- **FR-035**: 新信息编码协议 MUST 唯一编号为 **v16**(与实验代对齐的单一版本,不
  沿用 `v<A>_v<B>` 组合命名与多版本拆分,见 Assumptions),并经
  `$speckit-constitution` 修订宪法 Principle II 的现行契约声明,记录 Sync Impact
  Report。
- **FR-036**: 旧代码清理 MUST 在删除前做全仓库 rg 引用检查,零引用且测试通过才
  允许删除;按「每主题一个 commit、测试通过、可独立回滚」执行;checkpoint 与数据集
  一律不删除,v11 权重与 v14 资产仅冷存储保留。
- **FR-037**: 领域不变常量(136 TID、34 牌类、241 动作维、各 bucket 基数等)MUST
  收敛为单一命名常量、单一来源。
- **FR-038**: 每个版本配置 MUST 自包含写在自己的文件,禁止 overlay/继承式覆盖;
  版本号、checkpoint、数据集、对手模型、schema/契约 ID、种子、计数与间隔、路径
  一律经 CLI/配置传入,默认值不得锁定历史版本;V16 配置写于专属配置文件。
- **FR-039**: 产物 MUST 按规范存放:报告与脚本到 `audit/reports/v16/`(design/
  eval/report/scripts),评测输出到 `audit/reports/v16/eval`,进度与失败记录写
  `audit/reports/v16/report/PROGRESS.md`。
- **FR-040**: 评测机制 MUST 沿用现行机制常量单点定义:PPO 1v3 固定 10 进程 ×160
  =1600 hanchan、每 30 updates 一次;SFT 验证/启发式评测/checkpoint 保存固定每
  3000 steps 一次、最终评估 96 hanchan;不得在实验配置中复制或悄悄改动,如需改动
  必须走宪法修订。
- **FR-041**: 性能与训练测试 MUST 固定 target_kl=0.0、update_epochs=4、
  kyokus_per_worker=16,CUDA_DEVICE=0,1、learner_gpus=2、Conda 环境
  `Mahjong-AI`;默认跑 3 轮、首轮视为预热并单独报告后两轮,默认打印耗时监控与全部
  相关性能指标,冒烟测试结束 MUST 删除其产生的日志与结果文件。
- **FR-042**: README、docs、AGENTS.md MUST 与代码路径同步;目录按职责放置(模型/
  契约、训练、SFT、评测、工具、测试各自归位),新组件建独立目录/子包。

### Key Entities

- **V16 Actor-Critic 模型**:容量扩到约 7.5–7.8M 的 GQA+gated-FFN 网络,含 Shared/
  Actor-only/Critic-only 层与对称策略头。
- **统一 Action Query**:每个合法动作一对(Offense/Defense),各 10 个 categorical
  answer slot,聚合成单 token;是 Actor 的动作侧输入契约。
- **Actor Snapshot**:Objective Facts + 基础场况 + Score Pressure + 3×7 Opponent
  Summary 的紧凑当前状态。
- **Critic 状态**:公共编码 + 三家对手手牌 + 后续 5 张牌山 + Value Query 的特权
  状态。
- **Top-3 Q scorer**:对 [z_critic; detached h_a] 输出最多 4 个候选 Q 值的打分器。
- **GRP 模型**:约 50–70K 参数的两层 GRU 排名预测器,每小局边界执行,PPO 中冻结。
- **GRP 样本**:完整半庄按 4 视角旋转、每个 prefix 监督最终排名的序列样本。
- **奖励组件**:归一化 GRP delta(70%)与归一化小局分差(30%)、utility
  [12,4,-6,-10]、离线固定的 σ_GRP/σ_Score。
- **V16 数据集**:SFT 重编码数据集与 GRP 数据集,均由 Tenhou 2024/2025 原始数据
  构造并版本化命名。
- **模型输入协议契约**:版本号为 v16 的新信息编码协议及其协议文档(单一版本,与
  实验代对齐)。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 在抽样局面与全部边界局面清单上,所有合法动作的全部 20 个 query slot
  与独立 oracle 重算结果 100% 一致;任一 slot 不一致即判定失败。
- **SC-002**: Actor 输入隐藏信息检查 100% 通过,特权信息只出现在 Critic 输入。
- **SC-003**: 所有 categorical 因子在声明的 cardinality 内,越界样本数为 0。
- **SC-004**: 回放/桥接一致性测试 100% 通过(环境局面与编码 tensor 一致)。
- **SC-005**: V16 总参数 7.5–7.8M(容差 ±0.3M)、Actor 推理参数约 5.3M(容差
  ±0.3M)、GRP 参数 50–70K,由参数统计测试验证。
- **SC-006**: V16 SFT 从头训练可完成,验证集 Recall@3 ≥ 98% 作为进入 PPO 的前置
  检查,实际值记录于 `audit/reports/v16/report/PROGRESS.md`。
- **SC-007**: 性能基线 3 轮短训练跑通,首轮预热、后两轮性能统计单独报告,耗时监控
  与全部相关性能指标打印,冒烟结束后无残留日志与结果文件。
- **SC-008**: 旧代码删除目标全仓库 rg 零引用,全仓库测试通过。
- **SC-009**: 新模型输入协议文档与实现差异为 0;信息编码协议版本 v16 唯一且经宪法
  修订登记(Sync Impact Report 完整)。
- **SC-010**: 评测机制常量 diff 为空(PPO 1v3 与 SFT 节奏未被改动)。
- **SC-011**: PPO 训练前后 GRP 权重逐位一致(冻结验证),GRP 调用次数等于小局边界
  数而非动作数。

## Assumptions

- **A1**: 特性短名 `v16-model-rework`,目录 `specs/003-v16-model-rework`;实现所在
  分支为 `V16`(与 spec 目录名相互独立)。
- **A2**: 新信息编码协议采用单一版本号 **v16**(与 V16 实验代对齐);不再拆分
  token schema / feature schema / encoded-format / 内部分析版本等多套编号,也不
  沿用 `_v<A>_v<B>` 遗留组合命名;宪法 Principle II 修订预期为 MINOR
  (1.3.0→1.4.0),现行契约声明更新为「信息编码协议 v16」。
- **A3**: 数据集命名拍板:SFT 重编码数据集
  `datasets/tenhou_sft_2024_2025_encoded_40pct_v16`(单一协议版本后缀,沿用现有
  40% 采样与 train/validation 划分);GRP 数据集
  `datasets/tenhou_grp_2024_2025_v16`;既有数据集只保留不删除。
- **A4**: 设计文档未写死的 bucket 边界拍板(小值精确、大值截断,且覆盖设计文档
  例示值 3/7/12/9):O1 有效牌种类数={0,1,2,3,4,5,6,7,8,9,10+}(11 类);O2 有效牌
  剩余枚数={0,1-4,5-8,9-12,13-16,17-20,21+}(7 类);O3 合法等待种类数={N/A,1,…,13}
  (14 类);其余 slot 按设计文档给定值域。
- **A5**: 终局与边缘动作的 slot 约定拍板:自摸/荣和 → O0=AGARI,O1=0,O2=0,
  O3/O4/O5/O6/O8=N/A,O7=实际门清,O9=实际保留宝牌/赤牌数,D0–D5=N/A,D6–D8=0,
  D9=主牌公开出现数;流局类动作 → Offense 按动作后手牌现状计算、非听牌时 O3–O6=
  N/A、O8=N/A、D0–D5=N/A、D6–D8 按手中现物计算、D9=N/A;立直宣告 → D0–D5 按宣告
  牌计算现物/筋、O8=N/A;吃/碰/杠 → D0–D5=N/A、D6–D8 按动作后手牌计算、D9=主牌
  公开出现数。plan 阶段以独立 oracle 表格逐项落地。
- **A6**: 删除范围拍板:全部 Derived Features(现行 actor/critic 特征与语义契约中
  的派生特征)、Snapshot 中三家完整牌河/副露重复表示、策略头 zero-init Offense
  projection 分支;具体文件级删除清单由 plan 以全仓库 rg 引用检查生成,并按主题
  分 commit。
- **A7**: 分析函数归属初判:向听、有效牌、等待、有无役、基础番数、振听、门清、
  可立直等需规则/手牌结构评价的事实 → `riichienv-core`(优先复用既有
  shanten/手牌评价/yaku 分析);现物、筋、公开出现数、安全牌库存、对手 7 项摘要、
  立直巡目等由公开 MJAI 状态与自身信息可确定的事实 → `riichienv-state-machine`;
  Score Pressure 相对分差、O9 宝牌/赤牌聚合、GRP 输入构造等仅训练侧可组合事实 →
  模型输入转换侧;最终归属与理由写于 plan,禁止反向依赖。
- **A8**: V16 checkpoint 子目录规划:`checkpoints/train_riichi_v16/sft`、
  `checkpoints/train_riichi_v16/grp`、`checkpoints/train_riichi_v16/ppo`,均含
  配置快照。
- **A9**: σ_GRP 与 σ_Score 在训练数据上离线计算一次并固化,训练过程不动态修改。
- **A10**: 评测机制常量不改动;SFT 节奏沿用 `sft.yaml` 单点定义,PPO 1v3 常量沿用
  `riichi_ppo_v1/evaluation/mechanism.py` 单点定义。
- **A11**: 设计文档未覆盖的实现细节一律以「不改变设计意图」为原则由 spec/plan
  拍板,并回溯记录到本 Assumptions;本提示词声明的约束高于其他来源。
- **A12**: checkpoint 与数据集一律只归档移动、不删除;v11 权重与 v14 资产仅冷
  存储保留,不再被当前代码引用。
