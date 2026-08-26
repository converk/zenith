# Feature Specification: V18 GRP 输入扩展与重新训练

**Feature Branch**: `009-v18-grp`
**Created**: 2026-08-26
**Status**: Planned
**Input**: 以 21 维边界状态重新训练 V18 GRP(全局排名预测),增大模型与训练数据量;保持
Mortal 式纯 GRP reward 契约(24 类排列、冻结、只读)不变。

## 范围与保护边界

- 现行契约:V18;V16/V17 的 checkpoint、数据集、配置、日志和历史报告仅作冷存储,
  不改动、不删除、不迁移。
- 本特性只动 GRP 链路(输入契约、模型结构、数据构造、离线训练、计算快照与运行时
  特征装配)。不涉及 Actor/Critic 输入、PPO 训练与 1v3 评测机制。
- 训练数据选择:GRP 使用**全量数据**(denominator=1,remainders=(0))× 全部
  930 个 train shard(先移除 v17 的 max_shards=280 截断,子集从 40% →
  66.7% → 100%,约 36 万 train 半庄);与 SFT 60% 子集完全重叠——GRP 是
  冻结的奖励模型,仅用于评分,不构成对策略的训练污染。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 边界状态输入扩展到 21 维 (Priority: P1)

作为训练数据生产者,我希望 GRP 的每个小局边界输入包含 21 维状态,其中新增的
14 维全部是"一局结束后此刻的状态"(局风类型、上一小局结果类型、各玩家累计
和了/放铳/听牌流局次数),不包含任何手牌、牌河、未来牌等局内发展信息。

**Why this priority**: 当前 7 维输入无法区分东风/半庄/西风(同一 (风,局) 处特征
完全相同,而剩余局数不同),也无法直接得知各玩家的小局结果分布;这两类信息都是
公开历史状态,加入后能改善最终排名预测。

**Independent Test**: 构造带 4 个小局(含荣和、自摸、流局听牌)的样本,断言每行
21 维、字段顺序正确、累计计数随边界推进正确;局风类型由整局内容推导且东风/半庄/
西风分别映射 0/1/2。

**Acceptance Scenarios**:

1. **Given** 任意半庄边界序列, **When** 编码, **Then** 每行恰为 21 维,前 7 维与
   V17 完全一致,第 8-21 维为新增字段。
2. **Given** 荣和/自摸/流局小局, **When** 跨边界累计, **Then** 各玩家累计和
   次数/放铳次数/听牌流局次数在下一个边界的行中体现,且首局行全为 0。
3. **Given** 东风/半庄/西风局, **When** 解析整局内容, **Then** 局风类型为
   0/1/2,同一 (风,局) 在不同局风下的特征行不同。

### User Story 2 - 离线/在线特征装配一致性 (Priority: P1)

作为 PPO 装配者,我希望离线数据构造与在线边界推理产生完全一致的特征行。

**Why this priority**: GRP 在 PPO 阶段在小局边界实时装配特征;离线与在线数序若有
任何偏差,奖励信号与训练分布就会漂移。

**Independent Test**: 用同一个边界序列,分别走 `features_from_boundaries` 与逐边界
调用 `feature_row`(在线路径)得到相同的 [T,21] 矩阵。

**Acceptance Scenarios**:

1. **Given** 同一半个半庄的边界与结果, **When** 两种方式各自编码, **Then**
   np.array_equal 相等。
2. **Given** 运行时环境配置 `game_mode="4p-red-half"`, **When** 映射局风类型,
   **Then** 得到 1(半庄),未知模式抛 ValueError。

### User Story 3 - 模型微增与重训 (Priority: P1)

作为训练维护者,我希望 GRP 结构从 7→64×2 层 GRU 提升为 21→96×2 层 GRU
(fc 192→192→24,约 13.2 万参数),并用全量数据(约 36 万半庄)重新训练,
validation-loss best 落盘后完全冻结。

**Why this priority**: 输入维度扩充后需要匹配的容量;数据量从 4.34 万半庄加到约
14.5 万(≈3.3×),提升泛化。

**Independent Test**: 参数量断言 110,000–150,000;训练脚本在构造的小数据集上跑
通 CE 反向、冻结重写 best.pt 与 config_snapshot.json;数据集 dataset.json 的
format 为 `riichi-grp-v18` 且 input_size=21。

**Acceptance Scenarios**:

1. **Given** 训练结东的模型, **When** 保存, **Then** best.pt 携带 model_config
   (input_size=21、hidden=96、layers=2),权重 requires_grad=False。
2. **Given** 冻结后的 GRP, **When** PPO 加载, **Then** 依据 model_config 构造
   GRPModel,且不再更新参数。
3. **Given** 旧 v17 7 维 checkpoint, **When** 按现有契约加载, **Then** 不再被
   活跃代码支持(冷存储)。

## Functional Requirements

- **FR-001**: GRP 输入固定 21 维;字段顺序为 `[grand_kyoku, honba, kyotaku,
  s0..s3/1e4, game_type, prev_result_type, wins0..3, dealins0..3, tenpai0..3]`;
  常量在 `riichi_ppo_v1/model/grp.py` 单一来源。
- **FR-002**: `game_type` 语义 0=东风(仅 E 风 4 局)、1=半庄(E+S 8 局)、
  2=西风(E+S+W 12 局);离线由整局 `start_kyoku` 的 bakaze 集合推导,在线由
  `game_mode` 字符串映射(`single`/`east`→0、`half`→1、`west`→2)。
- **FR-003**: `prev_result_type` 0=首局、1=荣和、2=自摸、3=流局、4=中止;
  小局无结果(首局/中止)取 0 或 4。
- **FR-004**: 累计计数由边界链推导:每行反映"截至该小局开始"的累计值;首局
  行全 0;荣和给 winner 和了 +1、target 放铳 +1;自摸只给 winner 和了 +1;
  流局把 tenpai_mask 中每位玩家听牌流局 +1。
- **FR-005**: 模型结构 `input_size → 2 层 GRU(hidden=96) → concat hidden(192) →
  Linear(192,192) → ReLU → Linear(192,24)`;GRPModel 构造参数
  input_size/hidden_size/num_layers/num_classes 可配置,默认值取模型常量。
- **FR-006**: 数据集目录 `datasets/tenhou_grp_2024_2025_v18`,dataset.json
  format=`riichi-grp-v18`、input_size=21、记录 subsample、max_shards、counts
  与局风类型分布;train/validation 划分沿用原始数据(validation 1%,10 shard);
  tar shard 以 `--workers`(默认 6)个进程并行解析,记录按 shard 顺序拼接,
  输出与串行处理逐位一致;解析前先做只读 tar 头预扫描,把被 shard 边界切断
  的半庄所在相邻 shard 合并为分组,组内跨 tar 聚合,保证每场半庄只产出一条
  完整记录。
- **FR-007**: 配置 `riichi_ppo_v1/configs/v18_grp.yaml`(自包含)、checkpoint
  `checkpoints/train_riichi_v18/grp/best.pt`、日志 `logs/v18/`,均由脚本/配置
  显式指定,不硬编码。
- **FR-008**: 运行时 `GrpRollout` 每环境维护累计计数与局风类型,逐边界调用与
  离线相同的 `feature_row`;终局不追加行,奖励计算方式不变(纯 GRP delta,
  utility [1, 1/3, -1/3, -1])。
- **FR-009**: 验证节奏沿 V17 GRP:每 200 步验证、validation-loss best 保存、
  训练结束冻结后重写 best.pt;epochs=30、batch=2048、lr=1e-5(batch 增大
  不线性放大 LR,沿用 v17 配方),仅数据量与
  输入/结构变化。

## Success Criteria

- `pytest riichi_ppo_v1/tests` 全绿(更新后的 GRP 契约测试、新增一致性测试)。
- 构造小数据集(临时目录)走通 prepare→train→冻结→快照全链路。
- 一键脚本 `audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh`
  从原始数据准备到训练落盘,支持 `--skip-prepare`。
