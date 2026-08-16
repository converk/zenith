# V16 SFT/PPO 数据语义审计报告

**日期**: 2026-08-16
**分支**: `V16` | **基线 HEAD**: `9a5a1eb`
**数据源**: `datasets/tenhou_sft_2024_2025` 与
`datasets/tenhou_sft_2024_2025_encoded_40pct_v16`
**审计结论**: 通过;未发现会改变训练数据或模型输入语义的缺陷,记录 1 类低影响
元数据缺陷。

## 1. 审计结论

1. SFT 训练数据的 Objective Facts、Compact Snapshot、每动作一对
   Offense/Defense Query 的编码与业务语义一致。
2. PPO 采样阶段把公开局面状态转换进 Actor 输入的过程,与 SFT 编码逐决策一致;
   合法动作掩码、action_id 映射、Critic 特权段均正确;并已用独立解码器逐
   token 从原始 MJAI 事件重算比对。
3. 发现 1 类非致命缺陷:`query` 行中的 `source_seat` 字段在
   chi/pon/daiminkan/ron 上恒为 N/A(详见 §5)。

## 2. 审计范围与方法

本审计只读取训练数据与代码,未重新编码全量数据、未运行 GPU/Ray 训练、未修改
checkpoint 或数据集;仅新增以下可复现脚本:

| 脚本 | 用途 |
|---|---|
| `audit/reports/v16/scripts/audit_v16_static.py` | 契约/常量/文档哈希与 runtime 契约 |
| `audit/reports/v16/scripts/audit_v16_sft_dataset.py` | 全量结构扫描 + 抽样重编码比对 |
| `audit/reports/v16/scripts/audit_v16_semantic_oracle.py` | 独立手工语义 oracle |
| `audit/reports/v16/scripts/audit_v16_ppo_bridge.py` | PPO 离线回放与桥接一致性 |
| `audit/reports/v16/scripts/audit_v16_token_decoder.py` | 独立逐 token 解码器 + 分层抽样 |

抽样规则与训练期一致:`denominator=5, remainders=(0,1),
game_sample_denominator=1, game_sample_remainder=0`;抽样种子固定为
`v16-data-semantics-audit-v1`(SFT)与 `v16-ppo-bridge-audit-v1`(PPO)。

## 3. 静态契约审计

以下检查全部通过:

- `specs/003-v16-model-rework/contracts/actor-input-v16.md` 的 SHA-256 与
  `V16_ACTOR_INPUT_CONTRACT_SHA256` 一致:
  `56874dfb4738af3a506221c1001083ac67fe5188aa2042ae1576f5a852a7ed3b`。
- 协议版本唯一为 `16`、格式标识由常量派生;20 个 slot 基数、N/A 位置与 bucket
  边界和契约/文档一致。
- `riichi.ANALYSIS_VERSION == 4`、`riichienv.REPLAY_SEMANTICS_VERSION == 1`,
  即安装的 Rust 扩展与 Python 边界 runtime 契约一致。
- Rust `player.rs` 的事件/状态 token 与 `KyokuEventTupleProtocol.md` §2–4 一致;
  `cargo test --workspace` 124 个测试全部通过。
- 现有 V16 相关 pytest 基线 33 个测试全部通过。

## 4. SFT 数据审计

### 4.1 全量结构扫描

`audit_v16_sft_dataset.py --skip-sample --workers 8` 遍历全部 6,486 个编码
chunk,逐样本向量化校验 offsets、身份数组、legal mask、专家动作、query
成对/基数/主牌/来源编码、snapshot kind/数值范围与 4096 上下文上限。结果:

| 项 | 结果 |
|---|---|
| 扫描 chunk 数 | 6,486 |
| train 样本数 | 93,943,903 |
| validation 样本数 | 959,045 |
| 与 manifest counts 一致性 | 完全一致 |
| 结构/基数/上下文异常 | 0 |

manifest 的 `query_answer_out_of_range=0`、`snapshot_numeric_out_of_range=0`
与该扫描结论一致。

### 4.2 抽样重编码与存量数据比对

从原始数据确定性选取 120 train + 20 validation 局,重新运行
`encode_kyoku_v16`,并按 `shard` 与 selected ordinal 定位存量 chunk,对同一
`game_id + kyoku_index` 的样本逐字段比对:

| 项 | 结果 |
|---|---|
| 抽样小局数 | 140(120 train + 20 validation) |
| 比对决策样本数 | 8,880 |
| 比对合法动作/query 对数 | 74,945 |
| 离散字段不一致 | 0 |
| fp16 数值字段不一致(容差 6e-3) | 0 |

历史因素、snapshot、query rows、action_ids、legal mask 与专家动作全部与
存量编码一致;重编码与落盘后的 fp16 恢复误差在预期容差内。

### 4.3 独立语义 oracle

`audit_v16_semantic_oracle.py` 以手工期望值核对 9 项场景检查,全部通过:
立直宣告、暗杠、吃/碰/大明杠/加杠的结构性 N/A、荣和终局、门清 14 张 pass、
多重 dora+赤五、D9 公开可见数、Snapshot 场况/分数/对手摘要。现有
`test_v16_query_semantics.py` 的 20-slot 手工 oracle 仍作为基线通过。

## 5. PPO 采样/桥接一致性审计

`audit_v16_ppo_bridge.py` 对 40 个小局执行离线回放,把同一批 Observation 分别
送入 `encode_kyoku_v16` 与 `BatchedStateBridge.prepare_v16`,逐决策比较 Actor
输入;同时校验 Rust action_id 解码与 Critic 特权段:

| 项 | 结果 |
|---|---|
| 比对小局数 | 40 |
| 比对决策数 | 2,419 |
| 解码合法 action_id 数 | 20,848 |
| history/snapshot/query/legal 不一致 | 0 |
| query offense/defense 配对异常 | 0 |
| action_id 解码失败 | 0 |
| Critic 出现非特权段 | 0 |

说明:离线 `MjaiReplay` 的 `select_action_from_mjai` 对部分 replay 动作形态
(如 `pon`)更严格,因此本脚本用 Rust 解码 + representative 映射校验 action_id;
真实 `BatchedRiichiEnv` 上的 `bridge.decode` 回路由现有
`test_real_action_cases.py`、`test_batched_pipeline.py` 覆盖,不在离线回放中重复。

### 5.1 独立逐 token 解码扩展审计

为回答「每个编码 token 的语义是否与场面一致」,新增独立解码器脚本
`audit_v16_token_decoder.py`。它不复用生产端的 token 构造函数,而是:

- 从每个决策的 `Observation.new_events()` 原始 MJAI 事件,独立解码
  start_kyoku、dahai、chi、pon、daiminkan、ankan、kakan、dora、reach、
  reach_accepted 的 10 分类因子;
- 从 Observation 公开字段独立重算分数、局况计数、宝牌指示、自身手牌、摸牌与
  主动决策 flag 的状态后缀 token,以及 history/state 的八维周期数值特征;
- 独立重算 Compact Snapshot 的 kind/categorical/numeric 行;
- 独立重算每个合法动作 query 的头部字段与 D0–D9 防守答案,并校验
  offense 20 个 slot 的基数。

结果:

| 项 | 结果 |
|---|---|
| 分层抽样小局数 | 27 |
| 跨 shard 分布 | 27 个 shard(2024-01-01 至 2024-01-09) |
| 比对决策数 | 1,974 |
| 覆盖事件类型 | reach、chi、pon、daiminkan、ankan、kakan、dora、hora、ryukyoku 各 ≥3 局 |
| history/state 逐 token 不一致 | 0 |
| snapshot 逐行不一致 | 0 |
| query 头部/Defense answers 不一致 | 0 |
| offense answer 越界 | 0 |

独立解码器与生产实现完全一致。因此至少在本抽样覆盖的局阶段/鸣牌/杠/立直/
宝牌/和了/流局范围内,每个 token 的因子与场面状态一致;结合 §4.1 对全部
94.9M 样本的结构扫描,可以覆盖绝大多数字段与动作类型。

## 6. 发现与建议

### F1:`source_seat` 在 chi/pon/daiminkan/ron 恒为 N/A(低影响)

`model/action_query.py` 的 `_source_seat` 按 `last_discard[0]` 期望一个
`(seat, tile)` 元组,但当前 4 人局 `Observation.last_discard` 只返回牌 id
(`Option<u32>`)。因此这些动作的 query 行 `source_seat` 全部编码为 0/N/A。

- 影响:该字段目前未被 `QueryEmbedding` 消费,不影响模型输入向量或 SFT/PPO
  训练数值;但作为 query 行元数据,其业务语义未按代码注释意图落盘。
- 建议:由 MJAI 事件的 `actor/target` 或状态机中的最近弃牌者派生 source seat;
  或明确将 `source_seat` 定义为可选审计字段。修复后同步
  `actor-input-v16.md`/`v16_input_protocol.md` 的字段语义说明,并补充
  `test_v16_query_semantics.py` 或本审计脚本中的回归断言。

### 已排除的其他疑点

- `QueryEmbedding` 只消费 `query_type/action_id/answers`,与契约聚合公式
  `E_action + E_queryType + Σ E_slot(answer_i)` 一致,`action_type/primary_tile/
  source_seat` 定位为存储元数据而非模型输入。
- Snapshot 的分数、分差、局况、宝牌指示与对手摘要均由公开 Observation 直接
  计算,对手隐藏手牌篡改不改变 Actor 输入(现有 `test_v16_replay_bridge.py`
  已验证)。

## 7. 复现命令

```bash
conda run -n Mahjong-AI python -u audit/reports/v16/scripts/audit_v16_static.py
conda run -n Mahjong-AI python -u audit/reports/v16/scripts/audit_v16_sft_dataset.py --workers 8
conda run -n Mahjong-AI python -u audit/reports/v16/scripts/audit_v16_semantic_oracle.py
conda run -n Mahjong-AI python -u audit/reports/v16/scripts/audit_v16_ppo_bridge.py --count 40
```

其中 `audit_v16_sft_dataset.py` 默认同时执行全量扫描与 140 局抽样比对;如需
分别复跑,可加 `--skip-scan` 或 `--skip-sample`。
