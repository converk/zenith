# Data Model: V18 当前局面输入与 Actor 决策架构重构

## CurrentStateBatch（Rust/PyO3 产物）

| 属性 | 形状/类型 | 校验 |
|---|---|---|
| token_rows | int32 `[total_tokens, 32]` | 每行 [segment, kind, fields...]；域按 `contracts/v18-current-state-contract.md` §3 |
| token_numeric | float32 `[total_tokens, 8]` | 有限且 ∈ [-1,1]；未用槽位 0 |
| offsets | int64 `[B+1]` | 单调、首 0；diff 即每决策长度 |
| lengths | int64 `[B]` | = diff(offsets)，每决策 >0 |

state transition：原生 Observation（列表）→ Rust 批编码 → 扁平行 + offsets → Python 按
offsets 切分，再补 separator/query 行得到完整 Actor 序列。

## ProtocolRowCategory（每类别字段 schema，Python 单源）

| 字段 | 类型 | 校验 |
|---|---|---|
| segment | int | 1..5 |
| token_kind | int | §1 表格 |
| kind_class | enum | SIMPLE / DENSE / SEPARATOR |
| discrete_fields | list[(name, cardinality)] | 顺序即行偏移；0 语义按 §3 |
| numeric_fields | list[(name, normalize)] | 归一化规则（score/100000 clip） |
| slot_count | int | list 类（summary=6） |

关系：每个编码行由 schema 解释；契约 hash 由全部 schema 行 + query 行定义 + 版本生成。

## EncodedSample（SFT 样本）

| 属性 | 形状/类型 | 校验 |
|---|---|---|
| actor_factors | int32 `[T,32]` | 完整 Actor 序列（Shared+Analysis+Queries+separator），T≤256 |
| actor_numeric | float32 `[T,8]` | 与 actor_factors 同行 |
| query_rows | int32 `[2Q,15]` | Offense/Defense 交替，action_id 升序 |
| action_ids | int32 `[Q]` | 升序唯一，与 legal_mask 集合相等 |
| legal_mask | bool `[241]` | 每决策至少 1 个合法动作 |
| action | int | 监督动作 id 且合法 |
| year/game_id/kyoku_index/seat/decision_index | 标量 | 身份溯源 |

relationship：一个 kyoku 的每个决策（每座位）产出一个 EncodedSample；actor 序列不含
Critic 私有 token。

## StateSnapshotModelInput（模型前向）

| 属性 | 说明 |
|---|---|
| actor_factors/actor_numeric/actor_lengths | 完整 Actor 序列 |
| query_action_ids / query_pair_counts | 动作对元数据（用于 logits 映射/校验） |
| legal_mask | 非法动作 -inf |
| critic_factors/critic_lengths | 可选；Critic 私有行（SEP_CRITIC+闭手+未来） |

关系：模型从 actor 行类别推导 mask/位置/嵌入；Critic 复用共享 backbone 输出，私有段独立
backbone。

## Manifest（encoded dataset）

| 键 | 校验 |
|---|---|
| format | == `riichi-sft-encoded-v18` |
| encoding_protocol_version | == 18 |
| encoding_contract_sha256 | == 当前 `ACTOR_INPUT_CONTRACT_SHA256` |
| source_manifest_sha256 | 非空 |
| counts | train/validation kyoku/decision 均 >0 |
| state_protocol | `riichi-current-state-v18-1`（新键，schema 行散列） |
| numeric_dtype / legal_encoding | `float32` / `packbits-little-241` |

## CriticPrivateContext

| 属性 | 校验 |
|---|---|
| opponent_hands | 相对座次 1..3，各 ≥1 行（kind/tile/red/count） |
| future_wall | 恰 5 行，position 1..5 升序 |
| has no analysis/action rows | segment ∈ {4,5} 且 kind ∈ {111,13,14} |

## DenseEmbeddingReport

| 属性 | 校验 |
|---|---|
| total_parameters | ≤ 6,000,000 |
| by_root | embedding/shared/actor/critic/head 分项 |
| forbidden_keys | 无 q_scorer/candidate_q/dueling_q/MHA 分支 |
