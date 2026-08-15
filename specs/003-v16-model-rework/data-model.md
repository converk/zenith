# Data Model: V16 模型重构与训练

## 实体

### 1. 输入编码协议契约(EncodingProtocol v16)

- 字段:`encoding_protocol_version=16`(单一来源)、格式标识
  `riichi-sft-encoded-v16`、协议内容 sha256、Actor 输入各 segment 的 slot 语义/
  N/A 规则/bucket 基数。
- 校验:版本必须等于单一命名常量;任何 categorical 因子不得越出声明基数;Actor
  契约不得包含隐藏信息字段。
- 关系:被 SFT 数据集 manifest、checkpoint、Actor/Critic 编码器共同引用。

### 2. ActorSnapshot

- 字段:Objective Facts(历史事件、自身手牌、摸牌);基础场况(场风、局数、庄家、
  本场、立直棒、剩余牌数、宝牌指示、四家点数、当前顺位);Score Pressure(自身与
  三家相对分差 ×3);Opponent Summary(每对手 7 token ×3=21:是否立直、立直巡目、
  副露数、是否门清、舍牌数、手切次数、摸切次数)。
- 校验:仅由该观察者可见的公开信息构造;不含三家牌河/副露完整重复表示。

### 3. ActionQuery(Offense/Defense)

- 字段:`query_type`(OFFENSE/DEFENSE)、`action_id`、`action_type`、
  `primary_tile`、`source_seat`、`answer_0…answer_9`。
- 值域与 N/A 规则:见 `contracts/actor-input-v16.md`;终局/流局/立直/吃碰杠约定
  见 spec.md Assumptions A5。
- 关系:每个合法动作恰一对;聚合成一个 token 后由策略头对称融合。

### 4. V16ActorCritic(网络)

- 字段:`ModelConfig(d_model=256, query_heads=16, kv_heads=4, head_dim=16,
  ffn_dim=1088, shared_layers=4, actor_layers=1, critic_layers=2)`;策略融合头
  (concat 512→256→SiLU→MLP);Top-3 Q scorer(512→256→SiLU→1);Value head。
- 校验:d_model == Q×head_dim;Q 可被 KV 整除;总参数 7.5–7.8M、Actor 推理约
  5.3M(±0.3M);不存在 zero-init offense 分支与 241 维 Q head。

### 5. GRP 模型与样本

- GRP 模型:Linear→64、2 层 GRU(64)、Linear 64→32、SiLU、Linear 32→4、Rank
  Softmax;总参数 50–70K;PPO 中冻结。
- GRP 样本:完整半庄 ×4 视角(SELF/RIGHT/ACROSS/LEFT);每个小局边界输入
  「当前比赛状态 + 上一小局结果(首局 START)」;每个 prefix 监督该视角最终排名。
- 校验:旋转视角唯一;prefix 标签为最终排名;序列边界只在 kyoku 结束处。

### 6. 奖励组件

- 字段:utility `[12,4,-6,-10]`;σ_GRP、σ_Score(离线固定);clip 边界
  (±5 / ±12 / ±5);组合权重 0.7/0.3;终局使用真实排名 utility。
- 关系:GRP delta 来自冻结 GRP 的 V_{k+1}-V_k;分差来自终局点数 Δscore。

### 7. V16 数据集

- SFT:`datasets/tenhou_sft_2024_2025_encoded_40pct_v16`(train/validation,
  manifest 契约见 `contracts/sft-dataset-v16.md`)。
- GRP:`datasets/tenhou_grp_2024_2025_v16`(视角样本 + 归一化统计量 JSON)。
- 校验:manifest 的协议版本/契约 sha256/计数与独立统计一致;旧数据集不删除。

## 跨实体约束

- Actor 契约(category 3)不得引用 Critic 特权字段(对手手牌、后续牌山)。
- 奖励(6)只消费冻结 GRP(5)输出与终局点数,训练期不更新 GRP 参数。
- 网络(4)与契约(1)的版本必须经宪法 Principle II 登记后生效。

