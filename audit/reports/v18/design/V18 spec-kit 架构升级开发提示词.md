# V18 spec-kit 架构升级开发提示词

将下面整段提示词复制到新的 Codex 窗口中执行：

```text
请在项目 /mnt/disk1/hubowen/zenith 中完成 V18 输入与模型架构升级。

这是一项需要实际完成代码、测试和文档的开发任务，不是只输出方案或告诉我如何使用 spec-kit。请由你自主使用 spec-kit 完成完整流程，不要让我逐条发送 spec-kit 命令，也不要在生成 spec、plan 或 tasks 后停下来等待我提示继续。

你必须依次使用以下技能并遵守各自边界：

1. $speckit-constitution
2. $speckit-specify
3. 如确有无法从需求或仓库确定的关键歧义，使用 $speckit-clarify
4. $speckit-plan
5. $speckit-tasks
6. $speckit-analyze
7. $speckit-implement
8. $speckit-converge
9. 如果 converge 追加了任务，继续执行 $speckit-implement 和 $speckit-converge，直到没有剩余差距

$speckit-constitution 必须作为独立治理阶段执行，不能与 feature 工作混在同一个技能阶段；完成宪法升级后由你主动继续后续流程。$speckit-analyze 只做只读分析；发现阻断问题时，由你使用对应技能修正 spec、plan 或 tasks 后重新分析。不要把这些流程转化为需要我复制执行的命令清单。

除非遇到确实无法通过代码、现有文档和下述契约解决，并且会实质改变架构或验收结果的阻塞问题，否则不要中断任务向我提问。请持续推进到实现、测试、文档同步和 converge 全部完成。

## 一、环境与治理

- 工作目录固定为 /mnt/disk1/hubowen/zenith。
- 所有 Python、测试和训练相关命令使用 Mahjong-AI Conda 环境。
- 开始前读取并遵守 AGENTS.md、现行 constitution 和仓库内相关说明。
- 预期 feature short name 为 v18-input-architecture，目录为 specs/008-v18-input-architecture/。必须通过 spec-kit 脚本创建；如果脚本根据仓库实际状态分配了其他下一个可用编号，以脚本结果为准，禁止覆盖已有 feature。
- 开始 feature 工作前，先通过 $speckit-constitution 将 .specify/memory/constitution.md 从 v1.7.0 升级到 v1.8.0，并生成完整 Sync Impact Report。
- v1.8.0 宪法须将现行 encoding protocol 改为 V18，将 V16/V17 代码契约和实验资产转为冷存储，不再视为活跃契约。
- V16/V17 checkpoint、数据集、日志和历史报告继续归档保留，禁止删除、覆盖或破坏。活跃文档引用这些历史资产时必须明确其归档属性。
- PPO 既有评测机制、频率、规模和治理约束保持不变。本任务不修改 PPO 评测机制。
- 宪法升级完成后，自主继续创建和实现 V18 feature。

## 二、任务范围

V17 训练样本的平均输入长度约为 75–77 token。目前可能存在两类问题：部分有价值的局面隐藏信息没有被计算并编码；部分 token 又承载了过多异质信息，使语义向量无法充分区分麻将意义上差异很大的状态。

V18 的目标是：

- 将局面事实拆成更原子的 token；
- 在现有 2024、2025 年数据的既定 60% validation selection 上，将预计平均输入长度提高到约 99.80 token，验收范围为 97–103；
- 将 Actor-Critic 总参数量提高到约 5M；
- 保持 Actor 的公开信息边界；
- 保持 Critic 读取三家真实手牌和未来 5 张牌；
- 提供仅训练 Actor 的 BC SFT-ready 接口。

本 feature 只实现并验证：

- V18 输入协议与 schema；
- Rust/Python 编码桥接；
- V18 模型架构；
- Query metadata 和注意力隔离；
- Actor/Critic 信息边界；
- Actor-only SFT-ready 接口；
- 配置、测试、统计验证和文档。

本 feature 明确不包含：

- 不生成完整 60% V18 数据集；
- 不启动正式 SFT；
- 不设计、修改或启动 PPO 训练。

后续正式 SFT 数据集路径固定为：

datasets/tenhou_sft_2024_2025_encoded_60pct_v18

它仍使用与 V16/V17 一致的 2024、2025 年数据既定 60% selection。本次只能使用已有 selection 做不落完整训练集的统计、抽样验证或小型 fixture。

## 三、V18 公共输入与 29 个 Atomic Snapshot Token

保留现有历史事件 token、当前状态 suffix，以及每个合法动作一对 Query token 的整体组织方式。将 Snapshot 重构为固定 29 个 Atomic Snapshot Token，每个 token 只表达一个清晰事实。

29 个 Snapshot token 的顺序和语义必须固定为：

### 1. Placement 与分差压力：4 个

- 自己的当前名次：1 个；
- 自己相对另外三家的分差压力：每名对手 1 个，共 3 个。

分差压力使用单一、固定、可测试的归一化方式，不得根据单个样本动态缩放。

### 2. 三名对手摘要：15 个

按固定相对座次顺序遍历三名对手，每名对手恰好编码以下 5 个字段：

- 立直状态：NONE / DECLARED / ACCEPTED；
- 立直发生巡目；
- 副露面子数，暗杠不计入副露数，从而正确表达门清；
- 累计手切次数；
- 累计摸切次数。

不要单独增加河牌总数，因为手切数加摸切数已经表达；不要单独增加门清字段，因为副露面子数为 0 已表达门清。

### 3. 派生局面事实：10 个

- overall shanten：1 个；
- standard shanten：1 个；
- chiitoitsu shanten：1 个；
- kokushi shanten：1 个；
- 三名对手最近一次手切的牌：每名对手 1 个，共 3 个；
- 三名对手当前连续摸切长度：每名对手 1 个，共 3 个。

字段域必须明确、稳定并可测试：

- overall/standard shanten：AGARI 或 0..6；
- chiitoitsu shanten：开手时为 N/A，否则为 AGARI 或 0..6；
- kokushi shanten：开手时为 N/A，否则为 AGARI 或 0..13；
- 最近手切牌：N/A 或保留赤五区分的 37 种牌身份；
- 连续摸切长度：0、1、2、3、4+；
- 立直状态：NONE、DECLARED、ACCEPTED；
- 立直巡目：N/A、1..24、25+；
- 其他次数和计数在设计阶段确定单一、稳定、无歧义的离散域或桶，并通过测试锁定。

Snapshot 不得重复编码已经由当前状态 suffix 表达的以下内容：

- 场风；
- 局数；
- 庄家；
- 本场；
- 供托；
- 剩余牌数；
- 宝牌指示牌；
- 四家绝对点数；
- 自家手牌；
- 自摸牌。

Snapshot 批数据接口固定为：

- snapshot_factors: [B, 29, 4]，四列依次为 field_id、relative_seat、categorical_value、tile_value；
- snapshot_numeric: [B, 29, 1]；
- snapshot_lengths 对所有有效样本恒为 29。

V18 编码器、dataset schema、collator、模型前向和协议校验必须共享同一份字段顺序和域定义，禁止复制散落常量。

现有 60% validation selection 的基线统计约为：

- 历史 token 均值：53.84；
- 旧 Snapshot token 均值：6.12；
- Query pair 均值：8.48 对；
- 换成固定 29 个 Snapshot token 后，预计总均值：99.80。

验收时使用不生成完整训练数据集的统计或代表性 V18 编码验证，确认平均总输入 token 位于 97–103。

## 四、Query metadata 与动作对隔离注意力

每个合法动作继续只使用一个 Offense token 和一个 Defense token，不要把 Query 拆成更多 slot。

Query 行已有 action_type、primary_tile、source_seat 等 metadata。V18 embedding 必须实际使用这些字段，不得继续读取后忽略。

修复 Rust 到 Python 的 source_seat：

- chi、pon、daiminkan、ron 使用真实的相对供牌座次；
- 其他不适用的动作使用 N/A。

action_id、Query pair 与 raw logit 之间必须有明确且经过测试的映射协议。

实现 action-pair isolated attention：

- 每个 Offense/Defense 动作对可以看到全部公共前缀；
- 同一动作对中的 Offense 和 Defense token 可以双向互见；
- 一个动作对不能看到其他候选动作的任何 Query token；
- 公共 token 不能反向看到 Query token；
- 所有动作对复用相同的局部 position id，使动作排列置换不变性由结构保证，而不是依靠数据增强近似；
- 任意重排合法动作对后，raw logits 按 action_id 映射回原顺序时必须保持一致，只允许合理的浮点误差。

## 五、模型架构

采用保守的 Transformer/GQA 扩容。这一版不引入 ReBeL、belief search、oracle guiding 或其他会显著扩大训练目标与实现范围的结构。

保留：

- RoPE；
- RMSNorm；
- causal GQA；
- gated FFN。

核心超参数固定为：

- d_model = 256；
- query_heads = 16；
- kv_heads = 4；
- head_dim = 16；
- ffn_dim = 704；
- Actor 共 4 层，其中 3 层处理共享公共表示，1 层为 Actor-only；
- Critic 在共享公共表示后使用 2 层 Critic 分支。

Actor-Critic 总参数量必须位于 4.9M–5.1M。统计口径包含 embedding、Actor、Critic 和 value head，但不包含任何 Q 模块。必须提供统一参数统计入口和自动化测试。

彻底移除 Q scorer/Q-boosting：

- 不保留 q_scorer；
- 不保留 candidate-Q API；
- 不保留 Q-boost 参数或配置；
- V18 state dict 中不得存在相关 key；
- 活跃代码中不得存在实际生效的 Q scorer 分支。

清理前必须使用 rg 做全仓引用检查。只有确认零引用并通过相关测试后，才允许删除旧代码。

## 六、Actor 与 Critic 的信息边界

Actor 只能接收公开可观察信息、自家信息和合法动作 Query。Actor 不得读取另外三家的真实手牌或未来牌山。

必须增加信息隔离测试：改变对手真实手牌和未来牌山，同时保持 Actor 可见输入完全相同，验证 Actor logits 不变。

Critic 保留当前私有信息：

- 另外三家的真实手牌；
- 牌山未来 5 张牌。

Critic 输入为共享公共表示加上述私有 token 和 value query，不接收动作 Query。必须验证私有数据实际进入 Critic，并验证 shape、mask、顺序和边界正确。

## 七、Actor-only SFT-ready 接口

本 feature 只实现可供 SFT 使用的接口和小规模测试，不运行正式 SFT。

SFT 使用仅 Actor 的行为克隆：

- 前向只计算 Actor 合法动作 logits 和 BC loss；
- Critic 与 value 分支冻结；
- Critic/value 参数不产生梯度，不进入 optimizer，不被更新；
- 不增加 value、belief/oracle 或对手手牌辅助损失；
- Actor-only SFT 的前向、反向、optimizer step、保存和加载必须通过集成测试；
- 保存和加载只支持纯 V18 契约。

## 八、明确不保留 V16/V17 兼容性

不保留以下任何兼容能力：

- V16/V17 checkpoint 加载；
- V16/V17 协议适配；
- 旧输入转换；
- 双模型实现分支；
- legacy adapter；
- 旧字段映射；
- 兼容 flag 或旧 schema fallback；
- state-dict 迁移。

不要为了让旧 checkpoint 测试继续通过而污染 V18 实现。按升级后的宪法处理退出活跃契约的测试和引用。

V16/V17 checkpoint、数据集、日志和报告只能归档保留，绝不删除、覆盖或改写历史结果。删除任何旧代码或文件前，必须全仓 rg、确认零引用并通过测试。

## 九、代码与目录要求

实现应覆盖：

- V18 协议常量和 schema；
- Rust 状态与编码侧派生字段；
- PyO3 或当前桥接层；
- Python dataset 和 collator；
- 模型、embedding 和 attention mask；
- 参数统计；
- Actor-only SFT-ready 接口；
- 自包含 V18 配置；
- 单元、协议、集成和回放测试。

遵守现有职责边界：

- 模型、schema 与契约校验放在 riichi_ppo_v1/model；
- SFT 数据和训练接口放在 riichi_ppo_v1/sft；
- 环境、状态机和 MJAI 转换放在 RiichiEnv 对应模块；
- 生产校验入口放在 riichi_ppo_v1/tools；
- 独立职责应建立自描述文件，不要把无关逻辑塞进现有模块。

新增或修改的代码注释使用中文。领域不变常量必须单点定义。V18 配置必须完整自包含，不得使用 overlay 继承。通用实现不得硬编码实验版本、checkpoint、数据集、schema ID、默认路径或计数；版本配置可以提供 V18 的明确参数。

checkpoint、日志和 audit 产物遵守 AGENTS.md 的版本目录约定。冒烟测试结束后清理其产生的临时日志和结果文件。

## 十、文档同步是完成任务的硬门槛

以下文档工作必须作为 tasks.md 中的正式任务完成，不能只写在最终说明里：

1. 新增自描述的 V18 输入协议文档，完整说明 29 个 Snapshot token、字段域、固定顺序、Query metadata、attention mask、Actor/Critic 信息边界和 schema/version 契约。
2. 更新 riichi_ppo_v1/docs/KyokuEventTupleProtocol.md，说明 Rust/Python 事件与 V18 编码桥接以及 source_seat 语义。
3. 更新根 README。
4. 更新 riichi_ppo_v1 的 README 和训练框架相关文档。
5. 更新 SFT 使用说明、配置示例和 CLI 默认路径。
6. 更新 AGENTS.md 中的现行 encoding protocol、活跃训练代、现行/后续数据集和 V18 产物路径。V16/V17 必须描述为冷存储，而不是活跃契约。
7. 如果新增目录或改变目录职责，更新 docs/directory-responsibilities.md。
8. 全仓查找并更新所有仍把 V16/V17 描述为现行协议的活跃引用。历史报告本身保留原貌；活跃文档引用时明确其归档属性。
9. 在 audit/reports/v18/report/PROGRESS.md 持续记录实现过程、测试、参数统计、token 统计和验证结果。

文档、代码、配置、CLI 默认值和实际产物路径必须一致。文档未同步时不得宣告实现完成。

## 十一、验收标准

只有满足以下全部条件，任务才算完成：

1. constitution 已升级到 v1.8.0，Sync Impact Report 完整，V18 为现行协议，PPO 评测机制没有被改变。
2. spec、plan、tasks、实现、测试、配置和文档之间无矛盾；每条需求都能追溯到任务和验证证据。
3. Snapshot 对每个有效样本固定为恰好 29 token。
4. 在现有 60% validation selection 的不落完整训练集统计或代表性 V18 编码验证中，平均总输入 token 位于 97–103，目标预计值约为 99.80。
5. Actor-Critic 总参数量按统一口径位于 4.9M–5.1M。
6. V18 state dict 不含 Q scorer/Q-boosting key，代码和配置中没有活跃 Q API。
7. Query metadata 被 embedding 实际使用，source_seat 语义正确。
8. 重排动作 Query pair 后，raw logits 按 action_id 映射回去保持一致。
9. Actor 不受不可见的对手真实手牌和未来牌山变化影响。
10. Critic 保留并实际使用三家真实手牌和未来 5 张牌。
11. Actor-only BC SFT 的前向、反向、optimizer step、保存和加载通过；Critic/value 参数全程冻结且无梯度。
12. 相关 Python、Rust、协议、schema、集成和回放测试全部通过，覆盖正常情况、边界、N/A、离散桶和非法输入。
13. README、协议文档、SFT 文档、AGENTS.md、配置、CLI 与实际代码路径全部同步。
14. audit/reports/v18/report/PROGRESS.md 包含完整、可复现的验证记录。
15. 没有删除 V16/V17 checkpoint、数据集和历史报告。
16. 没有生成完整 V18 60% 数据集，没有启动正式 SFT，没有设计或执行 PPO。

## 十二、执行与交付方式

先调查当前代码和已有 V17 实现，再通过 spec-kit 形成需求、设计和任务，不要在不了解现有实现时直接重写。可以在 plan 阶段进行针对不完全信息博弈和 self-play 模型结构的必要调研，但本版优先采用上述已确定的保守 GQA 架构，不得因调研擅自扩大范围。

按 tasks.md 的依赖顺序实施，每完成并验证一个任务就标记为 [X]。运行与风险相称的测试；测试失败时定位并修复根因，不得通过降低验收标准、删除测试或增加 legacy 兼容分支绕过。

首次 implement 完成后必须运行 converge。若 converge 找到遗漏，让它把差距追加为任务，然后继续 implement；重复直到 converge 确认无剩余差距。

最终回复请简洁汇总：

- constitution 和 feature 产物路径；
- 核心实现文件；
- 29-token 和平均 token 统计；
- 模型参数量与无 Q state-dict 检查；
- Query 置换不变性、Actor/Critic 信息隔离和 Actor-only SFT 测试；
- Python、Rust、协议、集成和回放测试结果；
- 文档同步清单；
- converge 结论；
- 未执行完整数据生成、正式 SFT 和 PPO 的确认。

现在请直接开始并自主推进到全部完成，不要只向我复述计划，也不要让我手动驱动 spec-kit 的每一个阶段。
```
