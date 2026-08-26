# V18 当前局面输入逐 Token 审查与测试提示词

请在项目 `/mnt/disk1/hubowen/zenith` 中，对已经提交的 V18 当前局面输入、模型架构和
Actor-only SFT 数据链进行一次**独立、细致、可复现的正确性审查与测试**。

本任务不是复述现有设计、浏览测试名称后宣布通过，也不是仅运行现有测试套件。必须从
麻将局面真值和原始 Observation/MJAI replay 出发，建立不依赖生产编码器实现的审查
oracle，逐 token、逐字段、逐维度核对编码与解码，证明每个输入值确实表达了决策时刻
已经发生的事情和当前局面信息，并证明 SFT 预处理没有在任何阶段改变、遗漏、错位或
泄漏这些信息。

## 一、审查对象、提交边界与权威依据

本轮实现由以下五个提交组成，审查范围以它们的实际 diff 和当前 HEAD 为准：

| commit | 主题 |
| --- | --- |
| `f5e0c6f` | Rust/PyO3 当前局面批编码器与 schema 单源 |
| `a2fd884` | 当前局面快照模型协议与嵌入/架构重构 |
| `8feeb30` | Actor-only SFT 全链路与配置/统计工具 |
| `70247c8` | V18 输入/模型/SFT 测试与 fixture 重写 |
| `15fcd6b` | 协议、SFT、README、AGENTS、审计与 spec-kit 010 文档同步 |

开始前完整读取并遵守：

1. `AGENTS.md`；
2. `.specify/memory/constitution.md`，当前为 v1.8.0；
3. `audit/reports/v18/design/V18当前局面输入与Actor决策架构重构提示词.md`；
4. `specs/010-v18-current-state-input-sft/` 下的 `spec.md`、`plan.md`、`tasks.md`、
   `research.md`、`data-model.md`、`contracts/`、`quickstart.md` 和 checklist；
5. 上述五个提交的完整 diff，以及当前相关 Rust/Python/model/SFT/test/doc 源码；
6. `riichi_ppo_v1/docs/v18_input_protocol.md` 与
   `riichi_ppo_v1/docs/KyokuEventTupleProtocol.md`。

事实优先级为：宪法和原始重构需求 → 麻将规则与原生 Observation 的真实语义 → 可执行
源码 → spec/contract/docs → 现有测试和进度报告。文档、schema、Rust 常量和既有测试
可能一起复制了同一个错误，因此不得用“它们彼此一致”替代语义正确性证明。发现权威
材料之间冲突时，列出双方的文件、行号、具体值和影响，不能静默选一个。

只把 V18 当前局面输入、模型、SFT 预处理与 Actor-only 训练接口作为本轮阻塞范围。
PPO/rollout/1v3 仅做旧接口引用盘点，不迁移、不测试、不修复。不得生成完整 60% V18
数据集，不启动正式长时 SFT、GRP 或 PPO，不删除任何 checkpoint、数据集或历史报告。

## 二、执行方式与基本原则

### 2.1 先审查，后验证，最后才下结论

按以下顺序连续执行：

1. 保存起点信息：HEAD、工作区状态、Conda/Rust/PyO3 模块路径、模块版本、schema hash；
2. 对五个提交做逐文件 diff 审查，画出事实来源到模型入口的调用链；
3. 建立字段级审查矩阵和独立 oracle/decoder；
4. 运行现有测试作为基线，但不将其视为最终证据；
5. 补充逐字段、变形、差分、属性、反例和端到端测试；
6. 对真实 replay 做逐决策抽样审计，对边界局面做合成穷举；
7. 汇总问题、证据、影响和最小复现，不以“总测试数”掩盖覆盖空洞；
8. 做全仓旧契约与临时产物检查，输出最终审查结论。

本轮默认是**审查与测试任务，不直接修复生产实现**。发现缺陷时先固定最小复现、预期值、
实际值、根因位置和回归测试建议；不要为了让测试变绿而修改 oracle、放宽断言、截断输入、
增加 fallback 或改写需求。若用户后续明确要求修复，再以每个根因一个可回滚主题实施。

允许新增只读审查脚本、审查报告和不会掩盖缺陷的测试。若确认缺陷会使正式测试失败，
可将最小复现放在 `audit/reports/v18/scripts/` 并在报告中标为预期失败，避免把未经修复的
失败测试混入常规 suite。测试产生的临时 shard、日志和 checkpoint 必须使用临时目录并在
结束时清理。

### 2.2 独立性要求

审查 oracle 不得调用以下生产逻辑来计算 expected value：

- `prepare_current_state_batch`；
- `current_state.encode_batch`；
- 生产 `semantic_validation`；
- 生产的 shanten/advance/wait、筋、壁、宝牌、役牌或 query answer 聚合函数；
- 从生产 schema 动态读取字段顺序后原样比较自己。

可以读取原始 Observation 字段并使用独立、简单、审计友好的参考实现。对复杂算法至少
采用以下一种独立证据：第二实现、已验证的穷举器、规则表、手工小局面真值或与原生状态机
不同路径的交叉验证。oracle 中应显式写出字段名、列号、单位、基数、N/A、padding、bucket
边界和舍入规则，目的是发现 Rust/Python/schema 三者共同的错误。

### 2.3 sub-agent 分工

请积极使用 sub-agents 并行完成边界清晰的只读调查与测试设计，例如：

- Rust/Observation/牌河/副露/赤牌事实链；
- Python schema、decoder、embedding 和 mask；
- replay/precompute/shard/collator/SFT 标签链；
- 现有测试覆盖缺口、真实 replay 抽样和文档一致性。

主 agent 必须亲自确定 oracle、整合字段矩阵、复核所有反例、运行最终验证并作结论。
不同 agent 不得并行修改同一文件，不得回滚他人改动。

## 三、必须先产出的审查矩阵

在运行大规模测试前，先建立一张机器输入可追踪表。每个 token kind 的每个离散字段、
numeric 字段、隐式位置和 mask 角色都必须有一行，至少包含：

| 列 | 含义 |
| --- | --- |
| segment / kind / token 序号 | 它在 Actor 或 Critic 规范序列中的位置 |
| row offset / numeric offset | 实际张量维度，不得只写字段名 |
| 字段名与语义 | 表示的当前事实或已发生事实 |
| 原始真值来源 | Observation/Meld/river/action/replay 的具体字段和座次变换 |
| 编码公式 | 1-based/0-based、相对座次、bucket、cap、N/A、padding、归一化 |
| 合法域 | 最小、最大、特殊值及无效组合 |
| 独立解码规则 | 如何从行恢复为具名语义或等价类 |
| 有损性 | 无损、bucket 等价类、明确省略，不得混写 |
| 模型消费路径 | 哪个 embedding 表/投影/分支/mask 读取该维度 |
| 数据链路径 | replay → encode → shard key/offset → collator → forward 参数 |
| 自动化证据 | 测试名、fixture 和覆盖的边界值 |
| 审查结论 | PASS/FAIL/PARTIAL/UNTESTED 与证据链接 |

矩阵必须覆盖：BOS、11 个专属 separator、TABLE、SELF_HAND、SELF_STATE_ANALYSIS、
4 个 PLAYER、6 个 RIVER_SUMMARY、三家每张 RIVER_DISCARD、所有 MELD、固定 34 个
TILE_STATE、3 个 OPPONENT_ANALYSIS、每个合法动作的 Offense/Defense、三家 Critic
闭手、未来五张和 Value Query。不能用“一类 token 已测”代替该类所有字段维度。

同时生成以下映射：

- `token kind → segment → schema → Rust 写入位置 → Python embedding module`；
- `Observation 字段 → 所有受影响 token/维度`；
- `shard 数组和 offsets → EncodedSample → collator tensor → model.forward 参数`；
- `action JSON/MJAI 动作 → 241 action ID → query pair → raw logit → BC target`。

## 四、独立 decoder 与逐维编解码验收

### 4.1 decoder 要求

实现一个只读的审查 decoder，把 `actor_factors[T,32]`、`actor_numeric[T,8]`、
`query_rows[2Q,15]`、`action_ids[Q]`、critic rows 解码为具名结构。它必须：

- 严格检查 dtype、rank、shape、length、offset、segment、kind 和保留位；
- 严格检查每个字段域及跨字段约束；
- 区分合法 0、N/A、padding 和 bucket 饱和值；
- 将 numeric 按规定 scale 还原，并报告量化/clip 后的等价区间；
- 输出规范顺序和原始 row index，便于定位错位；
- 遇到未知 kind、错 segment、错序、重复/缺失 separator、非零保留位时 fail closed；
- 不导入生产 encoder/validator 来决定 expected value。

### 4.2 round-trip 定义

不能笼统要求从 token 恢复完整 Observation，因为协议有意省略历史事件并使用 bucket。
应定义一个 `CanonicalVisibleFacts`：只含 V18 契约要求表达的事实，并为每个有损字段保存
等价类。逐局面验证：

```text
Observation/replay
  → 独立提取 CanonicalVisibleFacts
  → 生产 encoder
  → 独立 decoder
  → DecodedCanonicalFacts
  → 按字段逐项比较
```

无损字段必须完全相等；bucket 字段必须落在正确等价类；numeric 必须在明确容差内；明确
省略字段必须做非干扰测试。禁止只比较 encoder 两次输出相同，或只验证值在 cardinality 内。

### 4.3 每维敏感性和不可混淆性

对每个有效字段至少构造一次单变量变化：保持其他可见事实不变，只改变该事实，确认：

1. 预期 token 的预期维度发生正确变化；
2. 不应变化的 token/维度保持不变；
3. decoder 得到新真值；
4. embedding 在固定权重下发生变化；
5. 对密集 token，梯度到达该字段自己的 embedding/projection；
6. padding、N/A、合法零值不会产生同一语义或错误梯度；
7. 超出合法域、互斥组合和非零保留位被拒绝，而不是 clamp 后悄悄接受。

对 bucket 的下界、上界、上界前一值、饱和值和溢出值逐一测试，例如 0/1、5/6/7、
15/16/17、19/20/21、24/25/26。对 numeric 测负分、零、正分、同分、接近 clip 边界和
超过 clip 边界。

## 五、逐 token 类别的语义审查

### 5.1 序列、segment、separator 和 padding

逐决策解码完整序列并验证：

- Shared 从 BOS 开始，按原始需求的固定类别顺序排列；
- 自手按 34 牌序，四家 PLAYER 为自己/下家/对面/上家；
- 三家牌河依次为下家/对面/上家且内部旧到新；
- MELD 按 owner_relative、meld_index 规范排序；
- TILE_STATE 恰好 34 个且 tile_type 1..34；
- Analysis 恰好三家且在 Action 前；
- Action pair 按 action ID 升序且 O/D 交替；
- Critic 是独立尾部，不含 Analysis/Action；
- 11 个 separator 的 ID、kind、segment、文档编号和实际行完全一致；
- 所有未使用整数列、numeric 列和批 padding 行严格为零；有效 token 不被零 padding 截断。

特别核对 contract 文档示例中的 separator kind 标注与 `kind=100+separator_id` 是否逐项
一致；核对 summary 的 `valid_length` 判定究竟是 `slot_index <= valid_length` 还是其他
规则。不能因为 validator 与 encoder 同错而通过。

### 5.2 TABLE

逐维核对场风、局索引、本场、立直棒、庄家、自身绝对座次、自风的可推导关系、决策模式、
当前摸牌、立直状态、五个宝牌指示槽、四家分数、自身名次和三家点差。覆盖：

- 东一到南四、连庄、多本场、多立直棒；
- 自身位于四个绝对座次的旋转等价测试；
- 四家同分及稳定排名 tie-break；
- 0 到 5 个宝牌指示、重复指示、红五指示；
- 主动摸牌、响应舍牌、响应加杠三种模式；
- drawn tile 缺失、普通牌、赤五，以及 `drawn_is_current` 的真实语义；
- 负分和大分下 numeric scale/clip。

明确证明没有 `tiles_left`、牌山估计、独立供牌公共 token 或任何历史 generation/cache。

### 5.3 SELF_HAND 与 SELF_STATE_ANALYSIS

SELF_HAND 必须无损反映当前暗手的 34 类计数、赤五、当前摸牌和立直锁定。覆盖同类普通五
与赤五共存、四枚同牌、暗杠、摸入后 14 张、响应时 13 张、立直宣言待受理/已受理。

SELF_STATE_ANALYSIS 逐字段用独立算法或手工牌例核对：门清、暗牌数、副露数、四种向听、
进张种数/实体数、等待种数/实体数、三类振听、确定基础番、宝牌、赤牌和役牌番。至少覆盖：

- standard、七对子、国士三条路线各自成为最优；
- 和牌、听牌、一向听、多向听和开放手的 N/A；
- 多面听、同牌四枚已见、进张为 0 实体；
- 永久振听、同巡振听、立直见逃及组合状态；
- 场风=自风的连风牌、役牌副露、暗杠、重复宝牌指示；
- 13/14 张归一化和 `kernel_shape` 扣牌策略是否改变真实分析。

如果“基础番”不是完整和牌番数，明确其定义和排除项，不允许名称与实现语义不一致。

### 5.4 PLAYER

四家分别核对绝对/相对座次、自风、庄家、分数、排名、点差、暗牌数、副露数、杠数、门清、
牌河长度、立直状态/巡目/宣言牌/立直后舍牌数、公开役牌番和可见宝赤数。

重点验证对手暗牌数不能读取真实闭手，只能从公开生命周期确定；chi/pon/daiminkan/ankan/
kakan 对暗牌数、meld_count、kan_count、menzen 的影响分别正确。kakan 只表示当前一个副露，
不能同时保留过去 pon。宣言牌被鸣、同牌种多次舍弃、立直未受理和终局附近长牌河必须测试。

### 5.5 三家 RIVER_SUMMARY 与 RIVER_DISCARD

对每家构造牌河长度 0、1、5、6、7、12、18、24：

- FIRST_SIX 始终取最早六张；RECENT_SIX 始终取最近六张且内部由旧到新；
- 两个摘要各恰好一个 token，每个有效子槽的牌、赤、手切/摸切、立直阶段正确；
- `valid_length` 与有效子槽完全一致，padding 不与合法值混淆且对 embedding 严格零贡献；
- 逐张 discard 的 relative seat、1-based local index、实体牌/赤牌、cut、立直三态、
  `supplied` 和 age bucket 正确；
- 摘要与逐张 token 针对同一牌河的值相互一致，但由独立 oracle 分别计算。

必须专门审查 `supplied`：当同一玩家先后舍出两张同牌种、只有其中一张被鸣时，只能标记
对应的那一个 river index/实体，不能因仅按牌种匹配而把多张都标记。覆盖红五/普通五同类、
同一牌种被不同副露使用、宣言牌被鸣和加杠后的当前形态。

### 5.6 MELD 与实体牌守恒

对 chi、pon、daiminkan、ankan、kakan 分别逐维核对 owner、完整构成、赤牌、called tile、
supplier、open、meld_index、役牌番、宝赤番。覆盖所有赤五在 called/consume 中的位置、三种
chi 形状、四个 owner 和所有合法 supplier 相对座次。

建立实体牌守恒 oracle，区分：

- 当前牌河区域中仍保留的被鸣牌；
- 副露完整构成中的 called tile；
- “公开区域计数”在语义上应按实体只计一次还是按区域表示计数；
- 宝牌指示、自己暗手与公开副露之间的 known/public/unknown 关系。

特别验证被鸣舍牌同时存在于 `discards` 和 `meld.tiles` 时不会在 TILE_STATE public/known 中
重复计算；若 Observation 的语义确实已从牌河移除被鸣牌，则以源码和 replay 证据证明，不能
凭假设。四枚上限的 cap 不得掩盖第五次错误计数。

### 5.7 固定 34 个 TILE_STATE

对每个 tile_type 逐维核对：自身暗手数、自身舍牌数/是否舍过、public、known、unknown、
all_seen、宝牌倍率/是否宝牌、场风/自风、赤五牌种、进张、和牌、对三家现物、三家筋、壁、
宝牌邻张。

必须验证以下不变量和反例：

- 对每种牌 `known + unknown = 4`，且各计数组成与实体事实一致；
- 赤五与普通五属于同一 tile_type 计数，但 red 身份在需要的位置仍无损；
- 重复宝牌指示正确增加倍率，不把指示牌自身误当宝牌；风牌/三元牌循环正确；
- 自己的逐张牌河虽然没有 token，但每种舍牌计数和三类振听不丢失；
- 现物只由对应对手的公开牌河语义产生，三家互不串位；
- 筋覆盖 1/4/7、2/5/8、3/6/9 边界、单侧/双侧与字牌 N/A；
- one-chance/no-chance 的相关牌、边张和公开计数定义正确；
- 宝牌邻张不跨花色、不循环到错误端点，字牌按契约为 0；
- is_advance/is_win 与 SELF_STATE_ANALYSIS 的计数及实体剩余数一致。

使用四个观察者座次旋转同一物理局面的变形测试：绝对事实不变，相对座次字段按规则置换，
Actor 看到的语义保持等价。

### 5.8 OPPONENT_ANALYSIS

每家逐维核对座次、立直状态/巡目/宣言牌、门清、暗牌数、副露数、杠数、公开役牌/宝赤、
立直后手切/摸切、最近六张手切/摸切、自己手中现物牌种/实体数和牌河长度。

证明这些字段只依赖 Actor 合法可见信息：固定所有公开事实和自身手牌，只改变对手真实闭手、
真实牌山、里宝牌或终局标签，三个 Analysis 的原始行必须字节级不变。逐字段消融时，目标
Analysis embedding 必须变化；在固定模型权重的受控 fixture 中，至少证明 Analysis 的变化
存在通向 action logits 的计算路径，而不是孤立或被 mask 屏蔽。

### 5.9 Action Offense/Defense Query

这是高风险项，不得只检查 query type、domain 和排序。对每个合法动作验证：

- action ID 唯一、在 0..240、与 legal mask、action JSON 和监督 target 一致；
- 每个 action 恰好 O/D 两行，按 ID 升序且同一对共享完全一致的 metadata；
- metadata 完整表达 action ID、动作类型、主牌、**完整 consume 组合及赤牌身份**、supplier、
  是否打出当前摸牌；
- chi/pon/daiminkan/ron supplier 为 1..3，其他为 N/A；
- O0–O9、D0–D9 每槽语义、N/A、bucket 和动作后状态分析正确；
- 同一 action type/primary tile 但 consume 组合、赤牌使用或 action ID 不同的合法动作不会得到
  完全相同且不可区分的 action embedding；
- `tsumogiri_mode` 来自动作真实语义，不能仅依赖未经验证的 action ID 数学规律；
- pass/打牌/reach/chi/pon/daiminkan/ankan/kakan/tsumo/ron/ryukyoku 全覆盖。

显式审查 `query_rows[15]` 转换到 `actor_factors[32]` 时哪些 metadata 被丢弃。若 action ID
只用于最终 scatter、未进入 token embedding，或完整 consume 根本不在 query row/token 中，
必须判定是否违反原始 FR-14，而不能以“241 action ID 已隐式编码”代替证明。至少构造红五
consume 和多种 chi/pon 组合的碰撞搜索，报告原始行、embedding 和 logits 路径。

## 六、按事件推进的差分与变形测试

选择包含完整生命周期的真实 MJAI replay，并在每个可决策点保存 Observation 与 decoded
facts。针对相邻事件做“预期变化白名单”：

| 事件 | 应变化的主要事实 | 不应变化的主要事实 |
| --- | --- | --- |
| `start_kyoku` | 桌况、初始手牌、宝牌、34 tile state | 不存在历史 token |
| `tsumo` | 当前摸牌、自手、向听/进张/等待、相关 tile state | 对手闭手不可见 |
| `dahai` | 对应牌河、自手/计数、手切摸切、年龄、振听 | 无关玩家事实 |
| `reach` / accepted | 立直状态、宣言牌/巡目、锁定、阶段 | 无历史 reach token |
| `chi` / `pon` | 当前副露、供牌标志、暗牌数、公开计数 | 不保留独立 call 事件 |
| `daiminkan` / `ankan` | 当前杠、公开/已知、暗牌与门清语义 | 不伪造普通副露历史 |
| `kakan` | pon 当前形态替换为 kakan | 不同时保留旧 pon |
| `dora` | 指示槽、宝牌倍率、已知/公开 | 不新增 dora 事件 token |
| `hora` / `ryukyoku` | 不得把事后标签回填给此前 Actor | 此前样本字节不变 |

对每个事件比较前后完整张量 diff，并由白名单解释每一个变化维度；出现意外变化或应变未变
即记录缺陷。再做以下变形测试：

- 打乱环境返回的合法动作顺序，规范编码、action IDs 和逐 ID logits 不变；
- 批次顺序置换后每个样本结果不变；单样本编码与 batch 编码一致；
- shard 边界、不同 batch size、不同 worker 数不改变样本字节；
- 改变明确省略的历史表示、事件 generation/cache，不改变当前快照；
- 两条不同历史若收敛到相同 `CanonicalVisibleFacts`，Actor 输入应相同；
- 当前可见事实有任何契约要求的差异时，decoder 必须能指出对应 token/维度差异。

## 七、RoPE、mask、嵌入和模型消费审查

### 7.1 RoPE 与 position

不要只测试 `_rope_values` 输出 finite。通过 forward hook 或等价手段验证每个分支实际送入
每一层 attention 的 Q/K 都应用了正确 RoPE：

- Actor 中所有非 padding token position 为 0..L-1，唯一、连续、单调；
- separator 占位置，类别边界不重置；
- 每个 action O/D 有独立连续位置，不复用局部 position；
- Critic 从 Shared 长度 P 继续，private、future、Value Query 都占唯一位置；
- Actor/Critic 是独立序列，分支间复用编号允许但分支内不允许；
- padding position 不得被任何有效 query 读取，也不得影响输出。

用位置置换、插入 separator、改变合法牌河顺序的受控反例证明 RoPE 确实影响表示，而不是
仅生成未消费的 cos/sin。

### 7.2 注意力 mask

对真实长度与多种 action 数显式导出 mask 矩阵并逐格与独立 oracle 比较：

- Shared ↔ Shared 全双向；Shared 不读 Analysis/Action；
- 三个 Analysis 读全部 Shared 和三个 Analysis，但不读 Action；
- 每个 O/D 读 Shared、三个 Analysis 和自己的 pair；
- 不同 action pair 互不可见；
- Critic 只含 Shared、三家闭手、未来五张、Value Query；
- Value Query 能读全部有效 Critic 输入；
- 所有 padding 列不可见，padding query 不产生有效输出。

做因果干预而不只看布尔矩阵：改变一个 action pair 的输入，其他 action raw logits 应保持在
数值容差内；改变 Shared/Analysis 可影响多个动作；改变 hidden private 只能影响 value。

### 7.3 embedding 逐槽消费

检查每个 schema 字段都实际被取出并送入自己的 embedding/projection，不存在：

- schema 声明了字段但切片漏读；
- 两个不同语义槽错误共享同一字段表或 slot ID；
- numeric 写入但模型固定传零；
- summary valid length 没有屏蔽 padding 子槽；
- code 0 同时表示有效类别与“严格零贡献”；
- action ID/consume 在预处理存在但进入模型前丢失；
- separator 只有 kind base embedding、没有类别专属可学习表示；
- 某字段 embedding 有梯度但永远收不到非零合法 code。

逐字段运行激活覆盖统计：训练抽样中 min/max、distinct、N/A 比例、饱和比例、永不出现值和
非法组合。随机合法组合的 embedding 碰撞搜索只能作为补充，不能替代语义碰撞测试。

## 八、Actor/Critic 信息边界

建立同局面成对 fixture：Actor 合法可见事实完全相同，分别改变三家真实闭手、未来五张、
更深牌山、里宝牌、终局结果。验证：

- `actor_factors`、`actor_numeric`、Analysis、Action Query 字节级相同；
- 按 action ID 对齐的 raw logits bitwise 相同或在确定性设备上的严格容差内；
- Critic private decoder 正确看到三家真实闭手与未来五张，顺序固定；
- 合法改变任一 private hand/future position 时，构造受控权重证明 value 有计算依赖；
- Critic shape、length、embedding 和 mask 中不存在 Analysis/Action；
- Actor-only SFT 不创建、前向、反传或优化 Critic/value 参数。

同时检查数据生成时间点，确保事后和牌、打点、终局名次或未来事件没有被预计算成 Actor
特征。仅做模型层“忽略 critic tensor”测试不够。

## 九、训练数据预处理逐阶段一致性

### 9.1 同一样本的张量指纹

为固定 replay 决策生成稳定 sample ID（来源文件/kyoku/decision index/observer seat），在
每个阶段保存 SHA256 和结构摘要：

```text
原始 replay + 决策动作
→ Observation
→ PyO3 rows/numeric/offsets
→ current_state.encode_batch
→ EncodedSample
→ precompute shard
→ iter_precomputed_samples
→ collate_samples
→ model.forward_actor
```

逐阶段比较有效区间必须字节级或张量级相等；padding 可不同但 length 内不得变化。核对
int32/int64、float32、bool、C-contiguous、端序、copy/view、offset 单位和 shape，防止 dtype
转换、拼接或 padding 产生静默变化。

### 9.2 shard、offset 与 manifest

至少测试空/单样本/多样本、不同 token/action 长度、跨 shard 边界和最后不足 shard：

- `actor_offsets`、`query_offsets`、`action_offsets` 从 0 开始、单调、尾值等于数组长度；
- 每个 sample 的 actor/query/action/label/legal/identity 字段仍属于同一决策；
- query 行数恰为 `2 * action_count`，target action 在 legal/action_ids 中且唯一；
- 保存后加载的 factors/numeric/query/action/legal/target 完全一致；
- shuffle/sampler/DDP rank 切分不改变样本内部对应关系，不重复或遗漏训练样本；
- manifest 对完整 schema、字段顺序、separator、模型关键配置、数据格式和 source selection
  fail closed，而不是只校验版本字符串；
- 修改任一 schema 字段名、cardinality、顺序、numeric scale、separator、context、query slot、
  dtype、shard key 或 hash 均被拒绝；缺键、额外危险键、坏 offsets、截断文件也被拒绝；
- 不得复用旧 V18/V16/V17 shard 或通过 fallback 读取 history/snapshot。

### 9.3 SFT 标签正确性

从 replay 中独立解码真实执行动作，核对观察者视角下的 241 action ID、supplier、consume、赤牌、
tsumogiri 和 legal set。覆盖 ron/tsumo、reach discard、赤五选择、多种 chi/pon consume、kan、pass。

逐样本证明：

- target 是该决策真正执行的动作，不是下一事件、其他玩家动作或绝对座次下的动作；
- target 对应唯一 query pair 和唯一 raw logit；
- BC loss 取的是该 action ID 的 logit；非法动作恒为 `-inf` 且不参与归一化；
- batch padding 的 action_id=0 不会伪造合法 action 0 或污染 loss；
- collator 之后 action_ids、legal_mask、query_pair_counts 和 target 仍一致；
- Actor-only optimizer 只包含允许训练的 Actor 参数，Critic/value 无梯度且 step 前后不变；
- checkpoint 保存/strict load 后模型输出、schema hash、manifest hash 和配置完全一致。

## 十、真实数据抽样、合成边界与覆盖标准

测试组合必须同时包含：

1. 手工可验证微型局面：每个字段能给出明确 expected；
2. 合成极端局面：牌河/副露/动作/context 上界和所有 bucket 边界；
3. 真实 replay：早巡、中巡、晚巡、四座次、立直、副露、杠、赤牌、重复宝牌、和牌响应；
4. 属性/变形测试：座次旋转、batch/action 顺序置换、同状态不同历史；
5. 受约束 fuzz：生成合法状态组合并检查守恒、domain、round-trip 和无截断。

真实 replay 不得只断言 `length <= 256` 和 forward finite。至少对每个 token kind 和字段输出：

- 非 padding 出现次数、distinct 值、min/max；
- N/A/零/padding/饱和 bucket 占比；
- 红牌、四种副露、三种决策模式、三种立直阶段、四个观察者座次覆盖数；
- 未覆盖字段和合法值清单。

在既定 60% selection 上只做不落完整数据集的流式代表性统计；报告 Shared/Actor/Critic 的
mean、p50、p95、p99、max 和各 segment 贡献。严格证明 context 256 上界，不得通过截断满足。
若无法用四麻规则证明上界，明确标为阻塞问题。

## 十一、现有测试的反向审查

逐个检查 `70247c8` 新增/重写的测试，回答：

- expected 是否来自被测生产函数或同一 schema，形成自证循环；
- fixture 是否真实改变了声称改变的底层 Observation；
- 测试是否只检查 shape/domain/finite，而没有检查语义值；
- `not allclose` 是否可能由无关位置、随机 dropout 或初始化引起；
- mask 测试是否覆盖三家 Analysis、多 action pair、padding 和 Critic；
- hidden-information 测试是在数据生成前改变隐藏事实，还是只在模型入口替换 critic tensor；
- dense gradient 测试是否真的覆盖**所有有效槽位表**，而不是“至少若干个”；
- padding 测试是否证明严格零贡献，还是仅证明两个 embedding 不相同；
- replay lifecycle 是否逐阶段比对字节，还是只检查能训练一步；
- 现有 fixture 是否覆盖赤牌、被鸣舍牌、重复同牌、kakan、bucket 边界和 action consume 碰撞。

给每个现有测试标注 PROVES / PARTIALLY PROVES / DOES NOT PROVE，并列出需要补充的断言。
测试数量和全绿状态不能替代此表。

## 十二、静态审查与一致性搜索

对 `02cd75e..15fcd6b` 和当前 HEAD 做静态审查，至少搜索：

- history、snapshot、54 行、atomic、event token、generation/cache、局部 query position；
- `tiles_left`、wall remaining、独立 current supplier/current tile；
- V16/V17 adapter、legacy fallback、state migration、旧 shard key；
- Q scorer、Q boost、candidate Q、MHA 双分支；
- schema 常量在 Rust/Python/docs/tests 的重复定义与不一致；
- hardcoded V18 路径、dataset、checkpoint、schema ID 和历史版本默认值；
- TODO、silent clamp、`unwrap_or`、`min/max/saturating_*` 是否掩盖非法状态；
- action ID 数学推断、tile code 转换、绝对/相对座次变换的散落实现。

历史报告可以保留旧描述，但必须与活跃文档区分。PPO 专用旧引用只分类记录，不在本轮处理。
任何发现都给出文件和行号，不只给搜索计数。

## 十三、测试命令与环境要求

所有 Python/扩展相关命令使用 `Mahjong-AI` Conda 环境。先确认加载的是本工作区刚编译的
`riichi` 和 `riichienv`，记录 `__file__`、协议版本和扩展构建信息，防止测试旧二进制。

CPU/Rust/PyO3 正确性测试不需要 GPU。模型 GPU 测试默认按项目约定使用
`CUDA_DEVICE=0,1`；本任务重点是正确性，除非需要验证设备/批次一致性，不运行正式训练。
涉及随机性的测试固定种子并关闭 dropout，说明 bitwise 或数值容差。

至少执行并记录：

```bash
git status --short
git log --oneline -8
git diff 02cd75e..15fcd6b --stat

env LD_LIBRARY_PATH=/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/lib \
  conda run --no-capture-output -n Mahjong-AI cargo test \
  --manifest-path RiichiEnv/Cargo.toml --workspace

conda run --no-capture-output -n Mahjong-AI \
  python -m pytest -q riichi_ppo_v1/tests/unit \
  riichi_ppo_v1/tests/protocol riichi_ppo_v1/tests/integration

conda run --no-capture-output -n Mahjong-AI \
  python -m pytest -q RiichiEnv/tests riichi_lab_bot/tests

conda run --no-capture-output -n Mahjong-AI \
  python -m riichi_ppo_v1.tools.validate --parameter-contract
```

此外执行本轮新增的 oracle/round-trip/event-delta/fuzz/shard-corruption/action-collision 测试。
每条命令记录 commit、环境、耗时、通过/失败/跳过数；失败不得只粘贴最后一行。

## 十四、缺陷分级与完成门槛

缺陷按以下口径分级：

- **P0 信息泄漏/监督错位**：Actor 读取隐藏或未来信息；SFT label 对错动作；样本跨决策错位；
- **P1 语义错误**：任一要求字段编码错误、漏字段、实体重复计数、座次错位、动作不可区分、
  mask/RoPE/critic 边界错误、截断；
- **P2 契约脆弱**：manifest 未 fail closed、padding/N/A 混淆、未使用字段、测试自证循环、
  文档与代码冲突；
- **P3 可观测性/维护性**：错误信息不足、统计缺失、重复常量或非阻塞文档问题。

只有全部满足才能给出“通过”：

1. 字段审查矩阵没有 FAIL、PARTIAL 或 UNTESTED；
2. 每个 token 的每个维度都有原始来源、编码公式、独立解码和自动化证据；
3. 无损字段 round-trip 完全一致，有损字段等价类正确，省略字段通过非干扰测试；
4. 真实 replay 每个决策的 decoded facts 与独立 oracle 一致；
5. 事件差分中的每一个 tensor 变化均能解释；
6. 实体牌、公开/已知/未知、座次、牌河、副露、振听、向听和动作语义不变量通过；
7. action metadata 足以区分全部 241 空间内的实际合法候选，target/query/logit 一一对应；
8. PyO3 → EncodedSample → shard → loader → collator → model 有效张量字节级一致；
9. RoPE、mask、Actor/Critic 隔离通过实际计算路径测试，而非只看辅助函数；
10. context 无截断、manifest fail closed、临时产物清理、归档资产未改动；
11. 所有新增和既有相关测试通过；若存在确认缺陷，则结论必须为不通过而不是“基本通过”。

无法证明不等于通过。对偶发失败、未覆盖合法值、依赖生产函数的 expected、只有 shape/domain
证据的字段，一律标 PARTIAL 或 UNTESTED。

## 十五、交付物

最终至少交付：

1. `audit/reports/v18/report/V18当前局面输入逐Token审查与测试报告.md`；
2. 字段级审查矩阵（可嵌入报告或放同目录自描述文件）；
3. 独立 decoder/oracle 与可复现审查脚本，放 `audit/reports/v18/scripts/`；
4. 新增的永久正确性测试，按职责放入 Rust 或 `riichi_ppo_v1/tests/`；
5. 缺陷清单：ID、等级、commit/文件/行、最小 replay/fixture、expected、actual、影响、
   回归测试建议；
6. 测试命令、环境、结果、覆盖统计、context/token/字段激活统计；
7. 全仓旧契约/PPO 待迁移分类和临时产物清理结果。

报告开头必须给出明确结论：PASS、FAIL 或 BLOCKED。随后按“结论 → P0/P1 问题 → 字段矩阵
摘要 → 数据链一致性 → 模型消费/信息边界 → 测试证据 → 未证明事项”的顺序组织。不要先写
大量过程再隐藏结论。

最终回复简洁列出：

- 总体结论和最高等级问题；
- 逐 token/逐维编解码是否全部得到独立证明；
- Observation/replay 到 SFT model input 是否字节级一致；
- action/query/label/logit 是否一一对应；
- Actor/Critic 隔离、RoPE、mask 和密集槽位是否通过；
- 新增证据文件与测试结果；
- 未修改 PPO、未生成完整数据集、未启动正式训练、未删除归档资产的确认。

现在请直接开始只读调查和审查，按上述流程连续推进。不要把现有 spec、schema、测试或
`PROGRESS.md` 的完成声明当作答案；审查目标正是验证这些声明是否真的由局面真值支持。
