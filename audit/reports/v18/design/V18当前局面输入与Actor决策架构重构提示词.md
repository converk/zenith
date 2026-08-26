# V18 当前局面输入与 Actor 决策架构重构提示词

将下面整段提示词复制到新的 Codex 对话中执行。

> 本文件只是下一次开发对话的任务提示词，提示词文件本身不需要用
> spec-kit 管理。后续实施对话必须使用 spec-kit 流程，并遵守 constitution、
> AGENTS.md 和项目既有治理规范。

````text
请在项目 `/mnt/disk1/hubowen/zenith` 中完成 V18 当前局面输入协议、数据预处理、
模型结构、Actor-only SFT 以及相关测试/文档的整体重构。PPO 阶段本次不考虑。

这是一项需要实际修改代码、测试、配置和文档并完成验证的开发任务，不是只输出方案。
V18 尚未开始正式训练，因此直接替换现行 V18 契约；不要新增 V19，也不要保留当前 V18
输入的兼容分支、旧数据转换器或 state-dict 迁移层。

## 一、spec-kit 执行流程与 sub-agent 分工

开始前完整读取并遵守仓库 `AGENTS.md`、`.specify/memory/constitution.md`、现行 V18
相关源码、协议文档和配置。仅在核对现行输入/模型/SFT 事实时参考
`specs/008-v18-input-architecture`；不修改或覆盖旧 feature。为本次重构创建一个新的
`specs/<NNN-v18-current-state-input-sft>/` feature，编号通过仓库现状确定，不要硬编。

主 agent 必须自己完整读取每个将使用的 spec-kit skill 指令，并按以下顺序连续推进，
不要让我在每个阶段之间手动继续：

1. 先做必要的只读调查，核对现行输入、Rust/PyO3、模型、SFT replay/precompute、
   dataset/collator、SFT 训练入口、测试和文档的真实调用链。
2. 使用 `$speckit-specify` 创建新 feature 的 `spec.md`，将本提示词中的输入契约、范围、
   信息隔离和验收条件完整落盘。
3. 只当存在会实质改变结果的真正歧义时使用 `$speckit-clarify`；能从代码、宪法和本
   提示词确定的事项不要反复提问。
4. 使用 `$speckit-plan` 生成实施计划和所需设计产物，明确协议/schema → Rust/PyO3 →
   replay/precompute/dataset → 模型与 mask → Actor-only SFT → 清理 → 测试/文档的依赖顺序。
5. 使用 `$speckit-tasks` 生成按依赖排序、可验收、可并行标记的 `tasks.md`；任务粒度以
   可执行和可分工为准，不为了形式拆出大量微小任务。
6. 使用 `$speckit-analyze` 做非破坏性一致性检查，先修复 spec/plan/tasks 中的阻塞性矛盾、
   遗漏和越界项，再开始实现。
7. 使用 `$speckit-implement` 按 `tasks.md` 分阶段实现；每完成一组依赖闭合的任务就运行
   针对性测试，失败时修复根因并同步任务状态。
8. 使用 `$speckit-converge` 对照代码与 spec/plan/tasks 检查未完成工作；如追加了任务，
   继续使用 `$speckit-implement` 完成，直到真正收敛。
9. 最后做全仓一致性复核：`rg` 旧契约引用并将 PPO 专用引用分类为后续待迁移项，
   核对本阶段配置/文档/schema、信息隔离、token/参数统计、测试证据和临时产物。

大量使用 sub-agents 承担路线明确、低风险或机械性的工作，让主 agent 把上下文集中在
契约、关键架构和整合验证上：

- 优先并行委派：全仓 `rg`/调用链盘点、文件责任映射、Rust/PyO3 字段传递、Python
  typing/schema 机械同步、dataset/collator 改造、独立测试 fixture、文档旧引用清理、
  token/参数/性能统计，以及边界清晰的测试失败诊断。
- 主 agent 必须亲自负责：读取和执行 spec-kit 与治理指令、确定最终 V18 契约、决定
  RoPE/mask/Actor-Critic 信息边界、处理破坏性删除、整合所有改动、运行最终验证和给出结论。
- 规划阶段可让 sub-agent 并行提供代码事实和风险输入，但 spec/plan/tasks 由主 agent 统一
  生成和收敛，不得由多个 agent 并行改同一份 spec-kit 产物。
- 实现阶段尽量将 `tasks.md` 中已标记可并行、且文件边界不重叠的任务同时交给
  sub-agents；每个 worker 都必须获得明确的文件/模块所有权、验收条件和禁止越界范围。
- 所有 worker 都必须知道其他 agent 也在修改仓库，不得回滚他人改动；不把同一文件交给
  多个 agent 并行修改。
- 主 agent 必须审查 sub-agent 结果与 diff，不得把子任务“完成”直接等同于整体验收通过。
- sub-agent 不得自行扩大范围、修改宪法、启动正式训练或擅自改动 spec-kit 主产物。

本任务不修改宪法：现行版本仍为 V18，Actor 仍只读取公开信息和自身私有信息，Critic
私有输入仍为三家真实闭手与未来五张牌，SFT 固定节奏不改变。PPO 代码、配置、训练、
评测和运行链均在本阶段范围外。如果实际调查发现
需求与宪法冲突，停止冲突部分并报告具体条款，不能静默修改宪法。

## 二、环境、治理与任务边界

- 工作目录固定为 `/mnt/disk1/hubowen/zenith`。
- 所有 Python、编码、测试和训练相关命令使用 `Mahjong-AI` Conda 环境。
- 涉及 GPU 的模型/SFT 性能测试默认使用 `CUDA_DEVICE=0,1`；仅显式单卡任务才
  使用 `CUDA_DEVICE=0`。
- V18 是唯一活跃输入、模型和 checkpoint 契约。V16/V17 资产只读归档，不得删除、
  覆盖或作为活跃输入。
- 当前未开始正式 V18 训练，不需要加载或迁移旧 V18 checkpoint。
- 后续正式 SFT 编码数据路径仍固定为
  `datasets/tenhou_sft_2024_2025_encoded_60pct_v18`，但本任务不生成完整 60% 数据集。
- 实现数据预处理代码、manifest/schema、抽样预计算、小型 fixture 验证与
  Actor-only SFT 完整小规模生命周期；不生成完整 60% 数据集，不启动正式长时 SFT、
  GRP 或 PPO 训练。
- 不修改或迁移 PPO 的 worker、rollout buffer、learner、推理/采样、训练配置、
  checkpoint/resume、1v3 评测与性能基线；也不恢复任何 Q scorer/Q boosting。
- 新的 V18 核心模型与 SFT 契约不得为 PPO 保留旧协议 adapter。如果旧 PPO 调用方因此
  暂时不兼容，在 V18 audit 记录为后续 PPO 阶段待迁移项，不纳入本次完成条件。只允许为
  保持包级 import 或通用 schema 编译正常而作必要的机械性修正，不得实质接入 PPO。
- 新增和修改的代码注释一律使用中文。
- 删除旧模块或文件前必须执行全仓 `rg` 引用检查；零引用且测试通过后才允许删除。
- 冒烟测试产生的临时日志和结果必须清理；正式验证记录写入既定 V18 audit/log 目录。

## 三、重构目标

现行 V18 使用“追加式历史事件 + 当前状态 suffix + 固定 Atomic Snapshot + 每动作一对
Query”。本次重构要彻底移除 Actor 的完整事件历史，将公共输入改成决策时刻的局面
快照：完整三家对手牌河、当前副露、当前手牌、34 种牌的确定性状态和少量高价值派生
信息。

核心原则：

1. 不输入“这一局依次发生了什么”的 MJAI 事件序列。
2. 保留“此刻桌面上仍然存在什么”的完整公开结构，尤其是三家对手牌河的逐张顺序。
3. 只加入确定、稳定、计算链较长的派生信息；不输入概率预测或人工综合危险分数。
4. 每个合法动作继续保留一个 Offense token 和一个 Defense token。
5. 三家 Opponent Analysis 是 Actor-only 决策输入，位于 Action Query 前；Critic 不读取。
6. 所有有效 token（包括 BOS、分隔符、摘要、Opponent Analysis、Action Query、Critic
   私有 token 和 Value Query）都必须应用 RoPE。
7. 不同输入类别之间使用类别专属的可学习分隔符。

## 四、规范序列布局

Shared 公共前缀固定按以下类别排列：

```text
[BOS]

桌况 token

[SEP_SELF_HAND]
自身手牌 token
SELF_STATE_ANALYSIS

[SEP_PLAYERS]
SELF_PLAYER
SHIMOCHA_PLAYER
TOIMEN_PLAYER
KAMICHA_PLAYER

[SEP_RIVERS]

[SEP_SHIMOCHA_RIVER]
SHIMOCHA_FIRST_SIX_SUMMARY
SHIMOCHA_DISCARD_1 ... SHIMOCHA_DISCARD_N
SHIMOCHA_RECENT_SIX_SUMMARY

[SEP_TOIMEN_RIVER]
TOIMEN_FIRST_SIX_SUMMARY
TOIMEN_DISCARD_1 ... TOIMEN_DISCARD_N
TOIMEN_RECENT_SIX_SUMMARY

[SEP_KAMICHA_RIVER]
KAMICHA_FIRST_SIX_SUMMARY
KAMICHA_DISCARD_1 ... KAMICHA_DISCARD_N
KAMICHA_RECENT_SIX_SUMMARY

[SEP_MELDS]
四家当前副露 token

[SEP_TILE_STATE]
固定 34 个 tile-state token
```

Shared 公共前缀到这里结束。

Actor-only 尾部：

```text
[SEP_OPPONENT_ANALYSIS]
SHIMOCHA_ANALYSIS
TOIMEN_ANALYSIS
KAMICHA_ANALYSIS

[SEP_ACTIONS]
ACTION_ID_ASC_1_OFFENSE
ACTION_ID_ASC_1_DEFENSE
ACTION_ID_ASC_2_OFFENSE
ACTION_ID_ASC_2_DEFENSE
...
```

Critic 使用独立尾部：

```text
Shared 公共前缀
[SEP_CRITIC]
三家真实闭手（固定相对座次顺序）
未来五张牌（固定摸牌顺序）
VALUE_QUERY
```

Critic 不接收 `[SEP_OPPONENT_ANALYSIS]`、三个 Opponent Analysis、`[SEP_ACTIONS]` 或任何
Action Query。Opponent Analysis 虽然只由公开信息派生，但按本协议明确归 Actor-only，
不得偷偷复制到 Critic 输入。

## 五、各类别输入信息

### 5.1 桌况

保留：

- 场风、局数、本场数、立直棒数；
- 庄家和观察者自风；
- 四家点数、观察者名次、观察者相对三家的点差；
- 所有公开宝牌指示牌，保留重复指示带来的宝牌倍率；
- 当前是主动决策、响应舍牌还是响应加杠；
- 自己当前摸牌；
- 自己立直状态。

明确不输入：

- `tiles_left` 或任何精确/估计剩余活牌数；
- 独立的“当前供牌者”和“当前供出的牌”公共 token；相关信息由最新牌河/当前加杠副露
  和需要 supplier 的 Action Query metadata 表达；
- 一发、流局满贯、双立直、海底、河底、岭上、抢杠等独立低频状态；
- 完整 MJAI 历史事件或历史 generation/cache 标识。

### 5.2 自身手牌与自身分析

自身手牌按固定 34 牌序，以当前非零牌种 token 表达：

- 牌种、张数；
- 是否包含赤五；
- 是否为当前摸牌；
- 当前立直状态下是否锁定。

`SELF_STATE_ANALYSIS` 是一个密集 token，包含：

- 是否门清、暗牌数、副露数；
- overall/standard/chiitoitsu/kokushi shanten；
- 当前进张牌种数、进张剩余实体牌数；
- 当前等待牌种数、等待剩余实体牌数；
- 永久振听、同巡振听、立直振听；
- 当前确定基础番和自身手牌/副露中的宝牌、赤牌数。

具体哪些牌是进张/和牌，不塞进该摘要，而是标记在固定 34 个 tile-state token 上。

### 5.3 四家玩家 token

每家一个紧凑 token：

- 相对座次、自风、是否庄家；
- 点数、名次、相对观察者点差；
- 暗牌数、副露数、杠数、是否门清；
- 牌河长度；
- 立直状态、立直巡目、立直宣言牌、立直后舍牌数；
- 副露中确定的役牌番和可见宝牌/赤牌数。

不得加入对手听牌率、预测等待、预测打点、染手/对对倾向或人工威胁等级。

### 5.4 自己牌河

自己牌河不使用逐张 Transformer token，以节省约 10–15 个平均 token。其必要信息必须
无损转移到：

- 34 个 tile-state 中每种牌的自己舍牌张数/是否舍过；
- SELF_PLAYER 中的牌河长度和立直宣言信息；
- SELF_STATE_ANALYSIS 中的精确振听类型。

不得因删除自己逐张牌河而破坏永久振听、同巡振听或立直见逃语义。

### 5.5 三家完整牌河

三家对手牌河逐张保留，每张当前牌河牌一个 token；按下家、对面、上家，且每家内部从
早到晚排列。每个 token 至少包含：

- 对手相对座次；
- 该玩家本地 river index；
- 牌种与赤牌；
- 手切或摸切；
- 立直前、立直宣言牌、立直后三态；
- 是否作为吃、碰、杠供牌；
- 相对该玩家最新舍牌的年龄桶。

这些 token 表达的是当前牌河区域，不是历史事件：不生成摸牌、reach、reach_accepted、
dora、chi、pon、kan 等独立事件 token。

每位对手额外固定两个摘要 token：

1. `FIRST_SIX_SUMMARY`：前六张舍牌，不足六张带有效长度；
2. `RECENT_SIX_SUMMARY`：最近六张舍牌，按由旧到新排列，不足六张带有效长度。

两个摘要都必须保留六个内部子槽位的顺序，子槽位至少包含牌种/赤牌、手切/摸切以及
立直阶段。摘要本身只占一个 Transformer token；内部六槽不是额外 token，使用固定
slot-id embedding 或等价机制区分 1..6，padding 不能与合法牌混淆。

### 5.6 当前副露

每个当前副露一个 token：

- 拥有者；
- chi/pon/daiminkan/ankan/kakan；
- 完整构成牌与赤牌；
- 被鸣牌、供牌者；
- 开放/暗置；
- 当前副露序号；
- 该副露确定贡献的役牌番和可见宝牌/赤牌数。

只表达当前形态；加杠不再同时保留过去的碰事件。

### 5.7 固定 34 个 tile-state token

每种牌固定一个 token，包含模型从散布的手牌、牌河、副露和指示牌中精确汇总较困难、
但完全确定的信息：

- 自己暗手张数；
- 自己舍牌张数/是否舍过；
- 总公开张数、包含自身手牌后的总已知张数、未知实体牌数、是否四张全见；
- 是否为宝牌及宝牌倍率；
- 是否场风/自风/赤五对应牌种；
- 是否为当前进张、是否为当前和牌；
- 对三家分别是否现物；
- 对三家的筋类别：非筋、单侧筋、双侧筋、不适用；
- 壁类别：无壁、one-chance、no-chance/四枚壁；
- 是否为宝牌邻张。

不输入人工合成危险分数或任何用隐藏手牌监督后直接回填的概率。

### 5.8 三个 Actor-only Opponent Analysis token

每名对手固定一个密集 token，像 Action Query 一样由确定性槽位聚合，不是空白 learned
query，也不是概率预测器。槽位至少包含：

- 相对座次；
- 立直状态、立直巡目、立直宣言牌；
- 门清/开放、暗牌数、副露数、杠数；
- 副露中确定役牌番、可见宝牌/赤牌数；
- 立直后手切数、立直后摸切数；
- 最近六张的手切数、摸切数；
- 自己手中对该家的现物牌种数、现物实体牌数；
- 对手牌河长度。

三个 Analysis token 位于 Actor 分支、Action Query 之前，并参与 action logits。不要声称
同一个 Transformer 层能让空白 Analysis token 先读取牌桌、再把更新后的输出传给同层
Action Query；标准自注意力没有这种层内顺序。当前正式方案使用预计算确定槽位形成的
Analysis embedding，因此保持一个 Actor-only 决策层即可。如果实现前要改成真正的
learned two-stage analysis，必须明确增加独立 Actor Analysis 层、重新核算参数和性能，
不能仅靠调整 token 排列假装实现。

### 5.9 Action Offense/Defense Query

每个合法 action 恰好两个 token，按 action ID 升序规范排序：先 Offense，后 Defense。
环境返回的合法动作顺序不得影响编码结果。

共享 metadata 至少包含：action ID、动作类型、主牌、完整 consume 组合、供牌相对座次、
是否打出当前摸牌。需要 supplier 的 chi/pon/daiminkan/ron 必须为 1..3；其他动作使用
N/A。不要新增独立公共供牌 token。

Offense 保留现行 O0–O9 语义：

- O0 动作后向听；
- O1 有效牌种数；
- O2 有效牌剩余数；
- O3 等待牌种数；
- O4 等待役覆盖；
- O5 基础番；
- O6 振听；
- O7 门清；
- O8 可立直；
- O9 保留宝牌/赤牌数。

Defense 保留现行 D0–D9 语义：

- D0–D2 对三家是否现物；
- D3–D5 对三家的筋关系；
- D6–D8 动作后手中对三家的现物库存；
- D9 候选牌公开出现数。

动作对仍相互隔离：每个动作对可读取全部 Shared 公共前缀、三个 Opponent Analysis 和
自己的 Offense/Defense 两 token；不同动作对不能读取彼此。公共 token 不能读取 Actor
尾部，Opponent Analysis 不能读取 Action Query。

## 六、RoPE、分隔符和注意力布局

### 6.1 所有 token 都使用 RoPE

每条分支中的所有有效 token 使用连续、唯一、单调递增的 position ID，padding 除外。
分隔符本身占位置，不在类别边界重置位置。

Shared 公共前缀位置为 `0..P-1`。Actor 尾部从 P 继续；Critic 尾部在自己的分支中也从
P 继续。Actor 与 Critic 是独立序列，因此分支后允许复用位置编号。

现行 `_isolated_action_layout` 让不同动作对复用两个局部 position ID；本次重构必须
改掉。Action Query 按 action ID 规范排序，每个 Offense/Defense token 获得自己的连续
RoPE 位置。原“任意重排已编码 Query pair 后 logits 不变”测试应替换为“任意打乱环境
合法动作输入，经规范排序编码后序列和 logits 不变”。

RoPE 只提供位置关系；每个 token 还必须有 token-type、segment/category、relative-seat
等内容 embedding。RoPE 不得替代这些类别信息。

### 6.2 分隔符

使用类别专属 learned separator，不共用一个无类型 `[SEP]`。所有分隔符进入 RoPE、mask、
length、manifest 和协议校验。分隔符集合单点定义，顺序严格校验，缺失、重复和错序均
fail closed。

### 6.3 Shared 公共注意力

新的公共输入是同一决策时刻的状态快照，不是自回归事件流。Shared 公共 backbone 使用
有效公共 token 之间的双向 GQA，而不是沿类别排列做 causal attention；不能让任意的类别
顺序限制桌况、手牌、牌河、副露和 tile-state 的相互建模。所有 Q/K 仍应用 RoPE。

### 6.4 Actor 结构化 mask

- Shared 公共 token 只读取 Shared 公共 token；
- 三个 Opponent Analysis 可读取全部 Shared 公共 token 和三个 Analysis，不能读取动作；
- 每个 Action Query 可读取全部 Shared 公共 token、三个 Analysis 和自己动作对；
- 不同动作对互不可见；
- padding 永不被有效 token 读取。

### 6.5 Critic mask

Critic 只读取 Shared 公共表示、三家真实闭手、未来五张牌和 Value Query。Value Query 必须
能读取全部有效 Critic 输入。Actor-only Analysis/Action token 不得进入 Critic shape、
length、embedding 或 mask。

## 七、密集 token 的嵌入设计

正式拓扑保持 `d_model=256`。不要因为一个 token 含有十几个槽位就把 Transformer 残差
宽度提高到 384/512；槽位数量不等于需要同等数量的正交维度。当前问题在于现行
`FactorEmbedding`/`QueryEmbedding` 主要把多个 d_model 向量直接求和，密集槽位之间只有
很弱的非线性交互。

为密集类别实现槽位感知的非线性融合，至少覆盖：

- FIRST_SIX/RECENT_SIX summary；
- SELF_STATE_ANALYSIS；
- player token；
- Opponent Analysis；
- Offense/Defense Query；
- 需要多个复合字段的 meld token。

推荐正式配置：

- `d_model=256`；
- 每个离散槽位使用独立字段/位置 embedding 表，不能让不同语义槽位误共享 ID；
- `dense_slot_dim=32`（如设计阶段证明某个高基数字段需要不同维度，必须说明）；
- 将固定槽位 embedding 按规范顺序 concat，而不是无类型求和；
- numeric 先做稳定归一化，再通过专属投影拼接；
- `dense_fusion_dim=512`；
- 使用 `RMSNorm + gated/SiLU MLP + projection` 投影回 `d_model=256`；
- padding 子槽必须为严格零贡献，并由 valid-length 显式区分。

简单的单事实 token 可以继续使用轻量 factor embedding；不要强迫所有 token 走同一个巨大
MLP。嵌入实现应按职责拆分为自描述模块，避免把所有 schema 和融合逻辑继续堆在
`architecture.py`。

必须测试：

- 每个有效槽位单独改变都会改变最终 token embedding；
- 交换前六/最近六内部两个位置会改变 embedding；
- padding 与合法零值不混淆；
- 梯度能到达所有槽位 embedding 和融合层；
- 激活槽位数量变化不会因简单求和导致未归一化幅度爆炸；
- 大批随机合法组合中不存在由编码器错误造成的完全相同 embedding；
- metadata 消融会影响对应 Action/Opponent Analysis embedding 和最终 logits。

## 八、模型宽度与 GQA/MHA 决策

生产契约保持：

- `d_model=256`；
- `query_heads=16`；
- `kv_heads=4`；
- `head_dim=16`；
- `ffn_dim=704`；
- 3 个 Shared 公共层；
- 1 个 Actor-only 决策层；
- 2 个 Critic 层；
- RMSNorm、RoPE、gated FFN；
- GQA，不改 MHA。

理由必须写入最终设计说明和 V18 审计记录：

1. 密集 token 的可区分性首先由槽位 embedding 和融合函数决定，不由 KV head 是否共享
   决定；MHA 不能修复无类型求和造成的表示碰撞。
2. 现行模型实测总参数为 4,940,802。仅把 `kv_heads` 从 4 改为 16（MHA），其余不变，
   参数即变为 5,530,626：每层多 98,304，6 层合计多 589,824，还会把独立 K/V 宽度和
   KV 存储提高四倍。
3. 16 个 Query heads 已提供多种关系查询，4 个 KV groups 对当前不超过约 256 token 的
  麻将状态足够；没有训练证据表明 MHA 的额外 K/V 子空间比新的输入融合更重要。
4. GQA 本来就是 MHA 与 MQA 之间的折中，目标是在接近 MHA 质量的同时减少 K/V 成本；
   参考 Ainslie 等人的 GQA 论文：
   `https://aclanthology.org/2023.emnlp-main.298/`。
5. RoPE 对所有 token 的要求与 GQA/MHA 无冲突；参考 RoFormer：
   `https://arxiv.org/abs/2104.09864`。

不要在生产代码中保留 GQA/MHA 双分支或实验 flag。可以在实现前的简短调查记录中给出理论参数、
FLOPs 和显存对比，但实现只保留最终 GQA 契约。新的密集融合会改变总参数量，因此不再
机械维持旧设计的 4.9M–5.1M 窄范围；必须提供统一参数统计，解释各模块增量，并将
最终总参数控制在 6.0M 以内。若超过 6.0M，先减少重复的类别专属融合层或共享合理组件，
不能擅自降低 d_model、层数或删除输入信息。

## 九、Actor/Critic 信息边界

Actor 只读取：

- 观察者自身手牌和自身私有振听状态；
- 当前公开桌况、三家完整牌河、四家公开副露、公开宝牌指示；
- 只由上述合法可见信息计算的 tile-state 和 Opponent Analysis；
- 合法动作 Offense/Defense Query。

Actor 不得读取：

- 对手闭手；
- 对手真实摸牌；
- 真实牌山顺序；
- 里宝牌；
- 事后和牌/打点/终局标签；
- Critic token。

Critic 私有输入保持现行契约：三家真实闭手和未来五张牌。不要扩大、缩小或改变顺序。

必须有信息隔离测试：固定全部 Actor 输入，只改变三家闭手或未来牌山，逐 action ID 的
Actor raw logits 必须不变；合法改变 Critic 私有输入时 value 必须能够变化。还要证明
三个 Opponent Analysis 只存在于 Actor branch，Critic 输入长度和 value 不受这些 token
的直接拼接影响。

## 十、编码、数据预处理和 SFT 数据一致性

实现必须覆盖并统一以下路径：

- RiichiEnv 原生 `Observation` → 当前状态事实；
- Rust 状态/分析与 PyO3 批编码；
- SFT replay/precompute；
- encoded dataset manifest、shard schema、collator；
- Actor-only SFT 前向、保存与加载；
- SFT 训练入口、loss、optimizer step、checkpoint 与 strict load。

同一局面从 Observation/replay fixture、PyO3 批编码、SFT precompute、encoded
shard 解码到 collator/模型入口，必须生成字节级/张量级一致的公共、Opponent
Analysis 和 Action Query 编码。不能保留“precompute 新协议、SFT 训练仍读旧历史
token”的双轨。

编码应直接以 `Observation` 当前字段（hands、discards、tsumogiri_flags、melds、scores、
riichi 状态、dora indicators、合法动作等）构造快照；不得为了模型输入继续维护追加式
历史 token。MJAI `new_events()` 可以继续用于环境同步、生命周期、奖励或动作执行，但
不能进入 Actor 输入。

在本阶段所属的 schema、预处理、模型与 SFT 路径中移除或重构现行：

- `history_factors/history_numeric/history_lengths/history_generations` 活跃模型契约；
- append-only event token cache 及其在数据预处理/模型/SFT 中的依赖；
- 固定 54 行 Atomic Snapshot 作为独立输入块；
- 与新类别重复的旧 opponent summary；
- 旧 causal-history position/mask 假设；
- 旧 Query pair 共享局部 position ID 的协议。

删除具体文件前先 `rg`；若模块仍承担 MJAI 动作解码、生命周期或其他独立职责，应拆分职责，
不能因删除历史输入而误删仍在使用的功能。

不修改 PPO 专用的 rollout KV cache、buffer 或 worker 调用链；它们对旧输入契约的依赖只做
引用盘点并记录为后续待迁移项，本次不清理、不兼容、不测试。

encoded manifest 继续使用 V18 版本标识，但 contract SHA256、字段 schema、segment 顺序、
分隔符、RoPE/position 语义、模型配置和数据格式必须全部更新并 strict 校验。任何现存旧
V18 encoded shard/checkpoint 都视为不兼容，不提供迁移；也不要在本任务中覆盖或删除
物质资产。

本任务只生成最小 fixture、抽样 shard 或临时预计算用于测试，完成后清理临时产物。不得
生成完整 `datasets/tenhou_sft_2024_2025_encoded_60pct_v18`。

## 十一、长度、排序与上下文契约

- Action pair 按 action ID 升序；三个对手始终按下家、对面、上家；三家牌河内部按本地
  river index 升序；34 tile-state 按固定 34 牌序。
- 所有 category separator、BOS 和有效 token 计入 length。
- 不允许静默截断牌河、摘要、meld、analysis、query 或 Critic 私有输入。
- 目标 `context_tokens=256`。实现前从四麻规则、最大三家牌河、副露、合法动作数和
  Critic 私有段证明理论上界，并以合成极端测试验证。若严格上界确实超过 256，使用下一个
  有充分余量的固定值并解释，不能靠截断满足 256。
- 在既定 60% selection 上做不落完整数据集的代表性统计，报告 public/Actor/Critic 的
  mean、p50、p95、p99、max，以及各 segment 的贡献。
- 当前预期 Actor 长度大致为早巡 85–105、中巡 105–135、晚巡 130–165、极端 185–215；
  这是审计参考而非篡改数据去满足的硬目标。偏差显著时查明原因并记录。

## 十二、测试与验收

本次验收只运行输入/schema、数据预处理、模型和 SFT 相关的 unit/integration/
performance 测试。不运行 PPO worker、rollout、learner、resume、1v3 或 PPO 端到端测试；
它们因待迁移接口而暂时不通不属于本次阻塞项，但必须在 audit 中如实列出。

至少完成以下自动化验证：

### 协议/schema

- segment、separator、字段域、固定顺序、padding、N/A、赤五、相对座次；
- 三家每人恰好一个 first-six 和 recent-six summary；
- summary 内部顺序、有效长度与不足六张 padding；
- 自己没有逐张 river token，但自己舍牌计数和振听完全正确；
- 三家完整牌河逐张、顺序、手切/摸切、立直阶段、被鸣标志正确；
- meld 当前形态正确，kakan 不重复历史 pon；
- 恰好 34 个 tile-state，已知/未知张数守恒；
- 不存在 `tiles_left`、独立当前供牌公共字段或 MJAI 历史事件 token。

### RoPE/position/mask

- Shared、Actor、Critic 每条分支所有非 padding token 都有 RoPE；
- position ID 连续唯一，分隔符占位，分支处正确续接；
- public 双向 mask；
- 三个 Opponent Analysis 与 Action Query 的结构化可见性；
- 不同 Action pair 隔离；
- Critic 不含 Analysis/Action；
- 打乱环境合法动作顺序，规范排序后的张量和 logits 不变；
- 改变合法牌河顺序/river index 会产生预期不同表示。

### 嵌入与模型

- dense slot sensitivity、内部顺序、padding、梯度与尺度测试；
- d_model、GQA heads、层数、FFN、RoPE 和 context 配置 strict 校验；
- 统一参数统计，总参数不超过 6.0M，并报告 embedding/shared/actor/critic/head 分项；
- state dict 不含 MHA 双分支、旧 Snapshot/history adapter、Q scorer/Q boost；
- raw logits 与 action ID 映射唯一，非法动作严格为 `-inf`；
- 三个 Opponent Analysis 的任一有效字段改变可以影响其 embedding，并在适当 fixture 中
  能影响对应 action logits。

### Actor/Critic 与训练接口

- 隐藏信息改变不影响 Actor logits；
- Critic 实际使用三家闭手与未来五张牌；
- Critic 不读取 Opponent Analysis 或 Action Query；
- Actor-only SFT 前向、BC loss、反向、optimizer step、保存、strict load；
- SFT 中 Critic/value 冻结且无梯度；
- Observation/replay fixture、PyO3、precompute、shard、collator 和模型入口编码一致；
- dataset manifest/hash/schema 错误 fail closed；
- 小型 replay → precompute → encoded shard → collator → Actor-only SFT 完整生命周期集成测试。

### 性能

- CPU/Rust/PyO3 编码吞吐和分段耗时；
- GPU Actor-only SFT 前向/反向、显存、tokens/s；可单独测量 Actor-Critic 模型前向以验证
  结构与显存，但不运行 PPO 性能/训练基线；
- 冒烟测试结束后清理日志与结果文件。

不要通过删除测试、放宽 fail-closed 校验、截断输入或新增 legacy fallback 来让测试通过。

## 十三、文档、配置和审计同步

必须完成以下文档和审计同步：

- 重写 `riichi_ppo_v1/docs/v18_input_protocol.md`；
- 更新 `KyokuEventTupleProtocol.md`，明确事件只用于同步、不再作为模型输入；
- 更新模型/SFT/encoded manifest 文档和配置；PPO/rollout 文档不做协议迁移，仅在
  V18 audit 中明确标记为后续阶段；
- 更新根 README、`riichi_ppo_v1` README、`AGENTS.md` 中的活跃 V18 描述；
- 如职责变化，更新 `docs/directory-responsibilities.md`；
- 更新生产校验入口和 CLI 默认值；
- 在 `audit/reports/v18/report/PROGRESS.md` 记录改动范围、schema hash、参数量、token 统计、
  性能、测试命令和结果；
- 全仓搜索仍把事件历史、54 行 Snapshot、旧局部 Query position 描述为现行协议的文档；
  更新本阶段的活跃文档，对 PPO 专用文档只添加“待迁移、当前不适用新 V18 输入”
  的明确标记，不实施 PPO 协议改造；
  历史报告本身保留原貌。

配置必须完整自包含。本阶段的 V18 模型/SFT 配置写明 d_model=256、16Q/4KV GQA、dense_slot_dim=32、
dense_fusion_dim=512、层数、FFN、RoPE base 和最终 context 上限。不得使用 overlay 或旧配置
隐式继承。

## 十四、完成条件和最终交付

只有以下全部满足才可宣告完成：

1. 最终实现、配置、测试和文档与宪法及本提示词一致；
2. 现行 V18 输入不再含事件历史、tiles_left、独立当前供牌 token 或 54 行 Snapshot；
3. 三家完整牌河、每家两个六张摘要、当前副露、34 tile-state 正确；
4. 所有类别分隔符和所有有效 token 应用 RoPE；
5. 三个 Opponent Analysis 位于 Actor-only 尾部、Action Query 之前，并影响 action logits；
6. Critic 不接收 Opponent Analysis/Action Query，私有输入仍严格为三家闭手 + 未来五张；
7. Offense/Defense O0–O9/D0–D9 保留，动作排序和 pair 隔离正确；
8. d_model=256、16Q/4KV GQA，密集槽位使用非线性融合，生产代码没有 MHA 双分支；
9. 总参数不超过 6.0M，context 理论/实测不截断，token 统计完整；
10. Observation/replay、PyO3、SFT precompute、shard、collator 与模型入口编码一致；
11. Actor-only SFT、Actor/Critic 隔离、RoPE/mask/schema/回放/集成/性能测试通过；
12. 本阶段范围内的旧活跃实现经引用检查后清理，无 V16/V17/V18 legacy adapter
    或 state migration；PPO 路径的待迁移引用已单独盘点；
13. 配置、协议、README、AGENTS、audit 记录与代码一致；
14. 全仓一致性复核确认本阶段活跃路径没有旧契约引用或遗漏实现，PPO 待迁移
    引用已分类记录，工作区没有未清理临时产物；
15. 没有生成完整 V18 数据集，没有启动正式长时 SFT、GRP 或 PPO，没有删除归档资产。

最终回复简洁列出：

- spec-kit 各阶段产物、任务完成情况和全仓一致性复核结论；
- 核心协议和序列布局；
- 主要 Rust/Python/model/precompute 文件；
- d_model/GQA/密集融合、参数量和上下文统计；
- RoPE、separator、Action mask、Actor-only Opponent Analysis、Critic 隔离证据；
- 预处理各阶段与 SFT 模型入口的一致性；
- 测试与性能结果；
- 文档同步清单；
- 未运行完整数据生成和正式长时 SFT 的确认；
- 本阶段未修改、未接入、未测试 PPO，以及已记录 PPO 后续迁移点的确认。

现在请直接开始调查，按照上述 spec-kit 流程连续自主推进，不要只复述提示词，也不要让我
手动驱动每个阶段。请积极使用 sub-agents 并行处理路线明确、低风险、机械性或不需要
主 agent 保留完整上下文的工作；最终契约、spec-kit 产物、集成和验证仍由主 agent 统一负责。
````

## 架构结论摘要

- 保持 Transformer `d_model=256`；密集 token 的问题用槽位专属 embedding 和
  `dense_fusion_dim=512` 的非线性融合解决，而不是扩大残差宽度。
- 保持 16Q/4KV GQA，不改 MHA。现行模型参数量为 4,940,802；仅把 KV heads 改成 16
  就会上升到 5,530,626，增加 589,824 个参数，但不能解决输入槽位线性相加的问题。
- 全部 token 使用 RoPE；公共快照使用双向 GQA，Actor 的 Opponent Analysis/Action
  Query 使用结构化隔离 mask，Critic 使用独立尾部。
