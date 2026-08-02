# v13 SFT 端到端独立语义审计

审计日期：2026-08-02（Asia/Shanghai）  
审计对象：`/mnt/disk1/hubowen/zenith`，分支 `sft`，HEAD `2ac9fd51e0cb4ec39aa0d4f628ab0781f3e4f3ec`

## 一句话结论

**不建议按当前 canonical 双卡配置开始正式 SFT。** 正式 v13 缓存本身通过了全量结构对账和 34 个定向/分层小局的原始数据重放，**不需要重做**；阻断项是：现有 checkpoint 的旧 `rank_steps` 不能安全套用当前全局连续分片计划，以及双卡下 group/rule 辅助损失按 rank 局部 eligible mean 求平均，目标并不等于全局样本 mean。

## 1. 严重问题

### P0-1：不得用当前 HEAD 直接续训现有 `latest.pt` / `best.pt`

- 位置：`riichi_ppo_v1/sft/train.py:686-701`；当前 loader 计划在 `riichi_ppo_v1/sft/precompute.py:538-590`。
- 直接证据：两个历史 checkpoint 都没有 `data_cursor`，仅有 `rank_steps=[15000,15000]` / `[18000,18000]`。当前 fallback 声称旧 `rank_steps` 可兼容；但上一提交的 loader 是 `paths[rank::world_size]` 文件轮转分片，而当前 HEAD 是全局 row stream 的连续 rank interval。checkpoint 时间为 05:02/05:04，当前 HEAD 提交时间为 14:34；且新保存器必定写入 `data_cursor`（`train.py:370-375`）。
- 触发：`--resume checkpoints/train_riichi_v13_sft/latest.pt` 或 `best.pt`。
- 影响：跳过的“第 N 个本地 batch”属于另一套身份顺序和 rank 划分；会重复或遗漏旧运行已消费的 identity，无法称为严格续训或可复现续训。
- 缓存：不污染缓存；无需重做缓存。
- 历史 checkpoint：模型权重可用于评估或作为新的初始化权重，但**不能连同 optimizer/scheduler/rank progress 当作当前 HEAD 的续训状态**。
- 建议：删除未版本化 fallback，或在 checkpoint 中记录并校验 `data_plan_version`；旧 checkpoint 必须明确拒绝恢复。若要迁移，只加载模型权重并启动一个全新的输出目录、optimizer、scheduler 与 epoch plan。

### P1-1：双卡辅助损失的 DDP 归一化语义错误

- 位置：group loss 在 `riichi_ppo_v1/sft/train.py:289-296`，rule teacher loss 在 `:312-320`，组合在 `:793-804`。
- 触发：`learner_gpus=2` 且 `group_coef=0.25`、`rule_coef>0`；即正式配置。
- 原因：每个 rank 对自己的 eligible 子集求 mean，DDP 再等权平均两个 rank 的梯度；当 eligible 数不同，这不等于所有 eligible 样本的全局 mean。
- 直接证据：正式训练流前 32 个双卡 batch 中，group eligible 数 32/32 次不相等，rule eligible 数 31/32 次不相等。group 两 rank 差值绝对值最大 158、均值 19.25；rule 最大 41、均值 16.875。示例第一步 group 为 73/55、rule 为 201/237。
- 影响：辅助目标被“每 rank 等权”而非“每样本等权”系统性重加权；历史 checkpoint 也已受影响。最后 48/47 样本的 policy CE 还有一次很小的同类尾批偏差。
- 缓存：不污染缓存；无需重做缓存。
- 建议：每项 loss 返回 local numerator/count，以全局 count 对 local sum 正确缩放（注意 DDP 自身还会除 world size），并补不等 eligible 数和空 eligible rank 的双卡测试。修复后才从随机初始化正式训练。临时把 `group_coef=rule_coef=0` 可绕开主要问题，但这不是当前 canonical 配置。

### P1-2：v13 replay semantics 没有进入 feature hash，也没有在 loader 查询运行时

- 位置：常量和 legacy contract 在 `riichi_ppo_v1/model/feature_schema.py:14-53`；v13 `FEATURE_CONTRACT` 从 `:78` 开始，但不含 replay version；hash 在 `:160-162`。v13 loader 只校验 feature/Rust/decision 三项（`precompute.py:518-524`），`assert_legacy_replay_runtime()` 只在 schema 11 调用（`:525-526`）。
- 触发：安装/加载 replay 语义不同但 feature/Rust/decision 常量不变的 `riichienv` 扩展。
- 影响：旧缓存可被静默当作兼容，离线/在线时序语义可能漂移。manifest 本身也没有显式 replay version。
- 当前实例：当前加载扩展报告 replay semantics `1`，扩展与本仓库 release artifact SHA256 完全相同；34 小局 fresh replay 与正式 NPZ 全行一致。因此这是**契约缺口，不是当前缓存已损坏的证据**。
- 缓存：当前无需重做；未来若语义确实改变，则必须重编码。
- 历史 checkpoint：当前扩展下 sampled/full-kyoku 结果一致；但 checkpoint/manifest 没有独立绑定这个运行时字段。
- 建议：将 replay semantics 纳入 v13 `FEATURE_CONTRACT` 和 manifest，并在 precompute、loader、checkpoint resume、head-to-head/heuristic checkpoint load 全部 fail closed 校验。

### P1-3：生成和 checkpoint 落盘的故障原子性不足

- `prepare.py:222` 不拒绝非空输出，tar/index/errors 直接写入（`:223-243`），manifest 直接写入（`:314-316`）。失败后重跑可能把 stale shard 混入。
- checkpoint 直接 `torch.save(..., path)`（`train.py:347-383`），中断可能破坏正式文件。
- precompute 的 NPZ 使用临时文件加 `os.replace`（`precompute.py:114`），且拒绝非空输出（`:362-363`），这部分良好；但最终 manifest 在 `:496` 直接写。失败会留下无 manifest 的不可训练目录，安全但难以恢复。
- 当前实例：raw/NPZ/manifest/checkpoint 均可完整读取、hash 和对账；没有当前污染证据，也不需要重做缓存。
- 建议：所有最终文件采用同目录临时文件、fsync（按需求）和 `os.replace`；`prepare` 默认拒绝非空目录，或使用 staging 目录整体提交。

## 2. P2 问题与解释性风险

1. 正式 manifest 缺 `policy_head_type`、action-space version、data cursor/plan version、training mode、model config、dataset manifest 自身 hash、显式 replay version；`sample_identity_contract=null`。checkpoint 补充了其中多项并绑定 manifest SHA，但数据入口无法单独 fail closed。
2. `train.py:681` 对缺失 `training_mode` 使用当前 expected mode 作为默认值；当前 checkpoint 有 `actor_only`，不会被 value/critic 配置误载，但更旧的缺字段 checkpoint 可能静默通过。
3. scheduler 在 optimizer 后 step（`train.py:814-819`），总步数 183,485 正确；`LambdaLR` 的 `max(step,1)` 使初始化和第一次 scheduler step 使用相同 warmup scale，min LR 在最后一次 optimizer 后的保存状态才到达。这不造成步数错位，但应文档化期望曲线。
4. 当前 validation 已修复为在 length bucketing 前 `islice`（`train.py:418-425`）。历史 checkpoint 保存的 validation loss 来自旧的“先 bucketing 后截断”实现，固定集偏向短 context；所以历史 `best.pt` 的 best 判据对旧运行内部一致，但其存储指标不能与当前 150k 直接比较。
5. heuristic evaluation 没有调用 metrics 的 `record_lineup`，所以 `opponents/current_seat_fraction` 和 `current_decision_fraction` 都为 0；这两个字段无效，应排除，不影响对局分差/名次等核心结果。

## 3. 当前运行版本与契约

| 项目 | 当前 Python/Rust | manifest | checkpoint | 结论 |
|---|---:|---:|---:|---|
| encoded format | `riichi-sft-encoded-v3` | 相同 | 间接绑定 | 一致 |
| token schema | 13 | 13 | 13 | 一致 |
| feature SHA256 | `ad8dc752…3036` | 相同 | 相同 | 一致，但不覆盖 replay version |
| Rust analysis | 4 | 4 | 4 | 一致 |
| decision analysis | 16 | 16 | 16 | 一致 |
| replay semantics | runtime=1 | 缺失 | 缺失 | 当前行为一致，契约不完整 |
| policy head | isolated action query | 缺失 | isolated action query | 模型/配置一致 |
| action space | 241，固定分段 | 无独立版本 | 无独立版本 | 数量与行为通过，版本字段缺失 |
| cursor | 当前 version 1 | 不适用 | 两文件均缺失 | 旧 checkpoint 不可安全恢复 |
| training mode | actor-only | `actor_only=true` | actor_only | 当前一致 |
| model config | mid | 缺失 | 完整 | 当前 shape 全匹配 |
| dataset manifest | SHA `01529575…e347` | 自身 | exact SHA | checkpoint 属于当前 dataset |

运行扩展：

- `riichienv`：包路径 `RiichiEnv/src/riichienv/__init__.py`；`.so` SHA256 `dd9e135f…b2a6`，Cargo 0.4.8，mtime 2026-08-02 11:17:38+08:00。与 `target/release/lib_riichienv.so` byte-identical。一个扩展同时覆盖 4p/3p。
- `riichi`：Conda site-packages；`.so` SHA256 `e443f1ea…b738`，Cargo 0.1.0，analysis version 4，mtime 2026-08-01 22:27:40+08:00。与本仓库 release artifact byte-identical。该策略桥路径面向 4p。
- mtime 仅用于记录；一致性结论来自 artifact hash、Rust/Python 行为测试、在线动作 round-trip 和 raw→fresh replay。

完整路径、时间和 artifact hash 见 `environment.json`。

## 4. 原始数据、选择、manifest、NPZ 与 identity

原始 manifest：363,312 games、3,846,384 kyokus、237,204,275 decisions、0 errors。train 为 3,807,907 kyokus / 234,834,558 decisions，validation 为 38,477 / 2,369,717。两个 source ZIP 的实际 SHA256 与 raw manifest 完全相同，raw manifest SHA256 也与 encoded manifest 的 `source_manifest_sha256` 相同。

流式读取全部 3,846,384 条 raw index 后：

- 选择确为 game hash 的 denominator=5、remainders=[0,1]；train 1,523,056 kyokus / 143,802 games，validation 15,574 / 1,461。
- 每个已选 game 的所有小局均被选中；record duplicate=0、tar/member location mismatch=0、whole-game failure=0。
- raw train/validation game overlap=0；选择后仍为 0，因此同 game、同小局和相邻小局均不会跨 split。
- 依照实际 split/tar/member 生成顺序重算 selection digest 为 `b0c4f9d7…f3da`，逐字匹配 encoded manifest。

全量 NPZ 扫描覆盖 6,420 个 train shard 和 66 个 validation shard，耗时 424.808 秒：

| split | kyokus | decisions | token rows | context min/max |
|---|---:|---:|---:|---:|
| train | 1,523,056 | 93,943,903 | 9,740,364,639 | 30 / 243 |
| validation | 15,574 | 959,045 | 99,348,992 | 33 / 223 |

每个 shard 均验证：offsets 为 int64、从 0 开始、严格正分段、终点等于 token rows；factors/numeric/target/mask/identity 行数和 shape 一致；categorical 全部在 cardinality 内；numeric 为 float16、每槽在 [-1,1] 内、NaN/Inf=0；legal/teacher 为 31-byte little-endian packbits；legal empty=0；expert illegal=0；teacher 越 legal=0；candidate rows=`2*legal_count`，失败=0；context 不超过 4096。

身份字段 dtype/range 正常，kyoku block duplicate=0，各 seat `decision_index` 连续失败=0，train/validation game 和 kyoku 交集均为 0。241 个 ID 的 legal count 和 expert count 全部逐项重算并与 manifest 相等；所有 241 个 ID 在 legal 和 expert 中都非零。legal 总计 804,925,629，expert 总计 94,902,948；单 ID 最小 legal count 3,178，最小 expert count 272。

逐 ID 数组、dtype/range 和 identity digest 见 `manifest_scan.json`、`identity_digest.json`。

## 5. 独立语义 replay 与 oracle

固定 seed 为 `20260802`。主样本为对**全部正式 action/offset 流**做确定性 reservoir，再按 expert action 分层：pass、手切、摸切、reach、chi、pon、daiminkan、ankan、kakan、hora、九种九牌各 2 个，另取最高 context 5 个，共 27 个互异小局。对每个小局重放其全部 decision，而非只比较命中行：1,833/1,833 decisions 的 factors、float16 numeric、legal、teacher、expert action、identity 完全一致。

另对全量被选 raw tar 做罕见事件定向搜索，加入普通流局、多家 ron、末巡自摸模式、抢杠、岭上、四风连打、末巡荣和模式 7 局，516/516 decisions 完全一致。合计 34 个互异小局、2,349 decisions，不一致样本为 0。

使用的 oracle：

1. **身份 oracle**：raw index 的 year/game/kyoku/tar/member 与 NPZ per-row identity 独立连接。
2. **事件 oracle**：直接解析原始 MJAI，维护独立公开账本，检查 actor/target、tile/consumed/tsumogiri、事件时间顺序、手牌/摸牌、河、副露、宝牌、分数、场风/自风、honba/kyotaku、相对座位、reach declared/accepted/tile/index；逐决策对照 `Observation.new_events` 和 Rust public history。自己摸牌可见、对手摸牌被遮蔽；没有发现未来事件、对手暗手、牌山或未来摸牌进入公开 token。
3. **重新编码 oracle**：用当前实际扩展从原始局 fresh replay，再与冻结 NPZ 全字段比较。此 oracle 证明 cache 与当前 runtime 一致，但不会被单独当作语义真值。
4. **朴素规则 oracle**：公开测试中的 scalar Python shanten/improving/ukeire 与 Rust 批量分析对账；8 个 defense row 与完整 Python reference 对账；构造筋、壁、现物、通过牌、honor、公开剩余牌的正反例；公开物理牌 ID 去重、红/普通五按同 tile type、重复 dora 指示牌按倍率累计；14 张时 ukeire=N/A；numeric 分母、裁剪和 N/A 逐字段断言。
5. **环境合法动作 oracle**：16 个固定 seed 在线环境对每一决策枚举 RiichiEnv legal actions，与 Rust 241 mask 做集合相等检查，并将 Rust decode / Python bridge decode 后动作送回环境，以 actor/target/tile/consumed/red/tsumogiri 语义签名对账。

主 27 局原始事件覆盖：1,382 tsumo、1,447 dahai、15 reach、14 reach_accepted、17 hora、36 pon、51 chi、3 daiminkan、4 ankan、4 kakan、11 dora、10 ryukyoku，以及各 27 次 start/end kyoku 和 start/end game。复杂度覆盖：12 局自摸、5 局荣和、26 局红五、3 局多家立直、17 局多副露、13 局长河、1 局重复宝牌指示。

身份清单见 `replay_sample_identities.json`；逐行结果见 `replay_comparison.json`、`replay_full_kyoku_comparison.json`；罕见事件见 `rare_replay_comparison.json`；覆盖汇总见 `action_coverage.json` 和 `special_event_coverage.json`。

边界：转换后的 MJAI 没有完整 yaku/reason 标签，因此“海底/河底/岭上/抢杠”由事件位置、70 次摸牌、kakan 后 hora 等时序证据识别，不是独立计番器的 yaku 名称 oracle。未穷举全部 1,538,630 正式小局的语义重放，故“无泄漏”结论限定为上述 34 局、构造反例、全量结构不变量和实现数据流审查。

## 6. 241 维动作空间

| ID | 语义 |
|---:|---|
| 0 | pass/none |
| 1–74 | `1 + 2*TILE37_index + tsumogiri`；37 种牌名 × 2 |
| 75 | reach |
| 76–132 | 57 个 chi consumed pair：每门 19 个，含红五组合 |
| 133–169 | 37 个 pon pair：非五牌 31 个，加三门普通五/含红五各 2 个 |
| 170 | daiminkan |
| 171–204 | ankan，34 tile types |
| 205–238 | kakan，34 tile types |
| 239 | hora；当前合法 request/state 区分 ron 与 tsumo |
| 240 | 环境合法时的 kyushu/abortive draw |

`TILE37` 是标准 34 牌名加 `5mr/5pr/5sr`；RiichiEnv 红五物理 ID 为 16、52、88，其他同牌型副本为普通五。chi/pon 的固定 ID 由 consumed 组合决定；daiminkan、hora、ryukyoku 依靠当时合法动作注册表保存完整 JSON。decode 返回注册时的 exact request，因此 consumed 顺序和红牌信息保留；同一状态若两个不同物理动作碰撞到同一 ID，Rust 明确拒绝，而不是任意覆盖。chi 同 consumed pair 可对应序列另一端，但当前 discard 唯一消歧。

验证结果：静态协议生成 241/241 template，Rust encode/decode exact；正式缓存 241/241 均有真实 legal 与 expert 样本；16 个在线随机游戏中，对所有自然出现的 legal action 均验证 mask 完整性、排他性、bridge decode 和环境接收。自然在线样本没有提供 ID 240，且缺 30 个罕见 ankan/kakan ID；这些 ID 由 241 静态 round-trip 和正式缓存真实出现补充，但没有做到每个 ID 都在在线自然状态实际执行。两步 reach、tile=None follow-up discard、reach marker/accepted 已由 replay 与环境回归覆盖。

## 7. 在线/离线状态机

- 离线 Tenhou replay：上述 34 局 fresh replay 全字段一致，含 kyoku/match 终止和下一局 reset。
- 在线 RiichiEnv：16 个随机游戏逐 decision 同步 legal/decode；日志的 64 次 start_game 是 16 局四个 player-view 事件，不是多跑游戏。
- Rust state machine + Python `PublicStateTracker`：独立 public ledger、现物/筋/壁/通过牌、公开 remaining 和 reach state 回归一致。
- RiichiEnv Python 全量测试同时覆盖 4p、3p、kyoku reset、match reset、结束环境和 reach(tile=None)；策略 bridge/当前 SFT 数据是 4p 范围，不据此宣称 3p 策略编码已完成同等审计。

## 8. 模型与 actor-only 运行

mid 展开：4 层，其中 shared 3、actor-specific 1、critic 2；`d_model=192`、query heads 8、KV heads 2、head_dim 24、FFN 576、context 4096、RoPE 10000、eps 1e-6、isolated action query head。

| 模块 | 总参数 | 可训练 |
|---|---:|---:|
| token embedding | 116,736 | 116,736 |
| public backbone | 1,272,960 | 1,272,960 |
| actor backbone | 424,512 | 424,512 |
| policy head | 37,441 | 37,441 |
| critic embedding | 115,200 | 0 |
| critic backbone | 848,832 | 0 |
| value query/head | 385 | 0 |
| **合计** | **2,816,066** | **1,851,649** |

冻结 964,417 参数。state_dict 51 tensors；optimizer 包含 33 个 parameter tensors / 1,851,649 参数，checkpoint shape 51/51 全匹配，optimizer state 33 entries，未包含冻结项。

CUDA actor-only canary：logits `[2,241]`；legal logits finite、illegal 为 `-inf`；bf16 loss finite；critic embedding/backbone/value head forward 次数均 0；所有冻结 grad 均为 None；fused AdamW 生效；optimizer state 33；peak 48.23 MiB。正式最大 context 243，远低于 4096。源码执行 `clip_grad_norm_` 后再 optimizer step（`train.py:813-815`），benchmark 中测得 pre-clip norm 可超过 1，故 clipping 路径实际被触发。

## 9. 配置、loss、scheduler 与 DDP plan

| 字段 | DEFAULT | sft.yaml / 实际 | checkpoint |
|---|---:|---:|---:|
| seed / GPUs | 1 / 2 | 1 / 2 | 相同 |
| model / context | mid / 4096 | mid / 4096 | 完整展开一致 |
| epochs / global batch | 1 / 512 | 1 / 512 | 相同 |
| LR / min LR | 1.5e-4 / 2e-5 | 相同 | scheduler state 一致 |
| warmup / WD | .02 / .01 | 相同 | 相同 |
| betas / eps | .9,.999 / 1e-8 | 相同 | 相同 |
| max grad norm | 1.0 | 1.0 | 相同 |
| critic / public value | false / false | false / false | actor_only |
| head | isolated | isolated | isolated |
| group / rule | .25 / .05 | .25 / .05 | 相同 |
| rule decay | .20 | .20 | 相同 |
| gamma / dtype | .99 / bf16 | 相同 | 相同 |
| bucket window | 32 batches | 32 | 相同 |
| checkpoint / validation / heuristic interval | 5000 / 6000 / 18000 | 相同 | 相同 |

policy CE 以 local batch mean 计算；group CE 只取至少两个合法 group；rule teacher 对 tied best action 使用均匀 `1/count` soft target，局部公式正确。问题是 P1-1 的跨 rank 分母。

正式 train `93,943,903 / 512 = 183,484` 余 95，所以 1 epoch 为 183,485 optimizer steps。当前计划：rank 0 为 46,971,952 rows，rank 1 为 46,971,951；各 183,485 batches，末批 48/47。两个连续 interval 的 union 是完整 train、intersection 为空；只有 `train-00464-003.npz` 是共享边界文件，选取 row slice 不重叠。相同 seed path/row order hash 完全一致，不同 seed 顺序变化而 identity 集不变。header scan 只读取 `actions.npy` NPY header，不解压 actions，耗时 0.773 秒；每个打开 NPZ 的主数组只 materialize 一次。

rule weight 用更新前的 `global_step`，在 step 0 为完整系数，约到 step 36,697 归零。checkpoint scheduler 的 `last_epoch==global_step`、`_step_count==global_step+1`，当前已保存状态内部一致。

新的 cursor canary 在固定 scheduler horizon=3 下比较连续 3 步、1+恢复到 3、1+恢复到 2+再恢复到 3：消费 identity、model、optimizer、scheduler、Torch/NumPy/Python RNG 全部 exact，cursor 累计为 1/2/3；world-size 改变被拒绝，fresh training 会拒绝非空输出。该结论只证明**新 cursor 逻辑**，不挽救 P0-1 的旧 checkpoint。

## 10. Validation

当前 `best.pt` 在当前代码下重新评估：

| 范围 | samples | CE/loss | top1 | top3 | group top1 | reach Brier | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 固定 150k | 150,000 | 0.58380862 | 0.783173 | 0.967240 | 0.918850 | 0.116793 | 21.96 s |
| 全 validation | 959,045 | 0.58317606 | 0.783355 | 0.967120 | 0.919576 | 0.117575 | 103.90 s |

150k identity SHA256 为 `4c71f14136daee854ce53fd0ee011c5891277065b39c64dc274f9587f4e60f7a`。validation 固定、不 shuffle，在 bucketing 前截断，不消费训练 RNG；actor-only 调 `forward_policy`，不走 value。CE/top-k/group/reach 与逐 action-group 累计分母独立复算。best 判据是 validation policy CE，不包含训练辅助 loss。

结论：**当前代码重新算出的 fixed/full 指标对当前模型和当前 dataset 可信**；历史 checkpoint 内保存的 0.5838886 等旧指标不应与当前 150k 横比，原因是旧截断位置导致样本集合不同。全量 validation 可作为更稳健最终报告。

## 11. Heuristic evaluation

相同正式设置（seed base 20260717、cycle 0、四座轮转、两个 opponent recipes 各 48、96 半庄、parallel 24）运行两次，除 elapsed 外所有核心字段 bitwise equal；耗时 156.47 / 152.22 秒。

- 1,011 kyokus；每局 point delta mean = 0.211276 千点（约 +211 点），标准误 = 0.157735 千点。
- win rate 0.223541；deal-in rate 0.113749。
- 96 matches；match point delta mean = +2,225 原始点；mean rank 2.17708；一位率 0.385417；top2 0.604167。
- kyoku point 字段单位是千点；match point delta 字段是原始点。
- `opponents/current_*` 两字段为 0 且无效，不能进入模型判断。

结论：核心指标对这 96 个固定 games 是确定且可复现的，但 +211 点仅约 1.34 个标准误，不能证明总体显著领先。长期用同一 96 局做 best selection 会产生选择过拟合；建议固定集只做回归/监控，另设不参与选择的 rotating held-out cycles 并报告置信区间。

## 12. 测试与 benchmark

| 检查 | 结果 | wall time |
|---|---|---:|
| `riichi cargo test --workspace` | 6 passed | 4.02 s |
| `RiichiEnv cargo test --workspace` | 117 passed | 2.72 s |
| `riichi_ppo_v1` pytest | 190 passed，2 warnings | 13.74 s |
| RiichiEnv pytest | 282 passed，2 skipped | 2.47 s |
| 正式 NPZ 全量 scan | 6486/6486 | 424.808 s |
| 分层 full-kyoku replay | 1833/1833 decisions exact | 结果文件保留，未单独计时 |
| 罕见事件 replay | 516/516 exact | 结果文件保留，未单独计时 |
| 241 static round-trip | 241/241 exact | 包含于测试运行 |
| 两次连续 resume | 全状态 exact | 小型 canary |
| heuristic 96 × 2 | bitwise deterministic | 156.47 / 152.22 s |
| `git diff --check` | passed | <1 s |

第一次 `riichi` Cargo 运行因动态链接器找不到 `libpython3.12.so` 失败；加入 Conda env 的 `lib` 到 `LD_LIBRARY_PATH` 后通过。这是测试启动环境问题，不是 Rust 测试失败。

双卡 3 轮 SFT benchmark（global batch 512；第一轮 warm-up）：

| iter | warm-up | elapsed | global samples/s | per-rank samples/s | effective tokens/s | padding | peak GPU | fwd/bwd/opt ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 是 | 3.0044 | 170.42 | 85.21 | 8,003 | .1304 | 674.1 MiB | 1806.46 / 67.74 / 6.52 |
| 2 | 否 | .0346 | 14,790.67 | 7,395.34 | 845,899 | .0388 | 766.1 MiB | 10.71 / 13.87 / .273 |
| 3 | 否 | .3661 | 1,398.55 | 699.28 | 85,434 | .0226 | 793.9 MiB | 345.28 / 9.97 / .273 |

后两轮合计 0.4007 s、global 2,555.47 samples/s、per-rank 1,277.74、mean padding .0307。GPU 0/物理 GPU 4 当时有无关长进程共享，iter 2/3 方差过大，不能据此给出稳定容量结论。`target_kl=0.0`、`update_epochs=4`、`kyokus_per_worker=1` 是 PPO rollout/update 参数，对 encoded SFT benchmark 不适用；没有伪造同名 SFT 参数。

## 13. 明确回答与修复顺序

1. **当前 v13 缓存需要重做吗？** 不需要。依据是 6,486/6,486 shard 全量结构扫描、241 ID 逐项 manifest 对账、raw selection/identity 对账和 34 局 2,349 decisions fresh replay exact。边界是没有语义重放全部 153.9 万局，且少数罕见 yaku 只有事件时序 oracle。
2. **当前代码可以从随机初始化重新训练吗？** 程序可运行，但当前双卡 canonical loss 不应正式运行。先修 P1-1；若临时禁用 group/rule loss，则主要问题可绕开，但改变了正式目标。
3. **双卡是否恰好消费全部 train identity 一次？** 对当前 HEAD 的一个新 epoch：是。数学 interval union/intersection、正式 count、共享边界 slice 和 seed 计划均已验证。对现有旧 checkpoint 的续训：否，无法保证，见 P0-1。
4. **scheduler step 正确吗？** optimizer step 数和保存状态一致，总数 183,485；存在首个 warmup scale 重复和 min LR 在末步后达到的曲线语义细节，但没有少/多 scheduler step。旧 checkpoint 的主要问题是 data plan，不是 scheduler state。
5. **validation 指标可信吗？** 当前代码重算的 fixed 150k 和 full validation 可信于当前模型/数据；历史存储指标及旧 best 选择不可与当前固定集直接比较。
6. **heuristic 指标可信吗？** 核心对局指标对固定 96 局可信且确定；统计充分性不足，`current_*` 两字段不可信，不能把这 96 局长期反复用于无偏 best selection。
7. **可以开始正式 SFT 吗？** **不建议。** 顺序：先禁止旧 checkpoint fallback/加入 plan version；再修双卡 eligible loss；再补 replay/action/model/training 契约；最后做原子保存改造。修复后用新目录、随机初始化（或仅权重初始化）做短双卡 canary，复跑 loss oracle、resume、validation，再开始正式训练。缓存不需重做。

## 14. 证据索引与残余风险

关键证据全部保存在本目录：`environment.json`、`git_status.txt`、`commands.txt`、`test_results.json`、`manifest_scan.json`、`identity_digest.json`、`replay_sample_identities.json`、`replay_comparison.json`、`replay_full_kyoku_comparison.json`、`rare_replay_comparison.json`、`action_coverage.json`、`action_roundtrip_random.json`、`ddp_plan.json`、`ddp_loss_weight.json`、`resume_comparison.json`、`validation.json`、`heuristic_run_1.json`、`heuristic_run_2.json`、`benchmark.json`、`model_checkpoint.json`、`actor_only_runtime.json`，以及 `scripts/` 下的独立审计 helper。

残余风险：没有对全部小局做独立规则引擎语义重放；没有让每个罕见 kan/ID 240 都在在线自然状态实际执行；converted MJAI 缺少部分 yaku/流局原因标签；GPU benchmark 受到共享负载污染；新 cursor 的两次恢复 canary 是固定 horizon 小数据，不是 18 万步正式 epoch。以上边界不改变“缓存无重做证据、当前正式训练被代码问题阻断”的结论。
