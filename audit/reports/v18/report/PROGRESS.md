# V18 输入与模型架构升级进度

## 范围与保护边界

- 开始日期:2026-08-25。
- 现行契约:V18;V16/V17 代码契约与实验产物仅作冷存储。
- 禁止修改或删除:`checkpoints/train_riichi_v16/`、`checkpoints/train_riichi_v17/`、
  `datasets/*v16`、`datasets/*v17`、`logs/v16/`、`logs/v17/`、
  `audit/reports/v16/`、`audit/reports/v17/`。
- 本次不生成完整 V18 数据集,不启动正式 SFT,不设计或运行 PPO,不改变 1v3 评测机制。

## Spec-kit 产物

- Constitution:`.specify/memory/constitution.md` v1.8.0。
- Feature:`specs/008-v18-input-architecture/`。
- 二次 analyze:41 条 FR/SC、70 项任务,阻断问题为 0。

## 初始引用审计

命令:

```bash
rg -n "q_scorer|q_scores_v16|dueling_candidate_q|q_boost|candidate_q" \
  riichi_ppo_v1 RiichiEnv riichi_lab_bot --glob '!**/tests/**' \
  --glob '!**/configs/v16_*' --glob '!**/configs/v17_*'
rg -n "V16PreparedBatch|prepare_v16|forward_v16|v16_rust_encoding|train_v16" \
  riichi_ppo_v1 RiichiEnv riichi_lab_bot --glob '!**/tests/**'
```

基线结果:

- 活跃 Q 引用 7 行:模型中的 scorer/API 以及 learner 的参数根/注释。
- 活跃 V16 API/模块引用 38 行,覆盖 bridge、模型、SFT、推理、评测、bot 与 PyO3。
- 清理策略:逐文件 `rg` 更新全部调用方,通过相关测试后才删除旧实现文件;历史配置、
  specs、checkpoint、数据集、日志与报告不属于删除目标。

## 实施记录

- 2026-08-25:完成 constitution/spec/plan/tasks/analyze;开始 implement Setup/Foundation。
- 2026-08-25:初始 Rust 29 字段 Atomic Snapshot schema、原生 fact/query bridge
  与真实 supplier 座次落地;Python 直接消费机器可读 schema。
- 2026-08-25:完成 256/16Q/4KV/16/704、3 Shared + 1 Actor + 2 Critic 的 GQA
  Actor-Critic;动作对使用隔离注意力和共享局部 position ID。
- 2026-08-25:拆分 tsumo/ron action-type code,使
  chi/pon/daiminkan/ron supplier=1..3、其他动作 supplier=N/A 可严格校验。
- 2026-08-25:移除活跃 Q scorer/candidate-Q 与 V16 命名入口,checkpoint format
  升为 4;旧 checkpoint、数据集、配置、日志与历史报告未修改或删除。
- 2026-08-25:完成 Actor-only BC freeze/optimizer/save/load、V18 manifest/config、
  生产参数校验和只读 token 统计入口。未生成完整数据集,未运行正式 SFT/PPO。
- 2026-08-25:同步根 README、训练 README、bot README、V18 输入/SFT/Kyoku
  协议、AGENTS.md、目录职责与归档标签。

## 验证结果

所有命令均从仓库根目录执行,Python 使用 `Mahjong-AI` 环境。

### 协议与统计

```bash
conda run --no-capture-output -n Mahjong-AI \
  python -m riichi_ppo_v1.tools.v18_token_statistics \
  --dataset datasets/tenhou_sft_2024_2025_encoded_60pct_v16
conda run --no-capture-output -n Mahjong-AI \
  python -m riichi_ppo_v1.tools.validate --parameter-contract
```

- 只读扫描既定 validation selection:102 shards、1,439,440 decisions。
- 初始 29 行基线:Objective Facts 均值 53.837822;Query token 均值
  16.965206(8.482603 pairs);总长均值 99.803028,范围 54–169。
- Actor-Critic 参数量 4,930,562;state keys 83;`forbidden_q_keys=[]`。
- Query pair 置换测试按 action ID 回填最大误差 `2.98e-08`,低于
  `atol=rtol=1e-5`。

### 信息与训练边界

- `test_v18_information_boundaries.py`:隐藏手牌/未来牌变化时 Actor raw logits
  不变;有效 private hand/future-five 变化会改变 Critic value。
- Critic 校验要求相对座次 1/2/3 三段真实闭手、随后严格位置 1..5,不接收动作
  Query;缺失、错序、数量错误均被拒绝。
- Actor-only 生命周期测试执行 forward、backward、AdamW step、save/load 与旧
  checkpoint 拒绝;Actor 参数发生更新,Critic/value 不在 optimizer、无梯度且逐参数
  保持不变。
- source-seat 测试覆盖三种相对对手以及四种 supplier 动作类型
  chi/pon/daiminkan/ron;tsumo/ron code 明确分离,非 supplier 必须 N/A。

### 全量测试

```bash
env LD_LIBRARY_PATH=/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/lib \
  conda run --no-capture-output -n Mahjong-AI cargo test \
  --manifest-path RiichiEnv/Cargo.toml --workspace -q
conda run --no-capture-output -n Mahjong-AI python -m pytest -q riichi_ppo_v1/tests
conda run --no-capture-output -n Mahjong-AI python -m pytest -q RiichiEnv/tests
conda run --no-capture-output -n Mahjong-AI python -m pytest -q riichi_lab_bot/tests
```

- Rust workspace:112 core + 4 correctness + 4 PyO3 facts + 15 state-machine =
  135 passed,0 failed;doc tests 0 failed。
- `riichi_ppo_v1/tests`:141 passed。
- `RiichiEnv/tests`:284 passed,2 skipped。
- `riichi_lab_bot/tests`:31 passed,1 skipped;跳过项要求 `CUDA_DEVICE=2,3` L20,
  不影响 CPU/V18 契约覆盖。

### Quickstart、审计与清理

- `quickstart.md` 的 unit/bridge/replay/information/statistics/production-validation/full-suite
  命令均已执行;1-game production smoke 成功。
- smoke 生成的 `audit/reports/v18/eval/v18_protocol_coverage.json` 已删除;它是可由
  同一命令重建的临时结果,未留下日志或正式评测产物。
- `git diff --check` 通过。
- 活跃 Q 审计只剩 `model/parameter_count.py` 中用于拒绝禁用 key 的检查字符串;
  活跃 V16 API/模块/兼容入口审计为 0。V16 协议文档已显式标为冷存储,历史
  specs/reports/configs 原样保留。

## 需求到证据

| 验收项 | 生产实现 | 自动化证据 |
| --- | --- | --- |
| 初始 29 行 Snapshot/schema 单一来源 | Rust `atomic_snapshot.rs` + Python schema 消费 | 初始 V18 验收测试 |
| 初始 token 均值 97–103 | 只读统计工具 | 99.803028 / 1,439,440 decisions |
| Query metadata/supplier/隔离 | native facts + isolated Actor mask | metadata ablation、supplier domain、pair permutation tests |
| Actor/Critic 信息隔离 | Actor-only/full forward + strict private sequence | hidden invariance/private-use/order/future-five tests |
| Actor-only SFT-ready | `sft/actor_bc.py` + V18 contract/config | optimizer/freeze/update/save/load/legacy-rejection tests |
| 约 5M且无Q | 参数统计入口 + 无 Q 模块 | 4,935,682;forbidden keys 0 |
| 单一 V18 契约 | V18-only loader/config/bot/eval bridge | config/artifact/validation/full suites |

## Converge 结论

`$speckit-converge` 在首次完整 implement 后检查 32 条 FR、9 条 SC、16 个验收场景、
4 组计划决策与 7 项宪法原则。missing、partial、contradicts、unrequested finding
均为 0,没有向 `tasks.md` 追加任务;现有 61 项任务全部完成。

## 49 行 Snapshot Objective Facts 扩展

- 2026-08-25:在未生成正式 V18 数据、未开始训练的前提下，固定 Snapshot 从 29 行
  原地扩展为 49 行；没有旧 V18 数据、checkpoint 或 schema fallback/migration 路径。
- 每名对手新增前六次舍牌万/筒/索/幺九字统计、公开副露役牌番、公开副露宝牌加赤宝
  番；全局新增四枚已可见牌种数和不同宝牌种的未知实体牌数。
- Rust `prepare_atomic_snapshots` 只读取观察者自身手牌/副露、公开牌河/副露和当前
  宝牌指示牌，不读取对手闭手、真实牌山或事后标签。重复宝牌种只在全局未知实体数
  中去重；连风按实际番数累计；暗杠不改变门清，但其已表示牌可计入宝牌/赤宝统计。
- 字段顺序为四项 placement/pressure、三组各 11 项对手摘要、四项向听、两项全局、
  三项最近手切和三项摸切连打。所有新字段均为 factorized categorical token；取值和
  溢出桶由 Rust schema 的 field ID 10–15、21–26、32–37、42–43 固定导出。

### 复现命令与结果

```bash
env LD_LIBRARY_PATH=/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/lib \
  conda run --no-capture-output -n Mahjong-AI cargo test \
  --manifest-path RiichiEnv/Cargo.toml --workspace -q
conda run -n Mahjong-AI python -m pytest -q riichi_ppo_v1/tests
conda run -n Mahjong-AI python -m pytest -q RiichiEnv/tests
conda run -n Mahjong-AI python -m pytest -q riichi_lab_bot/tests
conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.validate --parameter-contract
conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.v18_token_statistics \
  --dataset datasets/tenhou_sft_2024_2025_encoded_60pct_v16
conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.validate \
  --games 1 --seed 0 --output audit/reports/v18/eval/v18_protocol_coverage.json
```

- Rust workspace:138 passed,0 failed（112 core、4 correctness、7 PyO3 facts、15 state machine）。
- `riichi_ppo_v1/tests`:142 passed；`RiichiEnv/tests`:284 passed、2 skipped；
  `riichi_lab_bot/tests`:31 passed、1 skipped（L20 专用 checkpoint 测试）。
- 49-row 专项测试覆盖字段顺序、边界桶、不足六张舍牌、隐藏对手手牌不变性、重复宝牌、
  连风、赤宝和暗杠；Actor 前向输出 `raw_policy_logits` 形状为 `(1, 241)`。
- 只读扫描 102 个归档 validation shards、1,439,440 decisions：Snapshot 均值 `49.0`，
  总 token 均值 `119.803028`，范围 `74–189`。未物化、覆盖或修改归档数据。
- 参数量为 `4,935,682`，较 29 行 schema 增加 `5,120`（field-ID embedding 新增 20
  类 × 256）；仍在 4.9M–5.1M 契约内，83 个 state keys，禁用 Q key 为 0。
- 1-game 生产冒烟（`validate --games 1`，随机覆盖率、动作语义与 schema 校验）通过；
  临时 coverage JSON 已删除，未留下日志与正式评测产物。`git diff --check` 通过。
- `plan.md` 增补 Phase E 扩展策略（原 29 行 Phase A–D 文案同步为 49 行），与
  `research.md` Decision 11/12、`spec.md` FR-008a/b/c、`tasks.md` Phase 9 一致。

## 删除剩余牌山估计 token

- 2026-08-25:V18 Objective Facts 的状态后缀删除「剩余牌山数」计数器。MJAI 事件流
  无法精确推导剩余牌数（旧估计固定 70 起、仅按摸牌事件递减，无法还原杠导致的
  王牌变化、连庄配牌与终局差异），作为不可靠估计不再进入任何输入；其余
  `KIND_COUNTER` 字段重排为连续 1..7（场风、局数、本场、立直棒、庄家相对座次、
  自风、状态 flags）。
- 事件序列长度每决策固定减少 1 行：只读投影从 `119.803028` 修正为
  `118.803028`，范围从 `74–189` 变为 `73–188`；统计工具加入状态后缀差值常量
  校正。Snapshot 49 行 schema、Query 布局、参数量（4,935,682）均不受影响。
- 同时修复双包不同步问题:`install_conda_extension.sh` 现在同源安装
  `riichienv` 主扩展与 `riichi` 状态机模块。此前 `riichi` 模块独立安装,可能
  停留在旧编译产物,导致 Python 侧状态机逻辑与 Rust 源码不一致。

### 复现命令与结果

```bash
cd RiichiEnv && CONDA_PREFIX=/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI \
  bash scripts/install_conda_extension.sh
env LD_LIBRARY_PATH=/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/lib \
  conda run --no-capture-output -n Mahjong-AI cargo test \
  --manifest-path RiichiEnv/Cargo.toml --workspace -q
conda run -n Mahjong-AI python -m pytest -q riichi_ppo_v1/tests
conda run -n Mahjong-AI python -m pytest -q RiichiEnv/tests riichi_lab_bot/tests
conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.v18_token_statistics \
  --dataset datasets/tenhou_sft_2024_2025_encoded_60pct_v16
conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.validate --parameter-contract
```

- Rust workspace:138 passed,0 failed（112 core、4 correctness、7 PyO3 facts、15 state machine）。
- `riichi_ppo_v1/tests`:142 passed；`RiichiEnv/tests` + `riichi_lab_bot/tests`:
  315 passed、3 skipped（L20 与三麻复杂度专用跳过）。
- 协议矩阵 7 项全过:断言剩余牌山行不再出现、`KIND_COUNTER` 字段 1..7 连续、
  字段 5 仅承载庄家相对座次且 numeric 全 0。
- 只读统计 102 shards、1,439,440 decisions：Snapshot 均值 `49.0`，总 token 均值
  `118.803028`，范围 `73–188`。参数契约维持 `4,935,682`、83 state keys、
  禁用 Q key 为 0。
- `git diff --check` 通过;spec/plan/research/tasks 同步 FR-008d、Decision 13、
  Phase F 与 Phase 10,协议文档声明状态后缀不含剩余牌山数。

## 54 行 Snapshot 扩展:进度、立直特质与巡目

- 2026-08-25:Snapshot 从 49 行扩展到 54 行(无旧 V18 数据,原地重排):
  - 每家对手摘要 11 → 13 行:新增立直后摸切数(0..15,16=16+;不含宣言牌本身),
    立直宣言牌替换原"最新手切"(同 1..37 规范牌码,红五 5m/5p/5s=1/11/21)。
  - 全局新增自身进张数(0..39,40=40+)与听牌和牌张数(0=N/A 非听牌,1..39,40=40+),
    基于归一十三张形与合法已知区域剩余实体牌,复用 Rust `calculate_after_draws`。
  - Objective Facts 状态后缀新增当前巡目(已舍牌轮数+1,精确计数),与删除的
    剩余牌山估计抵消,后缀行数与旧基线一致。
- 弃置项:最新手切与事件前缀最后一条 dahai 完全重复;旧估计"剩余牌山数"维持
  不编码。所有新字段继续走 Rust 预处理路径,Python 只消费 schema 导出。

### 复现命令与结果

```bash
cd RiichiEnv && CONDA_PREFIX=/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI \
  bash scripts/install_conda_extension.sh
env LD_LIBRARY_PATH=/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/lib \
  conda run --no-capture-output -n Mahjong-AI cargo test \
  --manifest-path RiichiEnv/Cargo.toml --workspace -q
conda run -n Mahjong-AI python -m pytest -q riichi_ppo_v1/tests RiichiEnv/tests riichi_lab_bot/tests
conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.v18_token_statistics \
  --dataset datasets/tenhou_sft_2024_2025_encoded_60pct_v16
conda run -n Mahjong-AI python -m riichi_ppo_v1.tools.validate --parameter-contract
```

- Rust workspace:142 passed,0 failed(112 core、4 correctness、10 PyO3 facts、16 state machine),
  含进张/和牌张的听牌与非听牌边界、立直宣言牌红五编码、溢出桶与域拒绝测试。
- `riichi_ppo_v1/tests` + `RiichiEnv/tests` + `riichi_lab_bot/tests`:457 passed、3 skipped。
- 协议矩阵 7 项全过:字段 1..7 计数 + 巡目(field 8)断言、54 行顺序、N/A 语义。
- 只读统计 102 shards、1,439,440 decisions:Snapshot 均值 `54.0`,总 token 均值
  `124.803028`,范围 `79–194`。参数量 `4,940,802`(+5,120:field/categorical 基数扩展),
  83 state keys,禁用 Q key 为 0,仍在 4.9M–5.1M 契约内。
- V18 全前向:snapshot batch `(1,54,4)`、`snapshot_lengths=54`、`raw_policy_logits (1,241)`。
- 数据预处理链路本身无需改行:precompute/data 的编码入口(`encode_snapshot_rows` → Rust
  `prepare_atomic_snapshots`)与 offsets 全部由 Rust schema 动态传播,54 行自动生效。
  修复一处真实遗漏:`sft/contract.py` 的 `encoding_contract_sha256` 原为手工冻结常量,
  schema 自 29→49→54 行均未同步;现改为运行时从 Rust Schema 全表 + 协议常量推导
  (新增 `test_v18_sft_contract.py` 断言确定性与 fail-closed),旧值不再保留。
- `git diff --check` 通过;spec(FR-002/004/005/008a/008d)、research(Decision 14)、
  plan(Phase G)、tasks(Phase 11)、协议文档与 quickstart 全部同步。

## V18 GRP 输入扩展与重新训练准备(21 维/96×2/40%×全部 shard)

- 2026-08-26: 以 `specs/009-v18-grp/` 立项(spec/plan/tasks)。GRP 输入从 7 维
  扩展到 21 维:新增局风类型(东风/半庄/西风)、上一小局结果类型、各玩家累计
  和了/放铳/听牌流局次数(4×3);全部只来自公开小局结果与局风(边界状态),
  不含手牌/牌河/未来信息。模型 7→64×2 层 GRU 提升为 21→96×2 层
  (fc 192→192→24),参数量 131,832(≈2.25×)。数据从 40%×280 shard
  (43,407 半庄)扩到 40%×全部 930 shard(约 14.5 万半庄,≈3.3×),维持与
  SFT 60% 子集零重叠。
- 事实核查(40 shard 抽样):庄位 `oya` 恒等于 `(kyoku-1) mod 4`,是
  `grand_kyoku` 的纯推导,不作为新特征;局类型分布东风 88,957 / 半庄 74,011 /
  西风 872 局——7 维输入无法区分同一 (风,局) 下的东风与半庄(剩余局数不同),
  是本次扩展修复的核心信息缺口;运行时环境只有 `4p-red-half`,在线
  `game_type` 恒为 1。
- 实现: `model/grp.py`(21 维布局常量、GRP_HIDDEN=96、GRPModel 可配置构造)、
  `training/grp/prepare.py`(`game_type_from_content`/`game_type_from_mode`/
  `result_increment`/`feature_row`/离线按边界链推进计数、dataset.json
  `riichi-grp-v18`)、`training/grp/train.py`(快照用常量,去硬编码 64/2)、
  `training/worker.py`(GrpRollout 每环境维护累计计数、按 checkpoint
  `model_config` 构造、与离线共用 `feature_row` 保证逐位一致;结果类型常量
  收敛到 `model/grp.py` 单一来源)、`configs/v18_grp.yaml`、
  `audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh`(prepare → train,
  `--skip-prepare`)、`riichi_ppo_v1/docs/v18_grp.md` 协议文档、
  `audit/reports/v18/design/V18-GRP 输入扩展设计.md`。
- 测试: GRP 契约测试重写(21 维布局、局风映射、三类结果计数推进、离线/在线
  `np.array_equal` 逐位一致、参数预算 110K–150K、v18 dataset.json 格式);
  `test_v17_reward.py` 更名为 `test_grp_reward.py`。临时目录 prepare→train→
  冻结→`model_config` 构造全链路冒烟通过。`pytest riichi_ppo_v1/tests`:
  151 passed,0 failed。`git diff --check` 通过。
- 训练执行: 由维护者运行
  `bash audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh`
  (CUDA_DEVICE=0,1,产物 `checkpoints/train_riichi_v18/grp/best.pt`、
  `logs/v18/grp_prepare.log`、`logs/v18/grp_train.log`);训练完成后的 GRP
  供 V18 PPO 冻结只读。
- 数据准备并行化: `prepare` 改为按 tar shard 多进程解析(`--workers`,默认
  6,spawn 上下文),记录按 shard 顺序拼接,输出与串行处理逐位一致。新增
  单测 `test_prepare_grp_dataset_parallel_matches_serial`(workers=1 vs
  workers=3 全部 npz 数组 `np.array_equal`)。
- **线上事故与修复(半庄跨 shard)**: 首次真实运行在早期即因 fail-closed
  校验崩溃——`RuntimeError: game '2024010212gm-00a9-0000-90717c18' spans
  multiple shards`。实测确认 shard 边界普遍切断半庄(该局 10 个小局分居
  shard 1 尾与 shard 2 头;此前"整局为单位分 shard"的抽样核验结论有误,
  已更正)。修复: 解析前先做一次只读 tar 头的预扫描,按 game_id 归属用
  并查集把相邻 shard 合并为分组,worker 在组内跨 tar 聚合,每场半庄完整且
  只产出一条记录;新增单测
  `test_prepare_grp_dataset_merges_games_spanning_shards` 与真实数据
  max_shards=4 冒烟(4 个 train shard → 1 分组,603 局记录)。
  `pytest riichi_ppo_v1/tests`: 153 passed,0 failed。失败运行残留目录
  `datasets/tenhou_grp_2024_2025_v18`(空产物)已清理。

## 训练超参与数据量调整(batch 2048 / 66.7% 子集)

- 2026-08-26: 首次训练(40% 数据,batch 512)在 step 22,900 由维护者
  Ctrl+C 终止并删除全部训练产物(checkpoints/train_riichi_v18/grp、
  logs/v18/grp_train.log);该次已验证 21 维模型收敛:val loss 在
  step 22,600 已达 2.5570(低于 V17 的 2.6038)。
- `configs/v18_grp.yaml` batch_size 512 → 2048(lr 保持 1e-5,batch 增大
  不线性放大 LR;15 epochs 下步数约 1.9 万量级);train.py 默认值与文档同步。
- GRP 数据集子集从 40%(0,1/5)扩到 **66.7%(0,1,2,3/6)**——相比 40% 版
  新增约 9.6 万半庄(train 143,802 → 约 239,670;validation → 约 2,435),
  ≈5.5× V17。扩围进入 SFT 60% 区域,60/40 无重叠惯例主动放弃(GRP 为
  冻结奖励模型,无策略污染)。一键脚本 prepare 参数同步更新,旧 40% 数据集
  已移除以待重建。文档:protocol/design/spec/plan 全部同步。

## 全量数据 + 30 epochs 再训练(batch 2048)

- 2026-08-26: 66.7% 子集版(240,151 train 半庄,batch 2048,15 epochs)
  完整训练结束:**best val loss 2.5191 @ step 18,600**(validation 25,759
  样本),显著优于 V17(2.6038)与 40% 中途值(2.5570)。
- 决策:GRP 改用**全量数据**(denominator=1,与 SFT 60% 完全重叠,冻结
  奖励模型无策略污染)+ **epochs 30**(仍处欠拟合区间,长训练收益为正)。
  训练产物与 66.7% 数据集已按维护者要求清除;脚本 prepare 参数改为
  `--subset-denominator 1 --subset-remainders 0`,`configs/v18_grp.yaml`
  epochs 15 → 30;protocol/design/spec/plan 全部同步。预期 train 约 36 万
  半庄、约 5.6 万步(1–2 小时)。

## V18 当前局面输入与 Actor 决策架构重构（specs/010）

- 2026-08-27：按 `audit/reports/v18/design/V18当前局面输入与Actor决策架构重构提示词.md`
  完成输入协议重构（PPO 阶段不在本阶段范围）：
  - **协议**：移除 Actor 事件历史（history factors/numeric/lengths/generations）、
    54 行 Atomic Snapshot、旧局部 position Query；改为**当前局面状态快照**：
    Shared 公共前缀（桌况/自身手牌/SELF_STATE_ANALYSIS/四家 PLAYER/三家完整牌河+各两个
    六张摘要/当前副露/34 TILE_STATE）+ Actor-only 尾部（三个 OPPONENT_ANALYSIS +
    按 action_id 升序的 Offense/Defense Query，O0–O9/D0–D9 语义保留）。全 token RoPE
    连续位置、公共双向 GQA、结构化 Actor mask、Critic 独立尾部（SEP_CRITIC + 三家闭手 +
    未来五张 + Value Query）。schema 单点定义在 `riichi_ppo_v1/model/encoding_protocol.py`
    （`TOKEN_ROW_WIDTH=32`、`TOKEN_NUMERIC_WIDTH=8`、`CONTEXT_TOKENS=256`）。
  - **编码器**：新增 `RiichiEnv/riichienv-python/src/current_state_encoding.rs`
    （`riichienv.prepare_current_state_batch`，直接以 Observation 当前字段构造共享+分析行），
    复用 `riichi::shanten`/`analysis::suji_category`/`wall_class` 等内核；Python 装配层
    `model/current_state.py` 拼接 SEP_ACTIONS 与按 action_id 升序的 query 行。
  - **模型**：`architecture.py` 重写（`current_state_snapshot` 策略头、共享双向 backbone、
    结构化 Actor 层、Critic 尾部）；`model/dense_embedding.py` 新增槽位感知融合
    （`dense_slot_dim=32` 独立槽位表 + 共享输入投影 512 + 共享 gated MLP；
    简单类别 16 维 concat 共享投影）。`d_model=256`、16Q/4KV GQA、`ffn_dim=704`、
    3+1+2 层、`dense_fusion_dim=512`；**总参数 5,756,722（≤6.0M）**，无 MHA 双分支/
    Q 模块/旧 adapter（`forbidden_q_keys=()`）；分项 embedding 1,327,920 / shared
    2,115,072 / actor 705,280 / critic 1,410,560 / head 197,890。
  - **SFT**：`sft/data.py`、`precompute.py`、`trainer.py`、`actor_bc.py`、`contract.py`
    重写；`EncodedSample` 改为完整 Actor 序列（actor_factors/actor_numeric/query_rows/
    action_ids/legal/身份），shard 存 `actor_offsets/actor_factors/actor_numeric/...`；
    manifest 增加 `state_protocol=riichi-current-state-v18-1`；
    `encoding_contract_sha256=418b7b8f...`（由新 schema 生成）。
  - **测试**：重写/新增 unit/integration/protocol 共 16 个测试文件
    （协议单源、Rust 行结构、RoPE/结构化 mask、密集嵌入敏感性/内部顺序/padding/梯度、
    参数契约、Actor-only BC 隔离、契约 hash fail-closed、真实 replay 编码、Query 语义、
    信息边界、小规模 SFT 生命周期）。`pytest riichi_ppo_v1/tests`（PPO 相关除外）：
    **163 passed, 0 failed**；PPO 旧输入测试 `test_rollout_buffer::test_learner_accepts_only_rollout_buffer`
    因旧契约不兼容而失败，属预期待迁移项（见下）。
  - **上下文统计（抽样）**：真实 fixture 首个 kyoku 共 65 个决策，token_length=99
    （早巡样本），pair_count=13；`context_tokens=256` 严格上界核算见
    `specs/010/.../research.md §2.7`（Actor ≤204、Critic ≤185）。
  - **清理**：删除 `riichi_ppo_v1/model/snapshot.py`（全仓引用检查零命中）；删除后全仓
    import 冒烟通过（`riichi_lab_bot` 仅保留包级 import 兼容，其运行路径列入待迁移）。
  - **文档**：重写 `riichi_ppo_v1/docs/v18_input_protocol.md`、`v18_sft.md`、
    `KyokuEventTupleProtocol.md`；更新根 `README.md`、`riichi_ppo_v1/README.md`、
    `AGENTS.md`、`docs/directory-responsibilities.md`。
  - **PPO/rollout/bot 待迁移清单**：`training/worker.py`、`inference.py`、`learner.py`、
    `learner_ddp.py`、`rollout_buffer.py`、`trajectory.py`（Transition 旧字段）、
    `evaluation/policy_adapter.py`、`head_to_head_1v3.py`、`riichi_lab_bot/
    src/riichi_lab_bot/{model,policy,bridge}.py` 仍引用旧 history/snapshot/局部 position
    输入契约；本阶段不修改、不兼容、不测试，仅记录。
  注：`test_artifact_conventions::test_historical_audit_and_logs_are_removed_but_checkpoints_remain`
  断言本仓库存在 `checkpoints/train_riichi_v13/v14/v15` 目录；当前 checkout 仅有
  v16/v17/v18，属环境性差异（非本次改动引入）。
- **性能冒烟**：CPU Rust/PyO3 编码吞吐（真实 fixture 单 kyoku 65 decisions，3 轮取后两轮）：
  ~1,984–2,045 decisions/s；GPU Actor-only SFT 单卡（CUDA_DEVICE=0，L20，bf16 autocast，
  batch 16，2 步后两轮）: loss≈1.59，~5,836 tokens/s，峰值显存 191 MB（结构/显存验证，
  非训练基线）。临时脚本已删除。
- **60% selection 代表性 token 统计**（`--subset-denominator 5 --subset-remainders 0,1,2`
  × `--game-sample-denominator 500`，train 268,011 decisions，临时样本已删除）：
  actor mean **109.01**、p50 **108**、p95 **134**、p99 **144**、max **161**、min 73；
  分项 segment mean：shared **87.07**、analysis **4.00**、actions **17.94**；
  query_answer_out_of_range=0、actor_field_out_of_range=0。
  与提示词参考（早巡 85–105 / 中巡 105–135 / 晚巡 130–165 / 极端 185–215）一致；
  严格上界 `context_tokens=256` 保持成立（Actor ≤204、Critic ≤185，见
  `specs/010/.../research.md §2.7`），无截断。

## V18 当前局面输入审查缺陷修复（audit/reports/v18/design/V18当前局面输入审查缺陷修复提示词.md）

- 2026-08-27：按审查提示词完成 4 项 P1（V18-A1+A3 被鸣牌双计/supplied 失真、A2 concealed_count、
  A4 action_id 不进入嵌入）与 P2（B1–B11）修复。提交粒度：每个根因一个可回滚 commit。
- **P1 修复**：
  - `Meld` 新增 `called_tile_index`（serde default，记录被鸣牌在供牌者牌河中的 0 基下标）；
    状态机 `state/mod.rs`、`state/event_handler.rs`（apply_log_action）、3P 等价路径与
    `_resolve_kan` 全部写入；
  - `current_state_encoding.rs`：`entity_public_counts` 实体去重、`is_supplied` 按
    (from_who, called_tile_index) 精确标记、`concealed_count` 改为
    `13+pending-3×三张副露-4×杠`、`pending_draw_actor` 覆盖 tsumo/pon/chi/daiminkan/ankan/kakan；
  - Action Query 新增 `action_id`（241 维专用离散表）进入 token embedding
    （`encoding_protocol.py`/`current_state.py`/`semantic_validation.py`/`v18_fixtures.py`）。
- **模型/结构**：内容 token 保留 segment/kind 基础向量（相加而非覆盖）；RIVER_SUMMARY 槽内
  4 字段改 concat（每槽 5×dense_slot_dim）；SEP_ACTIONS 独立角色（只读自己，Action 行可读）；
  critic 空输入抛清晰 ValueError；重复 action_id 拒绝。
- **校验 fail-closed**：`semantic_validation.py` 增加 action_id 升序、summary valid_length 与
  河长一致、critic 字段域、TABLE 保留列恒 0、TILE_STATE 实体守恒、SELF_HAND 升序/域/is_drawn
  一致性、drawn_is_current==(mode==0)；`sft/contract.py::validate_manifest` 增加 storage 字段
  域校验并与运行时常量一致；`sft/precompute.py::iter_precomputed_samples` 增加三组 offsets
  严格校验；`trainer.py` collate 前默认执行 `assert_actor_input_semantics`（可显式关闭），
  BC loss 前重验 `target ∈ legal_mask`。
- **契约与参数**：`encoding_contract_sha256` 由 schema 自动更新为
  **`c60f867fec94b66f4a42d97fc1214685a78946bb618cf58fee597b2dd7caade0`**（旧 manifest/checkpoint
  fail closed）；总参数 **5,804,914（≤6.0M）**：embedding 1,376,112 / shared 2,115,072 /
  actor 705,280 / critic 1,410,817 / head 197,633；state keys 258。
- **oracle 转 PASS**（`v18_audit_oracle.py`，862 决策）：
  `concealed_bad=0`、`public_bad=0`、`known_bad=0`、`supplied_bad(real)=0`、
  合成反例 `river_marks=[(1,1),(2,0),(3,0)]`、`exact_collisions=0`；结论 PASS。
  `v18_model_structure_audit.py`：mask 逐格 `synthetic_mismatch=0` / `real_mismatch=0`、
  RoPE/padding/batch PASS、内容 token segment 变化输出 diff=0.258（相加保留）、
  critic 空输入 `ValueError: critic rows must not be empty`、重复 action_id 被拒。
- **测试结果**：Rust workspace `cargo test`：**148 passed, 0 failed**；
  `pytest riichi_ppo_v1/tests/unit + protocol + integration`：**182 passed, 2 failed**
  （仅既有两项：`test_historical_audit_and_logs_are_removed_but_checkpoints_remain` 环境性差异、
  `test_learner_accepts_only_rollout_buffer` PPO 待迁移断点）；`pytest RiichiEnv/tests`：
  **284 passed, 2 skipped**；`validate --parameter-contract` 通过（total 5,804,914）。
  新增永久化测试：`tests/unit/test_v18_buckets.py`、`tests/integration/test_v18_action_discriminability.py`、
  `tests/integration/test_v18_meld_fields.py`（红五/chi 三形状/kakan/ankan/daiminkan/supplied 精确标记），
  及 Rust `current_state_encoding.rs` bucket/entity/concealed 单测、`query_encoding.rs` bucket_o2 单测。
- **文档**：契约 separator 编号统一 101..111、valid_length `i<=valid_length`、§3.10 action_id
  进入 embedding、§5 SEP_ACTIONS 可见性；`data-model.md` 111/13/14；`v18_input_protocol.md`
  参数 5.80M、ACTION 字段说明；README 参数同步。
- **清理**：`logs/v18/` 调试脚本（audit_slots.py、debug_encoder.py、probe_*.py、smoke_encode.py）
  与 `smoke_out.txt`、`/tmp/dbg_riichi/` 已删除（全仓 rg 零引用）。
- **边界确认**：未修改 PPO/rollout/1v3/riichi_lab_bot 旧输入契约（仅包级 import 兼容）；
  未删除/覆盖任何 checkpoint、数据集、历史报告；未生成 60% 全量编码数据集；
  V16/V17 资产保持只读归档。

## 旧输入冗余代码清理（2026-08-27）

范围：删除 V18 重构后零引用的旧 54 行原子 Snapshot 族、旧 per-player semantic-token
路径与模型侧死代码/占位；PPO/rollout/learner/worker/inference/eval/1v3/GRP/riichi_lab_bot
训练路基（仍引用旧 history/snapshot 契约者）一律**未修改**，仅标记为待迁移。

命令（每主题提交前执行）：

```bash
rg -n "prepare_atomic_snapshots|AtomicSnapshotBatch|SNAPSHOT_FIELD_COUNT" riichi_ppo_v1 riichi_lab_bot   # 零命中
rg -n "SemanticToken|KIND_TILE_COUNT|SEGMENT_ACTOR_STATE|DecisionSnapshot|history_generations" \
  RiichiEnv/riichienv-state-machine riichi_ppo_v1 riichi_lab_bot                                       # 零命中
env LD_LIBRARY_PATH=$CONDA_PREFIX/lib conda run --no-capture-output -n Mahjong-AI \
  cargo test --manifest-path RiichiEnv/Cargo.toml --workspace
conda run --no-capture-output -n Mahjong-AI python -m pytest -q \
  riichi_ppo_v1/tests/unit riichi_ppo_v1/tests/protocol riichi_ppo_v1/tests/integration
conda run --no-capture-output -n Mahjong-AI python -m pytest -q RiichiEnv/tests
conda run --no-capture-output -n Mahjong-AI python -m riichi_ppo_v1.tools.validate --parameter-contract
conda run -n Mahjong-AI python audit/reports/v18/scripts/v18_audit_oracle.py
conda run -n Mahjong-AI python audit/reports/v18/scripts/v18_model_structure_audit.py
```

### 提交清单（每主题可独立回滚）

1. `refactor(constants)` — 溢出桶常量迁移到单源（6eb0b73）：
   `OPEN_MELD_YAKUHAI_HAN_OVERFLOW_BUCKET`(=6)/`VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET`(=8)
   迁入 `riichienv-state-machine/src/lib.rs` 常量区并导出到 `riichi` 扩展；
   `ENCODING_PROTOCOL_VERSION`(18) 导出随之迁入 lib.rs（原 atomic_snapshot::register 提供；
   为 C1 删除模块的前置）。`encoding_protocol.py` 增加 Python 镜像，
   `test_v18_encoding_protocol.py::test_overflow_bucket_constants_mirror_rust` 交叉验证
   （schema 基数 = 溢出桶 + 1）。
2. `chore(rust) C1`（0959f58）— 删除 `atomic_snapshot.rs` 模块及
   `prepare_atomic_snapshots`/`AtomicSnapshotBatch`/`global_visible_facts`/
   `self_progress_facts`/`opponent_riichi_traits`/`first_six_discard_counts` 与其单测；
   `_riichienv.pyi`/`riichienv/__init__.py`/`riichienv-python/src/lib.rs` 同步收口导出。
   保留 `open_meld_yakuhai_han`/`visible_meld_dora_aka_han`/`dora_kind` 等当前编码器活跃函数。
3. `chore(rust) C2`（9e6a1f9）— 删除 `player.rs` 旧 per-player semantic-token 路径与
   `semantic_token_tests.rs`；`manager.prepare_decisions` 只返回 241 维合法掩码（旧
   history/Snapshot 材料输出无任何活跃消费者，快照输入参数随之删除，bridge/sft/validation
   同步适配）；删除 `DecisionSnapshot`/`ACTOR_*`/`snapshot_tile` 与 table 的
   `history_generations`；`apply_events_batch` 保留事件解析/校验/边界同步与合法请求失效。
4. `chore(model) C3`（fa8d27d）— 删除 `DenseSlotFusion`（其槽位敏感性/顺序交换/padding
   断言已由 `StateTokenEmbedding` 等价用例覆盖，无需移植）、`_SEGMENT_ORDER_VALIDATOR`
   占位、`SEGMENT_KINDS`；清理 model/ 与 sft/ 全部未用 import（含 trainer.py 未用
   `grad_norm` 赋值；`validation.py` NUM_ACTIONS 改从 schema 单源导入）。`schema.py`
   保留 `NUM_ACTIONS` 再导出（8+ 处消费的单一来源枢纽，非死代码）。
5. 导出收口与过时注释（本次）— `__init__.py`/`_riichienv.pyi`/`lib.rs` 的导出删除已在
   C1/C2 同一提交内完成（符号删除与导出收口必须原子提交才能编译）；补充
   `validation.py` 模块文档与 `v18_input_protocol.md`/`KyokuEventTupleProtocol.md`
   残留的旧「54 行 Atomic Snapshot / history adapter」表述清理（负向表述保留「不再…」
   语义，仅删除对已删产物的引用）。

### 删除清单（含 rg 证据）

| 对象 | 位置 | 证据 |
| --- | --- | --- |
| `atomic_snapshot.rs`（601 行，含 SNAPSHOT_SCHEMA/encode/validate/register） | `riichienv-state-machine/src/` | 模块自迁出 overflow 常量后仅自测引用；`rg atomic_snapshot` 活跃路径零命中 |
| `prepare_atomic_snapshots`/`AtomicSnapshotBatch` | `riichienv-python/src/encoding_facts.rs` | `rg prepare_atomic_snapshots` 在 `riichi_ppo_v1/`、`riichi_lab_bot/`、`RiichiEnv/tests` 零命中；仅导出 |
| `global_visible_facts`/`self_progress_facts`/`opponent_riichi_traits`/`first_six_discard_counts` | `encoding_facts.rs` | 仅被 `prepare_atomic_snapshots` 调用（已证）；对应 5 个单测一并删除 |
| `SemanticToken`/`tokens()`/`append_state_tokens` 等全部令牌块 | `player.rs`（整文件删除） | 模块外零引用；`tokens()` 唯一调用方 `prepare_decisions` 同步改为掩码-only |
| `semantic_token_tests.rs`（3 测试） | `MjaiKyokuStateMachine/` | 仅测旧令牌格式 |
| `DecisionSnapshot`/`ACTOR_*`/`snapshot_tile`/`history_generations` | `types.rs`/`protocol.rs`/`table.rs` | 均为令牌路径专属，随 C2 删除 |
| `DenseSlotFusion` | `model/dense_embedding.py` | `rg DenseSlotFusion` 仅测试 import；测试实际只用 StateTokenEmbedding |
| `_SEGMENT_ORDER_VALIDATOR`/`SEGMENT_KINDS` | `architecture.py`/`encoding_protocol.py` | 零外部引用 |

### 保留清单（确认未删/未改语义）

- 当前 V18 全部路径：`prepare_current_state_batch`/`encode_query_batch`/
  `prepare_encoding_facts`/`analyze_encoding_yaku_batch`（保留函数
  `open_meld_yakuhai_han`/`visible_meld_dora_aka_han`/`dora_kind`/`decompose_melds`/
  `kernel_shape`/`tile_counts`/`count_dora_aka`/`dora_type`/`physical_tiles`/
  `remove_by_type`/`observation_facts`）、`model/{current_state,dense_embedding,
  architecture,semantic_validation,native_encoding,action_groups,validation,bridge,
  critic_features}.py`、`sft/*`、`configs/v18_*`。
- 状态机能力：`manager.prepare_decisions`（法律掩码）、`decode_actions`、
  `action_ids_with_source_indices`、`apply_events_batch`、`query_encoding.rs`、
  `analysis.rs`、`types.rs`（MjaiEvent/MjaiTile 常量）。
- PPO 训练路基：`training/{rollout_buffer,learner,worker,inference,metrics,
  trajectory}*`、`evaluation/*`、`training/grp/*`、`riichi_lab_bot/*` —— 原样保留；
  其中对旧 history/snapshot 字段的引用为待迁移项（`test_learner_accepts_only_
  rollout_buffer` 断点、`riichi_lab_bot/tests` collection 失败均未修复）。
- 归档资产：`specs/008-*`、`audit/reports/v16|v17`、`checkpoints/*`、`datasets/*`、
  `logs/v16|v17` —— 未修改未删除。

### 验证结果（与清理前一致）

- `cargo test --workspace`：148→134 通过（去除 3 个旧令牌测试与 5 个旧快照测试 =
  删除数量一致），0 failed；
- pytest unit+protocol+integration：467 passed, 2 failed —— 仅既有两项
  （`test_historical_audit_and_logs_are_removed_but_checkpoints_remain` 环境资产项、
  `test_learner_accepts_only_rollout_buffer` PPO 待迁移断点）；`RiichiEnv/tests` 2 skipped；
- `validate --parameter-contract`：通过，total **5,804,914** 不变（state keys 258）；
- 两个 oracle（`v18_audit_oracle.py`：862 决策 concealed/public/known=0、collision=0；
  `v18_model_structure_audit.py`：RoPE/padding/batch PASS）输出与清理前完全一致。

## V18 输入链路复审与性能审查（2026-08-27，HEAD 6a8e422）

按用户「审查与测试当前环境/状态机能否给出正确编码输入、输入编码解码一致、Python 侧是否引入多余计算」执行，新增两份只读审计脚本并在 HEAD 上复跑全部基线：

- **结论**：Q1（预处理逐 token 语义/V18 结构）PASS；Q2（输入编码一致 + 输出解码）输入 PASS、SFT 解码 PASS（7150 个合法 id 往返 0 失败），**在线 bridge 解码存在 P1 发现 V18-DEC-1**（27/7150 chi/pon 经 `decode_actions + select_action_from_mjai` 失败，集中在 called-in-consume 表示的回放；SFT 路径不受影响，PPO 待迁移项）；Q3（性能）发现 5 项 P3（逐批 Python 语义校验≈0.57s/batch@512、precompute 逐 token 统计≈2.9µs/token、encode_batch Python 装配≈52%、Rust 批编码不释放 GIL、encode_kyoku 双份 action_jsons+JSON 匹配）。
- **测试**：cargo 134 passed 0 failed；pytest unit+protocol+integration 183 passed 2 failed（仅既有两项）；RiichiEnv 284 passed 2 skipped；lab_bot collection 失败（既有待迁移）；`validate --parameter-contract` PASS（5,804,914/258 keys）；两个既有 oracle 全 PASS。
- **新增交付**：`audit/reports/v18/scripts/v18_decode_roundtrip.py`、`audit/reports/v18/scripts/v18_perf_review.py`、`audit/reports/v18/report/V18输入链路复审与性能审查报告.md`。
- **未改动**：PPO/rollout/1v3/lab_bot 仅盘点；未生成完整数据集；未启动正式训练；未删除归档资产；临时产物无。

## V18-DEC-1 修复与 v13 环境测试删除（2026-08-27 后续）

1. **V18-DEC-1 根修（bridge 解码缺陷，P1）**：`RiichiEnv/riichienv-core/src/observation/mjai_select.rs`
   在精确等长 consumed 匹配失败后，对 chi/pon/daiminkan 增加 `hand_only_consumed_matches` 回退
   （去掉与 `pai` 相同的被鸣牌后按序比较），兼容「consume_tiles 含被鸣牌」与「仅手牌侧」两种表示；
   新增 7 个 Rust 单元测试；重编译 `riichi`/`riichienv` 扩展。验证：`cargo test --workspace` 119+4+10+8 passed、
   `v18_decode_roundtrip.py` B2 27→**0**（7150 id 全通过）、`test_mjai_parity`/`test_action_to_mjai` 12 passed、
   两个 oracle 全 PASS、pytest 183 passed / 1 failed（仅 PPO 待迁移断点）。
2. **v13 checkpoint 环境测试删除（按用户决定）**：删除
   `test_artifact_conventions.py::test_historical_audit_and_logs_are_removed_but_checkpoints_remain`
   （V13 已停用；断言本机必须存在 `train_riichi_v13/v14/v15` 属环境资产项）。`rg` 确认仅测试自身与
   历史报告/归档 spec 引用（保留原貌）；测试文件重跑 11 passed。
3. 报告更新：`V18输入链路复审与性能审查报告.md` §2.2/§4/§5/§7 已同步；全部改动已提交。

## V18 precompute 性能优化（PERF-2/PERF-5，2026-08-27 后续）

生成正式数据集前按审查发现实施两项低风险 Python 热路径优化并全量验证：

1. **PERF-5**：`sft/data.py::encode_kyoku` 改用 `manager.action_ids_with_source_indices`
   （Rust 直接映射）索引 `legal_actions()`，替代 `decode_actions` + 每动作 Python JSON
   canonical 匹配；并消除每决策第二次 `action_jsons` 调用。
   cProfile（2 kyoku）：函数调用 86,722→39,642；`json.dumps 2610→219`、`json.loads 2740→349`、
   `action_jsons 262→131`。
2. **PERF-2**：`precompute._accumulate_field_statistics` 的 actor 行域检查按 kind 预计算
   列区间/基数上界后用 numpy 批量比较，替代逐 token × 逐字段 Python 循环。
   862 决策 95,291 token：273 ms → **113.7 ms**（≈1.2 µs/token，约 2.4×）。

验证：pytest unit+protocol+integration 183 passed / 1 failed（仅 PPO 待迁移断点）；
`v18_decode_roundtrip.py`（走生产 encode_kyoku）B1/B2 全 PASS；端到端冒烟
（`--game-sample-denominator 100000`，33 kyoku / 1,911 决策）precompute 成功，
manifest/offsets/field_statistics（越界 0/0）正常；冒烟产物已清理。

## SFT 训练首跑校验器 off-by-one 修复（V18-VAL-1，2026-08-27 后续）

1. **现象**：正式 60% 数据生成成功（validation 1,439,440 决策，actor_max=165），SFT 首批
   collate 抛 `AssertionError: SELF_HAND requires at least one nonzero kind`。
2. **根因**：合法「四副露+对子」和牌决策（例 `2024021011gm-00a9-0000-07d2a4be` kyoku0 seat3
   step19：3 碰+1 吃+对子 [44,44]，tsumo action=239）闭手只有一种牌 → SELF_HAND 恰好 1 行；
   `_assert_actor_canonical_order` 校验 SEP_SELF_HAND 后 `cursor += 2` 跳过第一行，1 行被误判为空。
   数据本身正确，无需重生成。
3. **修复**：`semantic_validation.py` `cursor += 2` → `cursor += 1` + 回归测试
   `test_v18_replay_bridge.py::test_single_self_hand_row_all_meld_pair_accepted`；
   真实异常 kyoku 74 决策全部通过，pytest 全绿。
4. **恢复命令**：`bash audit/reports/v18/scripts/run_v18_precompute_and_sft.sh --skip-precompute`
   （已有数据直接训练）。

## SFT 训练慢的诊断与修复（PERF-1 落地，2026-08-27）

- **现象**：用户报告 SFT 每 100 步要 ~80 秒（每步 ~0.8s，CPU 单核 101%），明显慢于
  V17/输入修改前的 V18。
- **定位**：`collate_samples(validate_semantics=True)` 每批跑全 Python 语义校验
  （B8 引入，默认 True）。实测 512 样本/批（~108 token/样本）：**collate+语义校验 547 ms，
  仅 collate 9 ms**——即每步 ~0.54s 的纯 CPU 冗余校验，是主导开销。
- **修复**：`v18_sft.yaml` 增加 `validate_semantics: false`（trainer 已支持该配置项）。
  安全性：precompute 阶段已断言域越界=0、`_assert_public_actor`、manifest/offsets fail-closed；
  载入端仍有 manifest/offsets 校验、每批 `_assert_targets_legal` 与模型 `_assert_structure` 兜底。
  需停止当前训练进程后重启生效。

## SFT 训练慢根因二：模型策略头逐行 Python 循环（已向量化，2026-08-27）

- **现象**：`validate_semantics=false` 后每步仍 ~0.45–0.9s；两个 worker 各占满一个 CPU 核，
  GPU 仅 32–49%——瓶颈在 Python 侧而非 GPU 算力（训练机为 2×L20）。
- **定位**：`architecture.py` 策略头对每行做 `torch.nonzero(query_mask[row])`（GPU→CPU 同步）
  + 逐行 scatter；local_batch=256 → 每 forward 256 次同步。
- **修复**：整批一次 `torch.nonzero(query_mask)`（单次同步）+ 行主序配对 + 扁平 scatter_add
  + 行内相邻重复检测（跨行重复合法）。L20 实测（local_batch=256）：每步 293ms→**109ms**
  （fwd 142→74ms，bwd 143→28ms，约 2.7×）。
- **验证**：18 个模型/查询/信息边界/BC 测试全过；结构审计（RoPE/mask/padding/批内一致）PASS；
  decode 往返 B1/B2 全 PASS；全量 pytest 184 passed / 1 failed（仅 PPO 迁移 WIP 项）。

## 性能优化与基准修正（2026-08-27 后续：嵌入/结构校验向量化）

**基准修正（重要）**：此前“02cd75e 旧版训练比当前慢 10 倍”的结论**有误**——CUDA 设备枚举顺序与
nvidia-smi 不一致，旧基准的 rank1 实际跑在 T400 4GB 上。用正确的 2×L20 重测：
**旧版 02cd75e 100 步 = 5.29s（≈53ms/步），旧版确实快**；当前优化后约 ~65ms/步，差距 ~1.2×。

**优化 1 StateTokenEmbedding 向量化**（`dense_embedding.py`）：
- 分隔符处理原为 `flat_kind.tolist()`（每批 28k 元素 CPU 往返）+ 逐分隔符 Python 循环；
  改为纯向量化 mask（kind∈[101,111] → id=kind-100）。
- 逐类别 `torch.nonzero(flat_kind.eq(kind))`（每类一次 GPU 操作）改为一次 stable `argsort`
  分段 + 每段切片。
- 输出与优化前**逐位一致**（max_abs_diff=0.0）；`token_embedding` 36ms → **7ms**。

**优化 2 `_assert_structure` 向量化**（`architecture.py`）：
- 原实现逐行 `tolist()` + Python 循环（一次 forward 实测 ~32ms）；改为整批 segment 查表 +
  boolean mask（保持全部失败语义/错误文案不变；`_segment_of_kind` 移除）。
- `_assert_structure` 31.6ms → **~1ms**；fwd 50 → **19ms**。

**torch.compile**：单卡 +~30%（57→40ms），但 **DDP 双卡下灾难性退化（8s/步，
IndexPutBackward0 图断裂）→ 配置 `torch_compile: false` 默认关闭，仅单卡可选**。

**最终实测（2×L20, batch 512, DDP 双卡）**：旧 02cd75e 53ms/步；当前 **~65ms/步**
（data 3 + fwd 19 + bwd 42 + opt 1）；经 precompute/encode 未改动，数据集无需重生成。
要生效需重启训练（`validate_semantics:false`、嵌入/校验优化均在进程启动时加载）。

## V18 PPO 训练性能优化（P0-1/P0-2/P1-4 回退/P1-5，2026-08-27）

按 `audit/reports/v18/design/V18 PPO训练性能优化审查提示词.md` 实施并 A/B 验证：

- **P0-1**：RolloutBuffer 因子行 uint8 压缩 + `_gather_padded` 单次分配 + collate 直出 int64
  （`training/rollout_buffer.py`）——`return_array_bytes` 10.2GB→**4.55GB（−55%）**、
  `transition_assembly_s` 2.93→1.30s。
- **P0-2**：learner collate 预取双缓冲线程（防死锁设计,`update_collate_prefetch` 默认 True;
  `training/learner.py`）+ learner 免 `query_rows` H2D。
- **P1-5**：推理 `collate_request_rows` 直出 int64 + pinned 缓冲复用 + `non_blocking` H2D
  （`training/inference.py`）。
- **P1-4（评估并回退）**：dummy 0 系数接入 policy 项 + `find_unused_parameters=False`
  使 bootstrap backward 104.2→248.4s（2.3×）,update_wall 251.5→401.4s——**REVERT**,
  列为 P2 观察项（动态重建 DDP 的 hook 风险中）。
- **A/B（512 半庄、mb2048、target_kl=0、update_epochs=4、双卡、3 轮）**：
  update_wall 251.5→227.2s（−9.7%）、187.3→162.0s（−13.5%）、374.3→351.1s（−6.2%）;
  sps +7.9% / +11.8% / +5.4%。第 3 轮受机器外部负载影响,以第 1/2 轮为准。
- **回归**：`pytest riichi_ppo_v1/tests` 201 passed（new 4 tests 含预取/uint8）;
  `cargo test --workspace` 141 passed;`pytest RiichiEnv/tests` 284 passed, 2 skipped;
  `git diff --check` 通过。详见
  `audit/reports/v18/report/V18_PPO训练性能优化实施记录.md`。

## 2026-08-28 update=10

- reward_mean=1.6517e-11 value_loss=0.31248 entropy=0.44712 actor_grad_norm=0.046277 critic_grad_norm=1.8799 shared_grad_norm=0.054686
- rollout_wall_s=246.75 update_wall_s=610.07 sps=1596.2 grp_calls=22133 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2955 top2_rate=0.5393 mean_rank=2.392 point_diff_mean=+2111.3 ci95=[1505.8575, 2721.4091666666664]

## 2026-08-28 update=10

- reward_mean=2.7881e-11 value_loss=0.32252 entropy=0.45077 actor_grad_norm=0.044847 critic_grad_norm=3.9424 shared_grad_norm=0.054813
- rollout_wall_s=235.43 update_wall_s=595.37 sps=1609.5 grp_calls=21656 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2955 top2_rate=0.5393 mean_rank=2.392 point_diff_mean=+2111.3 ci95=[1505.8575, 2721.4091666666664]

> 注:本条为第二轮(9e-5/8e-5)自动记录,其中 1v3 数字是训练循环按 update 编号
> 复用了旧 run 缓存(vs_sft_u010.json)的结果,非本轮真实评测。真实重跑
> (CUDA 2/3 → 物理 GPU 3/4):u5=0.2695/0.5065/2.470,u10=0.2860/0.5295/2.425,
> u15=0.3010/0.5527/2.371(first_place/top2/mean_rank)。eval 缓存复用的代码
> 缺陷(training/train.py:482-490)留待后续修复:缓存应绑定 checkpoint 身份。

## 2026-08-28 第二轮(actor 9e-5/critic 8e-5)停止归档

- 运行窗口:2026-08-28 17:08 – 20:39,完成 update 1–15 + update 5/10/15 三次 1v3 评测(全部真实)
- 配置:`riichi_ppo_v1/configs/v18_ppo.yaml`(第二轮,actor 9e-5/critic 8e-5/shared 7.5e-6,
  epochs=2、累积=10、2048 局/update;提交 f28b958)
- 归档:`logs/v18/ppo_lr9e5_20260828_1708/`(checkpoint_00005/00010/00015.pt + metrics/performance/
  tensorboard + v18_ppo_run2.log + eval/)
- 关键指标(u12→u15):EV(λ)=0.113→0.127、value_loss_raw≈0.014–0.015、grad_norm 回落至 1.4–2.2、
  KL/update≈1.0–1.03e-3(较第一轮 ~0.8e-3 提升约 25%,符合 1.5× LR 预期)
- 1v3 vs SFT:u5=0.2695、u10=0.2860、u15=0.3010(first_place);run1 同点为
  0.2652/0.2955/0.3118,差值均在 4000 半庄置信区间内——无回归、暂无明显提升
- 停止原因:维护者手动停训并归档;后续按需从 SFT init 重训或调参后再启

## 2026-08-28 V18 PPO 性能优化第二轮:阶段 A(A1-A5)

按 `audit/reports/v18/design/V18 PPO性能优化第二轮执行提示词.md` 实施,基线 tag
`perf-v18-round2-baseline`(b9af8a3),perftest 配置
`riichi_ppo_v1/configs/v18_ppo_perftest.yaml`(512 局、epochs 4、target_kl=0、双卡)。

- **A1(fix)**:1v3 评测缓存按 checkpoint sha256 内容校验(`summary_matches_checkpoint`),
  旧格式(无 `checkpoint_sha256` 记录)一律视为未命中并重跑原子覆盖;修复 run1/run2
  复用同名旧 JSON 的评测记录污染(即上文 update=10 注记遗留缺陷)。9524e60
- **A2(perf)**:loss 诊断字符串延迟构造,消除每 minibatch 4 次 float() GPU 同步。3cbe094
- **A3(perf)**:梯度累积非组末 minibatch 以 no_sync 包裹 forward+backward(单卡/无
  DDP 不受影响),有限性 all_reduce(MIN) 降到每累积组一次(与 optimizer step 同节拍;
  组边界由 learner_shard_indices 补齐后的 planned_minibatches 跨 rank 严格对齐)。
  新增双 rank gloo CPU 对照单测:与逐批 allreduce 最终参数 allclose(atol=1e-6)。
  cd267a4
- **A4(perf)**:`discounted_empirical_returns` gamma=1.0 走分段 float64 反向
  accumulate 后缀和(50 万行 178ms→3ms),与旧 Python 循环 `np.array_equal` 逐位一致;
  gamma≠1 保留旧循环按参数分派。新增 9 例单测。5d50409
- **A5(perf)**:collate 新增 `include_query_rows`(默认 True 保持其余调用方行为),
  learner 预取与直取路径传 False 跳过未消费的 query_rows。85ee52f

- **回归**:基线 `pytest riichi_ppo_v1/tests` = 1 failed, 200 passed(唯一失败为既有
  `test_v18_ppo_config_matches_stability_plan`:断言 actor LR 6e-5,而第二轮配置为
  9e-5,f28b958 引入,与本轮无关);阶段 A 后 = 1 failed, 213 passed(新增 13 测试
  全绿,不低于基线)。
- **A/B**(基线日志 `logs/v18/ppo_perf2_baseline_20260828.log`,阶段 A
  `logs/v18/ppo_perf2_phaseA_20260828.log`;第 1 轮预热,报第 2/3 轮):

  | 指标 | 基线 r2 | A 后 r2 | 基线 r3 | A 后 r3 |
  | --- | --- | --- | --- | --- |
  | algorithm_wall_s | 226.9 | 223.3(−1.6%) | 423.7 | 403.2(−4.8%) |
  | rollout/wall_s | 69.1 | 70.0(+1.3%) | 73.9 | 71.1(−3.8%) |
  | update/wall_s | 156.8 | 152.3(−2.8%) | 348.9 | 331.1(−5.1%) |
  | sps | 1652.6 | 1724.5(+4.4%) | 924.0 | 978.0(+5.8%) |
  | fwd ms/批/rank | 95.0 | 93.3 | 233.3 | 227.2 |
  | bwd ms/批/rank | 149.3 | 182.8(*) | 372.8 | 369.6 |

  (*) bootstrap 轮 backward stage 计时升高为同步点移除后的归因变化:基线每批
  backward 后的 `float(loss_is_finite)` + MIN allreduce + `.item()` 强制同步发生在
  计时 stage 之外;移除后主机先行入队,等待落入 stage 计时。端到端 update/wall_s
  与 sps 均改善,判定不劣化。r3(全 policy update,与生产结构一致)update −17.7s,
  按样本量外推生产规模约 −35s/轮,处于预期带(−35~70s)下沿。注:推理批组成受
  RPC 到达时序影响,bf16 内核随批形有微差,跨 run rollout 数据不逐位一致(既有
  非确定性),损失/熵/grad_norm 量级两侧均正常。
- **回滚**:单主题 `git revert <sha>`;全量 `git reset --hard perf-v18-round2-baseline`
  (本轮无 Rust/wheel 改动,无需重装扩展)。

## 2026-08-28 V18 PPO 性能优化第二轮:B1(前向去同步化)

- **改动**(669b1c1):forward 新增 `validate_structure`(默认 True)与
  `shared_capacity`/`critic_total_capacity`(host 预计算,默认 None 回退 .item());
  collate 额外输出两个 host 容量标量;learner/reference/rollout 推理按新配置键
  `update_validate_structure`(生产配置新增该键=false)传入;策略头以算术索引
  取代 torch.nonzero + repeat_interleave(变长输出同步 + compile 断裂源),校验
  路径新增 tail 窗口 canonical 契约检查(比旧数量检查更严)。新增 3 单测
  (契约断言/两态+容量两来源 torch.equal/违约拒绝);新增
  `riichi_ppo_v1/docs/v18_ppo_training.md` 开关说明。
- **同步点**:一次 forward 实测 45 个同步点 → lean 路径 23 个(剩余:dense
  embedding 16 个 = C2 目标;critic 组装 bool 索引 6 个;C1 前置已完成)。
- **等价性**:生产 bf16 autocast 路径与改前快照 `torch.equal` 逐位一致
  (validate 两态 × host 容量两来源 共 4 组合);fp32 路径仅 ~1e-7(6.7e-8,
  18/1928 元素)GEMM 形状舍入差,远小于 T2 的 1e-4 容差;回归 1 failed(既有)
  216 passed 不低于基线。
- **A/B**(同 perftest 协议,日志 `logs/v18/ppo_perf2_B1_20260828.log`,第 2/3 轮):

  | 指标 | 基线 r2/r3 | 阶段A r2/r3 | B1 r2/r3 |
  | --- | --- | --- | --- |
  | algorithm_wall_s | 226.9 / 423.7 | 223.3 / 403.2 | 222.0 / 408.1 |
  | update/wall_s | 156.8 / 348.9 | 152.3 / 331.1 | 145.7 / 333.9 |
  | rollout/wall_s | 69.1 / 73.9 | 70.0 / 71.1 | 75.3 / 73.2 |
  | fwd ms/批/rank(r3) | 233.3 | 227.2 | **154.1(−34%)** |
  | bwd ms/批/rank(r3) | 372.8 | 369.6 | 436.2(*) |
  | 推理 full_forward mean_ms(r3) | 22.4 | 21.9 | **19.9(−11%)** |

  (*) bootstrap 轮(1/2)bwd 与阶段 A 逐位一致(183ms);r3 的 bwd 上浮经
  微基准证伪为非代码因素:同权重同输入下,新算术索引路径的配对+策略头
  fwd+bwd 比旧 nonzero 路径快 8.0 倍(30.7→3.9ms/批),且运行窗存在外部
  CPU 负载(约 5.4)。r3 fwd −34% 显著超出预期带(−10~20%),收益方向兑现,
  判定保留。
- **回滚**:`git revert 669b1c1`(纯 Python/配置/文档,无 wheel)。

## 2026-08-28 V18 PPO 性能优化第二轮:B2(SFT reference 每 update 预计算)

- **改动**(5a8dcf4):`PPOLearner.update()` 在 epoch 循环前对整份 rank 分片以
  大批量(`update_reference_precompute_batch_size`,默认 8192)inference_mode +
  bf16 autocast 一次性预计算冻结 reference 的 policy_logits(fp32 [N,241] 驻留
  GPU);minibatch 内按 indices gather 取代每批一次 reference 前向(约占 update
  forward 40%)。bootstrap 期(sft_kl_coef=0)跳过。
- **等价性(T2)**:GPU 探针实测预计算(8192 块)与逐批前向(1536 行)的
  reference logits 差为 bf16 内核形状级(有限位 ≤1.7e-3 ≈1 ULP,-inf 位置
  完全一致),KL 差 ≤5.6e-5;按文档预案回退 1536/3072 分块仍为 2.5e-5/5.6e-5
  ——分块组成与分桶 minibatch 的 padding 维度不同,bf16 内核选择随形状变化,
  「1536 分块必然逐位一致」前提在本模型栈不成立。影响评估:sft_kl_coef≤0.0025
  时 loss 偏差 ≤1.4e-7(相对 ~1.4e-6);且旧路径逐 epoch 批组成不同,reference
  logits 本就带同类形状噪声。按 T2 精神(容差级一致、无训练语义变化)保留
  8192 默认并如实记录。回归 1 failed(既有)216 passed。
- **A/B**(日志 `logs/v18/ppo_perf2_B2_20260828.log`,第 2/3 轮):

  | 指标 | 基线 r3 | 阶段A r3 | B1 r3 | B2 r3 |
  | --- | --- | --- | --- | --- |
  | algorithm_wall_s | 423.7 | 403.2 | 408.1 | **375.6(较 B1 −8.0%)** |
  | update/wall_s | 348.9 | 331.1 | 333.9 | **305.2(较 B1 −8.6%)** |
  | rollout/wall_s | 73.9 | 71.1 | 73.2 | 69.4 |
  | fwd ms/批/rank | 233.3 | 227.2 | 154.1 | **141.5** |
  | bwd ms/批/rank | 372.8 | 369.6 | 436.2(*) | 368.9 |
  | reference_precompute s | — | — | — | 16.1(update 3;1/2 bootstrap 跳过) |
  | sps | 924.0 | 978.0 | 987.6 | **1064.8** |

  (*) B1 r3 的 bwd 上浮未被 B2 复现(同代码路径 368.9ms),确认为运行窗外部
  负载噪声。B2 收益方向兑现:reference 前向 4 遍 → 1 遍 + 大批量效率,
  update −28.7s/update(512 局规模),按样本量外推生产约 −55s,处预期带
  (−40~70s)内。
- **回滚**:`git revert 5a8dcf4`。

## 2026-08-28 V18 PPO 性能优化第二轮:B3 保留 / B4 放弃

- **B3(SDPA 后端锁定,保留,d6a9e50)**:GQA 的 SDPA 调用在 CUDA 上以
  `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` 包裹,结构化 bool mask 下
  约束不满足时 fail-fast 报错,杜绝静默回退 math 后端(实测 5.2 倍悬崖);
  CPU 路径保持默认调度(无 mem_efficient 内核)。单测证:默认调度本就选中
  mem_efficient(输出 torch.equal 逐位一致)、flash-only 上下文对该 mask 报错。
  B34 perftest 全程无后端回退报错,推理 mean_ms 20.5(B2 19.9,噪声带内)。
- **B4(推理批按长度排序,放弃,13dbd25 → revert 0f22103)**:
  A/B 实测 `padding_fraction_of_padded_tokens` 0.2552 与基线持平、rollout/wall_s
  72.4s(噪声带内),预期收益(−25~40s)未兑现。归因:padding = 行数 × 批内
  最大长度 − Σ长度,与批内顺序无关;排序仅在单批超过 max_batch=512 需要切块时
  有效,而实测 `full_forward_rows_mean ≈ 169 ≪ 512`(每个 16ms 等待窗的快照
  本身就是一批,排序改变不了批的长度跨度)。文档前提(批会跨 512 切块)与
  实际批动力学不符。真正的 padding 杠杆是跨等待窗累积凑批(需改批装配机制/
  等待参数,属本轮明确排除的配置/机制类变更),列为后续轮次候选。回归
  1 failed(既有)220 passed → revert 后复测通过。
- **B34 合并 A/B**(日志 `logs/v18/ppo_perf2_B34_20260828.log`,r3):algo
  372.2s(B2 375.6,基线 423.7)、update 298.9s(B2 305.2)、sps 1050.3。
- **回滚**:B3 `git revert d6a9e50`;B4 已 revert(0f22103)。

## 2026-08-28 V18 PPO 性能优化第二轮:C 组结论(C2 保留;C1/C3/C4 放弃;C5 跳过)

- **C1(torch.compile 三 Decoder,放弃)**:GPU 探针证实 fullgraph 编译成功且
  固定形状下 step 44.6→37.9ms(1.25×),数值在 T2 容差内;但 (a) 属性替换后
  state_dict 键引入 `_orig_mod` 前缀,破坏 checkpoint 契约;(b) `dynamic=False`
  要求静态形状,而生产 minibatch 的 token 容量 L 随分桶变化、推理 B×L 均变化,
  形状签名爆炸式重编译(探针 2 个形状已产生 6 张图);(c) 文档预设回退路径
  `dynamic=True` 实测 185ms/step(较 eager 44.6ms 慢 4.2 倍)。按放弃判据
  (回退仍无收益)放弃;形状分桶 padding 方案(改变 padding 计算与
  executed 语义)超出本轮"不改语义"边界,列为后续轮次候选。
- **C2(token_embedding host 行表,保留,8283f6d)**:`compute_kind_row_plan`
  在 collate/推理 host 装配时以纯 numpy 计算各类别升序行表(与旧 stable
  argsort 段内顺序逐位一致),forward 新增 `kind_row_plan`/
  `critic_kind_row_plan` 走静态键表迭代;分隔符改算术掩蔽(0 号 padding 行
  + 掩蔽,x+0.0==x);行表合并单次 pinned 异步 H2D(逐类别 pageable 拷贝
  每次都是一个同步点)。单测:两路 `torch.equal` 逐位一致(GPU 复验含全模型
  policy_logits/value);一次 forward 同步点 23→7(仅剩 critic 组装 6 + torch
  内部 1)。回归 1 failed(既有)219 passed。
- **C3(推理 CUDA Graph,放弃)**:图捕获要求区域内全静态形状且零设备同步;
  各批类别行组成是逐批动态数据,静态化需 K×桶静态索引缓冲 + dummy 行补齐
  (与 C2 行表路径冲突;旧 argsort 路径含 tolist 无法捕获);B4 实测批均值
  ~169 行 ≪ 512 也压缩了可寻址 launch 开销池。复杂度最高,允许放弃。
- **C4(FlexAttention,放弃)**:前置"C1 成功后追加"未满足,其融合 Triton
  内核同样依赖 torch.compile,受同一形状爆炸约束;三套 mask_mod 重写的 T2
  收益不确定性高。
- **C5(Rust 下沉 critic_feature_encode,跳过)**:文档明确仅在 B4/C3 落地后
  worker CPU 成为新约束时才有意义;两者均未落地,worker CPU 仍被推理 RPC
  等待掩盖。
- **C2 A/B**(日志 `logs/v18/ppo_perf2_C2_20260828.log`,r3,对比 B34):

  | 指标 | B34 r3 | C2 r3 |
  | --- | --- | --- |
  | algorithm_wall_s | 372.2 | **363.9(−2.2%)** |
  | update/wall_s | 298.9 | **289.2(−3.2%)** |
  | bwd ms/批/rank | 369.6 | **324.8(−12.1%)** |
  | collate_soa_gather s | 14.8 | 28.2(plan 计算入预取线程,净收益仍正) |
  | sps | 1050.3 | **1105.6(+5.3%)** |
- **回滚**:C2 `git revert 8283f6d`。

## 评测失败记录

- update=5：1v3 shard 子进程失败，训练已中止；已尝试路径与证据见下方失败详情。
  - shard 0: returncode=1            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;   File "/mnt/disk1/hubowen/zenith/riichi_ppo_v1/model/architecture.py", line 112, in forward;     value = F.scaled_dot_product_attention(;             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 44.42 GiB of which 22.00 MiB is free. Process 2631739 has 39.84 GiB memory in use. Process 2632187 has 0 bytes memory in use. Process 578837 has 666.00 MiB memory in use. Process 578789 has 666.00 MiB memory in use. Including non-PyTorch memory, this process has 488.00 MiB memory in use. Process 578885 has 392.00 MiB memory in use. Process 578933 has 392.00 MiB memory in use. Of the allocated memory 124.41 MiB is allocated by PyTorch, and 17.59 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
  - shard 3: returncode=1            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;   File "/mnt/disk1/hubowen/zenith/riichi_ppo_v1/model/dense_embedding.py", line 227, in forward;     base = self.segment(segment.clamp(0, 5)) + self.kind(kind.clamp(0, 127));            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 44.42 GiB of which 22.00 MiB is free. Process 2631739 has 39.84 GiB memory in use. Process 2632187 has 0 bytes memory in use. Process 578837 has 666.00 MiB memory in use. Process 578789 has 666.00 MiB memory in use. Process 578787 has 488.00 MiB memory in use. Including non-PyTorch memory, this process has 392.00 MiB memory in use. Process 578933 has 392.00 MiB memory in use. Of the allocated memory 55.75 MiB is allocated by PyTorch, and 6.25 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
  - shard 4: returncode=1            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;   File "/mnt/disk1/hubowen/zenith/riichi_ppo_v1/model/dense_embedding.py", line 227, in forward;     base = self.segment(segment.clamp(0, 5)) + self.kind(kind.clamp(0, 127));            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 44.42 GiB of which 22.00 MiB is free. Process 2631739 has 39.84 GiB memory in use. Process 2632187 has 0 bytes memory in use. Process 578837 has 666.00 MiB memory in use. Process 578789 has 666.00 MiB memory in use. Process 578787 has 488.00 MiB memory in use. Process 578885 has 392.00 MiB memory in use. Including non-PyTorch memory, this process has 392.00 MiB memory in use. Of the allocated memory 55.75 MiB is allocated by PyTorch, and 6.25 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## 2026-08-29 第三轮训练启动与 u5 评测 OOM 处置

- **启动**:v18_ppo.yaml(tmux 会话 ppo,日志 logs/v18/v18_ppo_r3_20260829.log)。
  u1-u5 全部完成:u1=608s(上轮同配置 825-900s,u_forward 194→96s)、u3(首个
  全 policy)828s、u4=816s;显存 learner rank0 峰值 ~40GiB(B2 reference 预计算
  常驻 + 激活缓存,高于上轮 ~35GiB),无 OOM、无异常。
- **事故**:u5 评测 3/10 分片 CUDA OOM(全部在 GPU0):learner rank0 进程占
  39.84GiB,GPU0 仅余 ~3GiB,评测分片(每卡 5 个、各 ~0.4-0.7GiB)挤爆;
  GPU1(11GiB 余量)5 分片全部成功。机制按设计中止训练(checkpoint_00005.pt
  已先行保存)。失败详情由机制写入本文件。
- **处置**:按维护者指示下调 minibatch,同时保持有效批量不变——
  `v18_ppo_resume_mb1024.yaml`(自包含副本):mb 1536→1024、accumulation
  10→15(1024×15×2=30720 与原一致)、resume checkpoint_00005.pt、init_model
  置空。
- **u5 评测补跑**(空闲 GPU,`logs/v18/rerun_eval_u5_20260829.log`):10/10 分片
  成功,first_place_rate=0.2685、top2_rate=0.4965、mean_rank=2.500、
  point_diff_mean=+365.60 ci95=[-311.98, 1021.26](与上轮 u5 0.2695 交叉印证)。
  vs_sft_u005.json 已含 checkpoint_sha256(A1 修复生效)。
- 随后以 resume 配置续训(u6 起,150 updates 目标不变)。

## 2026-08-29 update=10

- reward_mean=-3.0923e-12 value_loss=0.31339 entropy=0.44784 actor_grad_norm=0.0471 critic_grad_norm=2.2242 shared_grad_norm=0.053643
- rollout_wall_s=239.2 update_wall_s=527.91 sps=1796.9 grp_calls=22212 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2910 top2_rate=0.5260 mean_rank=2.430 point_diff_mean=+1547.2 ci95=[948.0366666666667, 2178.010416666667]

## 2026-08-29 续训(u6 起,mb1024+accum15)监控结论

- u6=866.9s、u7=788.2s、u8=774.1s、u9=775.0s、u10=811s 级,全部无错误;
  mb1024 后 learner 驻留显存 37.4/34.4GiB(mb1536 时 rank0 曾达 ~40GiB)。
- **u10 评测在 learner 驻留显存之上成功运行(10/10 分片)**:first_place_rate=
  0.2910、top2_rate=0.5260、mean_rank=2.430、point_diff_mean=+1547.22
  ci95=[948.04, 2178.01]——与上轮 u10(0.2860/0.5295/2.425)轨迹一致,
  mb1024+accum15 的显存余量修复得到验证。
- 训练在 tmux 会话 ppo 中按 v18_ppo_resume_mb1024.yaml 继续向 150 updates
  运行,转入免值守。

## 评测失败记录

- update=15：1v3 shard 子进程失败，训练已中止；已尝试路径与证据见下方失败详情。
  - shard 4: returncode=1            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;   File "/mnt/disk1/hubowen/zenith/riichi_ppo_v1/model/architecture.py", line 205, in forward;     x = x + self.down(F.silu(gate) * value);                       ~~~~~~~~~~~~~^~~~~~~; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 34.00 MiB. GPU 0 has a total capacity of 44.42 GiB of which 20.00 MiB is free. Process 1840348 has 38.14 GiB memory in use. Process 1840808 has 0 bytes memory in use. Process 2970691 has 854.00 MiB memory in use. Process 2970617 has 856.00 MiB memory in use. Process 2970616 has 1010.00 MiB memory in use. Including non-PyTorch memory, this process has 710.00 MiB memory in use. Process 2970713 has 906.00 MiB memory in use. Of the allocated memory 265.29 MiB is allocated by PyTorch, and 98.71 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## 2026-08-29 u15 评测 OOM(1/10 分片)与根因修复

- **事故**:u15 评测 1/10 分片 CUDA OOM(GPU0 44.42GiB 仅余 20MiB,shard 4 差
  34MiB;失败详情由机制写入上文)。mb1024 只是把余量从 3GiB 抬到 ~7GiB,
  仍属边缘;根因是 **learner 进程的 PyTorch 缓存分配器在 update 后保留大量
  cached-unallocated 块**(参考 logits 预计算 + 激活缓存),同卡评测分片
  无法使用这部分显存。
- **根因修复**(非调参):评测前显式释放缓存显存——`PPOLearner.release_cache()`
  (gc + `torch.cuda.empty_cache()`,活跃张量不受影响)、LearnerDDP 新增
  `_CMD_RELEASE_CACHE` 命令、推理 actor 新增 `release_cache` 远程方法、
  train.py 在 `run_1v3_evaluation` 前统一调用。显存余量恢复到接近空卡水平。
- **配置回归标准**:根因修复后不再需要降批,从头重跑使用标准
  `v18_ppo.yaml`(mb1536 + accumulation 10)。
- 归档:本次中止运行的 checkpoints(u5/u10/u15)与本轮 eval 产物(u5/u10 评测、
  u15 部分分片)移至 `logs/v18/v18_ppo_r3_20260829/`(run.log、rerun_eval_u5、
  checkpoints、eval);随后以标准 v18_ppo.yaml 从头重跑(日志
  `logs/v18/v18_ppo_r4_20260829.log`)。
- 注:1c86401 的 learner.py 部分同时包含当时工作区中维护者并行重构的片段
  (ACTOR_ROOTS/SHARED_ROOTS 收敛与 scheduled_u_coefficient 移除),与其
  model 重构同属一套、套件当时全绿,特此说明。
- **修复验证(r4,u5)**:learner 满配显存(mb1536)下评测前 release_cache
  生效,u5 评测 10/10 分片成功:first_place_rate=0.2557、top2_rate=0.5078、
  mean_rank=2.483、point_diff_mean=+441.02 ci95=[-190.19, 1043.09]
  (u5 距 SFT init 仅 4 update,与 r3 0.2685/run2 0.2695 交叉一致)。
  训练在 tmux ppo 继续向 150 updates 运行,转入免值守。
