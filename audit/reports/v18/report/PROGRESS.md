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
