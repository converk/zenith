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
- 二次 analyze:41 条 FR/SC、61 项任务,阻断问题为 0。

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
- 2026-08-25:新增 Rust 单一 29 字段 Atomic Snapshot schema、原生 fact/query bridge
  与真实 supplier 座次;Python 直接消费机器可读 schema。
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
- Objective Facts 均值 53.837822;Snapshot 固定 29;Query token 均值
  16.965206(8.482603 pairs);总长均值 99.803028,范围 54–169,满足 97–103。
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
| 固定 29 Snapshot/schema 单一来源 | Rust `atomic_snapshot.rs` + Python schema 消费 | Rust 5 个边界测试、V18 snapshot/bridge/replay tests |
| token 均值 97–103 | 只读统计工具 | 99.803028 / 1,439,440 decisions |
| Query metadata/supplier/隔离 | native facts + isolated Actor mask | metadata ablation、supplier domain、pair permutation tests |
| Actor/Critic 信息隔离 | Actor-only/full forward + strict private sequence | hidden invariance/private-use/order/future-five tests |
| Actor-only SFT-ready | `sft/actor_bc.py` + V18 contract/config | optimizer/freeze/update/save/load/legacy-rejection tests |
| 约 5M且无Q | 参数统计入口 + 无 Q 模块 | 4,930,562;forbidden keys 0 |
| 单一 V18 契约 | V18-only loader/config/bot/eval bridge | config/artifact/validation/full suites |

## Converge 结论

`$speckit-converge` 在首次完整 implement 后检查 32 条 FR、9 条 SC、16 个验收场景、
4 组计划决策与 7 项宪法原则。missing、partial、contradicts、unrequested finding
均为 0,没有向 `tasks.md` 追加任务;现有 61 项任务全部完成。
