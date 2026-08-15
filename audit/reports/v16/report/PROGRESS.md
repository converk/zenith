# V16 进度记录

**特性**: specs/003-v16-model-rework(V16 模型重构与训练)
**分支**: `V16` | **基线 HEAD**: `4d4bb2a428c7c80322684bd0f68f0b9162f0c5a9`
**开始时间**: 2026-08-16

## 阶段进度

- [x] Phase 1 Setup:产物目录骨架、变更前基线、分支快照
- [x] Phase 2 Foundational:宪法修订 + 协议 v16 常量
- [x] Phase 3 US1:输入契约与语义落地(硬门槛通过;bridge 装配待 US2 联网)
- [ ] Phase 4 US2:网络 + SFT 重编码训练
- [ ] Phase 5 US3:GRP 模型/数据集/训练
- [ ] Phase 6 US4:PPO 集成与性能基线
- [ ] Phase 7 US5:治理闭环
- [ ] Phase 8 Polish:全链复跑与一致性收口

## 基线记录

- 变更前测试基线:pytest 243 passed(0 failed);cargo test --workspace 121 passed
  (111 core + 4 agari + 6 riichi state-machine);cargo 需
  `LD_LIBRARY_PATH=<conda env>/lib` 加载 libpython3.12
- 协议版本:新信息编码协议 v16(单一版本)

## 已完成工作记录

- 2026-08-16:宪法 v1.3.0→v1.4.0(Principle II:信息编码协议 v16,活跃实验代 v16,
  Sync Impact Report 已记录)
- 2026-08-16:新增 `model/encoding_protocol.py`(协议 v16 单一来源、20 slot 语义/
  基数/bucket)+ `tests/unit/test_encoding_protocol.py`(通过);`model/schema.py`
  版本常量改为引用协议常量
- 2026-08-16:Rust 内核完成并通过单测:`riichienv-core/offense_analysis.rs`
  (analyze_offense_v16,O0–O6/O8)与 `riichienv-state-machine/analysis.rs`
  (analyze_defense_v16,D0–D9);两个扩展已 maturin develop --release 重装
- 2026-08-16:产物约定测试扩到 v16;`audit/reports/v15/` 两个历史 eval 归档目录
  已按规范归档移动到 `v15/eval/`(只移动未删除)
- 2026-08-16:US1 语义落地:core `offense_analysis.rs` 拆为役/番数内核(O4/O5,
  `YakuAnalysisV16`),state-machine 新增 `analyze_offense_v16`(O0–O3/O6/O8)与
  `analyze_defense_v16`(D0–D9)+ `public_opponent_summary`;Python 侧新增
  `model/snapshot.py` 与 `model/action_query.py`;语义硬门槛
  (test_v16_query_semantics.py 8 项 + test_v16_replay_bridge.py 3 项)全部通过,
  覆盖门清/副露打牌、自摸/荣和终局、pass/吃牌、振听、N/A 与基数边界
- 2026-08-16:测试基线更新:pytest 258 通过;Rust workspace 124 通过

## 关键指标

- 语义正确性:待记录(20 slot 独立 oracle 比对)
- SFT 验证集 Recall@3:待记录(≥98% 为 PPO 前置)
- PPO 性能基线(后两轮):待记录
- 宪法修订:待记录(预期 1.3.0→1.4.0 MINOR)

## 接续指引(新会话交接)

- 工作目录 `/mnt/disk1/hubowen/zenith`,分支 `V16`,Conda 环境 `Mahjong-AI`。
- 继续执行 `$speckit-implement`,任务清单与勾选状态在
  `specs/003-v16-model-rework/tasks.md`;设计在 `plan.md`、契约在 `contracts/`。
- 已全部完成:Phase 1(T001–T003)、Phase 2(T004–T007)、Phase 3 除 T015 外
  (T008–T014、T016);T015(bridge 装配)随 US2 模型联网一起做。
- **下一个任务:Phase 4(US2)**,从 T017 开始:参数量测试 → T018 `ModelConfig`
  v16 preset(d_model=256/Q=16/KV=4/head_dim=16/FFN=1088/shared=4/actor=1/
  critic=2)+ Offense/Defense 对称融合(删除 `offense_fusion` zero-init 分支与
  241 维 `q_head`)→ T019 Critic 适配 → T020–T023 SFT 编码器/manifest 契约
  (单一 `encoding_protocol_version=16`)→ T024 canary → T025 40% 全量编码 →
  T026 SFT 冒烟(Recall@3)。
- 关键环境事实:
  - `cargo test` 需要 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`;
  - 扩展重装:`cd RiichiEnv && python -m maturin develop --release` 与
    `cd RiichiEnv/riichienv-state-machine && python -m maturin develop --release`
    (需 `PYO3_PYTHON=$CONDA_PREFIX/bin/python`);
  - 赤五物理牌号 = {16, 52, 88};core 的 `calculate_shanten` 对部分手牌会 panic,
    向听一律用 state-machine(`riichi.analyze_hands` / `analyze_offense_v16`);
  - 役/番数内核:`riichienv.analyze_offense_v16`(O4/O5),等待牌取普通五
    (wait*4 为赤五时 +1);
  - 当前基线:pytest 258 通过,Rust workspace 124 通过。
- 测试文件:`tests/integration/test_v16_query_semantics.py`、
  `tests/integration/test_v16_replay_bridge.py`、
  `tests/unit/test_encoding_protocol.py`。
