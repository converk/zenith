# V19 实施进度记录

> 本文件按阶段记录 V18→V19 架构升级：改动文件、关键决策与理由、测试结果。
> 设计依据：`audit/reports/v19/design/` 三册 D1–D32 与 `AGENTS.md`。

## 阶段 0：契约与常量（进行中/已完成部分）

改动文件：
- `RiichiEnv/riichienv-state-machine/src/lib.rs`：`ENCODING_PROTOCOL_VERSION` 18→19。
- `riichi_ppo_v1/model/encoding_protocol.py`：V19 schema 全量变更——删除
  `KIND_CRITIC_FUTURE(14)/SEGMENT_CRITIC_FUTURE(5)/KIND_RIVER_SUMMARY(6)`；
  新增 `KIND_RIICHI_CARD(14)/KIND_BELIEF(15)/SEGMENT_BELIEF(5)`；
  `CONTEXT_TOKENS=256→320`；PLAYER/RIVER_DISCARD/MELD/TILE_STATE/
  OPPONENT_ANALYSIS 字段按 D9 收敛；MELD +meld_turn/called_tsumogiri；
  RIICHI_CARD schema 按 §5。
- `riichi_ppo_v1/model/architecture.py`：ModelConfig `layers=5/critic_layers=1`、
  `preset("v19")`、`_segment_map`/`_assert_structure` 更新（V19 段表 + 立直卡/
  信念 kind；critic 删除 future 校验）。
- `riichi_ppo_v1/model/critic_features.py`：删除 future wall 全部代码；
  改为优先从 Observation.privileged_hands 取四家真手（在线/回放同一数据源）。
- `riichi_ppo_v1/model/bridge.py`：prepare 删除 walls 参数与 future 传参。
- `RiichiEnv/riichienv-core/src/observation/mod.rs` + `state/mod.rs`：
  Observation 新增 `temp_furiten` / `permanent_furiten`（全状态标记）并
  `privileged_hands` 在线填充（仅训练/Rust 侧使用，不进 Actor 输入）。
- `RiichiEnv/riichienv-python/src/current_state_encoding.rs`：河区重构
  （删 SUMMARY/被鸣锚行/relative_seat/supplied）、RIICHI_CARD 恒发射、
  MELD 新字段、TILE_STATE/OPPONENT_ANALYSIS 收敛；新增信念五头标签批量导出
  `prepare_belief_labels_batch`（D26：上帝视角、反事实、无未来信息）。
- `riichi_ppo_v1/model/belief_labels.py`：新增 Python 标签边界（新文件）。
- `riichi_ppo_v1/sft/contract.py`：V19 契约版本与文案。
- `riichi_ppo_v1/model/parameter_count.py`、`tools/validate.py`：
  V19 参数审计入口（阈值 7.2M）。
- `riichi_ppo_v1/configs/v19_ppo.yaml`、`v19_sft.yaml`：自包含 V19 配置
  （PPO 超参沿用 v18_ppo.yaml；新增信念键）。
- `riichi_ppo_v1/docs/v19_input_protocol.md`：新写 V19 输入协议文档。
- `riichi_ppo_v1/tests/v18_fixtures.py`：迁移为 V19 合成张量夹具
  （供测试使用；文件名仅历史遗留，后续阶段统一清理为 v19 命名）。

关键决策与理由：
- 信念 token 注入位置定稿为 SEP_ACTIONS 之后、第一对 Query 之前：满足
  “最后一个信念 token 距第一对 query 恒距 1” 的输入分册 §6 不变式。
- 在线 Observation 全量填充 privileged_hands/temp_furiten 是训练侧特权
  数据源；Actor 编码路径不消费，语义验收将反向断言无泄漏。
- 标签 Loss 返回原始点数，训练侧按 /24000 clip 归一化（训练分册 §9 既定口径）。

测试结果（本阶段已做）：
- `cargo check`/`cargo test -p riichienv-python` 通过（12 tests，含新增的
  `encode_one_emits_riichi_cards_and_removes_summaries` 与
  `compute_labels_marks_tenpai_wait_and_danger`）。
- 编码器冒烟：真实 fixture `encode_kyoku` 产出 V19 序列（含 RIICHI_CARD，
  无 RIVER_SUMMARY）；belief 标签批量导出形状 [N,102]/[N,3]/[N,105]/[N,102]/[N,102] 通过。
- 期间修复 `suji_category` 在 debug 构建下 `rank<3` 时 `tile-3` 下溢的潜在
  缺陷（`then` 改为惰性闭包），属共享编码器正确性修复。

## 阶段 2：模型与信念网络（已完成，commit 3996472）

改动文件：
- `riichi_ppo_v1/model/belief_network.py`（新）：BeliefNetwork——z_pool=mean
  (shared_hidden) → Linear(256→512)+SiLU → 五头；三家共享
  Linear(282→10×256) 转换矩阵；输出六键
  （hand/shanten/wait/danger/loss_logits+sigmoid/summary/tokens）。
- `riichi_ppo_v1/model/architecture.py`：belief_network 挂载、
  `belief_public_grad_scale`（critic 同款 detach+重标度）、SEP_ACTIONS 后
  Query 前插入 30 信念 token、增广 seq/kind/length、mask 信念规则
  （query 读 belief、belief 读 shared∪belief、analysis 不读 belief）、
  策略头按增广长度取尾窗；policy_only 也返回 belief 六键。
- `riichi_ppo_v1/model/parameter_count.py`：上界注释 7.2M。
- 新增 `tests/unit/test_v19_belief_network.py`（5 测试）与
  `tests/unit/test_v19_architecture.py`（4 测试）。

关键决策与理由：
- 信念网络、转换矩阵均归属 actor 分支（训练侧不新增信念学习率），
  与实际计算图一致。
- 实际参数量 **7,039,600**：belief 网络 1,290,062（与设计 §2.6 一致），
  差值 -52,272 来自 phase0 嵌入表字段净增删；仍在 7.0M–7.2M 契约内，
  以 `parameter_count.py` 实测为准。

测试结果：`test_v19_belief_network.py test_v19_architecture.py` 8 passed；
梯度缩放数值验证（×0.25 → public_backbone 梯度 ≈1/4）通过。
- 业务语义脚本 `audit/reports/v19/scripts/verify_v19_semantics.py` 首轮通过：
  初始状态 + 20 步真实环境局中决策的全套正/反向断言（结构、RIICHI_CARD、
  critic 真手、信念标签 13 张/danger⊆wait/loss⇔danger、无信息泄漏）。
- 显存实测（脚本 `measure_v19_memory.py`，CUDA_VISIBLE_DEVICES=0 / L20）：
  B=256 peak 2.688GB；B=1024 peak 10.572GB；**B=2048 peak allocated
  21.03GB（reserved 23.30GB）≤35GB 验收线**，无需梯度检查点预案。

## 阶段 4：SFT 管线与数据重编码（已完成，commit 2b506de）

改动文件（摘要）：`sft/contract.py`（BELIEF_LABEL_SHAPES + manifest
fail-closed）；`sft/data.py`（EncodedSample 五头标签 + encode_kyoku 离线标签）；
`sft/precompute.py`（npz 五数组读写 + manifest belief 键 + v19 行序种子）；
`sft/trainer.py`（V19 建模、collate 五字段、联合损失
`L_BC + 1.0·Σλ·L_k + 0.05·L_wait_danger`、评估五头指标）；
`sft/tensorboard.py`、`actor_bc.py`、`checkpoint.py`、`train.py`、README/docs。
删除 `configs/v18_sft.yaml`（旧配置不再活跃；git 历史可回看），
`docs/v18_sft.md` 迁移为 `v19_sft.md`。

关键决策：SFT 联合损失中 Loss 目标归一化 `min(raw,24000)/24000`；
`belief_sft_coef=1.0`、五头 λ=1.0、`λ_c=0.05`。
测试：`test_v19_sft_contract.py` + `test_v19_sft_lifecycle.py` 4 passed；
v19 actor/encoding/artifact/cleanup 相关 31 passed；真实信念网络 2 步
CPU 训练集成跑通（指标非 NaN）。

## 阶段 6：bot 适配与完整 MJAI 事件日志（已完成，commit d4fe7ed）

改动文件（摘要）：`riichi_lab_bot/src/.../{bridge,policy,client,telemetry,
local_play,cli,audit,observation}.py` 与 README；`telemetry.py` 新增
`MjaiEventLogger`（logs/v19/bot_mjai/<session>-<game>-<ts>.jsonl，行含
log_no/game_no/seat/timestamp/event）与 `replay_mjai_log`（MjaiReplay
完整回放重建终局）；client 单点接入保证 exactly once；本地对局也写完整流。
测试：bot tests 21 passed + 1 skipped（CUDA 对比）；本地对局冒烟使用临时
V19 随机 checkpoint 跑 1018 steps + 回放 12 rounds 成功。

关键决策：bot 推理直接复用 `current_state.encode_batch`（与训练同一编码
路径）；`tiles_left` 保留用于规则判定、不入模型输入；真实 checkpoint
依赖阶段 4 产出的 v19 SFT，临时随机 checkpoint 已覆盖拓扑加载路径。

## 阶段 3/5：训练侧与 1v3 评测接入（已完成，commit a53e948）

改动文件（摘要）：`training/trajectory.py`（Transition 五头标签）、
`training/rollout_buffer.py`（SoA 五字段 + belief_present + collate）、
`training/worker.py`（全 policy 生成标签、current 写 buffer）、
`training/belief.py`（新：五头损失+Wait-Danger 软约束+纯 torch AUC）、
`training/learner.py`（BELIEF_ROOTS 并入 actor 组、grad_scale 转发、
五头损失加入 total、checkpoint format 5）、`learner_ddp.py`/`tensorboard.py`/
`metrics.py`/`train.py`/`inference.py`；`evaluation/policy_adapter.py`
（V19PolicyAdapter，无旧类名 alias）与 `head_to_head_1v3.py`（belief 指标面）；
`model/architecture.py` 修复 bf16 autocast 下 belief token dtype 不匹配；
`configs/training.yaml`/`v19_ppo.yaml` 同步。

关键决策：信念网络参数归 actor 优化器组（设计未给独立 belief LR）；
D25 全路径生成标签，但只有 current policy 决策写入 PPO buffer（非 current
座位不是策略学习者）。
测试：`test_v19_ppo_config.py` + `test_rollout_buffer.py` +
`test_v19_learner_belief_loss.py` 28 passed（含 CUDA reference 编译对照）。

## 集成与剩余验收（已完成，主会话）

- 全量测试迁移：删除/重命名全部 v18 前缀测试为 v19（架构/参数量/快照/
  buckets/dense-embedding/integration 六件套），修正 batched_pipeline 的
  walls 参数与 artifact/cleanup 的 checkpoint_dir 断言；
  `tests/v18_fixtures.py → tests/v19_fixtures.py`。
- **全量 pytest（riichi_ppo_v1 + riichi_lab_bot）245 passed, 1 skipped**
  （skip 为 bot CUDA L20 bf16 仅需 CUDA_DEVICE=2,3 的已知项）。
- 业务语义脚本 `verify_v19_semantics.py`：初始 + 20 步真实环境局中决策通过
  （正向：RIICHI_CARD×3、critic 真手、信念标签 13 张/危险⊆待牌/loss⇔danger；
  反向：无 critic/信念段、无 RIVER_SUMMARY、数值域合法）。
- 显存实测通过：B=2048 peak allocated 21.03GB（reserved 23.30GB）≤35GB，
  未触发梯度检查点预案。
- SFT 一体化脚本 `--smoke` 自检通过：mini 首 shard 重编码（5251 kyokus）→
  2 步 CPU SFT（loss 7.57→7.54），临时产物由 trap 清理。
- 记录与说明：critic explained variance 的“不低于 V18 基线”无法从现有
  V18 日志/checkpoint 指标直接取证（V18 metrics.jsonl 未含该字段）；
  V19 短程 PPO update 已在 `test_v19_learner_belief_loss` 中实际跑通并获得
  有限 value/EV 指标；设计已保留风险预案（critic_layers 可回退 2 层，
  +705,280 参数）供正式训练 A/B 验证。其余验收线全部达标。
