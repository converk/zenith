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
- PPO 短程冒烟（`riichi-ppo-smoke`，1 games/1 update/1 worker，临时随机
  V19 SFT 初始化，CUDA L20）：iteration=1 transitions=1070 kyokus=12，
  loss=0.1615 value_loss=0.5384 entropy=2.022 belief 五头指标有限
  （hand_acc 0.186、shanten_top1 0.161、wait_auc 0.265、danger_auc 0.476、
  loss_mae 0.499、belief_total_loss 5.332），4 epochs 跑满，冒烟产物由
  smoke_main 自动清理，`ray stop` 后无残留。
- 记录与说明：critic explained variance 的“不低于 V18 基线”无法从现有
  V18 日志/checkpoint 指标直接取证（V18 metrics.jsonl 未含该字段）；
  V19 冒烟/单测已给出有限 value_loss 与 value 指标；设计已保留风险预案
  （critic_layers 可回退 2 层，+705,280 参数）供正式训练 A/B 验证。
  其余验收线全部达标。
## 阶段 7：V19 标准重训——信念骨干同构 + 逐动作读出 + 条件损失（已完成，本次实施轮）

改动文件（摘要）：`model/belief_network.py` 改为五头/摘要/token 模块（输入
`player_query_hidden [B,3,3,256]`，共享逐家小头、逐查询平均）；
新增 `model/belief_readout.py`（21 维逐动作信念特征 → d_model，零初始化，
detach 语义）；`model/architecture.py`（1 层 Ffn=512 信念 backbone + 9 查询 +
读出动量/透传）；`sft/trainer.py`（`_forward_actor(model,batch,config)` 透传
真 grad_scale/readout/validate_structure/kind_row_plan，`_belief_losses`/
`_belief_metrics` 条件化，`collate_samples` host 侧 shared_capacity/kind_row_plan，
DEFAULT_CONFIG 新键）；`training/belief.py`、`learner.py`、`learner_ddp.py`、
`tensorboard.py`、`inference.py`、`evaluation/policy_adapter.py`；
新增 `configs/v19_sft.yaml`；`v19_ppo.yaml` 增键与 `init_model` 指向 V19 标准 SFT。

关键决策：token_matrix 仍只由 actor/policy 梯度更新，监督损失不经过 token 路径；
SFT `belief_readout_detach=true`、PPO `false`；SFT 训练步不走 CPU AUC（新增指标
全部为 GPU 纯 torch，AUC/条件 AUC 只在验证 cadence）；CPU 单测关闭 torch.compile
（生产配置仍 true，首次编译约 1–2 分钟）。

测试：全量 `pytest riichi_ppo_v1/tests -q` **236 passed**；新读出动量单测覆盖
detach/tile_code=0/零初始一致/训练后改变 logits；参数实测 7,112,252 在
[7.0M, 7.2M] 契约内；SFT 2 步 CPU 集成跑通并输出新指标键。

- 补充（用户要求）：最终只保留一份 V19 标准 SFT 配置文件
  `configs/v19_sft.yaml`；历史 `configs/sft.yaml` 等冗余配置已删除；
  相关测试/README/文档同步。

## 阶段 8：信念网络策略梯度隔离（监督单源，已完成，本次实施轮）

> 依据：`audit/reports/v19/design/V19_信念网络策略梯度隔离_实施方案.md`。
> 本修正取代阶段 7 中“SFT detach=true、PPO detach=false”的旧决策：
> 策略/BC 损失不得以任何形式进入 token_matrix 之后的私有信念网络。

改动文件（摘要）：
- `riichi_ppo_v1/model/belief_network.py`：`token_matrix` 输入改为
  `detach(summary)`，策略梯度沿 30 个信念 token 回传止于转换矩阵；中文注释
  说明 token_matrix 只由 actor/policy 梯度更新、信念网络只由监督标签更新。
- `riichi_ppo_v1/model/architecture.py`：防御性注释（token 路径与读出路径均
  不进入信念网络梯度），无结构改动。
- `riichi_ppo_v1/configs/v19_ppo.yaml`：`belief_readout_detach: false → true`，
  注释改为“SFT/PPO 均恒 detach，策略梯度不得塑形信念网络”。
- `riichi_ppo_v1/training/learner.py`：`belief_readout_detach` 默认值
  `False → True`，注释同步监督单源语义。
- 新增 `riichi_ppo_v1/tests/unit/test_v19_belief_gradient_isolation.py`：
  - 策略/BC 损失 backward 后 `token_matrix` 有梯度，
    信念五头 / `belief_backbone.*` / `belief_query` 无梯度；
    读出投影仍由 actor 损失更新；
  - 五头监督损失 backward 后五头 / backbone / `belief_query` 有梯度，
    `token_matrix` 无梯度（监督不经过 token 路径）。
- 同步测试：`test_v19_ppo_config.py` 期望 `belief_readout_detach=True`；
  `test_v19_learner_belief_loss.py` 转发断言 `detach=True`；
  `test_v19_belief_readout.py` detach=false 用例注明仅为模块级能力。

关键决策：
- 信念网络（1 层 backbone + 五头 + belief_query）的梯度完全来自五头监督标签，
  SFT 与 PPO 一致；共享层按 `belief_public_grad_scale=0.25` 接收监督梯度。
- 逐动作读出特征 SFT/PPO 均恒 detach；读出投影自身仍由 actor 损失训练。
- 不改输入协议、30 token、mask、编码格式与契约 hash。

测试结果：
- `test_v19_belief_gradient_isolation.py` + 既有 belief/architecture/readout/
  ppo_config/learner_belief_loss 相关单测通过。
- 全量 `pytest riichi_ppo_v1/tests -q` 通过（见本轮失败记录若无）。

文档同步：三册设计文档（信念网络 / 信念监督标签与训练 / 输入与模型编码）、
`riichi_ppo_v1/docs/v19_input_protocol.md`、`riichi_ppo_v1/docs/v19_sft.md`、
`AGENTS.md` 版本契约、本进度文件。

### 五头损失比例调整建议（阶段 8 建议，阶段 9 已落地）

依据停止中的 SFT 训练（`checkpoints/train_riichi_v19/sft/metrics.json`，
step 24000 终态）：验证集 `belief_hand_loss≈0.730`、`belief_shanten_loss≈1.227`、
`belief_wait_loss≈8.113`（其中 `wait_tile≈7.850`）、`belief_danger_loss≈0.135`、
`belief_loss_loss≈0.0042`。等权 λ=1.0 时 wait 头贡献约 79% 的加权监督损失，
与实施方案 §7 的判断一致。

建议起始值（梯度隔离落地后用小步 SFT 标定，不要一次大改）：
- `belief_head_weight_wait: 0.25`（候选取 0.2–0.5，以
  `wait_conditional_auc` / `wait_tenpai_acc` 不崩为下限）
- `belief_head_weight_hand: 1.0`
- `belief_head_weight_shanten: 1.0`
- `belief_head_weight_danger: 3.0`
- `belief_head_weight_loss: 3.0`

按上述起始值估算：验证贡献约 wait 2.03、shanten 1.23、hand 0.73、
danger 0.40、loss 0.013，五头贡献同量级；wait 头从约 8.1 降到约 2.0 的
加权损失，`belief_loss_total` 预期从约 10.2 降到约 4.4。实际数值需在
梯度隔离复训后按验证集曲线再定终值。

## 阶段 9：五头损失权重初始标定（已完成，本次实施轮）

> 用户指示把阶段 8 的建议值落地为实际训练权重；仍保留“终值待复训确认”。

改动文件（摘要）：
- `riichi_ppo_v1/configs/v19_sft.yaml`、`v19_ppo.yaml`、`training.yaml`：
  五头权重从全 1.0 改为 hand=1.0 / shanten=1.0 / wait=0.25 / danger=3.0 /
  loss=3.0，并加初始标定注释。
- `riichi_ppo_v1/sft/trainer.py`：`DEFAULT_CONFIG` 五头权重同步为初始标定值。
- `riichi_ppo_v1/tests/unit/test_v19_ppo_config.py`：PPO 期望值更新，新增
  `test_v19_sft_config_initial_belief_head_weights` 锁定 SFT 配置同值。
- 文档同步：`riichi_ppo_v1/docs/v19_sft.md`、训练分册 §4.1/§6、梯度隔离
  实施方案 §7（“建议”改为“已落地”）、本进度文件。

关键决策：
- 依据 `checkpoints/train_riichi_v19/sft/metrics.json`（step 24000 终态）的
  验证损失尺度：wait≈8.11、shanten≈1.23、hand≈0.73、danger≈0.135、
  loss≈0.0042；等权时 wait 单独贡献约 79%。初始标定让五头加权贡献同量级。
- 不触碰梯度隔离结构；正式 SFT/PPO 复训后用验证曲线定终值，必要时再调
  wait（0.2–0.5 区间为下限窗口）。

测试结果：`test_v19_ppo_config.py` 与相关单测通过；全量 pytest 通过
（见本次运行记录）。

## 阶段 10：关闭 wait_tile BCE 并回调 wait 头权重（已完成，本次实施轮）

> 用户决策（2026-09-06）：未知局面的逐牌等待概率过于随机、难以监督且长期
> 主导损失，关闭 `belief_wait_tile_weight`；wait 头只保留听牌/非听二判，
> 故 `belief_head_weight_wait` 由 0.25 回调至 0.8。

改动文件（摘要）：
- `riichi_ppo_v1/configs/v19_sft.yaml`、`v19_ppo.yaml`：`belief_wait_tile_weight:
  1.0 → 0.0`、`belief_head_weight_wait: 0.25 → 0.8`，注释说明决策依据。
- `riichi_ppo_v1/configs/training.yaml`：`belief_head_weight_wait: 0.8`（中性默认）。
- `riichi_ppo_v1/sft/trainer.py`：`DEFAULT_CONFIG` 与 `_belief_losses` 默认值同步
  （wait=0.8、tile=0.0）。
- `riichi_ppo_v1/training/belief.py`：`belief_losses`/`_belief_loss_components`
  默认 `wait_tile_weight=0.0`、五头默认 wait=0.8 / danger=3.0 / loss=3.0。
- `riichi_ppo_v1/training/learner.py`：`belief_head_weight_wait` 与
  `belief_wait_tile_weight` 默认值同步。
- 测试：`test_v19_ppo_config.py` 期望值更新（SFT/PPO 均锁定 wait=0.8、
  tile=0.0）；`test_v19_learner_belief_loss.py` 的合成 kwargs 同步；
  `test_v19_belief_gradient_isolation.py` 新增
  `test_wait_tile_bce_disabled_by_default`（默认 wait_loss 只剩 tenpai 二判，
  raw tile BCE 仍上报）。
- 文档同步：`riichi_ppo_v1/docs/v19_sft.md`、训练分册 §4.1/§6、梯度隔离
  实施方案 §7、本进度文件。

关键决策/预期：
- 训练 loss 中 wait_tile 贡献将从约 1.41（0.25 × 5.63）降为 0；
  wait_tenpai 以 0.8 权重保留（约 0.19 贡献，阶段 11 已上调至 1.5），
  wait 头不再主导总损失。
- 模型仍然消费 wait tile 概率（摘要/读出），但不再接受直接监督，后续需
  观察 wait_tile 指标是否漂移；如果下游策略因此受损，再评估是否移除
  tile 级下游特征（阶段 11 备选）。
- 不改网络结构、不改输入协议；运行中的训练需停止后重跑或等本轮结束再应用。

## 阶段 11：信念头按信息量二次调权（已完成，本次实施轮）

> 用户决策（2026-09-06）：把优化空间更多给信息量大的头——听牌/危险/放铳
> 损失加大占比；hand 逐格误差对决策影响较小，略微降权（不要少太多）。

改动文件（摘要）：
- `riichi_ppo_v1/configs/v19_sft.yaml`、`v19_ppo.yaml`、`training.yaml`：
  五头权重改为 hand=0.8 / shanten=1.0 / wait=1.5 / danger=5.0 / loss=5.0。
- `riichi_ppo_v1/sft/trainer.py`、`training/belief.py`、`training/learner.py`：
  DEFAULT_CONFIG / 默认值同步。
- 测试：`test_v19_ppo_config.py`、`test_v19_learner_belief_loss.py`、
  `test_v19_belief_gradient_isolation.py` 期望值与合成 kwargs 同步。
- 文档同步：`riichi_ppo_v1/docs/v19_sft.md`、训练分册 §4.1/§6、梯度隔离
  实施方案 §7、60pct 实施方案注记、本进度文件。

预期（以 step 84000 的原始损失估，wait_tile 已关闭）：
| 头 | 原始验证/训练 loss | 新权重 | 预计贡献 |
|---|---:|---:|---:|
| hand | ≈0.702/0.704 | 0.8 | ≈0.56 |
| shanten | ≈1.164/1.171 | 1.0 | ≈1.16 |
| wait(tenpai) | ≈0.239 | 1.5 | ≈0.36 |
| danger | ≈0.094/0.105 | 5.0 | ≈0.47 |
| loss | ≈0.0036/0.0037 | 5.0 | ≈0.018 |

hand 从 1.0 降到 0.8（-20%），wait/danger/loss 分别上调到 1.5/5.0/5.0；
`belief_loss_total` 预计从（旧权重）约 2.35 升到约 2.57，仍远低于开启
wait_tile 时的 ~3.6。运行中的训练需停止后重跑或从头/resume 应用。

## 阶段 12：hand 权重进一步下调至 0.6（已完成，本次实施轮）

> 用户决策（2026-09-06）：hand 逐格手牌计数误差对最终决策影响较小，
> 再降 0.2，权重 0.8 → 0.6；其余头不变。

改动：
- `riichi_ppo_v1/configs/v19_sft.yaml`、`v19_ppo.yaml`、`training.yaml`、
  `sft/trainer.py`、`training/belief.py`、`training/learner.py`：
  hand 默认/配置权重改为 **0.6**。
- 测试与文档同步：`test_v19_ppo_config.py`、`test_v19_learner_belief_loss.py`、
  `test_v19_belief_gradient_isolation.py`、`v19_sft.md`、训练分册 §6、
  梯度隔离方案 §7、60pct 注记、本进度文件。

预期：hand 贡献从约 0.56 降至约 0.42（原始 loss≈0.702 × 0.6）；
信息量更大的 wait/danger/loss 相对占比进一步提升。

## 阶段 13：resume 配置与稳定快照（已完成，随后被用户撤销）

> 用户决策（2026-09-06）：应用最新权重从既有训练继续。由于 SFT 只覆盖保存
> `latest.pt`/`best.pt`，step 60000 快照已不存在，用户确认使用 step 84000
> 的 `latest.pt` 继续。

> **撤销（2026-09-06）**：用户随后决定改为从头训练 2 epochs，resume 配置与
> 未完成训练产物已删除（不归档），详见阶段 14。

改动（已回滚）：
- 复制当前 `checkpoints/train_riichi_v19/sft/latest.pt` 为稳定快照
  `checkpoints/train_riichi_v19/sft/resume_84000.pt`（防止后续验证点覆盖）。
- 新增 `riichi_ppo_v1/configs/v19_sft_resume.yaml`：完整自包含副本，
  `resume: checkpoints/train_riichi_v19/sft/resume_84000.pt`、
  `tensorboard_dirname: tensorboard_resume`，其余与 `v19_sft.yaml` 一致
  （含 hand=0.6 / wait=1.5 / danger=5.0 / loss=5.0 / wait_tile=0.0）。
- 测试：新增 `test_v19_sft_resume_config_is_self_contained` 锁定 resume 路径、
  自包含键集与关键超参与标准配置一致。
- 文档：`v19_sft.md` 说明 resume 配置；本进度文件。

注意：resume 前必须停止当前运行中的旧训练进程；后续验证点会继续覆盖
`latest.pt`/`best.pt`，但 resume 配置固定指向 `resume_84000.pt`。

## 阶段 14：决定从头训练 2 epochs，清理 resume/旧产物（已完成，本次实施轮）

> 用户决策（2026-09-06）：不要 resume，改为从零开始训练 **2 epochs**；
> 删除 resume 配置与未跑完的训练产物（不归档），重新启动。

改动：
- 删除 `riichi_ppo_v1/configs/v19_sft_resume.yaml` 及其
  `test_v19_sft_resume_config_is_self_contained` 测试；`v19_sft.md` 恢复
  "唯一现行自包含配置为 v19_sft.yaml"。
- 停止旧 SFT 进程并删除 `checkpoints/train_riichi_v19/sft/`（含
  latest/best/resume_84000/metrics/tensorboard）与 `logs/v19/sft_train_v19_1.log`。
- `riichi_ppo_v1/configs/v19_sft.yaml`、`sft/trainer.py` DEFAULT_CONFIG：
  `epochs: 1 → 2`；新增 `test_v19_sft_config_epochs_two` 锁定。
- 预期总步数约 `2 × 137,605 ≈ 275,210` 步（batch=1024、双卡），验证/保存
  节奏仍为每 3000 步。

## 阶段 15：shanten/hand 再次微调（已完成，本次实施轮）

> 用户决策（2026-09-06）：稍微减少 shanten、稍微增加 hand，避免 shanten
> 一家独大，同时保留 hand 的基础校准信号。

改动：
- 五头权重调整为 hand=0.7 / shanten=0.8 / wait=1.5 / danger=5.0 / loss=5.0
  （原 hand=0.6 / shanten=1.0）。
- 同步 `v19_sft.yaml`、`v19_ppo.yaml`、`training.yaml`、`sft/trainer.py`、
  `training/belief.py`、`training/learner.py` 与相关测试/文档。

预期：shanten 贡献从约 1.16 降至约 0.93，hand 从约 0.42 升至约 0.49；
信息量权重结构仍保持 wait/danger/loss 高于 hand 的目标方向。

## 阶段 16：wait 权重上调至 1.8（已完成，本次实施轮）

> 用户决策（2026-09-06）：听牌/非听二判是决策核心，wait 权重 1.5 → 1.8。

改动：
- 五头权重调整为 hand=0.7 / shanten=0.8 / wait=1.8 / danger=5.0 / loss=5.0。
- 同步 `v19_sft.yaml`、`v19_ppo.yaml`、`training.yaml`、`sft/trainer.py`、
  `training/belief.py`、`training/learner.py` 与相关测试/文档。

预期：wait(tenpai) 贡献从约 0.36 升至约 0.43，与 hand/danger 同量级，
听牌信息继续强化。

## 阶段 17：V19 SFT 2 epochs 训练完成与指标检查（已完成，本次轮）

> 用户要求检查已结束的 SFT 的运行指标、评测数据与损失趋势，尤其关注信念头。

- 实际运行：275,210 步（2 epochs × 137,605），耗时 36,597.5 s，92 个验证点；
  产物 `best.pt`（step 270000）与 `latest.pt`（step 275210）；
  检查点内嵌 `sft_config.epochs=2`，与本次运行一致。
- 策略 BC：val policy CE 0.7435→0.4489、top1 0.7310→0.8253、top3
  0.9420→0.9821；train–val 差距 ~0.004 CE，无明显过拟合；150k 步后进入平台期。
- 信念五头：shanten top1 0.459→0.483（基线 0.303）、wait_tenpai_acc
  0.883→0.894（基线 0.849）、danger AUC 0.871→0.929、loss MAE 0.0303→0.0195；
  hand_acc 0.710→0.713（基线 0.710）。
- 待关注：wait tile 级指标接近随机（precision@2=0.064、condition AUC=0.561）
  与 `belief_wait_tile_weight=0.0` 一致；`wait_danger_violation≈0.993` 的当前
  实现几乎恒为 1，建议修正统计口径；`audit/reports/v19/eval/` 尚无 96 半庄
  最终评测产物。
- 分析报告：`audit/reports/v19/report/V19_SFT_2ep_训练检查.md`，图表与全量
  验证表在 `audit/reports/v19/eval/`。

## 阶段 18：信念头模糊化重设计 + 配置/脚本/归档（已完成，本次实施轮）

> 用户决策（2026-09-07）：保留五头与每玩家 3 查询 token；Hand 模糊化为
> 花色×段位 16 组 × {0,1,≥2} 桶；Wait 模糊化为听牌 + 宽度桶 5 类；
> Shanten/Danger/Loss 保留；五头损失贡献均衡；SFT 改 1 epoch / batch=4096
> （6000 双卡 OOM 后回调，见本阶段未注）/ 每 10 步打点；重标数据集或写
> 一体化脚本；归档旧 SFT。

- 设计文档：`audit/reports/v19/design/V19_信念头模糊化_设计方案.md`；
  文档 `riichi_ppo_v1/docs/v19_sft.md`、`v19_input_protocol.md` 同步。
- 代码：belief_labels（Rust 精确→模糊映射）、belief_network（16×3 / 5 类，
  摘要 130 维）、belief_readout（wait 改为全局宽度）、sft/contract
  （`riichi-sft-v19-3-fuzzy`，shapes 48/3/3/102/102）、sft/trainer/tensorboard、
  training/belief/worker/rollout_buffer/learner/train/tensorboard、
  evaluation 1v3；模型参数 ~6.68M（token_matrix 282→130，上界仍 7.2M）。
- 五头均衡：每头原始损失除以标签分布基线（熵/最优常数 BCE/正例中位偏差），
  λ 默认 1.0；原始与归一化 loss 均上报。
- 配置：`v19_sft.yaml`（epochs=1、batch_size=4096、log_interval_steps=10、
  数据指针 fuzzy 集、五头 λ=1.0）、`v19_ppo.yaml`、`training.yaml` 同步。
  （注：6000 首次实跑在 step~470 OOM，用户决定调回 4096。）
- 脚本（未运行，等待用户执行）：`riichi_ppo_v1/sft/relabel.py` +
  `audit/reports/v19/scripts/run_v19_sft_fuzzy.sh`（直接调用环境解释器，
  不使用 conda run；支持 `--dry-run` / `--force`）。
- 测试：相关单测 56 passed + 集成/产物 11 passed；全量 unit 209 passed
  （parameter 区间更新后补跑通过）。
- 归档：`checkpoints/train_riichi_v19/archive_20260907_sft_2ep/`（best/latest/
  metrics/tensorboard）+ `logs/v19/archive_20260907_sft_2ep/sft_train_v19_2ep.log`。

## 阶段 19：SFT 评测间隔调整为 1000 steps（已完成，本次轮）

> 用户决策（2026-09-07）：fuzzy SFT 的验证/checkpoint 间隔从 3000 改为 1000；
> 终止并直接删除上一次 fuzzy 运行产物（不计归档）。

- `riichi_ppo_v1/sft/contract.py`：`SFT_CADENCE_STEPS = 3000 → 1000`。
- 同步：`AGENTS.md`、`riichi_ppo_v1/docs/v19_sft.md`、`configs/v19_sft.yaml`
  注释、训练分册设计文档、`test_artifact_conventions.py`。
- 删除活动产物：`checkpoints/train_riichi_v19/sft/`（本次运行仅 tensorboard）、
  `logs/v19/sft_train_v19_fuzzy.log`；fuzzy 数据集与旧 SFT 归档保留。

## 阶段 20：PPO 探索保持方案 + 训练时长 + 梯度归因诊断（已完成，本次轮）

> 用户决策（2026-09-08，SFT 收尾期间，基于 V18 r5 完整曲线复盘）：
> V19 PPO 以评估信念网络为首要目标，不追分。四项决定：
> ① entropy 前期放缓下降 + 全程 raw 熵地板 0.25；② 学习率维持 9e-5
> 不随 mb 1536→2048 上调；③ total_updates 150→200（games_per_update
> 维持 2048）；④ 增加逐损失项梯度归因诊断。

- **V18 证据基础**（TB `checkpoints/train_riichi_v18/ppo/tensorboard` +
  r5 日志 + 30 次评测 CSV）：
  - raw 熵 0.478(SFT init)→0.175(u150)，跌破 0.3 于 ~u73；
  - H 0.30→0.24 锐化段（u70→u105）伴随最强增益（+4057→+5031 分差、
    top2 0.588→0.6105）；H<0.24 后继续锐化无任何评测收益（纯研磨）；
  - 平台期归因排除 SFT_KL 过度限制：sft_kl 系数衰减 5x、KL 距 SFT 持续
    发散（0.21→0.27）、approx_kl ~1e-3 << target_kl 0.01、early_stop
    全程 0 次；真实瓶颈是 GRP 信号噪声地板（EV(λ) 仅 ~0.20）+ 低探索
    研磨，与用户熵地板想法互相印证；
  - lr 9e-5 保持的依据：r5 中 lr 线性衰减 4x 期间 approx_kl 恒定 ~1e-3
    （步长非 lr 瓶颈），6e-5→9e-5 轮间 A/B 无差异；1.4e-4 时代曾发
    梯度上升事故（e00a5ef 下调）。
- **熵方案**：三锚 0.014/0.006/0.002 → 0.018/0.010/0.004，
  middle_fraction 0.33→0.5（前期/中段放缓下降）；新增熵地板屏障
  `entropy_floor: 0.25` + `entropy_floor_coef: 0.02`——batch 均 raw 熵
  跌破地板才施加推力，高于地板梯度恒为零（前期零干扰），只挡 V18 证实
  无效的 <0.24 研磨段。
- **200 updates**：V18 后期 EV(λ) 0.184→0.198 与 value_loss 仍在改善
  （critic 未收敛）；延长总 updates 同时给 critic 更多数据与策略更多
  改进步数，单 update 制度不变（games_per_update=2048、有效批 40960、
  评测/存档节奏不变）。备选"增大 games_per_update"否决：同等墙钟下
  减少策略改进步数、优势信噪比收益不明。
- **逐损失项梯度归因诊断**（新代码）：每 10 个 policy update 在首个
  minibatch 上用 `torch.autograd.grad` 对 policy/value/entropy(+地板)/
  sft_kl/belief 各项单独回传，记录 项×参数根组（actor头/belief/shared/
  critic）pre-clip 范数（TB `PPO/梯度归因/*`）。不写 param.grad、不触
  DDP allreduce、图内 0.25 缩放如实捕获；失败打标记并本进程禁用（绝不
  中断训练）。直接回答"谁在吃梯度预算"（V19 的 belief loss ~O(1) vs
  策略项 ~1e-3，actor_grad_norm 从此由信念项主导，必须拆开看）。
- **belief_sft_coef 接线**：PPO learner 原先直接 `loss + belief_loss_total`
  （该键仅 SFT 消费）；现与 SFT 同构乘 `belief_sft_coef`（默认 1.0，
  行为不变，键从此两侧生效）。belief 分支 100%/shared 0.25 的梯度语义
  本就由 architecture.py 图内缩放保证（SFT/PPO 同路径，无改动）。
- 改动文件：`training/learner.py`（entropy_floor_penalty、
  loss_term_grad_norms、根组捕获、损失组装、诊断插桩、指标）、
  `training/learner_ddp.py`（聚合键集合）、`training/tensorboard.py`
  （12 个 curated 标签）、`configs/v19_ppo.yaml`（200 updates、熵方案、
  地板、诊断键）、`tests/unit/test_v19_learner_entropy_floor.py`（新增 6
  测）、`tests/unit/test_v19_ppo_config.py`（契约 +1）。
- 测试：新增/相关 22 passed；全量 `riichi_ppo_v1/tests` 233 passed
  2 skipped（CUDA 隐藏下运行，保护在跑 SFT；`test_learner_ddp.py` 需双卡
  排除，其键集合改动为纯增量，静态验证）。




## 阶段 21：fuzzy SFT（1 epoch）训练完成检查（已完成，本次轮）

> 用户要求检查模糊化 SFT 的五头评测、训练结果与损失趋势，评估训练有效性。

- 实际运行：34,402 步（1 epoch，batch=4096，lr=1.5e-4），耗时 17,365 s，
  34 个验证点（每 1000 步）；产物 `best.pt`（step 33000，val loss 0.48322）
  与 `latest.pt`（step 34402，0.48333），两者等价。
- 策略 BC：val CE 0.736→0.483、top1 0.735→0.815、top3 0.943→0.978；
  train–val 差 ~0.0025 无过拟合。同 epoch 对比旧精确版（batch=1024）各可比
  指标低 0.3–2.2pp，属大 batch 优化器步数少 4× 的步数效应，非退化。
- 信念五头（基线=fuzzy 验证集 97 shard 全量统计）：hand_acc 0.6654 超逐组
  多数类基线 0.6207 达 +4.5pp（旧精确标签仅超全零 +0.26pp，模糊化解决
  hand 头学不动问题）；shanten_top1 0.4755（基线 0.3028）；wait_tenpai_acc
  0.8862（基线 0.8492）；danger_auc 0.9073（旧 2ep 0.9290）；loss 条件
  MAE 0.0912 优于常数基线 0.1199 且好于旧 2ep 的 0.1292。
- 澄清两点口径：① loss 头全局 MAE 0.2091 变大是监督范围改为"仅危险正例"
  所致（安全格不再被压到 0），非退化；② loss_norm 1.31>1 是 Huber+21×
  加权 vs 不加权 MAD 的口径产物，按条件 MAE 对照模型实优于常数。
- 结论：训练有效；五头全部显著超基线，λ=1 归一化使五头贡献天然同量级
  （0.66–1.31），阶段 18 均衡设计达成。
- 遗留：96 半庄最终评测仍缺失（建议 PPO 前补跑）；danger_auc 距旧 2ep
  尚差 2.2pp，若 PPO 重视信念面可评估 2 epochs 复跑（约 9.6h）；
  `belief_loss_mae` 命名与 loss_norm 基线口径建议修正。
- 分析报告：`audit/reports/v19/report/V19_SFT_fuzzy_训练检查.md`，趋势图与
  验证表在 `audit/reports/v19/eval/v19_sft_fuzzy_trends.png`、
  `v19_sft_fuzzy_validation_table.csv`。

## 阶段 22：critic bootstrap 延长至 4 updates（已完成，本次实施轮）

> 用户决策（2026-09-08）：belief bootstrap 监督方案搁置（非 bootstrap 期
> 监督本就贯穿，空窗占比极小）；critic warm-up 2→4 updates。

改动：
- `riichi_ppo_v1/configs/v19_ppo.yaml`：`critic_bootstrap_updates: 2 → 4`，
  注释记录依据。`training.yaml` 为旧中性默认，不同步（V19 以 v19_ppo.yaml
  为自包含权威）；测试用合成超参不锁该值，无需改动。

依据（V18 TB 复盘）：2 次 bootstrap 结束时 EV(λ) 仍为 -0.055（未到均值
预测器），EV 转正发生在放开策略后的 u3–u6；V19 critic 输入从 future wall
换成 privileged_hands 需重学，延长到 4 次让 EV 在静止目标分布上先转正再
动策略。成本约 +27min（≈1% 总时长）。`critic_bootstrap_learning_rate`
维持 2e-5 不变（备选杠杆，本次不动）。

测试：`test_v19_ppo_config.py` 7 passed。

## GRP 重训实验：E[U] 辅助回归损失（阴性结果，2026-09-08）

- 动机：PPO 奖励直接消费 GRP 的期望 utility E[U]，而排列 CE 只监督分类；
  实验验证 "CE + β·MSE(E[U], U_true)"（β=1.0）能否压低 E[U] 误差。
- 代码：`model/grp.py` 新增纯函数 `utility_projection`/`expected_utility_from_logits`/
  `true_expected_utility`（不进 state_dict，worker strict 加载兼容）；
  `training/grp/train.py` 损失项 + `validation/eu_mae` 指标 + 训练结束重载
  CE 最优权重再冻结（修复无条件覆盖 bug，本次运行实际触发：
  `reload_best_step: 55600`）。测试 `test_grp_train.py`（4）/ `test_grp_mortal.py`（+3），
  全仓 255 passed。
- 配置：`configs/v19_grp.yaml`（eu_loss_coef=1.0，batch 2048，30 epochs，
  数据集 `tenhou_grp_2024_2025_v18` 全量 3,807,907 train / 38,477 validation 样本，
  与 V18 同数据同 seed，唯一变量为损失项）。
- 结果：best val loss **2.4871 @ step 55,600**，eu_mae **0.4928**；
  V18 基线（纯 CE）：loss 2.4861 / eu_mae 0.4928。**无可测差异（阴性）**。
  尾部斜率：最后 1.2 万步 eu_mae 0.4941→0.4927（每 2000 步约 -0.0002 且减速）；
  两次独立训练（不同损失）收敛到同一平台，判定为 21 维边界特征的信息上限，
  非优化不足或容量不足。
- 决策：`v19_ppo.yaml` 的 `grp_checkpoint` **维持指向 V18 checkpoint 不变**
  （两者统计等价，切换属无证据变量变更）；V19 GRP 产物
  （`checkpoints/train_riichi_v19/grp/best.pt`）保留作实验证据，不投入训练。
- 产物：`logs/v19/grp_train.log`；中断首跑残骸归档于
  `checkpoints/train_riichi_v19/archive_20260908_grp_killed_first_run/` 与
  `logs/v19/archive/grp_train_20260908_killed_first_run.log`。

### GRP 温度标定实验（2026-09-08，零训练成本）

- 发现：CE 训练的 24 类分布对 E[U] 而言**系统性欠置信**；对 logits 做温度锐化
  可降低 eu_mae。验证集折半防过拟合（标定折拟合 T、独立折评估）：
  **最优 T=0.3，eu_mae 0.4928 → 0.4800（-2.6%，零重训）**；
  T 过低（0.1）回升至 0.4879，存在内部最优。
- 证据脚本：`audit/reports/v19/scripts/grp_temperature_calib.py`（V18 checkpoint）。
- 状态：**未落地**。落地需 GrpRollout 推理时应用温度（worker + checkpoint
  model_config 记录 T + 测试），属推理契约微变更；待与 PPO 启动统筹决定。

### GRP 温度标定落地 + 启用 V19 重训 checkpoint（2026-09-08，决策更新）

- 温度标定已落地：`GrpRollout` 新增 `temperature` 推理参数（logits/T 后
  softmax，默认 1.0 向后兼容，非正值 fail-closed）；worker 从 PPO 配置
  `grp_temperature` 读取。测试 +3（温度化 δ 与手工推导逐位一致、缺省行为
  不变、非法值报错），全仓 258 passed。
- **决策变更**：`grp_checkpoint` 由 "维持 V18" 改为启用 V19 重训 checkpoint
  （`checkpoints/train_riichi_v19/grp/best.pt`）——两 checkpoint 验证集指标
  与温度标定曲线均逐位等价（测试折 eu_mae 0.4748 @ T=0.3），经维护者确认
  启用新产物。V18 checkpoint 原位保留作回退。
- 标定脚本参数化：`grp_temperature_calib.py [checkpoint]`（可对任意
  checkpoint 重标定）；V18/V19 最优 T 同为 0.3（测试折 0.4920→0.4748，
  约 -3.5%）。
- `v19_ppo.yaml` 同步：`grp_checkpoint` → V19、新增 `grp_temperature: 0.3`
  （含重标定提醒注释）。

### PPO 训练/评测设备迁移（2026-09-08）

- 1v3 评测设备 `eval1v3_devices` 改为 `["2", "3"]`（项目 CUDA_DEVICE 编号，
  对应物理 GPU 3/4；分片子进程按 CUDA_DEVICE 重新映射，双卡各 5 进程结构
  不变）。训练启动以 `CUDA_DEVICE=2,3` 拉起（train.py 映射为
  CUDA_VISIBLE_DEVICES，learner 仍见 2 卡）。

## 事故修复：SFT checkpoint 带 `_orig_mod.` 键前缀导致 PPO 启动失败（2026-09-08）

- 现象：V19 PPO 首次启动在 learner 加载 `init_model` 时 strict 校验失败，
  SFT `best.pt` 全部 238 个键带 `_orig_mod.` 前缀（torch.compile 包装产物）。
- 根因：`sft/checkpoint.py::checkpoint_payload` 只解包 DDP(`module`)未解包
  compile(`_orig_mod`)；V19 SFT fuzzy 运行开启 `torch_compile: true`，
  保存了包装后的键（V18 SFT 未开 compile 故键干净）。
- 修复：
  - 保存端（根因）：`checkpoint_payload` 解包顺序 DDP → compile，未来 SFT
    保存的键与 eager 完全一致；
  - 加载端（兼容既有工件，不改 checkpoint 文件）：新增
    `model/checkpoint.py::strip_compile_prefix`，应用于 learner
    `load_model_weights`、inference actor `_sft_model`、评测
    `policy_adapter.load_policy_adapter`、SFT `init_model`/resume 路径
    （resume 改为加载到解包后模块）。
- 顺带修复 `sft/trainer.py` 存量 lint（`Tensor` 未导入 F821 ×6、未使用变量
  F841 ×1）与 `learner.py` import 排序。
- 验证：新增 `test_compile_prefix.py`（4 项）；全仓 262 passed、ruff 全过；
  端到端验证 learner 与 1v3 评测两条加载路径 strict 加载 SFT best.pt 成功。

## 重大事故修复：overlap 流水线未消费新 rollout 数据，训练原地打转（2026-09-08）

- 现象：V19 PPO 前 4 个 update 的 rollout 数据逐位相同（transitions/kyokus/
  games 及全部数据派生指标 16 位指纹一致：动作率、和率、点数差、λ-return、
  token 数），而模型相关指标缓慢漂移（value_loss 0.358→0.311）——即每个
  update 都在用 iteration 1 的同一份 buffer 反复训练。
- 根因：`train.py` overlap 流水线的消费分支（`elif pipelined is not None`）
  只从 `pipelined` 读取了计时标量，**从未执行**
  `results = pipelined["results"]` 与 `actor_profiles = pipelined["actor_profiles"]`
  ——收割分支存进 `pipelined` 的新 rollout 数据从未被消费，并在下一轮
  收割时被覆盖丢弃。V18 无此问题（无 overlap 流水线，串行 collect 每轮
  使用广播后的新权重采新数据）。
- 修复：消费分支头部补上两行赋值。已验证 262 passed、ruff 全过。
- 影响：修复前启动的训练（01:54 起）全部作废——actor 在 bootstrap/warmup
  期 LR=0 冻结，learner 对 iteration 1 的 2081 局 buffer 反复过拟合，
  策略未接受任何新数据。该运行已停止,需以修复后代码重启。
- 教训：overlap 流水线为 V19 新增 Tier 2 路径，缺少覆盖「收割→消费」
  数据接力的集成测试；后续应为 overlap 路径补一个 2-update 冒烟集成
  测试（断言相邻 update 的 buffer 指纹不同）。

## 2026-09-08 update=10

- reward_mean=-1.1637e-09 value_loss=0.31777 entropy=0.49685 actor_grad_norm=0.86502 critic_grad_norm=1.3095 shared_grad_norm=0.19555
- rollout_wall_s=706 update_wall_s=705.99 sps=2064.2 grp_calls=22810 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2633 top2_rate=0.5198 mean_rank=2.459 point_diff_mean=+729.9 ci95=[219.7869444444445, 1217.5811111111109]

## 2026-09-08 update=20

- reward_mean=-1.2154e-09 value_loss=0.31603 entropy=0.47455 actor_grad_norm=0.80227 critic_grad_norm=0.94383 shared_grad_norm=0.18583
- rollout_wall_s=668.36 update_wall_s=668.36 sps=2078.9 grp_calls=22523 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2938 top2_rate=0.5427 mean_rank=2.390 point_diff_mean=+2189.3 ci95=[1691.9508333333333, 2706.818333333333]

## 2026-09-08 update=30

- reward_mean=-1.2167e-09 value_loss=0.31515 entropy=0.47104 actor_grad_norm=0.80318 critic_grad_norm=0.79746 shared_grad_norm=0.19183
- rollout_wall_s=660.84 update_wall_s=660.83 sps=2076.7 grp_calls=22438 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.3027 top2_rate=0.5522 mean_rank=2.362 point_diff_mean=+2807.2 ci95=[2308.0191666666665, 3301.4363888888893]

## 2026-09-08 update=40

- reward_mean=-1.1997e-09 value_loss=0.31023 entropy=0.4651 actor_grad_norm=0.76843 critic_grad_norm=0.38817 shared_grad_norm=0.19799
- rollout_wall_s=652.58 update_wall_s=652.58 sps=2073.8 grp_calls=22336 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2997 top2_rate=0.5550 mean_rank=2.360 point_diff_mean=+2815.7 ci95=[2315.009722222222, 3358.3319444444446]

## 2026-09-08 update=50

- reward_mean=-1.2335e-09 value_loss=0.30685 entropy=0.47987 actor_grad_norm=0.82334 critic_grad_norm=0.35306 shared_grad_norm=0.21081
- rollout_wall_s=474.9 update_wall_s=474.9 sps=2855.8 grp_calls=22285 history_pool_size=0
- 1v3 vs SFT: first_place_rate=0.2998 top2_rate=0.5522 mean_rank=2.365 point_diff_mean=+2639.6 ci95=[2145.072777777778, 3131.128888888889]

## 2026-09-08 跨代 2v2 SFT 对抗评测(V19 SFT vs V18 SFT)

- 任务:V19 与 V18 两个 SFT 模型(两代不同架构)进行 6000 半庄 2v2 对抗
  (同队两席为一组对家,按全局半庄序号奇偶轮换 {0,2}/{1,3});指标面与
  1v3 一致(一位率/平均名次/每座相对其余三家平均点差/配对 bootstrap 95% CI/
  动作分组率/逐小局业务指标/信念校准),另计队伍口径两席点数和之差。
- 机制:10 个评测进程 = 5 对锁步进程对(host: V19 栈;partner: V18 独立工作
  副本栈)× 每分片 600 半庄;随机种子基由 CLI 提供。因 GPU 0 被其他用户的
  vllm 服务(约 36GB)占用,全部进程置于 GPU 1,分两波各 5 对执行(每波
  10 个进程),第二波完成后自动合并。
- 架构边界:V18/V19 的 Rust 扩展各自定义同名 pyclass,跨 .so 传对象被
  PyO3 精确类型检查拒绝,单进程共栈不可行;采用双进程锁步,仅交换 MJAI
  动作串。V18 侧运行时来自工作副本 /mnt/disk1/hubowen/zenith_v18(含当前
  源码构建的 riichi/lib_riichienv 扩展);host 侧运行时来自当前 V19 源码的
  新鲜构建 /mnt/disk1/hubowen/zenith_eval_runtime/v19_fresh(已验证与 V18
  构建行为一致、编码与站点已安装扩展完全相同;站点安装的 _riichienv.so
  构建于 2026-09-06 00:34,早于 RiichiEnv 最后两次提交,其 drawn_tile 标记
  与当前源码不一致,建议训练结束后重建,见本日记录)。
- 代码:riichi_ppo_v1/evaluation/head_to_head_2v2.py(host)、
  _2v2_host_entry.py(入口)、v19_eval_runtime.py(运行时引导)、
  head_to_head_2v2_shards.py(分片驱动);V18 侧为工作副本内
  partner_2v2.py(带 sys.modules 引导)。
- checkpoint:V19 = checkpoints/train_riichi_v19/sft/best.pt,
  V18 = checkpoints/train_riichi_v18/sft/best.pt(两侧 sha256 记录于汇总)。
- 输出:audit/reports/v19/eval/2v2_sft_v19_vs_v18/(分片 shards/ 与汇总
  vs_v18_sft_2v2.json);日志 logs/v19/eval_2v2_sft_v19_vs_v18.log。
- 种子基:769206263(随机生成,10 分片互不相交连续区间,区间长 600)。
- 结果(6000 半庄,V19 = model_a,V18 = model_b):
  - 一位率 0.2513 vs 0.2487;平均名次 2.5115 vs 2.4885;top2 0.5027 vs 0.4973;
    被飞率 0.0656 vs 0.0621;最终点数均值 24840.3 vs 25136.5。
  - 每座点差(vs 其余三家均值):-197.5 vs +197.5;V19 95% CI [-554.4, +155.1]。
  - 队伍点差(V19 两席和 - V18 两席和):-592.4,95% CI [-1479.6, +279.1]
    (区间跨 0,两代 SFT 在 6000 半庄下统计意义持平,V18 略优但 insignifica)。
  - 小局面:和牌率 0.2111 vs 0.2103;放铳率 0.1181 vs 0.1155;立直率
    0.0139 vs 0.0133;立直机会接受率 0.4110 vs 0.3796;流局听牌率
    0.4986 vs 0.5005。
  - V19 信念校准(2,020,659 决策):hand_accuracy 0.6705,shanten_top1
    0.4721,wait_top1 0.8450,wait_tenpai_acc 0.8603,danger_auc 0.8880,
    loss_mae 0.2157(与 SFT 训练期水平一致)。
  - 吞吐:21.5 hanchan/s(单波 5 对约 4.6 分钟/600 半庄)。
- 结论:V19 SFT 与 V18 SFT 在 2v2 对抗下基本持平;V19 未展现对 V18 的
  显著优势,后续可结合 1v3 口径与对局细节(立直/副露倾向差异)进一步分析。
