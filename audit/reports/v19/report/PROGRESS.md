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
  wait_tenpai 以 0.8 权重保留（约 0.19 贡献），wait 头不再主导总损失。
- 模型仍然消费 wait tile 概率（摘要/读出），但不再接受直接监督，后续需
  观察 wait_tile 指标是否漂移；如果下游策略因此受损，再评估是否移除
  tile 级下游特征（阶段 11 备选）。
- 不改网络结构、不改输入协议；运行中的训练需停止后重跑或等本轮结束再应用。

