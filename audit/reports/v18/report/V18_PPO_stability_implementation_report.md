# V18 PPO stability implementation report

Date: 2026-08-27

## 修改文件

- `riichi_ppo_v1/configs/v18_ppo.yaml`: 新增 V18 PPO 正式自包含配置。
- `riichi_ppo_v1/training/learner.py`: 接入 λ-return value target、分支独立 gradient clipping、normalized entropy loss、三点 entropy schedule、step 级 target KL guardrail、rank-specific bucketing seed、accumulation 尾组缩放修正与新增 PPO 指标。
- `riichi_ppo_v1/model/architecture.py`: 增加 `critic_private_embedding_grad_scale`,只缩放 critic private token embedding 回流共享 `token_embedding` 的梯度,不改变 forward 数值或参数拓扑。
- `riichi_ppo_v1/training/rollout_buffer.py`: length bucketing 改为 coarse/windowed bucketing,保留 padding 优势并提升 minibatch 随机性。
- `riichi_ppo_v1/training/learner_ddp.py`: 汇总新增 PPO 指标,DDP 聚合分支裁剪与 λ/MC target 诊断指标。
- `riichi_ppo_v1/training/inference.py`: rollout autocast 改为服从 `inference_dtype` 配置。
- `riichi_ppo_v1/evaluation/policy_adapter.py`: 删除旧 history/snapshot/`isolated_action_query` 评测适配路径,改为当前 `current_state_snapshot` 批格式。
- `riichi_ppo_v1/training/metrics.py`: 扩展训练/评测共用业务指标 accumulator。
- `riichi_ppo_v1/training/tensorboard.py`: 接入新增 PPO 与业务指标的中文 TensorBoard 展示名。
- `riichi_ppo_v1/tests/unit/*`: 增加和更新 V18 PPO 稳定性测试。

## 解决的问题

- Global clipping 改为 actor/shared/critic 三分支独立裁剪,避免 critic 大梯度统一缩小 actor 与 shared。
- Critic target 从 MC return 改为 `old_value + raw_gae_advantage`;MC return 仅保留为 diagnostic。
- Actor loss 继续使用归一化 advantage,不会污染 critic 的 λ-return target。
- Entropy loss 支持 `raw`/`normalized`,V18 正式配置使用 normalized entropy 与三点分段 schedule。
- Target KL 从 epoch 末检查改为每 8 个 optimizer step 聚合检查,触发后只提前结束当前 update。
- Adam β1 提高到 0.95,不实现 gradient queue/EMA/accumulation 伪平均。
- Accumulation 只在 optimizer step 前裁剪,尾组按实际 minibatch 数缩放。
- Rollout/update dtype 由 `inference_dtype` 统一控制。
- 训练和评测业务指标复用 `SemanticMetrics`,TensorBoard 展示名使用中文。
- 1v3 评测的业务指标按「小局结束」逐局结算,全小局/player-kyoku 口径(和牌率、放铳率、流局率、立直率、流局听牌率等),不再只统计最终小局。
- 训练侧 `record_kyoku` 接入真实流局听牌掩码(牌山耗尽前的逐座听牌判定),`draw_tenpai` 只在荒牌流局累计,与文档口径一致。
- 1v3 评测摘要投影为 `eval/...` 标量写入 TensorBoard,与训练曲线同图对比。
- 旧模型架构训练兼容代码已移除:当前 PPO learner 要求显式 V18 三分支裁剪与 normalized entropy;1v3 adapter 只加载 `current_state_snapshot` checkpoint。

## 最终 PPO 参数

- 训练规模: `iterations=150`, `total_updates=150`, `games_per_update=2048`
- PPO: `gamma=1.0`, `gae_lambda=0.95`, `ppo_clip=0.20`, `update_epochs=4`, `minibatch_size=512`
- LR: actor/critic `4e-5 -> 1.5e-5`, shared `5e-6 -> 2.5e-6`, bootstrap critic `2e-5`
- Optimizer: AdamW β1/β2/eps = `0.95/0.999/1e-5`, `weight_decay=0.0`
- Clipping: actor/shared/critic = `0.5/0.5/1.0`
- Entropy: normalized, `0.020 -> 0.012 -> 0.0045`, middle fraction `0.33`
- KL guardrail: `target_kl=0.01`, `target_kl_check_interval=8`
- Critic gradient scaling: public `0.25`, private embedding `0.25`
- Bucketing: `bucket_window_multiplier=8`
- Checkpoint/eval interval: 5 updates(与宪法 1v3 机制一致,`checkpoint_interval_updates=5`/`eval1v3_interval_updates=5`)

## 与 V17 的核心差异

- V17 的 global clipping 会让 critic 梯度影响 actor effective LR;V18 改为分支裁剪。
- V17 critic 拟合 MC return;V18 critic 拟合 λ-return,降低 high-variance target 噪声。
- V17 entropy 使用 raw entropy 线性退火;V18 使用 normalized entropy 三点退火。
- V17 KL guardrail 主要在 epoch 后触发;V18 按 optimizer step 间隔触发。
- V17 PPO weight decay 为 0.01;V18 PPO fine-tuning 使用 0.0。
- V18 rollout bucketing 在长度窗口内 shuffle,减少连续同质 minibatch。

## 测试结果

- `conda run -n Mahjong-AI python -m compileall -q riichi_ppo_v1`: 通过。
- `conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests`: 190 passed, 2 warnings。
- 双卡极短 smoke:
  `CUDA_DEVICE=0,1 ... riichi-ppo-smoke --config riichi_ppo_v1/configs/v18_ppo.yaml --kyokus 1 --num-workers 2 --learner-gpus 2 --envs-per-worker 1 --minibatch-size 8`
  通过。输出包含有限的 value loss、entropy、approx KL,DDP 未死锁,smoke checkpoint 目录已清理。

## 当前风险

- 业务指标仍依赖环境/MJAI 事件字段命名;已覆盖 `hora`、`ryukyoku`、`reach/riichi` 与常见副露事件,若环境新增事件别名需同步扩展。
- `find_unused_parameters=True` 在非 bootstrap smoke 中产生 PyTorch 性能 warning;bootstrap 阶段仍需要兼容未使用 actor 参数,因此本次未改。
- 未执行完整 2048 半庄/update 的正式训练,最终 entropy/KL 曲线仍需在首轮长跑中确认。
- V16/V17 YAML 配置文件按要求保留为历史记录;当前训练代码不再保证这些旧架构配置可直接运行。
