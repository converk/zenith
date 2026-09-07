# V19 Actor-only SFT（当前局面快照 + 模糊化信念监督）

V19 SFT 的入口为 `riichi-sft-precompute` 与 `riichi-sft-train`。**唯一现行
自包含配置为 `riichi_ppo_v1/configs/v19_sft.yaml`**（V19 标准 SFT，数据
`datasets/tenhou_sft_2024_2025_encoded_60pct_v19_fuzzy`，模糊标签由旧精确
数据集重标生成，`epochs=1`、`batch_size=4096`、`log_interval_steps=10`）；
不再保留其他 SFT 配置文件。不得覆盖归档旧版本数据。

## 数据契约

manifest 必须包含 `format=riichi-sft-encoded-v19`、`encoding_protocol_version=19`、
`state_protocol=riichi-current-state-v19-1`、运行时从 schema 推导的 contract SHA256、
`belief_labels=true` 与 `belief_shape`（48/3/3/102/102），以及正数的
train/validation 局数和决策数。训练加载器会 fail closed，拒绝旧格式、未知 hash、
缺失信念标签或形状不完整。

每条样本保存完整 Actor 序列（`actor_factors[T,32]`、`actor_numeric[T,8]`、长度）、
Query rows（`[2Q,15]`）、action IDs、legal mask、监督动作，以及模糊信念五头标签
（hand `[48]`、shanten `[3]`、wait `[3]`、danger `[102]`、loss `[102]`）。
V19 不保留旧格式适配层，旧读写路径已移除。

## 模糊信念头（2026-09-07 用户决策）

| 头 | 输出 | 语义 | 损失 |
|---|---|---|---|
| Hand | `[B,3,16,3]` | 花色×段位 16 组 × 计数桶 `{0,1,≥2}` | softmax CE |
| Shanten | `[B,3,9]` | 0..8 向听 | softmax CE |
| Wait | `[B,3,5]` | 非听 / 1 面 / 2 面 / 3-5 面 / ≥6 面 | softmax CE |
| Danger | `[B,3,34]` | 逐牌可荣危险度 | sigmoid BCE（pos_weight=5） |
| Loss | `[B,3,34]` | 逐牌反事实放铳打点 | 危险正例子集加权 Huber |

Hand/Wait 由 Rust 精确标签在 Python 边界映射为模糊标签；摘要维度
48+9+5+34+34 = 130；逐动作读出仍为 21 维（danger、loss 逐牌 + tenpai、
shanten、max_danger、max_loss、wait_width 全局）。

**损失均衡**：每头原始损失除以标签分布基线（CE 熵 / 最优常数加权 BCE /
正例中位偏差），λ_k 默认 1.0 时五头加权贡献同量级；`belief_head_weight_*`
可再微调。SFT 与 PPO 使用同一套 `sft/trainer.py` / `training/belief.py` 逻辑。

## Actor-only 生命周期

`actor_only: true`、`train_critic: false`、`train_public_value: false` 时，优化器仅
接收 Actor 参数（token_embedding、public/actor backbone、行动作融合、策略头、
信念 backbone/查询/读出与 `belief_network`）；Critic backbone/value 参数冻结且无梯度。
SFT 目标为 `L_BC + belief_sft_coef·Σλ_k·L_k_norm`，默认 `belief_sft_coef=1.0`。

`belief_public_grad_scale=0.25`、`belief_readout_enabled=true`、
`belief_readout_detach=true`（信念头只由标签校准）；Loss 目标按
`min(raw, 24000)/24000` 归一化；danger pos_weight=5.0、loss 正例加权 20。

梯度隔离（监督单源，SFT 与 PPO 一致）：`token_matrix` 的输入为
`detach(summary)`，策略梯度沿 30 个信念 token 回传止于转换矩阵；逐动作读出
特征恒 detach；信念五头、1 层 belief backbone 与 `belief_query` 只由五头监督
标签更新，共享层仅按 `belief_public_grad_scale=0.25` 接收监督梯度。

`torch_compile: true`、`validate_structure: false` 一起开启；首次编译约 1–2 分钟
属正常。固定验证与 checkpoint 间隔为 3000 steps，最终评估为 96 半庄，不能在实验配置里覆盖。
正式运行前先执行：

```bash
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest \
  riichi_ppo_v1/tests/unit/test_v19_actor_sft.py \
  riichi_ppo_v1/tests/integration/test_v19_sft_lifecycle.py
```

标签重标（不重编码）：`/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m
riichi_ppo_v1.sft.relabel --source <旧数据集> --output <新数据集>`；一体化脚本
`audit/reports/v19/scripts/run_v19_sft_fuzzy.sh`（不使用 conda run）。

