# V19 Actor-only SFT（当前局面快照 + 信念监督）

V19 SFT 的入口为 `riichi-sft-precompute` 与 `riichi-sft-train`。**唯一现行
自包含配置为 `riichi_ppo_v1/configs/v19_sft.yaml`**（V19 标准 SFT，数据
`datasets/tenhou_sft_2024_2025_encoded_60pct_v19`，已预处理完成）；不再保留
其他 SFT 配置文件。不得覆盖归档旧版本数据。

## 数据契约

manifest 必须包含 `format=riichi-sft-encoded-v19`、`encoding_protocol_version=19`、
`state_protocol=riichi-current-state-v19-1`、运行时从 schema 推导的 contract SHA256、
`belief_labels=true` 与 `belief_shape`（102/3/105/102/102），以及正数的
train/validation 局数和决策数。训练加载器会 fail closed，拒绝旧格式、未知 hash、
缺失信念标签或形状不完整。

每条样本保存完整 Actor 序列（`actor_factors[T,32]`、`actor_numeric[T,8]`、长度）、
Query rows（`[2Q,15]`）、action IDs、legal mask、监督动作，以及信念五头标签
（hand `[102]`、shanten `[3]`、wait `[105]`、danger `[102]`、loss `[102]`）。
V19 不保留旧格式适配层，旧读写路径已移除。

## Actor-only 生命周期

`actor_only: true`、`train_critic: false`、`train_public_value: false` 时，优化器仅
接收 Actor 参数（token_embedding、public/actor backbone、行动作融合、策略头、
信念 backbone/查询/读出与 `belief_network`）；Critic backbone/value 参数冻结且无梯度。
SFT 目标为 `L_BC + belief_sft_coef·Σλ_k·L_k + λ_c·L_wait_danger`，默认
`belief_sft_coef=1.0`、λ_k 初始标定为 hand=1.0 / shanten=1.0 / wait=0.25 /
danger=3.0 / loss=3.0（依据停止中的 SFT step24000 验证损失尺度，见
PROGRESS.md 阶段 9）、`belief_wait_danger_weight=0.05`，Loss 目标按
`min(raw, 24000)/24000` 归一化。V19 标准配置下信念损失为条件/加权式（wait N/A 二判 +
仅听牌行 34 牌、danger pos_weight=5.0、loss 正例加权 huber），逐动作读出
`belief_readout_enabled=true`、`belief_readout_detach=true`（信念头只由标签校准），
`belief_public_grad_scale=0.25` 已由 `_forward_actor` 透传。

梯度隔离（监督单源，SFT 与 PPO 一致）：策略/BC 损失不得进入信念网络——
`token_matrix` 的输入为 `detach(summary)`，策略梯度沿 30 个信念 token 回传
止于转换矩阵；逐动作读出特征恒 detach；信念五头、1 层 belief backbone 与
`belief_query` 只由五头监督标签更新，共享层仅按 `belief_public_grad_scale=0.25`
接收监督梯度。

`torch_compile: true`、`validate_structure: false` 一起开启；首次编译约 1–2 分钟
属正常。固定验证与 checkpoint 间隔为 3000 steps，最终评估为 96 半庄，不能在实验配置里覆盖。
正式运行前先执行：

```bash
conda run -n Mahjong-AI python -m pytest \
  riichi_ppo_v1/tests/unit/test_v19_actor_sft.py \
  riichi_ppo_v1/tests/integration/test_v19_sft_lifecycle.py
```

本阶段只验证 SFT-ready 接口；完整数据集生成、SFT 与 PPO 按 V19 排期执行。
