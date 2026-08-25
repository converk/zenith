# V18 Actor-only SFT

V18 SFT 的入口为 `riichi-sft-precompute` 与 `riichi-sft-train`，当前自包含配置为
`riichi_ppo_v1/configs/v18_sft.yaml`。数据仍使用 2024–2025 既定 60% selection
（remainder `0,1,2 / 5`），但输出格式必须是新建的
`datasets/tenhou_sft_2024_2025_encoded_60pct_v18`；不得覆盖归档 V16 数据。

## 数据契约

manifest 必须包含 `format=riichi-sft-encoded-v18`、
`encoding_protocol_version=18`、冻结的 contract SHA256、源 manifest SHA256，以及
正数的 train/validation 局数和决策数。训练加载器会 fail closed，拒绝旧格式、未知
hash 或不完整计数。

每条样本保存 Objective Facts、`[29,4]` Snapshot factors、`[29,1]` Snapshot
numeric、Query pairs、action IDs、legal mask 和监督动作。V18 不保留旧格式适配层。

## Actor-only 生命周期

`actor_only: true`、`train_critic: false`、`train_public_value: false` 时，优化器仅
接收公共嵌入、3 个共享层、Actor-only 层、Query embedding 和 policy 输出参数。
Critic/private embedding/value 参数冻结且无梯度；保存文件只包含 V18 Actor 权重、
精确 `model_config` 与 contract 元数据，加载时 strict 校验。

固定验证与 checkpoint 间隔为 3000 steps，最终启发式评估为 96 半庄，不能在实验
配置里覆盖。正式运行前先执行：

```bash
conda run -n Mahjong-AI python -m pytest \
  riichi_ppo_v1/tests/unit/test_v18_actor_sft.py \
  riichi_ppo_v1/tests/integration/test_v18_actor_sft_lifecycle.py
```

本升级只验证 SFT-ready 接口；不会生成完整数据集、启动正式 SFT 或启动 PPO。
