# Contract: V16 数据集命名与 manifest

## 1. 命名(单一协议版本)

```text
原始数据:  datasets/tenhou_sft_2024_2025           (既有,不删除)
SFT 编码:  datasets/tenhou_sft_2024_2025_encoded_40pct_v16
GRP 数据:  datasets/tenhou_grp_2024_2025_v16
```

`_v16` 即输入编码协议 `encoding_protocol_version = 16`;不再出现
`_v<A>_v<B>` 组合后缀。

## 2. SFT manifest

```json
{
  "format": "riichi-sft-encoded-v16",
  "encoding_protocol_version": 16,
  "encoding_contract_sha256": "<协议契约内容哈希>",
  "source_manifest_sha256": "<来源 tar 清单哈希>",
  "subset_denominator": 5,
  "subset_remainders": [0, 1],
  "counts": {"train_kyokus": ..., "validation_kyokus": ...,
             "train_decisions": ..., "validation_decisions": ...}
}
```

- `format` 由单一协议常量派生(`riichi-sft-encoded-v{version}`)。
- 删除 token_schema_version / feature_schema_sha256 / rust_analysis_version /
  decision_analysis_version 多版本字段,以单一版本 + 契约哈希替代。
- 40% 划分(train:validation ≈ 99:1)沿用现有 prepare/precompute 参数。
- canary 运行必须满足:每个合法/专家动作组非零、数值越界计数为 0
  (沿用 v13 的 audit 门槛,见 `riichi_ppo_v1/docs/v13_sft.md`)。

## 3. GRP 数据

- 每半庄 4 个视角样本;prefix → 最终排名标签。
- 归一化统计量(σ_GRP、σ_Score)离线计算一次,写入数据集内固定 JSON,训练期只读。

