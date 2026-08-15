# Quickstart: V16 验证场景

环境:Conda `Mahjong-AI`,默认 `CUDA_DEVICE=0,1`、`learner_gpus=2`。命令在仓库根
`/mnt/disk1/hubowen/zenith` 执行。契约细节见
[contracts/](contracts/actor-input-v16.md);数据模型见
[data-model.md](data-model.md)。

## 场景 1:语义正确性硬门槛(必须先全绿)

```bash
conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests/integration/test_v16_query_semantics.py riichi_ppo_v1/tests/protocol -q
```

预期:每个 action query 的 20 个 slot 与独立 oracle 重算 100% 一致;N/A 规则、
bucket 边界、终局/流局/立直/吃碰杠约定全命中;categorical 越界样本为 0;Actor 无
隐藏信息、特权信息只出现在 Critic。任一 slot 不一致即判定失败,不得进入后续场景。

## 场景 2:编码 canary 与 manifest

```bash
conda run -n Mahjong-AI python -m riichi_ppo_v1.sft.precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_canary_v16 \
  --subset-denominator 5 --subset-remainders 0,1 \
  --game-sample-denominator 100 --game-sample-remainder 0 \
  --workers 16 --require-complete-action-coverage
```

预期:manifest 为 `format=riichi-sft-encoded-v16`、单一
`encoding_protocol_version=16` 与契约 sha256;每个合法/专家动作组非零、数值越界
计数为 0。验证后删除 canary 目录。

## 场景 3:40% 全量重编码

```bash
conda run -n Mahjong-AI python -m riichi_ppo_v1.sft.precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_2024_2025_encoded_40pct_v16 \
  --subset-denominator 5 --subset-remainders 0,1 --workers 16
```

预期:train/validation 决策数与独立统计一致;manifest 哈希稳定;契约版本字段唯一。
训练前与 canary/audit 报告核对后再启动。

## 场景 4:V16 SFT 从头训练

```bash
env -C /mnt/disk1/hubowen/zenith CUDA_DEVICE=0,1 PYTHONUNBUFFERED=1 \
  conda run -n Mahjong-AI python -m riichi_ppo_v1.sft.train \
  --config configs/v16_sft.yaml 2>&1 | tee logs/v16/v16_sft.log
```

预期:每 3000 steps 验证/启发式评测/保存一次;tensorboard 输出
`train/top3` 与 `validation/top3`(Recall@3),进入 PPO 前验证集 Recall@3 ≥ 98%
并记录到 `audit/reports/v16/report/PROGRESS.md`;checkpoint 含配置快照。

## 场景 5:GRP 数据集、训练与冻结

```bash
conda run -n Mahjong-AI python -m riichi_ppo_v1.training.grp.prepare \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_grp_2024_2025_v16
conda run -n Mahjong-AI python -m riichi_ppo_v1.training.grp.train \
  --config configs/v16_grp.yaml
```

预期:每半庄 4 个旋转视角、prefix→最终排名标签正确;σ_GRP/σ_Score 固化到数据集
JSON;GRP 参数 50–70K;验证集排名预测显著优于均匀随机;训练后权重冻结,PPO 前后
逐位一致。

## 场景 6:PPO 性能基线(3 轮,首轮预热)

```bash
env -C /mnt/disk1/hubowen/zenith RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,1 \
  PYTHONUNBUFFERED=1 conda run -n Mahjong-AI python -m riichi_ppo_v1.training.train \
  --config configs/v16_ppo.yaml --device cuda --learner-gpus 2 \
  --target-kl 0.0 --update-epochs 4 --kyokus-per-worker 16 \
  2>&1 | tee logs/v16/v16_ppo_baseline.log
```

预期:3 轮跑通,首轮视为预热,单独报告后两轮的耗时监控与全部相关性能指标;
Top-3 Q-boost 候选 = Top-3 ∪ 行为动作(≤4)、动作表示 detach;结束后
`ray stop --force` 清理,冒烟产生的日志与结果文件删除。

## 场景 7:治理门

```bash
rg -n "<删除目标符号清单>" riichi_ppo_v1 RiichiEnv riichi_lab_bot || echo ZERO
conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests -q
cargo test --manifest-path RiichiEnv/Cargo.toml
git diff -- riichi_ppo_v1/evaluation/mechanism.py configs/sft.yaml
```

预期:删除目标全仓零引用;Python 与 Rust 测试全绿;评测机制常量 diff 为空;协议
文档(`docs/v16_input_protocol.md`、`KyokuEventTupleProtocol.md`)与实现差异为 0;
宪法 Principle II 已修订为「信息编码协议 v16」并记录 Sync Impact Report;
README/docs/AGENTS 路径与产物一致。

