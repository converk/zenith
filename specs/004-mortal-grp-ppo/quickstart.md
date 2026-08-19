# Quickstart: V17 PPO(基于 Mortal GRP)

**环境**: `conda activate Mahjong-AI`;工作目录 `cd /mnt/disk1/hubowen/zenith`。

## 0. 一键脚本(推荐)

```bash
bash audit/reports/v17/scripts/run_v17_grp_then_ppo.sh
```

自动依次执行:GRP 数据集构造 → GRP 单卡训练(15 epochs、冻结)→ PPO 双卡训练
(CUDA_DEVICE=0,2) → 1V3 best-checkpoint 选择。默认使用 40% 子集 + 280 个
train shards(实测约 1 shard ≈ 1477 个 prefix 样本,**总计约 40 万样本**);
可用环境变量覆盖:`MAX_SHARDS=200`、`GRP_EPOCHS=6`、`PPO_ITERATIONS=100`。

## 1. 宪法修订(先做,如未完成)

- `$speckit-constitution`:原则 IV 修订 1v3 为 4000 半庄、每 5 updates;原则 II
  登记 v17 实验代。

## 2. GRP 数据集构造(40 万样本 ≈ 280 shards)

```bash
python -m riichi_ppo_v1.training.grp.prepare \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_grp_2024_2025_v17 \
  --subset-denominator 5 --subset-remainders 0,1 \
  --kyokus-per-shard 512 \
  --max-shards 280
```

## 3. GRP 离线训练(单卡,4 epochs)

```bash
env -C /mnt/disk1/hubowen/zenith CUDA_DEVICE=0 PYTHONUNBUFFERED=1 \
  python -m riichi_ppo_v1.training.grp.train \
  --dataset datasets/tenhou_grp_2024_2025_v17 \
  --config riichi_ppo_v1/configs/v17_grp.yaml \
  --epochs 4 \
  2>&1 | tee logs/v17/grp_train.log
```

产出:`checkpoints/train_riichi_v17/grp/best.pt`(validation loss 最低)。

## 4. PPO 启动(双卡 CUDA_DEVICE=0,2)

```bash
env -C /mnt/disk1/hubowen/zenith RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,2 \
  PYTHONUNBUFFERED=1 python -m riichi_ppo_v1.training.train \
  --config riichi_ppo_v1/configs/v17_ppo.yaml --device cuda --learner-gpus 2 \
  2>&1 | tee logs/v17/ppo_run1.log
```

- `--init-model` 由 v17_ppo.yaml 内 `init_model` 提供(V16 SFT best.pt)。
- 每 5 updates 保存 checkpoint + 4000 半庄 1v3 vs V16 SFT;u100 结束。

## 5. 最佳 Checkpoint 选择

```bash
python -m riichi_ppo_v1.evaluation.select_best_checkpoint \
  --eval-dir audit/reports/v17/eval \
  --metric point_diff_vs_mean_opponent_mean \
  --output audit/reports/v17/eval/best_checkpoint.json
```

## 6. 冒烟(可选,限小规模快速验证)

```bash
python -m riichi_ppo_v1.training.train --smoke \
  --config riichi_ppo_v1/configs/v17_ppo.yaml \
  --iterations 2 --kyokus 8 --minibatch-size 128
```

## 7. 产物

- checkpoint:`checkpoints/train_riichi_v17/ppo/`
- 日志:`logs/v17/`
- 评测:`audit/reports/v17/eval/`
- TensorBoard:`checkpoints/train_riichi_v17/ppo/tensorboard`