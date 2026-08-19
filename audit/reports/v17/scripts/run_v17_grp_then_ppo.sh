#!/usr/bin/env bash
# V17 全流程一键脚本:GRP 数据集构造 → GRP 单卡训练(冻结) → PPO 双卡训练。
#
# GRP 说明:
#   - 单卡训练(模型 ~58K 参数,无需 DDP),CUDA_DEVICE=0;
#   - 数据:40% 子集(denominator=5, remainders=0,1),实测 1 个 train shard
#     约产出 1477 个 prefix 样本;默认 --max-shards 140 保证 ≥ 20 万样本,
#     可经环境变量 MAX_SHARDS 覆盖;
#   - 训练 4 个 epoch(GRP_CONFIG 内 epochs: 4),保存 validation loss 最低
#     的 checkpoint 到 checkpoints/train_riichi_v17/grp/best.pt,训练后冻结。
#
# PPO 说明:
#   - 双卡 DDP(CUDA_DEVICE=0,2 → 物理 GPU 0,3),learner_gpus=2;
#   - 从 checkpoints/train_riichi_v16/sft/best.pt 初始化,GRP reward 只读;
#   - 每 5 updates 保存 checkpoint 并 1v3 vs V16 SFT 评测 4000 半庄,
#     评测分片按 5/5 分摊到 CUDA_DEVICE=0,2;
#   - 结束后按 1V3 表现选择最佳 checkpoint。
#
# 前置:
#   - Conda 环境 Mahjong-AI 已激活(PYTHON_BIN 可覆盖 python 路径);
#   - datasets/tenhou_sft_2024_2025 原始数据存在;
#   - checkpoints/train_riichi_v16/sft/best.pt 存在(PPO 初始化 + 1v3 对手)。
#
# 用法:
#   bash audit/reports/v17/scripts/run_v17_grp_then_ppo.sh
#
# 可覆盖的环境变量:
#   PYTHON_BIN        python 解释器(默认 conda 环境内 python)
#   MAX_SHARDS        每个 split 最多处理的 tar shard 数(默认 280 → ~40w 样本)
#   GRP_EPOCHS        GRP 训练 epoch 数(默认 15,覆盖 v17_grp.yaml 的 epochs)
#   PPO_ITERATIONS    PPO 训练 update 数(默认 100,覆盖 v17_ppo.yaml)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_SHARDS="${MAX_SHARDS:-280}"
GRP_EPOCHS="${GRP_EPOCHS:-15}"
PPO_ITERATIONS="${PPO_ITERATIONS:-100}"

GRP_CONFIG="riichi_ppo_v1/configs/v17_grp.yaml"
PPO_CONFIG="riichi_ppo_v1/configs/v17_ppo.yaml"
GRP_SOURCE="datasets/tenhou_sft_2024_2025"
GRP_DATASET="datasets/tenhou_grp_2024_2025_v17"
GRP_DIR="checkpoints/train_riichi_v17/grp"
GRP_CKPT="${GRP_DIR}/best.pt"
PPO_DIR="checkpoints/train_riichi_v17/ppo"
SFT_CKPT="checkpoints/train_riichi_v16/sft/best.pt"

GRP_LOG="logs/v17/grp_train.log"
PPO_LOG="logs/v17/ppo_train.log"
mkdir -p logs/v17 "$GRP_DIR" "$PPO_DIR"

echo "==> 前置检查"
if [[ ! -e "$GRP_SOURCE" ]]; then
    echo "错误:缺少原始数据目录 $GRP_SOURCE" >&2
    exit 1
fi
if [[ ! -e "${GRP_SOURCE}/train/train-00000.tar" ]]; then
    echo "错误:原始训练数据不可用 ${GRP_SOURCE}/train/" >&2
    exit 1
fi
if [[ ! -e "$SFT_CKPT" ]]; then
    echo "错误:缺少 V16 SFT checkpoint $SFT_CKPT(PPO 初始化 + 1v3 对手)" >&2
    exit 1
fi

cleanup_ray() {
    if command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
    else
        "$PYTHON_BIN" -m ray stop --force >/dev/null 2>&1 || true
    fi
}

echo "==> [1/3] GRP 数据集构造(40% 子集,max_shards=${MAX_SHARDS})"
if [[ -e "$GRP_DATASET" ]] && [[ -n "$(find "$GRP_DATASET" -mindepth 1 -print -quit)" ]]; then
    echo "    数据集已存在,跳过构造: $GRP_DATASET"
    echo "    (如需重建,请先删除该目录)"
else
    CUDA_DEVICE=0 PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m riichi_ppo_v1.training.grp.prepare \
        --source "$GRP_SOURCE" \
        --output "$GRP_DATASET" \
        --subset-denominator 5 --subset-remainders 0,1 \
        --kyokus-per-shard 512 \
        --max-shards "$MAX_SHARDS"
    echo "    GRP 数据集就绪: $GRP_DATASET"
fi

echo "==> [2/3] GRP 单卡训练(epochs=${GRP_EPOCHS}, CUDA_DEVICE=0)"
if [[ -e "$GRP_CKPT" ]]; then
    echo "    GRP checkpoint 已存在,跳过训练: $GRP_CKPT"
    echo "    (如需重新训练,请先删除该 checkpoint)"
else
    CUDA_DEVICE=0 PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m riichi_ppo_v1.training.grp.train \
        --dataset "$GRP_DATASET" \
        --config "$GRP_CONFIG" \
        --epochs "$GRP_EPOCHS" \
        2>&1 | tee "$GRP_LOG"
    echo "    GRP 训练完成,冻结产物: $GRP_CKPT"
fi

echo "==> [3/3] PPO 双卡训练(iterations=${PPO_ITERATIONS}, CUDA_DEVICE=0,2, learner_gpus=2)"
cleanup_ray
env RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,2 PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m riichi_ppo_v1.training.train \
    --config "$PPO_CONFIG" \
    --device cuda \
    --learner-gpus 2 \
    --iterations "$PPO_ITERATIONS" \
    2>&1 | tee "$PPO_LOG"

echo "==> 按 1V3 vs SFT 表现选择最佳 checkpoint"
"$PYTHON_BIN" -m riichi_ppo_v1.evaluation.select_best_checkpoint \
    --eval-dir audit/reports/v17/eval \
    --output audit/reports/v17/eval/best_checkpoint.json

cleanup_ray
echo "==> 全部完成"
echo "    GRP checkpoint:  $GRP_CKPT"
echo "    PPO checkpoint:   ${PPO_DIR}/checkpoint_*.pt"
echo "    Best(1V3 最优):   audited 于 audit/reports/v17/eval/best_checkpoint.json"
echo "    日志:             ${GRP_LOG} / ${PPO_LOG}"