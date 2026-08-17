#!/usr/bin/env bash
# V16-small 全流程:先双卡 SFT,再用 SFT checkpoint 双卡启动 PPO。
# SFT/PPO 训练使用 CUDA_DEVICE=0,3(物理 GPU 0 与 4)、learner_gpus=2;
# PPO 的 1v3 评测分片也按 5/5 分摊到这两张卡。
#
# 前置:
#   - Conda 环境 Mahjong-AI 已激活(PYTHON_BIN 可覆盖 python 路径);
#   - datasets/tenhou_sft_2024_2025_encoded_60pct_v16 已就绪;
#   - datasets/tenhou_grp_2024_2025_v16 已就绪;
#   - checkpoints/train_riichi_v16/grp/best.pt 已就绪。
#
# 用法:
#   bash audit/reports/v16/scripts/train_v16_sft_then_ppo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "riichi_ppo_v1/configs/v16_ppo.yaml" ]]; then
    echo "错误:无法定位仓库根目录,请从仓库内运行本脚本" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

cleanup_ray() {
    if command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
    else
        "$PYTHON_BIN" -m ray stop --force >/dev/null 2>&1 || true
    fi
}

SFT_CONFIG="riichi_ppo_v1/configs/v16_sft.yaml"
PPO_CONFIG="riichi_ppo_v1/configs/v16_ppo.yaml"
SFT_DIR="checkpoints/train_riichi_v16/sft"
PPO_DIR="checkpoints/train_riichi_v16/ppo"
SFT_CKPT="checkpoints/train_riichi_v16/sft/best.pt"
GRP_CKPT="checkpoints/train_riichi_v16/grp/best.pt"
GRP_DATASET="datasets/tenhou_grp_2024_2025_v16"
SFT_DATASET="datasets/tenhou_sft_2024_2025_encoded_60pct_v16"
PPO_LOG="logs/v16/v16_ppo_from_scratch.log"
SFT_LOG="logs/v16/v16_sft_from_scratch.log"

echo "==> 前置检查"
for path in "$SFT_DATASET" "$GRP_CKPT" "$GRP_DATASET"; do
    if [[ ! -e "$path" ]]; then
        echo "错误:缺少前置产物 $path" >&2
        exit 1
    fi
done

mkdir -p logs/v16 "$SFT_DIR" "$PPO_DIR"

echo "==> 清理 Ray 残留会话"
cleanup_ray

echo "==> [1/2] V16-small SFT 训练(双卡 CUDA_DEVICE=0,3、learner_gpus=2)"
env RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,3 PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m riichi_ppo_v1.sft.train \
    --config "$SFT_CONFIG" --device cuda --learner-gpus 2 \
    2>&1 | tee "$SFT_LOG"

if [[ ! -f "$SFT_CKPT" ]]; then
    echo "错误:SFT 训练结束但缺少 checkpoint $SFT_CKPT" >&2
    exit 1
fi

echo "==> [2/2] V16-small PPO 训练 + 双卡 1v3 评测(CUDA_DEVICE=0,3、learner_gpus=2、1200 updates)"
env RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,3 PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m riichi_ppo_v1.training.train \
    --config "$PPO_CONFIG" --device cuda --learner-gpus 2 \
    2>&1 | tee "$PPO_LOG"

echo "==> 训练结束,清理 Ray"
cleanup_ray
echo "完成。SFT 日志:${SFT_LOG};PPO 日志:${PPO_LOG}"
