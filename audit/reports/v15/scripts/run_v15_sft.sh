#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/v15"
LOG_FILE="${LOG_DIR}/v15_sft_$(date '+%Y%m%d_%H%M%S').log"

cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"

# 使用逻辑设备 0、1；训练入口会将其映射到 CUDA_VISIBLE_DEVICES。
export CUDA_DEVICE="0,1"
export PYTHONUNBUFFERED=1

if [[ "${CONDA_DEFAULT_ENV:-}" != "Mahjong-AI" ]]; then
    echo "警告：当前 Conda 环境为 '${CONDA_DEFAULT_ENV:-未激活}'，预期为 'Mahjong-AI'。"
    echo "如依赖不可用，请先执行：conda activate Mahjong-AI"
fi

exec > >(tee "${LOG_FILE}") 2>&1

echo "V15 SFT 日志：${LOG_FILE}"
echo "阶段 A：15,000 steps，仅训练 offense projection，CUDA_DEVICE=${CUDA_DEVICE}"

python -m riichi_ppo_v1.sft.train \
    --dataset datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16 \
    --config riichi_ppo_v1/configs/v15_sft_offense_warmup.yaml \
    --device cuda \
    --learner-gpus 2

echo "阶段 A 已完成，开始阶段 B：30,000 steps，解冻完整 Actor。"

python -m riichi_ppo_v1.sft.train \
    --dataset datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16 \
    --config riichi_ppo_v1/configs/v15_sft_actor_finetune.yaml \
    --device cuda \
    --learner-gpus 2

echo "V15 两阶段 SFT 已完成。"
echo "阶段 A checkpoints：checkpoints/train_riichi_v15/sft/stage_a"
echo "阶段 B checkpoints：checkpoints/train_riichi_v15/sft/stage_b"
