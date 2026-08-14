#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/train_riichi_v15/ppo"
FINAL_SFT="${PROJECT_ROOT}/checkpoints/train_riichi_v15/sft/final.pt"
LOG_DIR="${PROJECT_ROOT}/logs/v15"
LOG_FILE="${LOG_DIR}/v15_ppo_train.log"

cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"

export CUDA_DEVICE="0,1"
export PYTHONUNBUFFERED=1

if [[ "${CONDA_DEFAULT_ENV:-}" != "Mahjong-AI" ]]; then
    echo "当前 Conda 环境为 '${CONDA_DEFAULT_ENV:-未激活}'，请先执行：conda activate Mahjong-AI" >&2
    exit 1
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "$(date --iso-8601=seconds) V15 PPO start"
echo "CUDA_DEVICE=${CUDA_DEVICE}; learner_gpus=2; updates=1200; kyokus_per_worker=16"
echo "SFT initialization=${FINAL_SFT}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "1v3 devices=0,1; processes=10; hanchans_per_process=160"

if [[ ! -f "${FINAL_SFT}" ]]; then
    echo "missing final SFT checkpoint: ${FINAL_SFT}" >&2
    exit 1
fi
if [[ -e "${CHECKPOINT_DIR}" ]] && find "${CHECKPOINT_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite non-empty PPO checkpoint directory: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

python -m riichi_ppo_v1.training.train \
    --config riichi_ppo_v1/configs/v15_ppo.yaml \
    --device cuda \
    --learner-gpus 2

echo "$(date --iso-8601=seconds) V15 PPO completed"
