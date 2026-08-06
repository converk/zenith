#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_BIN="${CONDA_BIN:-/mnt/disk1/hubowen/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-Mahjong-AI}"

SFT_CUDA_DEVICE="${SFT_CUDA_DEVICE:-2}"
PPO_CUDA_DEVICE="${PPO_CUDA_DEVICE:-0,3}"

SFT_DATASET="${SFT_DATASET:-datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16}"
SFT_CONFIG="${SFT_CONFIG:-riichi_ppo_v1/configs/sft.yaml}"
SFT_OUTPUT="${SFT_OUTPUT:-checkpoints/train_riichi_v13_sft}"
SFT_LEARNER_GPUS="${SFT_LEARNER_GPUS:-1}"
SFT_CHECKPOINT_FOR_PPO="${SFT_CHECKPOINT_FOR_PPO:-${SFT_OUTPUT}/best.pt}"
RUN_SFT="${RUN_SFT:-auto}"

PPO_TRAINING_CONFIG="${PPO_TRAINING_CONFIG:-riichi_ppo_v1/configs/training.yaml}"
PPO_OVERLAY_CONFIG="${PPO_OVERLAY_CONFIG:-}"
PPO_OUTPUT="${PPO_OUTPUT:-checkpoints/train_riichi_ppo}"
PPO_LEARNER_GPUS="${PPO_LEARNER_GPUS:-2}"
PPO_ITERATIONS="${PPO_ITERATIONS:-4000}"
PPO_NUM_WORKERS="${PPO_NUM_WORKERS:-}"
PPO_ENVS_PER_WORKER="${PPO_ENVS_PER_WORKER:-}"
PPO_KYOKUS_PER_WORKER="${PPO_KYOKUS_PER_WORKER:-}"
PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS:-}"
PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-}"
PPO_TARGET_KL="${PPO_TARGET_KL:-}"

mkdir -p logs "${SFT_OUTPUT}" "${PPO_OUTPUT}"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S %Z"
}

checkpoint_is_loadable() {
    local checkpoint="$1"
    [[ -s "${checkpoint}" ]] || return 1
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python -c 'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu", weights_only=False); assert "model" in p and "model_config" in p' \
        "${checkpoint}" >/dev/null
}

validate_dataset() {
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python - "${SFT_DATASET}" <<'PY'
import sys
from pathlib import Path

from riichi_ppo_v1.sft.contract import load_manifest, validate_v13_manifest

dataset = Path(sys.argv[1])
validate_v13_manifest(load_manifest(dataset))
PY
}

run_sft() {
    local log_path="logs/train_sft_then_ppo_sft_$(date +%Y%m%d_%H%M%S).log"
    local sft_cmd=(
        "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}"
        python -m riichi_ppo_v1.sft.train
        --dataset "${SFT_DATASET}"
        --config "${SFT_CONFIG}"
        --learner-gpus "${SFT_LEARNER_GPUS}"
        --output "${SFT_OUTPUT}"
    )

    if [[ -n "${SFT_RESUME:-}" ]]; then
        sft_cmd+=(--resume "${SFT_RESUME}")
    fi
    if [[ -n "${SFT_INIT_MODEL:-}" ]]; then
        sft_cmd+=(--init-model "${SFT_INIT_MODEL}")
    fi
    if [[ -n "${SFT_MAX_TRAIN_STEPS:-}" ]]; then
        sft_cmd+=(--max-train-steps "${SFT_MAX_TRAIN_STEPS}")
    fi
    if [[ -n "${SFT_STOP_AFTER_STEPS:-}" ]]; then
        sft_cmd+=(--stop-after-steps "${SFT_STOP_AFTER_STEPS}")
    fi

    echo "$(timestamp) starting SFT: output=${SFT_OUTPUT}, cuda=${SFT_CUDA_DEVICE}, learner_gpus=${SFT_LEARNER_GPUS}"
    CUDA_DEVICE="${SFT_CUDA_DEVICE}" PYTHONUNBUFFERED=1 "${sft_cmd[@]}" 2>&1 | tee "${log_path}"
}

run_ppo() {
    local log_path="logs/train_sft_then_ppo_ppo_$(date +%Y%m%d_%H%M%S).log"
    local ppo_cmd=(
        "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}"
        python -m riichi_ppo_v1.training.train
        --device cuda
        --training-config "${PPO_TRAINING_CONFIG}"
        --learner-gpus "${PPO_LEARNER_GPUS}"
        --init-model "${SFT_CHECKPOINT_FOR_PPO}"
        --checkpoint-dir "${PPO_OUTPUT}"
    )

    if [[ -n "${PPO_OVERLAY_CONFIG}" ]]; then
        ppo_cmd+=(--config "${PPO_OVERLAY_CONFIG}")
    fi
    if [[ -n "${PPO_ITERATIONS}" ]]; then
        ppo_cmd+=(--iterations "${PPO_ITERATIONS}")
    fi
    if [[ -n "${PPO_NUM_WORKERS}" ]]; then
        ppo_cmd+=(--num-workers "${PPO_NUM_WORKERS}")
    fi
    if [[ -n "${PPO_ENVS_PER_WORKER}" ]]; then
        ppo_cmd+=(--envs-per-worker "${PPO_ENVS_PER_WORKER}")
    fi
    if [[ -n "${PPO_KYOKUS_PER_WORKER}" ]]; then
        ppo_cmd+=(--kyokus-per-worker "${PPO_KYOKUS_PER_WORKER}")
    fi
    if [[ -n "${PPO_UPDATE_EPOCHS}" ]]; then
        ppo_cmd+=(--update-epochs "${PPO_UPDATE_EPOCHS}")
    fi
    if [[ -n "${PPO_MINIBATCH_SIZE}" ]]; then
        ppo_cmd+=(--minibatch-size "${PPO_MINIBATCH_SIZE}")
    fi
    if [[ -n "${PPO_TARGET_KL}" ]]; then
        ppo_cmd+=(--target-kl "${PPO_TARGET_KL}")
    fi

    echo "$(timestamp) starting PPO: output=${PPO_OUTPUT}, init_model=${SFT_CHECKPOINT_FOR_PPO}, cuda=${PPO_CUDA_DEVICE}, learner_gpus=${PPO_LEARNER_GPUS}"
    CUDA_DEVICE="${PPO_CUDA_DEVICE}" PYTHONUNBUFFERED=1 "${ppo_cmd[@]}" 2>&1 | tee "${log_path}"
}

echo "$(timestamp) validating SFT dataset: ${SFT_DATASET}"
validate_dataset

case "${RUN_SFT}" in
    auto)
        if checkpoint_is_loadable "${SFT_CHECKPOINT_FOR_PPO}"; then
            echo "$(timestamp) found loadable SFT checkpoint; skipping SFT: ${SFT_CHECKPOINT_FOR_PPO}"
        else
            run_sft
        fi
        ;;
    1|true|yes)
        run_sft
        ;;
    0|false|no)
        echo "$(timestamp) RUN_SFT=${RUN_SFT}; skipping SFT"
        ;;
    *)
        echo "RUN_SFT must be auto, true, or false; got: ${RUN_SFT}" >&2
        exit 2
        ;;
esac

if ! checkpoint_is_loadable "${SFT_CHECKPOINT_FOR_PPO}"; then
    echo "$(timestamp) SFT checkpoint for PPO is missing or incomplete: ${SFT_CHECKPOINT_FOR_PPO}" >&2
    exit 1
fi

run_ppo
