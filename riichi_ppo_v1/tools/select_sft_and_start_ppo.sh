#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_BIN="${CONDA_BIN:-/mnt/disk1/hubowen/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-Mahjong-AI}"
MODEL_A="${MODEL_A:-checkpoints/train_riichi_v11_sft_40pct/best_heuristic.pt}"
MODEL_B="${MODEL_B:-checkpoints/train_riichi_v11_sft_40pct/best.pt}"
SELECTION_DIR="${SELECTION_DIR:-checkpoints/train_riichi_v11_sft_40pct_2v2_selection}"
PPO_CHECKPOINT_DIR="${PPO_CHECKPOINT_DIR:-checkpoints/train_riichi_v11_ppo_selected}"
HANCHANS="${HANCHANS:-320}"
PARALLEL_HANCHANS="${PARALLEL_HANCHANS:-24}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0,3}"
PPO_CUDA_DEVICE="${PPO_CUDA_DEVICE:-0,3}"
PPO_LEARNER_GPUS="${PPO_LEARNER_GPUS:-2}"
PPO_KYOKUS_PER_WORKER="${PPO_KYOKUS_PER_WORKER:-1}"

mkdir -p "${SELECTION_DIR}" "${PPO_CHECKPOINT_DIR}"
RESULT_JSON="${SELECTION_DIR}/result.json"
SNAPSHOT_A="${SELECTION_DIR}/best_heuristic.snapshot.pt"
SNAPSHOT_B="${SELECTION_DIR}/best.snapshot.pt"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S %Z"
}

checkpoint_is_loadable() {
    local checkpoint="$1"
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python -c 'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu", weights_only=False); assert "model" in p and "model_config" in p' \
        "${checkpoint}" >/dev/null
}

snapshot_checkpoint() {
    local source="$1"
    local destination="$2"
    local temporary="${destination}.tmp"
    while true; do
        if [[ -s "${source}" ]] && checkpoint_is_loadable "${source}"; then
            cp --reflink=auto "${source}" "${temporary}"
            if checkpoint_is_loadable "${temporary}"; then
                mv -f "${temporary}" "${destination}"
                return
            fi
        fi
        rm -f "${temporary}"
        echo "$(timestamp) waiting for a complete checkpoint: ${source}"
        sleep 30
    done
}

echo "$(timestamp) scheduled selector started"
snapshot_checkpoint "${MODEL_A}" "${SNAPSHOT_A}"
snapshot_checkpoint "${MODEL_B}" "${SNAPSHOT_B}"
echo "$(timestamp) checkpoint snapshots are ready"

CUDA_DEVICE="${EVAL_CUDA_DEVICE}" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python -m riichi_ppo_v1.sft.head_to_head \
    --model-a "${SNAPSHOT_A}" \
    --model-b "${SNAPSHOT_B}" \
    --hanchans "${HANCHANS}" \
    --parallel-hanchans "${PARALLEL_HANCHANS}" \
    --device cuda \
    --model-a-device cuda:0 \
    --model-b-device cuda:1 \
    --output "${RESULT_JSON}"

SELECTED_CHECKPOINT="$(
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"])' \
        "${RESULT_JSON}"
)"
printf '%s\n' "${SELECTED_CHECKPOINT}" >"${SELECTION_DIR}/selected_model.txt"
echo "$(timestamp) selected PPO base model: ${SELECTED_CHECKPOINT}"

echo "$(timestamp) starting PPO on CUDA_DEVICE=${PPO_CUDA_DEVICE}"
exec env CUDA_DEVICE="${PPO_CUDA_DEVICE}" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    riichi-ppo-train \
    --device cuda \
    --learner-gpus "${PPO_LEARNER_GPUS}" \
    --kyokus-per-worker "${PPO_KYOKUS_PER_WORKER}" \
    --init-model "${SELECTED_CHECKPOINT}" \
    --checkpoint-dir "${PPO_CHECKPOINT_DIR}"
