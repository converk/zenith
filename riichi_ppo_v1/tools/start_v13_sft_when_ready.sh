#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_ROOT="/mnt/disk1/hubowen/zenith"
readonly DATASET="${SFT_DATASET:-$PROJECT_ROOT/datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16}"
readonly CONFIG="${SFT_CONFIG:-$PROJECT_ROOT/riichi_ppo_v1/configs/sft.yaml}"
readonly OUTPUT="${SFT_OUTPUT:-$PROJECT_ROOT/checkpoints/train_riichi_v13_sft}"
readonly CONDA="/mnt/disk1/hubowen/miniconda3/bin/conda"
readonly LOCK="$PROJECT_ROOT/logs/sft-v13-v16.lock"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date '+%F %T') another v13 SFT launcher or training process holds $LOCK"
    exit 1
fi

echo "$(date '+%F %T') waiting for completed cache: $DATASET"
while [[ ! -s "$DATASET/manifest.json" ]]; do
    sleep 60
done

"$CONDA" run -n Mahjong-AI python - "$DATASET/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "format": "riichi-sft-encoded-v3",
    "token_schema_version": 13,
    "rust_analysis_version": 4,
    "decision_analysis_version": 16,
}
for name, value in expected.items():
    if manifest.get(name) != value:
        raise SystemExit(
            f"incompatible completed cache: {name}={manifest.get(name)!r}, expected {value!r}"
        )
PY

if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit)" ]]; then
    echo "$(date '+%F %T') refusing to overwrite non-empty output: $OUTPUT"
    exit 1
fi

echo "$(date '+%F %T') cache complete; starting actor-only SFT on CUDA_DEVICE=0,3"
exec env CUDA_DEVICE=0,3 PYTHONUNBUFFERED=1 \
    "$CONDA" run --no-capture-output -n Mahjong-AI \
    python -m riichi_ppo_v1.sft.train \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --learner-gpus 2 \
    --output "$OUTPUT"
