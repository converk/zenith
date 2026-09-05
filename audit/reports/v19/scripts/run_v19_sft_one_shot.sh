#!/usr/bin/env bash
# V19 SFT 一体化启动：① 两年 60% 数据重编码 + 五头信念标签 → ② SFT 训练启动。
# 自描述命名、可重入（已编码则跳过）、支持 --force 与 --smoke（最小规模自检并清理）。
# 用法:  bash audit/reports/v19/scripts/run_v19_sft_one_shot.sh [--force] [--smoke]
# 依赖:  已激活 Mahjong-AI Conda 环境（或 PYTHON 指向该环境解释器）;
#        RiichiEnv 扩展已安装 V19（ENCODING_PROTOCOL_VERSION=19）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
FORCE=false
SMOKE=false
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=true ;;
    --smoke) SMOKE=true ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SOURCE="${V19_SFT_SOURCE:-datasets/tenhou_sft_2024_2025}"
ENCODED="${V19_SFT_ENCODED:-datasets/tenhou_sft_2024_2025_encoded_60pct_v19}"
SFT_CONFIG="${V19_SFT_CONFIG:-riichi_ppo_v1/configs/v19_sft.yaml}"
LOG_DIR="${V19_LOG_DIR:-logs/v19}"
mkdir -p "$LOG_DIR"

# V18 同口径抽样：subset 3/5（60%），game-level 100%。
SUBSET_DENOM=5
SUBSET_REMAINDERS="0,1,2"
GAME_SAMPLE_DENOM=1
GAME_SAMPLE_REMAINDER=0

encode_dataset() {
  local src="$1" out="$2" workers="$3" log="$4"
  mkdir -p "$(dirname "$log")"
  "$PYTHON" -m riichi_ppo_v1.sft.precompute \
    --source "$src" --output "$out" \
    --subset-denominator "$SUBSET_DENOM" \
    --subset-remainders "$SUBSET_REMAINDERS" \
    --game-sample-denominator "$GAME_SAMPLE_DENOM" \
    --game-sample-remainder "$GAME_SAMPLE_REMAINDER" \
    --workers "$workers" \
    --kyokus-per-shard 256 \
    --progress-every-kyokus 32 \
    --require-complete-action-coverage \
    2>&1 | tee "$log"
}

run_sft() {
  local config="$1" output="$2" log="$3"
  mkdir -p "$(dirname "$log")"
  "$PYTHON" -m riichi_ppo_v1.sft.train --config "$config" 2>&1 | tee "$log"
}

if $SMOKE; then
  SMOKE_ROOT="$(mktemp -d /tmp/v19_sft_smoke.XXXXXX)"
  trap 'rm -rf "$SMOKE_ROOT"' EXIT
  echo "[v19-sft] smoke: source mini shards -> $SMOKE_ROOT"
  mkdir -p "$SMOKE_ROOT/src/train" "$SMOKE_ROOT/src/validation"
  cp "$SOURCE/manifest.json" "$SMOKE_ROOT/src/manifest.json"
  cp "$(ls "$SOURCE"/train/*.tar | head -1)" "$SMOKE_ROOT/src/train/"
  cp "$(ls "$SOURCE"/validation/*.tar | head -1)" "$SMOKE_ROOT/src/validation/"
  SMOKE_OUT="$SMOKE_ROOT/encoded"
  SMOKE_CKPT="$SMOKE_ROOT/sft"
  encode_dataset "$SMOKE_ROOT/src" "$SMOKE_OUT" 2 "$SMOKE_ROOT/precompute.log"
  SMOKE_CFG="$SMOKE_ROOT/smoke_sft.yaml"
  "$PYTHON" - "$SFT_CONFIG" "$SMOKE_OUT" "$SMOKE_CKPT" "$SMOKE_CFG" <<'PY'
import sys
import yaml
base, dataset, ckpt, out = sys.argv[1:]
cfg = yaml.safe_load(open(base, encoding="utf-8"))
cfg.update({
    "dataset": dataset,
    "checkpoint_dir": ckpt,
    "device": "cpu",
    "learner_gpus": 1,
    "batch_size": 8,
    "epochs": 1,
    "max_train_steps": 2,
    "stop_after_steps": 2,
    "validation_samples_per_run": 16,
    "validation_max_samples": 16,
    "tensorboard_enabled": False,
    "torch_compile": False,
    "log_interval_steps": 1,
})
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  run_sft "$SMOKE_CFG" "$SMOKE_CKPT" "$SMOKE_ROOT/sft.log"
  echo "[v19-sft] smoke OK; artifacts cleaned by trap"
  exit 0
fi

# 正式路径：编码已存在则跳过（可重入），--force 强制重编码。
if $FORCE; then
  rm -rf "$ENCODED"
fi
if [[ -f "$ENCODED/manifest.json" ]]; then
  echo "[v19-sft] encoded dataset exists and will be reused: $ENCODED"
else
  echo "[v19-sft] step 1: re-encode 2-year/60% subset with V19 + belief labels"
  encode_dataset "$SOURCE" "$ENCODED" 16 "$LOG_DIR/sft_precompute.log"
  echo "[v19-sft] step 1 done -> $ENCODED/manifest.json"
fi

echo "[v19-sft] step 2: launch SFT training (config=$SFT_CONFIG)"
run_sft "$SFT_CONFIG" "checkpoints/train_riichi_v19/sft" "$LOG_DIR/sft_train.log"
