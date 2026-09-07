#!/usr/bin/env bash
# V19 模糊信念 SFT 一体化脚本（2026-09-07）：① 只重标信念标签（不重编码）
# → ② 用新配置跑 1 epoch / batch=6000 / 每 10 step 打点。
#
# 用法:
#   bash audit/reports/v19/scripts/run_v19_sft_fuzzy.sh            # 直接执行
#   bash audit/reports/v19/scripts/run_v19_sft_fuzzy.sh --dry-run  # 只打印命令
#   bash audit/reports/v19/scripts/run_v19_sft_fuzzy.sh --force    # 强制重标数据集
#
# 说明: 脚本直接调用 Mahjong-AI 环境解释器，**不使用 conda run**（conda run
# 会吞/阻塞终端输出）。可用 PYTHON 环境变量覆盖解释器路径。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python}"
OLD_DATA="${OLD_DATA:-datasets/tenhou_sft_2024_2025_encoded_60pct_v19}"
NEW_DATA="${NEW_DATA:-datasets/tenhou_sft_2024_2025_encoded_60pct_v19_fuzzy}"
CONFIG="${V19_SFT_CONFIG:-riichi_ppo_v1/configs/v19_sft.yaml}"
LOG_DIR="${V19_LOG_DIR:-logs/v19}"
SFT_LOG="$LOG_DIR/sft_train_v19_fuzzy.log"

DRY_RUN=false
FORCE=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --force) FORCE=true ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

run() {
  if $DRY_RUN; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

mkdir -p "$LOG_DIR"

echo "[v19-fuzzy] 1/2 relabel belief labels (no re-encoding): $OLD_DATA -> $NEW_DATA"
if $FORCE; then
  run rm -rf "$NEW_DATA"
fi
if [[ -f "$NEW_DATA/manifest.json" && ! $FORCE ]]; then
  echo "[v19-fuzzy] fuzzy dataset exists, reuse: $NEW_DATA"
else
  run "$PYTHON" -m riichi_ppo_v1.sft.relabel \
    --source "$OLD_DATA" --output "$NEW_DATA" \
    --workers 16 --progress-every 500
fi

echo "[v19-fuzzy] 2/2 SFT train: config=$CONFIG, log=$SFT_LOG"
if $DRY_RUN; then
  echo "[dry-run] $PYTHON -m riichi_ppo_v1.sft.train --config $CONFIG 2>&1 | tee $SFT_LOG"
else
  "$PYTHON" -m riichi_ppo_v1.sft.train --config "$CONFIG" 2>&1 | tee "$SFT_LOG"
fi
