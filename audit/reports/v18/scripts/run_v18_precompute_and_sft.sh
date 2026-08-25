#!/usr/bin/env bash
# V18 编码数据生成 + Actor-only SFT 训练一键脚本。
#
# 产物地址(与 v16/v17 版本惯例一致,均可在 AGENTS.md 溯源):
#   - 编码数据:  datasets/tenhou_sft_2024_2025_encoded_60pct_v18(60% selection: 0,1,2/5)
#   - 模型保存:  checkpoint_dir 在 v18_sft.yaml 中固定为 checkpoints/train_riichi_v18/sft
#   - 运行日志:  logs/v18/(本脚本: v18_precompute_60pct.log、v18_sft_from_scratch.log、统计快照)
# 训练节奏(3000 steps 验证/保存、96 半庄终评)在 sft/contract.py 单点定义,勿在配置复制。
#
# 用法:
#   bash audit/reports/v18/scripts/run_v18_precompute_and_sft.sh             # 生成数据 + 训练
#   bash audit/reports/v18/scripts/run_v18_precompute_and_sft.sh --skip-precompute  # 已有数据仅训练
# 环境变量(可选): CONDA_ENV=Mahjong-AI  WORKERS=16
# 注意: 脚本不处理 resume;恢复训练请用自包含 resume 配置手工启动(参照 v17_ppo_resume.yaml 做法)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-Mahjong-AI}"
WORKERS="${WORKERS:-16}"
CUDA_DEVICE="${CUDA_DEVICE:-0,1}"
SOURCE_DIR="datasets/tenhou_sft_2024_2025"
DATASET_OUT="datasets/tenhou_sft_2024_2025_encoded_60pct_v18"
SFT_CONFIG="riichi_ppo_v1/configs/v18_sft.yaml"
LOG_DIR="logs/v18"
CKPT_DIR="checkpoints/train_riichi_v18/sft"

SKIP_PRECOMPUTE=0
for arg in "$@"; do
  case "$arg" in
    --skip-precompute) SKIP_PRECOMPUTE=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

export CUDA_DEVICE
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$CKPT_DIR"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "错误: 源数据目录不存在: $SOURCE_DIR" >&2
  exit 1
fi

# 目标目录非空即拒绝:precompute 不支持增量/续跑(输出非空会失败),
# 上次中断的残留必须手工清理后重跑,防止新旧 schema 数据混用。
if [[ "$SKIP_PRECOMPUTE" -eq 0 ]] && [[ -d "$DATASET_OUT" ]] && [[ -n "$(ls -A "$DATASET_OUT" 2>/dev/null || true)" ]]; then
  echo "错误: $DATASET_OUT 已存在且非空。请确认是否为未完成生成残留并手工处理后重跑。" >&2
  exit 1
fi

# ── 阶段 1: 编码数据生成(60% selection) ──────────────────────────────────
if [[ "$SKIP_PRECOMPUTE" -eq 1 ]]; then
  echo "[v18] 跳过数据生成(使用已有 $DATASET_OUT)"
else
  echo "[v18] 阶段 1: 生成 V18 编码数据 → $DATASET_OUT"
  conda run --no-capture-output -n "$CONDA_ENV" riichi-sft-precompute \
    --source "$SOURCE_DIR" \
    --output "$DATASET_OUT" \
    --subset-denominator 5 --subset-remainders 0,1,2 \
    --workers "$WORKERS" \
    2>&1 | tee "$LOG_DIR/v18_precompute_60pct.log"
fi

# ── 阶段 2: 数据契约校验 + 只读统计自检 ──────────────────────────────────
echo "[v18] 阶段 2: 校验 manifest 契约与 54 行 Snapshot 统计"
conda run -n "$CONDA_ENV" python -c "
from pathlib import Path
from riichi_ppo_v1.sft.contract import load_manifest, validate_manifest
from riichi_ppo_v1.tools.v18_token_statistics import calculate
ds = Path('$DATASET_OUT')
validate_manifest(load_manifest(ds))
stats = calculate(ds, 'validation')
assert stats['snapshot_mean'] == 54.0, stats
print('manifest OK; validation decisions=%d snapshot_mean=%.1f total_mean=%.6f' % (
    stats['decisions'], stats['snapshot_mean'], stats['total_mean']))
" 2>&1 | tee "$LOG_DIR/v18_precompute_verify.log"

# ── 阶段 3: Actor-only SFT 训练(前台,日志落盘) ──────────────────────────
echo "[v18] 阶段 3: 启动 V18 Actor-only SFT 训练 → $CKPT_DIR"
conda run --no-capture-output -n "$CONDA_ENV" riichi-sft-train \
  --config "$SFT_CONFIG" \
  2>&1 | tee "$LOG_DIR/v18_sft_from_scratch.log"

echo "[v18] 全部完成。"
echo "      模型保存: $CKPT_DIR"
echo "      日志保存: $LOG_DIR/ (precompute / verify / sft)"
