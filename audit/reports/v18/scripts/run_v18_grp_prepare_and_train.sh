#!/usr/bin/env bash
# V18 GRP 数据准备 + 离线训练一键脚本。
#
# 产物地址(均可在 AGENTS.md 溯源):
#   - GRP 数据集:  datasets/tenhou_grp_2024_2025_v18(全量数据 denominator=1,
#                   约 36 万 train 半庄;与 SFT 60% 子集完全重叠——GRP 是冻结
#                   奖励模型,仅用于评分,无策略污染)
#   - 模型保存:    checkpoint_dir 在 v18_grp.yaml 中固定为 checkpoints/train_riichi_v18/grp
#   - 运行日志:    logs/v18/(本脚本: grp_prepare.log、grp_train.log)
# 输入契约与训练节奏(21 维、96×2 层 GRU、批次 2048、每 200 步验证、
# validation-loss best 冻结)见 model/grp.py 与 configs/v18_grp.yaml,勿在脚本内复制。
#
# 用法:
#   bash audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh             # 准备数据 + 训练
#   bash audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh --skip-prepare  # 已有数据仅训练
#   bash audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh --prepare-only  # 仅准备数据
# 环境变量(可选): CONDA_ENV=Mahjong-AI  CUDA_DEVICE=0,1
# 注意: 训练不处理 resume;重训请先归档/迁移既有 checkpoint 再运行。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-Mahjong-AI}"
CUDA_DEVICE="${CUDA_DEVICE:-0,1}"
SOURCE_DIR="datasets/tenhou_sft_2024_2025"
DATASET_OUT="datasets/tenhou_grp_2024_2025_v18"
GRP_CONFIG="riichi_ppo_v1/configs/v18_grp.yaml"
LOG_DIR="logs/v18"

SKIP_PREPARE=0
PREPARE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-prepare) SKIP_PREPARE=1 ;;
    --prepare-only) PREPARE_ONLY=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

export CUDA_DEVICE
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "错误: 源数据目录不存在: $SOURCE_DIR" >&2
  exit 1
fi

# 目标目录非空即拒绝:prepare 不支持增量/续跑,残留必须手工清理后重跑。
if [[ "$SKIP_PREPARE" -eq 0 ]] && [[ -d "$DATASET_OUT" ]] && [[ -n "$(ls -A "$DATASET_OUT" 2>/dev/null || true)" ]]; then
  echo "错误: $DATASET_OUT 已存在且非空。请确认是否为未完成生成残留并手工处理后重跑。" >&2
  exit 1
fi

# ── 阶段 1: GRP 数据集构造(全量数据,6 进程并行解析) ────────────────────
if [[ "$SKIP_PREPARE" -eq 1 ]]; then
  echo "[v18-grp] 跳过数据准备(使用已有 $DATASET_OUT)"
else
  echo "[v18-grp] 阶段 1: 构造 GRP 数据集(全量)→ $DATASET_OUT"
  conda run --no-capture-output -n "$CONDA_ENV" python -m \
    riichi_ppo_v1.training.grp.prepare \
    --source "$SOURCE_DIR" \
    --output "$DATASET_OUT" \
    --subset-denominator 1 \
    --subset-remainders 0 \
    2>&1 | tee "$LOG_DIR/grp_prepare.log"
fi

# ── 阶段 2: GRP 离线训练(前台,日志落盘) ──────────────────────────────────
if [[ "$PREPARE_ONLY" -eq 1 ]]; then
  echo "[v18-grp] --prepare-only:仅完成数据准备,跳过训练。"
  exit 0
fi
echo "[v18-grp] 阶段 2: 启动 GRP 离线训练 → checkpoints/train_riichi_v18/grp"
conda run --no-capture-output -n "$CONDA_ENV" python -m \
  riichi_ppo_v1.training.grp.train \
  --dataset "$DATASET_OUT" \
  --config "$GRP_CONFIG" \
  --device cuda \
  2>&1 | tee "$LOG_DIR/grp_train.log"

echo "[v18-grp] 全部完成。"
echo "      数据集: $DATASET_OUT"
echo "      模型保存: checkpoints/train_riichi_v18/grp/best.pt"
echo "      日志保存: $LOG_DIR/ (grp_prepare.log / grp_train.log)"
