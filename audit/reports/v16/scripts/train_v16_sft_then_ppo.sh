#!/usr/bin/env bash
# V16 PPO-only 训练:跳过 SFT 重训,直接使用现有 SFT checkpoint 初始化 PPO
# (1200 updates,单卡 GPU 0、learner_gpus=1)。
#
# 前置:
#   - Conda 环境 Mahjong-AI 已激活(PYTHON_BIN 可覆盖 python 路径);
#   - datasets/tenhou_grp_2024_2025_v16 已就绪;
#   - checkpoints/train_riichi_v16/sft/best.pt 与
#     checkpoints/train_riichi_v16/grp/best.pt 均已就绪。
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
PPO_CONFIG="riichi_ppo_v1/configs/v16_ppo.yaml"
PPO_DIR="checkpoints/train_riichi_v16/ppo"
SFT_CKPT="checkpoints/train_riichi_v16/sft/best.pt"
GRP_CKPT="checkpoints/train_riichi_v16/grp/best.pt"
GRP_DATASET="datasets/tenhou_grp_2024_2025_v16"
PPO_LOG="logs/v16/v16_ppo_from_scratch.log"

echo "==> 前置检查"
for path in "$SFT_CKPT" "$GRP_CKPT" "$GRP_DATASET"; do
    if [[ ! -e "$path" ]]; then
        echo "错误:缺少前置产物 $path" >&2
        exit 1
    fi
done

mkdir -p logs/v16 "$PPO_DIR"

echo "==> 清理 Ray 残留会话"
"$PYTHON_BIN" -m ray stop --force >/dev/null 2>&1 || true

echo "==> [1/1] V16 PPO 训练(跳过 SFT,单卡 GPU 0、learner_gpus=1、1200 updates)"
env RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0 PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m riichi_ppo_v1.training.train \
    --config "$PPO_CONFIG" --device cuda --learner-gpus 1 \
    2>&1 | tee "$PPO_LOG"

echo "==> PPO 结束,清理 Ray"
"$PYTHON_BIN" -m ray stop --force >/dev/null 2>&1 || true
echo "完成。PPO 日志:${PPO_LOG}"
