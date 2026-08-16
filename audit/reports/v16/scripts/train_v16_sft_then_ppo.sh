#!/usr/bin/env bash
# V16 从头训练:先以修复后的 forward_v16 重训 SFT,完成后记录 Recall@3 并
# 自动衔接单卡 V16 PPO(1200 updates,不设 Recall@3 硬门槛)。
#
# 前置:
#   - Conda 环境 Mahjong-AI 已激活(PYTHON_BIN 可覆盖 python 路径);
#   - datasets/tenhou_sft_2024_2025_encoded_40pct_v16 与
#     datasets/tenhou_grp_2024_2025_v16 均已就绪;
#   - checkpoints/train_riichi_v16/grp/best.pt(冻结 GRP)已就绪。
#
# 用法:
#   bash audit/reports/v16/scripts/train_v16_sft_then_ppo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SFT_CONFIG="riichi_ppo_v1/configs/v16_sft.yaml"
PPO_CONFIG="riichi_ppo_v1/configs/v16_ppo.yaml"
SFT_DIR="checkpoints/train_riichi_v16/sft"
PPO_DIR="checkpoints/train_riichi_v16/ppo"
GRP_CKPT="checkpoints/train_riichi_v16/grp/best.pt"
GRP_DATASET="datasets/tenhou_grp_2024_2025_v16"
SFT_LOG="logs/v16/v16_sft_from_scratch.log"
PPO_LOG="logs/v16/v16_ppo_from_scratch.log"

mkdir -p logs/v16 "$SFT_DIR" "$PPO_DIR"

echo "==> 前置检查"
for path in "$GRP_CKPT" "$GRP_DATASET"; do
    if [[ ! -e "$path" ]]; then
        echo "错误:缺少前置产物 $path" >&2
        exit 1
    fi
done

echo "==> [1/2] V16 SFT 从头训练(修复后的 forward_v16;单卡 GPU 0、learner_gpus=1)"
env RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0 PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m riichi_ppo_v1.sft.train --config "$SFT_CONFIG" \
    2>&1 | tee "$SFT_LOG"

echo "==> 记录 SFT 验证集 Recall@3(仅记录,不做门槛拦截)"
"$PYTHON_BIN" - "$SFT_DIR/metrics.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
recall = metrics.get("validation/top3")
if recall is None:
    print("警告:SFT 产物 metrics.json 缺少 validation/top3,跳过记录")
else:
    print(f"SFT 验证集 Recall@3 = {float(recall):.4f}")
PY

echo "==> 清理 Ray 残留会话"
"$PYTHON_BIN" -m ray stop --force >/dev/null 2>&1 || true

echo "==> [2/2] V16 PPO 训练(单卡 GPU 0、learner_gpus=1、1200 updates)"
env RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0 PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m riichi_ppo_v1.training.train \
    --config "$PPO_CONFIG" --device cuda --learner-gpus 1 \
    2>&1 | tee "$PPO_LOG"

echo "==> PPO 结束,清理 Ray"
"$PYTHON_BIN" -m ray stop --force >/dev/null 2>&1 || true
echo "完成。SFT 日志:${SFT_LOG}  PPO 日志:${PPO_LOG}"
