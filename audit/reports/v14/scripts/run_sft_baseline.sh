#!/usr/bin/env bash
# SFT self-play baseline for the fixed 1v3 protocol:
# model_a = model_b = best_heuristic.pt, 10x160 hanchans, same seeds/seat
# rotation as every V14 checkpoint eval.  Runs only after the V14 training
# process exits, so it never contends with training for the two GPUs.
set -euo pipefail

cd /mnt/disk1/hubowen/zenith
mkdir -p logs/v14
log="logs/v14/sft_baseline.log"
: > "$log"

while pgrep -f "riichi_ppo_v1.training.train" >/dev/null 2>&1; do
  sleep 60
done
sleep 20

if [ -f "audit/reports/v14/eval/vs_sft_u000.json" ]; then
  echo "SFT baseline summary already exists; skip." >> "$log"
  exit 0
fi

env CUDA_DEVICE=0 conda run --no-capture-output -n Mahjong-AI python - <<'PY' >> "$log" 2>&1
from riichi_ppo_v1.evaluation.head_to_head_1v3_shards import run_sharded_1v3

summary = run_sharded_1v3(
    "checkpoints/train_riichi_v13/sft/best_heuristic.pt",
    "checkpoints/train_riichi_v13/sft/best_heuristic.pt",
    update=0,
    processes=10,
    hanchans_per_process=160,
    parallel_hanchans=160,
    devices=("0", "2"),
    seed_base=20260812,
    output_dir="audit/reports/v14/eval",
)
model_a = summary["model_a"]
print(
    "SFT_BASELINE first",
    round(model_a["first_place_rate"], 4),
    "top2",
    round(model_a["top2_rate"], 4),
    "rank",
    round(model_a["mean_rank"], 3),
    "pd",
    round(model_a["point_diff_mean"], 1),
    "ci",
    model_a["point_diff_bootstrap_ci95"],
    flush=True,
)
PY
