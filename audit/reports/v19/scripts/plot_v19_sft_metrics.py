"""V19 SFT 2 epochs 指标可视化与分析制品生成脚本（只读 TensorBoard，不修改训练产物）。"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ROOT))))
EVENT = os.path.join(
    REPO,
    "checkpoints/train_riichi_v19/sft/tensorboard/events.out.tfevents.1788707051.RAIDIX.2432310.0",
)
METRICS_JSON = os.path.join(REPO, "checkpoints/train_riichi_v19/sft/metrics.json")
EVAL_DIR = os.path.join(REPO, "audit/reports/v19/eval")
os.makedirs(EVAL_DIR, exist_ok=True)

ea = EventAccumulator(EVENT, size_guidance={"scalars": 0})
ea.Reload()
S = ea.Scalars


def series(tag: str):
    return [(x.step, x.value) for x in S(tag)]


TAGS = {
    "val_ce": "SFT/验证/policy_ce",
    "val_top1": "SFT/验证/top1",
    "val_top3": "SFT/验证/top3",
    "val_hand_acc": "SFT/验证/信念/hand_acc",
    "val_shanten_top1": "SFT/验证/信念/shanten_top1",
    "val_wait_tenpai_acc": "SFT/验证/信念/wait_tenpai_acc",
    "val_wait_topk": "SFT/验证/信念/wait_topk",
    "val_wait_prec2": "SFT/验证/信念/wait_precision_at_2",
    "val_wait_cond_auc": "SFT/验证/信念/wait_conditional_auc",
    "val_danger_auc": "SFT/验证/信念/danger_auc",
    "val_danger_recall": "SFT/验证/信念/danger_recall_at_topk",
    "val_loss_mae": "SFT/验证/信念/loss_mae",
    "val_loss_cond_mae": "SFT/验证/信念/loss_conditional_mae",
    "val_hand_loss": "SFT/验证/信念/hand_loss",
    "val_shanten_loss": "SFT/验证/信念/shanten_loss",
    "val_wait_loss": "SFT/验证/信念/wait_loss",
    "val_danger_loss": "SFT/验证/信念/danger_loss",
    "val_loss_loss": "SFT/验证/信念/loss_loss",
    "val_loss_total": "SFT/验证/信念/loss_total",
    "tr_loss": "SFT/训练/总损失 (loss)",
    "tr_ce": "SFT/训练/策略交叉熵 (policy_ce)",
    "tr_top1": "SFT/训练/Top-1 准确率 (top1)",
    "tr_top3": "SFT/训练/Top-3 准确率 (top3)",
    "tr_hand_acc": "SFT/训练/信念手牌逐格精度 (belief_hand_acc)",
    "tr_shanten_top1": "SFT/训练/信念向听 Top-1 (belief_shanten_top1)",
    "tr_wait_tenpai_acc": "SFT/训练/信念听牌二判精度 (belief_wait_tenpai_acc)",
    "tr_danger_recall": "SFT/训练/信念真值可荣牌 top-k 召回 (belief_danger_recall_at_topk)",
    "tr_loss_mae": "SFT/训练/信念打点 MAE (belief_loss_mae)",
    "tr_belief_total": "SFT/训练/信念加权损失 (belief_loss_weighted)",
    "tr_step_time": "SFT/性能/平均 Step 耗时·秒 (step_time_s)",
}
series_map = {k: series(v) for k, v in TAGS.items()}
steps = [x[0] for x in series_map["val_ce"]]

with open(os.path.join(EVAL_DIR, "v19_sft_2ep_validation_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step"] + list(TAGS.keys()))
    for s in steps:
        w.writerow([s] + [dict(series_map[k])[s] for k in TAGS])

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
x = np.array(steps)
ax = axes[0, 0]
ax.plot(x, [dict(series_map["tr_loss"])[s] for s in steps], label="train total loss", color="#1f77b4")
ax.plot(x, [dict(series_map["tr_ce"])[s] for s in steps], label="train policy CE", color="#ff7f0e")
ax.plot(x, [dict(series_map["val_ce"])[s] for s in steps], label="val policy CE", color="#2ca02c", linestyle="--")
ax.set_xlabel("step"); ax.set_ylabel("loss / CE"); ax.set_title("Loss & policy CE"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(x, [dict(series_map["val_top1"])[s] for s in steps], label="val top1", color="#1f77b4")
ax.plot(x, [dict(series_map["val_top3"])[s] for s in steps], label="val top3", color="#d62728")
ax.plot(x, [dict(series_map["tr_top1"])[s] for s in steps], label="train top1", color="#1f77b4", linestyle=":")
ax.axhline(0.82525, color="#1f77b4", alpha=0.2)
ax.set_xlabel("step"); ax.set_ylabel("accuracy"); ax.set_title("Top-1 / Top-3"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
for k, label, color in [
    ("val_hand_loss", "hand", "#1f77b4"),
    ("val_shanten_loss", "shanten", "#ff7f0e"),
    ("val_wait_loss", "wait(tenpai)", "#2ca02c"),
    ("val_danger_loss", "danger", "#d62728"),
    ("val_loss_loss", "loss", "#9467bd"),
]:
    ax.plot(x, [dict(series_map[k])[s] for s in steps], label=label, color=color)
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("validation belief loss (log)")
ax.set_title("Belief head losses (val)"); ax.legend(); ax.grid(alpha=0.3, which="both")

ax = axes[1, 1]
for k, label, color in [
    ("val_hand_acc", "hand acc", "#1f77b4"),
    ("val_shanten_top1", "shanten top1", "#ff7f0e"),
    ("val_wait_tenpai_acc", "wait tenpai acc", "#2ca02c"),
    ("val_danger_auc", "danger AUC", "#d62728"),
    ("val_wait_cond_auc", "wait cond AUC", "#9467bd"),
]:
    ax.plot(x, [dict(series_map[k])[s] for s in steps], label=label, color=color)
ax.set_xlabel("step"); ax.set_ylabel("metric"); ax.set_title("Belief head quality (val)")
ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, "v19_sft_2ep_trends.png"), dpi=140)
plt.close(fig)

metrics = json.load(open(METRICS_JSON, encoding="utf-8"))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axes[0]
heads = ["hand", "shanten", "wait(tenpai)", "danger", "loss"]
weights = [0.7, 0.8, 1.8, 5.0, 5.0]
raw_loss = [
    metrics["validation/belief_hand_loss"],
    metrics["validation/belief_shanten_loss"],
    metrics["validation/belief_wait_tenpai_loss"],
    metrics["validation/belief_danger_loss"],
    metrics["validation/belief_loss_loss"],
]
contrib = [w * r for w, r in zip(weights, raw_loss)]
xpos = np.arange(len(heads))
ax.bar(xpos, raw_loss, color="#9ecae1", label="raw loss")
ax.bar(xpos, contrib, color="#3182bd", label="weighted contribution")
for i, (r, c) in enumerate(zip(raw_loss, contrib)):
    ax.text(i, r * 1.02, f"{r:.3f}", ha="center", va="bottom", fontsize=7)
    ax.text(i, c * 1.02, f"{c:.3f}", ha="center", va="bottom", fontsize=7, color="#08519c")
ax.set_xticks(xpos); ax.set_xticklabels(heads); ax.set_ylabel("val loss")
ax.set_title("Belief head loss & weighted contribution"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

ax = axes[1]
actions = ["pass", "discard", "reach", "chi", "pon", "kan", "hora", "ryukyoku"]
act_top1 = [metrics[f"train/action/{a}/top1"] for a in actions]
act_top3 = [metrics[f"train/action/{a}/top3"] for a in actions]
xpos = np.arange(len(actions))
ax.bar(xpos - 0.18, act_top1, width=0.36, color="#3182bd", label="top1")
ax.bar(xpos + 0.18, act_top3, width=0.36, color="#a1d99b", label="top3")
ax.set_xticks(xpos); ax.set_xticklabels(actions, rotation=25, ha="right"); ax.set_ylim(0, 1.05)
ax.set_ylabel("accuracy"); ax.set_title("Final action-type accuracy (train window)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

ax = axes[2]
belief_metrics = {
    "hand_acc": metrics["validation/belief_hand_acc"],
    "shanten_top1": metrics["validation/belief_shanten_top1"],
    "wait_tenpai_acc": metrics["validation/belief_wait_tenpai_acc"],
    "danger_auc": metrics["validation/belief_danger_auc"],
    "wait_cond_auc": metrics["validation/belief_wait_conditional_auc"],
    "danger_recall": metrics["validation/belief_danger_recall_at_topk"],
    "loss_mae": metrics["validation/belief_loss_mae"],
    "loss_cond_mae": metrics["validation/belief_loss_conditional_mae"],
}
names = list(belief_metrics.keys())
vals = [belief_metrics[n] for n in names]
ax.barh(names[::-1], vals[::-1], color="#756bb1")
for i, v in enumerate(vals[::-1]):
    ax.text(v, i, f"{v:.4f}", va="center", fontsize=8)
ax.set_xlabel("metric value"); ax.set_title("Final validation belief metrics"); ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, "v19_sft_2ep_final.png"), dpi=140)
plt.close(fig)

print("OK", len(steps))
