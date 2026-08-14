from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import riichi_ppo_v1.sft.train as t
from riichi_ppo_v1.model import ModelConfig
from riichi_ppo_v1.model.feature_schema import DECISION_ANALYSIS_VERSION, RUST_ANALYSIS_VERSION, feature_schema_sha256
from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION
from riichi_ppo_v1.sft.precompute import _write_chunk

ROOT = Path(tempfile.mkdtemp(prefix="zenith_v13_resume_fixture_"))
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
(DATA / "train").mkdir(exist_ok=True)
(DATA / "validation").mkdir(exist_ok=True)
source = Path("datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16")
train_rows = []
stream = t.iter_split_samples(source, "train", gamma=.99, seed=1, shuffle=False, include_critic=False)
for _ in range(24): train_rows.append(next(stream))
valid_rows = []
stream = t.iter_split_samples(source, "validation", gamma=.99, seed=1, shuffle=False, include_critic=False)
for _ in range(8): valid_rows.append(next(stream))
_write_chunk(DATA / "train" / "train-00000-000.npz", train_rows)
_write_chunk(DATA / "validation" / "validation-00000-000.npz", valid_rows)
manifest = {
    "format": "riichi-sft-encoded-v3", "token_schema_version": TOKEN_SCHEMA_VERSION,
    "feature_schema_sha256": feature_schema_sha256(), "rust_analysis_version": RUST_ANALYSIS_VERSION,
    "decision_analysis_version": DECISION_ANALYSIS_VERSION,
    "counts": {"train_decisions": len(train_rows), "validation_decisions": len(valid_rows)},
}
(DATA / "manifest.json").write_text(json.dumps(manifest)+"\n")

t._model_config = lambda _cfg: ModelConfig(layers=2, shared_layers=1, critic_layers=1, d_model=32, query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64, context_tokens=4096, policy_head_type="isolated_action_query")
original_collate = t.collate_samples
current_log = None
def logged_collate(rows, *args, **kwargs):
    if current_log is not None:
        current_log.extend((r.year, r.game_id, r.kyoku_index, r.seat, r.decision_index) for r in rows)
    return original_collate(rows, *args, **kwargs)
t.collate_samples = logged_collate

base = t.load_config(None)
base.update({
    "device": "cpu", "learner_gpus": 1, "epochs": 1, "batch_size": 4,
    "checkpoint_interval_steps": 0, "validation_interval_steps": 0,
    "validation_max_samples": 8, "heuristic_evaluation_enabled": False,
    "tensorboard_enabled": False, "length_bucket_window_batches": 1,
})

def run(output: Path, max_steps: int, resume: Path | None = None):
    global current_log
    cfg = copy.deepcopy(base)
    cfg["max_train_steps"] = max_steps
    cfg["checkpoint_dir"] = str(output)
    cfg["resume"] = str(resume) if resume else None
    current_log = []
    t.train_worker(0, 1, cfg, DATA, output)
    log = current_log
    current_log = None
    return torch.load(output / "latest.pt", map_location="cpu", weights_only=False), log
