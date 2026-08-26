"""V18 小规模 SFT 生命周期：replay→precompute→shard→collate→train→checkpoint。"""

from __future__ import annotations

import json
import tarfile
import io
from pathlib import Path

import numpy as np
import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.sft.actor_bc import actor_parameters, freeze_critic, is_actor_parameter
from riichi_ppo_v1.sft.checkpoint import checkpoint_payload, load_exact_resume
from riichi_ppo_v1.sft.contract import dataset_manifest_hash
from riichi_ppo_v1.sft.data import iter_split_samples
from riichi_ppo_v1.sft.precompute import precompute
from riichi_ppo_v1.sft.trainer import collate_samples
from riichi_ppo_v1.tests.v18_fixtures import first_kyoku_record


def _build_source(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    (source / "train").mkdir(parents=True, exist_ok=True)
    record, _game_id = first_kyoku_record()
    def add(split: str, member: str) -> None:
        (source / split).mkdir(parents=True, exist_ok=True)
        with tarfile.open(source / split / f"{split}-00000.tar", "w") as archive:
            payload = record.encode("utf-8")
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    add("train", "2024-test-1-00.mjson")
    add("validation", "2025-test-2-00.mjson")
    (source / "manifest.json").write_text(json.dumps({
        "format": "riichi-sft-kyoku-v1",
        "years": [2024, 2025],
        "counts": {"train": 1, "validation": 1,
                   "train_decisions": 65, "validation_decisions": 65},
    }), encoding="utf-8")
    return source


def test_small_sft_lifecycle(tmp_path: Path) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "encoded"
    precompute(source, output, denominator=1, remainders=(0,), kyokus_per_shard=1, workers=1)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["train_kyokus"] == 1
    assert manifest["counts"]["train_decisions"] > 0

    samples = list(iter_split_samples(output, "train", seed=0, shuffle=False, include_critic=False))
    assert samples
    batch = collate_samples(samples[:4], torch.device("cpu"))
    model = KyokuTransformerActorCritic()
    freeze_critic(model)
    loss = torch.nn.functional.cross_entropy(
        model(
            actor_factors=batch["actor_factors"], actor_numeric=batch["actor_numeric"],
            actor_lengths=batch["actor_lengths"], query_action_ids=batch["action_ids"],
            query_pair_counts=batch["pair_counts"], legal_mask=batch["legal_mask"], policy_only=True,
        )["policy_logits"].float(),
        batch["actions"],
    )
    loss.backward()
    actor_grads = 0
    for name, parameter in model.named_parameters():
        if is_actor_parameter(name):
            if parameter.grad is not None:
                actor_grads += 1
        else:
            assert parameter.grad is None, name
    # 参与 Actor 前向的参数应获得梯度；Critic 专用表（13/14）在 actor-only 下无梯度。
    assert actor_grads >= 20
    optimizer = torch.optim.AdamW(list(actor_parameters(model)), lr=1e-4)
    optimizer.step()

    # checkpoint 保存/严格加载（actor-only 模式）。
    hash_value = dataset_manifest_hash(output)
    payload = checkpoint_payload(
        model, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0),
        config={"train_critic": False, "device": "cpu", "learner_gpus": 1,
                "model_size": "v18", "context_tokens": 256,
                "dense_slot_dim": 32, "dense_fusion_dim": 512,
                "policy_head_type": "current_state_snapshot"},
        manifest_hash=hash_value, mode="actor_only", epoch=0, global_step=1,
        rank_batches_consumed=[1], best_validation_loss=float("inf"),
        metrics={}, rank_rng_states=[{"torch": torch.get_rng_state(), "cuda": None,
                                      "numpy": np.random.get_state(), "python": __import__("random").getstate()}],
    )
    path = tmp_path / "latest.pt"
    torch.save(payload, path)
    loaded = load_exact_resume(path, model_config=model.config, training_mode="actor_only",
                               dataset_manifest_hash=hash_value, world_size=1,
                               trainable_scope="full_actor")
    assert loaded["global_step"] == 1
    # 旧 format 拒绝。
    bad = dict(payload)
    bad["sft_contract_version"] = "riichi-sft-v16-1"
    bad_path = tmp_path / "old.pt"
    torch.save(bad, bad_path)
    try:
        load_exact_resume(bad_path, model_config=model.config, training_mode="actor_only",
                          dataset_manifest_hash=hash_value, world_size=1)
        raise AssertionError("legacy checkpoint must be rejected")
    except RuntimeError:
        pass
