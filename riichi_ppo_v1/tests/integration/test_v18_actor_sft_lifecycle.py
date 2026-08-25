"""V18 Actor-only artifact 的严格保存、加载与旧契约拒绝。"""

import torch
import pytest

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.sft.actor_bc import load_actor, save_actor


def test_actor_only_roundtrip_and_legacy_rejection(tmp_path) -> None:
    model = KyokuTransformerActorCritic()
    path = tmp_path / "actor.pt"
    save_actor(path, model)
    loaded = load_actor(path)
    assert loaded.config == model.config
    legacy = torch.load(path, weights_only=False)
    legacy["sft_contract_version"] = "riichi-sft-v16-1"
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(RuntimeError, match="pure V18"):
        load_actor(legacy_path)
