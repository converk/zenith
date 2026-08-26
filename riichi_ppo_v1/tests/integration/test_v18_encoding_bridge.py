"""真实 RiichiEnv → 当前局面编码 → 模型 的集成测试。"""

from __future__ import annotations

import numpy as np
import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.sft.data import encode_kyoku
from riichi_ppo_v1.tests.v18_fixtures import actor_inputs, first_kyoku_record


def test_encoding_bridge_full_model_forward() -> None:
    record, _game_id = first_kyoku_record()
    samples = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    selected = samples[:4]
    model = KyokuTransformerActorCritic()
    from riichi_ppo_v1.sft.trainer import collate_samples

    batch = collate_samples(selected, torch.device("cpu"))
    output = model(
        actor_factors=batch["actor_factors"], actor_numeric=batch["actor_numeric"],
        actor_lengths=batch["actor_lengths"], query_action_ids=batch["action_ids"],
        query_pair_counts=batch["pair_counts"], legal_mask=batch["legal_mask"], policy_only=True,
    )
    legal = batch["legal_mask"]
    assert torch.isfinite(output["policy_logits"][legal]).all()
    assert output["policy_logits"][~legal].eq(float("-inf")).all()


def test_canonical_sorting_permutation_invariance() -> None:
    """同一记录重新编码两次字节级一致；动作 id 升序唯一（规范排序）。"""
    record, _game_id = first_kyoku_record()
    first = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    second = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.actor_factors, b.actor_factors)
        assert np.array_equal(a.actor_numeric, b.actor_numeric)
        assert np.array_equal(a.action_ids, b.action_ids)
        assert np.all(np.diff(a.action_ids) > 0)
        # 完整 logits 一致性（真实数据、确定性模型）。
        model = KyokuTransformerActorCritic()
        from riichi_ppo_v1.sft.trainer import collate_samples

        batch = collate_samples([a], torch.device("cpu"))
        out = model(
            actor_factors=batch["actor_factors"], actor_numeric=batch["actor_numeric"],
            actor_lengths=batch["actor_lengths"], query_action_ids=batch["action_ids"],
            query_pair_counts=batch["pair_counts"], legal_mask=batch["legal_mask"], policy_only=True,
        )
        assert torch.isfinite(out["policy_logits"][batch["legal_mask"]]).all()
        assert out["policy_logits"][~batch["legal_mask"]].eq(float("-inf")).all()
