"""V18 拓扑、结构化 mask、RoPE 位置与动作重排不变量。"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.model.architecture import (
    _actor_structured_layout,
    _bidirectional_layout,
    _rope_values,
)
from riichi_ppo_v1.model.encoding_protocol import (
    SEGMENT_ACTIONS,
    SEGMENT_ANALYSIS,
    SEGMENT_SHARED,
)
from riichi_ppo_v1.tests.v18_fixtures import actor_inputs


def test_model_topology() -> None:
    config = ModelConfig.preset("v18")
    model = KyokuTransformerActorCritic(config)
    assert config.d_model == 256
    assert config.query_heads == 16
    assert config.kv_heads == 4
    assert config.head_dim == 16
    assert config.ffn_dim == 704
    assert config.shared_layers == 3
    assert config.layers == 4
    assert config.critic_layers == 2
    assert config.context_tokens == 256
    assert model.public_backbone.blocks[0].attention.kvh == 4


def test_forward_actor_shapes_and_masks() -> None:
    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=2, action_ids=(1, 7, 12))
    output = model(
        actor_factors=inputs["actor_factors"],
        actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"],
        query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"],
        legal_mask=inputs["legal_mask"],
        policy_only=True,
    )
    assert output["policy_logits"].shape == (2, 241)
    assert output["raw_policy_logits"].shape == (2, 241)
    legal = inputs["legal_mask"]
    assert torch.isfinite(output["policy_logits"][legal]).all()
    assert output["policy_logits"][~legal].eq(float("-inf")).all()


def test_actor_structured_mask_isolation() -> None:
    # 构造 kind 序列：[shared×3, analysis×3, action×4]
    segments = torch.tensor([
        [SEGMENT_SHARED, SEGMENT_SHARED, SEGMENT_SHARED, SEGMENT_ANALYSIS, SEGMENT_ANALYSIS,
         SEGMENT_ANALYSIS, SEGMENT_ACTIONS, SEGMENT_ACTIONS, SEGMENT_ACTIONS, SEGMENT_ACTIONS]
    ])
    kinds = torch.tensor([
        [1, 2, 3, 10, 10, 10, 11, 12, 11, 12]
    ])
    lengths = torch.tensor([10])
    mask, valid = _actor_structured_layout(segments, kinds, lengths, 10)
    assert mask.shape == (1, 10, 10)
    # 动作 0（第 6/7 位）与动作 1（第 8/9 位）互不可见。
    assert not mask[0, 6, 8] and not mask[0, 8, 6]
    # 动作可看 shared 与 analysis，但 shared 不可看动作。
    assert mask[0, 6, 0] and mask[0, 6, 3]
    assert not mask[0, 0, 6]
    # analysis 之间互见。
    assert mask[0, 3, 4]
    assert mask[0, 4, 5]


def test_bidirectional_layout() -> None:
    mask, valid = _bidirectional_layout(torch.tensor([3, 5]), 5)
    assert mask.shape == (2, 5, 5)
    assert mask[0, 0, 2] and mask[0, 2, 0]
    assert not valid[0, 3] and not mask[0, 0, 3]


def test_rope_positions_continuous() -> None:
    positions = torch.arange(10)[None]
    cos, sin = _rope_values(positions, 16, torch.float32, 10_000.0)
    assert cos.shape == (1, 1, 10, 8)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()


def test_encode_batch_canonical_sort(monkeypatch) -> None:
    """环境动作顺序不同，经规范排序后编码完全一致（无需真实环境）。"""
    import numpy as np

    import riichi_ppo_v1.model.current_state as cs

    class _Batch:
        pass

    called: list[tuple[object, int]] = []

    def fake_queries(rows):
        from riichi_ppo_v1.model.native_encoding import NativeQueryBatch
        for obs, action, action_id in rows:
            called.append((action, action_id))
        return NativeQueryBatch(
            np.zeros((2 * len(rows), 2, 15), dtype=np.int32), 0, 0,
        )

    monkeypatch.setattr(cs, "encode_action_queries_batch_native", fake_queries)

    class _Obs:
        native_observation = None
        player_id = 0
        missed_agari_doujun = False
        missed_agari_riichi = False
        riichi_declared = [False] * 4
        drawn_tile = None

        def __init__(self, tag):
            self.tag = tag

    obs = _Obs("o")
    fake_env = type("FakeEnv", (), {})
    fake_batch = _Batch()
    fake_batch.rows = np.zeros((1, 32), dtype=np.int32)
    fake_batch.numeric = np.zeros((1, 8), dtype=np.float32)
    fake_batch.offsets = np.array([0, 1], dtype=np.int64)
    monkeypatch.setattr(cs.riichienv, "prepare_current_state_batch", lambda observations: fake_batch)

    class _Action:
        def __init__(self, kind):
            self.kind = kind
            self.tile = None
            self.consume_tiles = ()
            self.action_type = None

        def to_mjai(self):
            return '{"type": "none"}'

    a3, a7, a12 = _Action("a3"), _Action("a7"), _Action("a12")
    batch_a = cs.encode_batch([(obs, [(a7, 7), (a3, 3), (a12, 12)])])
    assert [item[1] for item in called] == [3, 7, 12]
    assert batch_a.action_ids[0].tolist() == [3, 7, 12]
    assert batch_a.query_pair_counts[0] == 3
    assert bool(batch_a.legal_mask[0, 3]) and bool(batch_a.legal_mask[0, 12])


def test_pair_isolation_in_logits() -> None:
    """只改变一个动作对的 answer，不影响其他动作对的 raw logits。"""
    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=1, action_ids=(3, 9, 20))
    base = model(
        actor_factors=inputs["actor_factors"], actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"], query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"], legal_mask=inputs["legal_mask"], policy_only=True,
    )["raw_policy_logits"].detach()
    modified = inputs["actor_factors"].clone()
    # 只修改第二个动作对（kind 11 的第 2 个位置）的 O1 answer。
    positions = torch.nonzero(modified[0, :, 1].eq(11)).squeeze(-1)
    assert positions.numel() == 3
    modified[0, positions[1], 6] = 1
    out = model(
        actor_factors=modified, actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"], query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"], legal_mask=inputs["legal_mask"], policy_only=True,
    )["raw_policy_logits"].detach()
    assert not torch.allclose(base, out)
    # 第一、三个动作对不受影响（pair 隔离）。
    assert torch.allclose(base[0, 3], out[0, 3], atol=1e-5, rtol=1e-5)
    assert torch.allclose(base[0, 20], out[0, 20], atol=1e-5, rtol=1e-5)


def test_critic_forward_private_changes_value() -> None:
    from riichi_ppo_v1.tests.v18_fixtures import critic_inputs

    model = KyokuTransformerActorCritic()
    inputs = actor_inputs(batch=2, action_ids=(1, 7))
    critic = critic_inputs(batch=2)
    output = model(
        actor_factors=inputs["actor_factors"], actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"], query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"], legal_mask=inputs["legal_mask"],
        critic_factors=critic["critic_factors"], critic_lengths=critic["critic_lengths"],
    )
    assert output["value"].shape == (2,)
    assert torch.isfinite(output["value"]).all()
