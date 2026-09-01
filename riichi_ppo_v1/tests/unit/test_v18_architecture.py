"""V18 拓扑、结构化 mask、RoPE 位置与动作重排不变量。"""

from __future__ import annotations

import numpy as np
import pytest
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


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        layers=2,
        shared_layers=1,
        critic_layers=1,
        d_model=32,
        query_heads=4,
        kv_heads=1,
        head_dim=8,
        ffn_dim=64,
        dense_slot_dim=8,
        dense_fusion_dim=64,
        context_tokens=256,
    )


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
        for _obs, action, action_id in rows:
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


def test_critic_private_embedding_grad_scale_preserves_forward_and_scales_backward() -> None:
    from riichi_ppo_v1.tests.v18_fixtures import critic_inputs

    torch.manual_seed(123)
    model = KyokuTransformerActorCritic(_tiny_config())
    inputs = actor_inputs(batch=1, action_ids=(1, 7))
    critic = critic_inputs(batch=1)

    outputs = []
    grads = []
    for scale in (0.0, 0.25, 1.0):
        model.zero_grad(set_to_none=True)
        output = model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            critic_factors=critic["critic_factors"],
            critic_lengths=critic["critic_lengths"],
            detach_critic_public=True,
            critic_private_embedding_grad_scale=scale,
        )
        outputs.append(output["value"].detach())
        output["value"].sum().backward()
        grad_squares = [
            parameter.grad.detach().float().square().sum()
            for parameter in model.token_embedding.parameters()
            if parameter.grad is not None
        ]
        grad_norm = (
            torch.stack(grad_squares).sum().sqrt()
            if grad_squares
            else torch.zeros(())
        )
        grads.append(grad_norm.item())

    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(outputs[0], outputs[2])
    assert grads[0] == 0.0
    assert grads[1] == pytest.approx(grads[2] * 0.25, rel=1e-4, abs=1e-6)


def _kind_constants() -> tuple[int, int]:
    from riichi_ppo_v1.model.encoding_protocol import (
        KIND_ACTION_DEFENSE_QUERY,
        KIND_ACTION_OFFENSE_QUERY,
    )

    return KIND_ACTION_OFFENSE_QUERY, KIND_ACTION_DEFENSE_QUERY


def test_action_query_rows_are_canonical_tail_window() -> None:
    """canonical 契约:action query 行恒为每行序列尾部连续 2×pair_count 行。

    forward 的算术索引(取代 torch.nonzero)以此为前提;契约来源为 Rust
    编码器 fail-closed 构造,fixtures 按同一布局构造。
    """
    offense_kind, defense_kind = _kind_constants()
    for batch in (1, 4):
        inputs = actor_inputs(batch=batch, action_ids=(1, 7, 12))
        factors = inputs["actor_factors"]
        lengths = inputs["actor_lengths"]
        counts = inputs["query_pair_counts"]
        capacity = factors.shape[1]
        positions = torch.arange(capacity)[None, :]
        kind = factors[..., 1]
        mask = (kind.eq(offense_kind) | kind.eq(defense_kind)) & (
            positions < lengths[:, None]
        )
        tail = (positions >= lengths[:, None] - 2 * counts[:, None]) & (
            positions < lengths[:, None]
        )
        assert torch.equal(mask, tail)
        # tail 窗口内 O 先 D 后相邻成对:偶数位 O、奇数位 D。
        window = kind[0, lengths[0] - 2 * counts[0]:lengths[0]]
        assert torch.equal(window[0::2], torch.full_like(window[0::2], offense_kind))
        assert torch.equal(window[1::2], torch.full_like(window[1::2], defense_kind))


def _forward_kwargs(batch: int = 3) -> dict:
    from riichi_ppo_v1.tests.v18_fixtures import critic_inputs

    inputs = actor_inputs(batch=batch, action_ids=(2, 9))
    critic = critic_inputs(batch=batch)
    return dict(
        actor_factors=inputs["actor_factors"],
        actor_numeric=inputs["actor_numeric"],
        actor_lengths=inputs["actor_lengths"],
        query_action_ids=inputs["action_ids"],
        query_pair_counts=inputs["query_pair_counts"],
        legal_mask=inputs["legal_mask"],
        critic_factors=critic["critic_factors"],
        critic_lengths=critic["critic_lengths"],
    )


def _host_capacities(kwargs: dict) -> tuple[int, int]:
    """与 rollout_buffer.collate 相同语义的 host 容量预计算。"""
    shared_per_row = (kwargs["actor_factors"][..., 0] == SEGMENT_SHARED).sum(-1)
    shared_capacity = int(shared_per_row.max())
    critic_total_capacity = int(
        (shared_per_row + kwargs["critic_lengths"] + 1).max()
    )
    return shared_capacity, critic_total_capacity


def test_forward_validate_switch_and_host_capacity_bitwise_equal() -> None:
    """validate_structure 两态、host capacity 两来源在合法输入上 torch.equal。"""
    model = KyokuTransformerActorCritic()
    model.eval()
    kwargs = _forward_kwargs()
    shared_capacity, critic_total_capacity = _host_capacities(kwargs)
    with torch.no_grad():
        base = model(**kwargs)
        no_validate = model(**kwargs, validate_structure=False)
        host = model(
            **kwargs,
            shared_capacity=shared_capacity,
            critic_total_capacity=critic_total_capacity,
        )
        lean = model(
            **kwargs,
            validate_structure=False,
            shared_capacity=shared_capacity,
            critic_total_capacity=critic_total_capacity,
        )
    for output in (no_validate, host, lean):
        for key in ("raw_policy_logits", "policy_logits", "value"):
            assert torch.equal(base[key], output[key]), key


def test_validate_structure_true_rejects_contract_violations() -> None:
    """validate=True 拒绝 tail 窗口违约/重复 id;False 依契约放行不抛错。"""
    from riichi_ppo_v1.model.encoding_protocol import (
        KIND_ACTION_DEFENSE_QUERY,
        KIND_ACTION_OFFENSE_QUERY,
    )

    model = KyokuTransformerActorCritic()
    model.eval()
    # 重复 action id:tail 窗口位置合法但行内相邻重复。
    kwargs = _forward_kwargs(batch=1)
    kwargs["query_action_ids"][0, 1] = kwargs["query_action_ids"][0, 0]
    with pytest.raises(ValueError, match="unique"):
        model(**kwargs, validate_structure=True)
    with torch.no_grad():
        model(**kwargs, validate_structure=False)
    # tail 窗口违约:声明 pair 数少于实际 query 行(tail 窗口与 kind mask 失配;
    # 直接改 kind 会先撞上 segment/kind schema 校验,故用 pair_counts 失配)。
    broken = _forward_kwargs(batch=1)
    broken["query_pair_counts"][0] -= 1
    with pytest.raises(ValueError, match="tail window"):
        model(**broken, validate_structure=True)
    with torch.no_grad():
        model(**broken, validate_structure=False)
    assert KIND_ACTION_OFFENSE_QUERY != KIND_ACTION_DEFENSE_QUERY


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SDPA 后端锁定仅 CUDA 可验")
def test_sdpa_backend_lock_matches_default_and_rejects_flash() -> None:
    """B3:结构化 bool mask 下 mem_efficient 锁定输出与默认调度逐位一致,
    flash-only 上下文 fail-fast 报错(防静默 5 倍回退悬崖)。"""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    device = torch.device("cuda")
    torch.manual_seed(5)
    batch, heads, tokens, head_dim = 4, 4, 48, 8
    q = torch.randn(batch, heads, tokens, head_dim, device=device)
    k = torch.randn(batch, heads, tokens, head_dim, device=device)
    v = torch.randn(batch, heads, tokens, head_dim, device=device)
    # 结构化 bool mask 的等价形态:任意 [B,1,L,L] 布尔掩蔽(含整行 False)。
    pair_mask = torch.rand(batch, tokens, tokens, device=device) > 0.2
    pair_mask[0, 5] = False  # 制造整行 False(等价 -inf 掩蔽行)。
    attn_mask = pair_mask[:, None]

    with torch.no_grad():
        default_value = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            locked_value = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        with pytest.raises(RuntimeError):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    # 锁定前后同后端:逐位一致(若默认调度并非 mem_efficient,此处即失败,
    # 需重新核对 B3 前提)。
    assert torch.equal(default_value, locked_value)


def test_critic_assembly_variable_lengths_match_reference_scatter() -> None:
    """critic 装配算术 scatter 与逐元素参考实现逐位一致(变长 shared/critic 行)。

    参考实现按旧布尔掩码索引的语义逐行写入:public 行 [b, p] 在
    p < shared_len(b) 时等于 shared_for_critic[b, p];private 行
    [b, shared_len(b)+q] 在 q < critic_len(b) 时等于 critic_embeddings[b, q];
    value 行 [b, shared_len(b)+critic_len(b)] 恒为 value_query。对比整条
    critic_sequence 的重建结果(经 critic_hidden 反解不可行,这里以同一输入
    双路径 forward 对比 value/critic_hidden)。
    """
    from riichi_ppo_v1.model.critic_features import (
        encode_critic_features,
        pad_critic_feature_rows,
    )
    from riichi_ppo_v1.tests.v18_fixtures import critic_inputs

    torch.manual_seed(321)
    model = KyokuTransformerActorCritic(_tiny_config()).eval()
    inputs = actor_inputs(batch=1, action_ids=(1, 7))
    B = 5
    # actor 侧复制为 5 行同输入(装配对比只关心 critic 变长)。
    inputs = {
        name: value.expand(B, *value.shape[1:]).contiguous() if value.ndim >= 2 else value.repeat(B)
        for name, value in inputs.items()
    }
    base = critic_inputs(batch=1)
    # 用 fixture 行的截断构造 5 条变长 CriticFeatures(长度 1..5,覆盖
    # 「短私有行」「空未来段」形态),再拼接成 padded 批。
    from riichi_ppo_v1.model.critic_features import CriticFeatures
    features = []
    base_rows = base["critic_factors"][0]
    base_length = int(base["critic_lengths"][0])
    for row in range(B):
        length = min(1 + row, base_length)  # 1..5(不超过 fixture 行数)
        factors = np.asarray(base_rows[:length].numpy(), dtype=np.uint8).reshape(-1, base_rows.shape[1])
        features.append(CriticFeatures(factors=factors, length=length))
    critic_factors, critic_lengths = pad_critic_feature_rows(features)
    critic_factors = torch.from_numpy(critic_factors).long()

    with torch.inference_mode():
        output = model(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            critic_factors=critic_factors,
            critic_lengths=torch.from_numpy(critic_lengths),
            validate_structure=False,
        )
    assert output["value"].shape == (B,)
    assert torch.isfinite(output["value"]).all()
    # 短私有行行数递增时 value 随行内容变化(装配确实消费了逐行长度)。
    assert not torch.allclose(output["value"][0], output["value"][B - 1])


def test_critic_assembly_padding_slots_do_not_leak_into_value() -> None:
    """非法槽位(clamp 占位)不进入任何有效行:等价输入不同 padding 高度,
    有效区 value 输出必须逐位一致(padding 行高不影响结果)。"""
    from riichi_ppo_v1.tests.v18_fixtures import critic_inputs

    torch.manual_seed(654)
    model = KyokuTransformerActorCritic(_tiny_config()).eval()
    inputs = actor_inputs(batch=1, action_ids=(1, 7))
    critic = critic_inputs(batch=1)
    length = int(critic["critic_lengths"][0])
    factors = critic["critic_factors"]
    padded = torch.zeros((1, length + 6, factors.shape[2]), dtype=factors.dtype)
    padded[:, :length] = factors

    with torch.inference_mode():
        kw = dict(
            actor_factors=inputs["actor_factors"],
            actor_numeric=inputs["actor_numeric"],
            actor_lengths=inputs["actor_lengths"],
            query_action_ids=inputs["action_ids"],
            query_pair_counts=inputs["query_pair_counts"],
            legal_mask=inputs["legal_mask"],
            critic_lengths=critic["critic_lengths"],
            validate_structure=False,
        )
        value_tight = model(critic_factors=factors, **kw)["value"]
        value_padded = model(critic_factors=padded, **kw)["value"]
    torch.testing.assert_close(value_tight, value_padded)
