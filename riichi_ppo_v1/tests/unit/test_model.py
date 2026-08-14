import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.model.architecture import isolated_action_layout


def config(*, context_tokens: int = 64) -> ModelConfig:
    return ModelConfig(layers=2, shared_layers=1, critic_layers=1, d_model=32, query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64, context_tokens=context_tokens)


def isolated_inputs(action_ids=(0, 5)):
    length = 3 + 2 * len(action_ids)
    factors, numeric = semantic_token_inputs(1, length)
    factors[:, 2, 0] = 2
    legal = torch.zeros(1, 241, dtype=torch.bool)
    for pair, action in enumerate(action_ids):
        offense, defense = 3 + 2 * pair, 4 + 2 * pair
        factors[0, offense, 0] = factors[0, defense, 0] = 7
        factors[0, offense, 2] = factors[0, defense, 2] = action + 1
        factors[0, offense, 9] = 1
        factors[0, defense, 9] = 2
        legal[0, action] = True
    return factors, numeric, legal, torch.tensor([length])


def isolated_config() -> ModelConfig:
    return ModelConfig(
        layers=2, shared_layers=1, critic_layers=1, d_model=32,
        query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64,
        context_tokens=64, policy_head_type="isolated_action_query",
    )


def v15_config() -> ModelConfig:
    return ModelConfig(
        layers=2, shared_layers=1, critic_layers=1, d_model=32,
        query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64,
        context_tokens=64, policy_head_type="isolated_action_query",
        offense_fusion=True, critic_head_type="action_value",
    )


def test_default_model_topology_is_three_plus_one_plus_two() -> None:
    model = KyokuTransformerActorCritic(ModelConfig())
    assert len(model.public_backbone.blocks) == 3
    assert len(model.actor_backbone.blocks) == 1
    assert len(model.critic_backbone.blocks) == 2


def semantic_token_inputs(batch: int, length: int):
    factors = torch.zeros(batch, length, 10, dtype=torch.long)
    numeric = torch.zeros(batch, length, 8)
    if length:
        factors[:, 0] = torch.tensor([1, 1, 4, 1, 1, 1, 0, 0, 2, 1])
    if length > 1:
        factors[:, 1] = torch.tensor([2, 2, 1, 1, 0, 0, 0, 0, 0, 0])
        numeric[:, 1, 0] = 1.0
    return factors, numeric


def test_model_masks_actions_and_backpropagates() -> None:
    model = KyokuTransformerActorCritic(config())
    factors, numeric, legal, lengths = isolated_inputs((0, 10))
    factors = torch.cat((factors, factors))
    numeric = torch.cat((numeric, numeric))
    legal = torch.cat((legal, legal))
    lengths = torch.cat((lengths, lengths))
    output = model(factors, numeric, legal, lengths)
    assert output["policy_logits"].shape == (2, 241)
    assert output["policy_logits"].dtype is torch.float32
    assert output["value"].dtype is torch.float32
    assert torch.isneginf(output["policy_logits"][:, 1]).all()
    (output["value"].mean() + output["policy_logits"][:, 0].mean()).backward()
    assert model.policy_head[1].weight.grad is not None
    assert model.value_head.weight.grad is not None


def test_right_padding_does_not_change_query_output() -> None:
    torch.manual_seed(3)
    model = KyokuTransformerActorCritic(config()).eval()
    short_f, short_n, legal, lengths = isolated_inputs((0, 5))
    padded_f = torch.zeros(
        (1, short_f.shape[1] + 2, short_f.shape[2]), dtype=short_f.dtype,
    )
    padded_n = torch.zeros(
        (1, short_n.shape[1] + 2, short_n.shape[2]), dtype=short_n.dtype,
    )
    padded_f[0, : short_f.shape[1]] = short_f[0]
    padded_n[0, : short_n.shape[1]] = short_n[0]
    with torch.no_grad():
        a = model(short_f, short_n, legal, lengths)
        b = model(padded_f, padded_n, legal, lengths)
    torch.testing.assert_close(a["raw_policy_logits"], b["raw_policy_logits"])
    torch.testing.assert_close(a["value"], b["value"])


def test_context_overflow_is_not_truncated() -> None:
    model = KyokuTransformerActorCritic(config(context_tokens=4))
    factors, numeric, legal, lengths = isolated_inputs((0,))
    try:
        model(factors, numeric, legal, token_lengths=lengths)
    except ValueError as error:
        assert "context overflow" in str(error)
    else:  # pragma: no cover
        raise AssertionError("context overflow must be rejected")


def test_critic_private_inputs_do_not_change_policy_logits() -> None:
    torch.manual_seed(5)
    model = KyokuTransformerActorCritic(config()).eval()
    factors, numeric, legal, lengths = isolated_inputs((0, 5))
    critic_factors_a = torch.ones(1, 2, 10, dtype=torch.long)
    critic_factors_b = torch.full((1, 2, 10), 2, dtype=torch.long)
    critic_lengths = torch.tensor([2])

    with torch.no_grad():
        first = model(
            factors,
            numeric,
            legal,
            lengths,
            critic_factors=critic_factors_a,
            critic_lengths=critic_lengths,
        )
        second = model(
            factors,
            numeric,
            legal,
            lengths,
            critic_factors=critic_factors_b,
            critic_lengths=critic_lengths,
        )

    torch.testing.assert_close(first["raw_policy_logits"], second["raw_policy_logits"])
    assert not torch.allclose(first["value"], second["value"])


def test_critic_receives_every_shared_public_token() -> None:
    torch.manual_seed(6)
    model = KyokuTransformerActorCritic(config()).eval()
    factors, numeric, legal, token_lengths = isolated_inputs((0, 5))
    critic_factors = torch.ones(1, 2, 10, dtype=torch.long)
    critic_lengths = torch.tensor([2])
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, inputs):
        captured["sequence"] = inputs[0].detach().clone()
        captured["lengths"] = inputs[1].detach().clone()

    handle = model.critic_backbone.register_forward_pre_hook(capture)
    with torch.no_grad():
        model(
            factors, numeric, legal, token_lengths=token_lengths,
            critic_factors=critic_factors, critic_lengths=critic_lengths,
        )
        shared, _actor, state_mask, _defense_ids, _valid = model._isolated_public(
            factors, numeric, token_lengths, legal,
        )
        batch = factors.shape[0]
        public_capacity = int(state_mask.long().sum(-1).max().item())
        packed = shared.new_zeros((batch, public_capacity, model.config.d_model))
        source_rows, source_positions = torch.nonzero(state_mask, as_tuple=True)
        packed_positions = state_mask.long().cumsum(dim=1)[
            source_rows, source_positions,
        ] - 1
        packed[source_rows, packed_positions] = shared[source_rows, source_positions]
    handle.remove()

    public_length = packed.shape[1]
    torch.testing.assert_close(captured["sequence"][0, :public_length], packed[0])
    assert int(captured["lengths"][0]) == public_length + int(critic_lengths[0]) + 1


def test_critic_public_gradient_scale_only_reduces_shared_value_gradient() -> None:
    torch.manual_seed(7)
    full = KyokuTransformerActorCritic(config())
    scaled = KyokuTransformerActorCritic(config())
    scaled.load_state_dict(full.state_dict())
    factors, numeric, legal, lengths = isolated_inputs((0, 5))
    critic_factors = torch.ones(1, 2, 10, dtype=torch.long)
    kwargs = {
        "token_lengths": lengths,
        "critic_factors": critic_factors,
        "critic_lengths": torch.tensor([2]),
    }
    full(factors, numeric, legal, critic_public_grad_scale=1.0, **kwargs)["value"].sum().backward()
    scaled(factors, numeric, legal, critic_public_grad_scale=0.25, **kwargs)["value"].sum().backward()

    full_shared = full.public_backbone.blocks[0].attention.qkv.weight.grad
    scaled_shared = scaled.public_backbone.blocks[0].attention.qkv.weight.grad
    full_critic = full.critic_backbone.blocks[0].attention.qkv.weight.grad
    scaled_critic = scaled.critic_backbone.blocks[0].attention.qkv.weight.grad
    assert full_shared is not None and scaled_shared is not None
    assert full_critic is not None and scaled_critic is not None
    torch.testing.assert_close(scaled_shared, full_shared * 0.25)
    torch.testing.assert_close(scaled_critic, full_critic)


def test_critic_padding_does_not_change_value() -> None:
    torch.manual_seed(8)
    model = KyokuTransformerActorCritic(config()).eval()
    factors, numeric, legal, lengths = isolated_inputs((0, 5))
    short_critic = torch.ones(1, 1, 10, dtype=torch.long)
    padded_critic = torch.zeros(1, 4, 10, dtype=torch.long)
    padded_critic[:, :1] = short_critic
    kwargs = {"token_lengths": lengths, "critic_lengths": torch.tensor([1])}
    with torch.no_grad():
        short = model(factors, numeric, legal, critic_factors=short_critic, **kwargs)
        padded = model(factors, numeric, legal, critic_factors=padded_critic, **kwargs)
    torch.testing.assert_close(short["raw_policy_logits"], padded["raw_policy_logits"])
    torch.testing.assert_close(short["value"], padded["value"])


def test_critic_context_accounts_for_public_and_private_tokens() -> None:
    model = KyokuTransformerActorCritic(config(context_tokens=5))
    factors, numeric, legal, lengths = isolated_inputs((0,))
    common = {"token_lengths": lengths}
    model(factors, numeric, legal, critic_factors=torch.ones(1, 1, 10, dtype=torch.long), critic_lengths=torch.tensor([1]), **common)
    try:
        model(factors, numeric, legal, critic_factors=torch.ones(1, 2, 10, dtype=torch.long), critic_lengths=torch.tensor([2]), **common)
    except ValueError as error:
        assert "critic context overflow" in str(error)
    else:  # pragma: no cover
        raise AssertionError("combined public and critic context overflow must be rejected")


def test_value_loss_updates_only_shared_public_and_critic_branches() -> None:
    torch.manual_seed(7)
    model = KyokuTransformerActorCritic(config())
    factors, numeric, legal, lengths = isolated_inputs((0, 5))
    output = model(
        factors,
        numeric,
        legal,
        lengths,
        critic_factors=torch.ones(1, 2, 10, dtype=torch.long),
        critic_lengths=torch.tensor([2]),
    )
    output["value"].sum().backward()

    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name.startswith(("actor_backbone", "policy_head"))
    )
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith(("token_embedding", "public_backbone", "critic_embedding", "critic_backbone", "value_query", "value_head"))
    )


def test_mid_parameter_count_matches_the_three_plus_one_plus_two_budget() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("mid"))
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_816_066


def test_isolated_action_queries_are_permutation_invariant() -> None:
    torch.manual_seed(12)
    model = KyokuTransformerActorCritic(isolated_config()).eval()
    first = isolated_inputs((0, 5, 75))
    second = isolated_inputs((75, 0, 5))
    with torch.no_grad():
        a = model(*first)
        b = model(*second)
    torch.testing.assert_close(a["raw_policy_logits"][:, [0, 5, 75]], b["raw_policy_logits"][:, [0, 5, 75]])
    torch.testing.assert_close(a["value"], b["value"])


def test_v15_zero_projection_preserves_v13_policy_logits_and_receives_gradient() -> None:
    torch.manual_seed(13)
    legacy = KyokuTransformerActorCritic(isolated_config()).eval()
    v15 = KyokuTransformerActorCritic(v15_config()).eval()
    common = {
        name: value for name, value in legacy.state_dict().items()
        if name in v15.state_dict() and v15.state_dict()[name].shape == value.shape
    }
    missing, unexpected = v15.load_state_dict(common, strict=False)
    assert not unexpected
    assert set(missing) == {
        "offense_projection.weight", "offense_projection.bias",
        "q_head.weight", "q_head.bias",
    }
    inputs = isolated_inputs((0, 5, 75))
    with torch.no_grad():
        old_logits = legacy(*inputs)["raw_policy_logits"]
        new_logits = v15(*inputs)["raw_policy_logits"]
    torch.testing.assert_close(new_logits, old_logits, rtol=0.0, atol=0.0)

    v15.train()
    v15(*inputs)["raw_policy_logits"][:, [0, 5, 75]].sum().backward()
    assert v15.offense_projection.weight.grad is not None
    assert torch.count_nonzero(v15.offense_projection.weight.grad) > 0


def test_v15_q_critic_outputs_fixed_action_space_and_is_permutation_invariant() -> None:
    torch.manual_seed(14)
    model = KyokuTransformerActorCritic(v15_config()).eval()
    first = isolated_inputs((0, 5, 75))
    second = isolated_inputs((75, 0, 5))
    critic = torch.ones(1, 2, 10, dtype=torch.long)
    with torch.no_grad():
        a = model(*first, critic_factors=critic, critic_lengths=torch.tensor([2]))
        b = model(*second, critic_factors=critic, critic_lengths=torch.tensor([2]))
    assert a["q_values"].shape == (1, 241)
    assert "value" not in a
    torch.testing.assert_close(
        a["raw_policy_logits"][:, [0, 5, 75]],
        b["raw_policy_logits"][:, [0, 5, 75]],
    )


def test_isolated_action_queries_mask_illegal_and_handle_single_action() -> None:
    model = KyokuTransformerActorCritic(isolated_config()).eval()
    factors, numeric, legal, lengths = isolated_inputs((239,))
    output = model(factors, numeric, legal, lengths)
    assert torch.isfinite(output["raw_policy_logits"]).all()
    assert torch.isneginf(output["policy_logits"][~legal]).all()
    probabilities = output["policy_logits"].softmax(-1)
    assert probabilities[0, 239] == 1
    entropy = -(probabilities * output["policy_logits"].log_softmax(-1).nan_to_num()).sum(-1)
    assert entropy.item() == 0.0


def test_isolated_action_queries_reject_missing_pair() -> None:
    model = KyokuTransformerActorCritic(isolated_config())
    factors, numeric, legal, lengths = isolated_inputs((0, 5))
    lengths[0] -= 1
    try:
        model(factors, numeric, legal, lengths)
    except ValueError as error:
        assert "each legal action" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a missing defense query must be rejected")


def test_isolated_layout_enforces_visibility_and_4096_boundary() -> None:
    state_tokens = 4096 - 2 * 241
    factors = torch.zeros((1, 4096, 10), dtype=torch.long)
    factors[0, :state_tokens, 0] = 2
    legal = torch.ones((1, 241), dtype=torch.bool)
    for action in range(241):
        offense = state_tokens + 2 * action
        defense = offense + 1
        factors[0, offense:defense + 1, 0] = 7
        factors[0, offense:defense + 1, 2] = action + 1
        factors[0, offense, 9] = 1
        factors[0, defense, 9] = 2
    mask, valid, positions, state, defense_ids = isolated_action_layout(
        factors, torch.tensor([4096]), legal,
    )
    assert valid.all() and int(state.sum()) == state_tokens
    first_offense, first_defense = state_tokens, state_tokens + 1
    second_offense = state_tokens + 2
    assert not bool(mask[0, 0, 0, first_offense])
    assert bool(mask[0, 0, first_offense, state_tokens - 1])
    assert bool(mask[0, 0, first_defense, first_offense])
    assert not bool(mask[0, 0, first_defense, second_offense])
    assert int(positions[0, first_offense]) == state_tokens
    assert int(positions[0, second_offense]) == state_tokens
    assert int(defense_ids[0, -1]) == 240
