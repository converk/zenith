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
    factors, numeric = semantic_token_inputs(2, 3)
    legal = torch.zeros(2, 241, dtype=torch.bool)
    legal[:, [0, 10]] = True
    output = model(factors, numeric, legal, torch.tensor([2, 2]))
    assert output["policy_logits"].shape == (2, 241)
    assert output["policy_logits"].dtype is torch.float32
    assert output["value"].dtype is torch.float32
    assert torch.isneginf(output["policy_logits"][:, 1]).all()
    (output["value"].mean() + output["policy_logits"][:, 0].mean()).backward()
    assert model.policy_head.weight.grad is not None
    assert model.value_head.weight.grad is not None


def test_right_padding_does_not_change_query_output() -> None:
    torch.manual_seed(3)
    model = KyokuTransformerActorCritic(config()).eval()
    short = semantic_token_inputs(1, 3)
    padded = semantic_token_inputs(1, 7)
    padded[0][:, :3] = short[0]
    padded[1][:, :3] = short[1]
    legal = torch.ones(1, 241, dtype=torch.bool)
    with torch.no_grad():
        a = model(*short, legal, torch.tensor([2]))
        b = model(*padded, legal, torch.tensor([2]))
    torch.testing.assert_close(a["raw_policy_logits"], b["raw_policy_logits"])
    torch.testing.assert_close(a["value"], b["value"])


def test_context_overflow_is_not_truncated() -> None:
    model = KyokuTransformerActorCritic(config(context_tokens=4))
    factors, numeric = semantic_token_inputs(1, 4)
    try:
        model(factors, numeric, token_lengths=torch.tensor([4]))
    except ValueError as error:
        assert "context overflow" in str(error)
    else:  # pragma: no cover
        raise AssertionError("context overflow must be rejected")


def test_critic_private_inputs_do_not_change_policy_logits() -> None:
    torch.manual_seed(5)
    model = KyokuTransformerActorCritic(config()).eval()
    factors, numeric = semantic_token_inputs(1, 3)
    legal = torch.ones(1, 241, dtype=torch.bool)
    critic_factors_a = torch.ones(1, 2, 10, dtype=torch.long)
    critic_factors_b = torch.full((1, 2, 10), 2, dtype=torch.long)
    critic_lengths = torch.tensor([2])

    with torch.no_grad():
        first = model(
            factors,
            numeric,
            legal,
            torch.tensor([2]),
            critic_factors=critic_factors_a,
            critic_lengths=critic_lengths,
        )
        second = model(
            factors,
            numeric,
            legal,
            torch.tensor([2]),
            critic_factors=critic_factors_b,
            critic_lengths=critic_lengths,
        )

    torch.testing.assert_close(first["raw_policy_logits"], second["raw_policy_logits"])
    assert not torch.allclose(first["value"], second["value"])


def test_critic_receives_every_shared_public_token() -> None:
    torch.manual_seed(6)
    model = KyokuTransformerActorCritic(config()).eval()
    factors, numeric = semantic_token_inputs(1, 4)
    token_lengths = torch.tensor([2])
    critic_factors = torch.ones(1, 2, 10, dtype=torch.long)
    critic_lengths = torch.tensor([2])
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, inputs):
        captured["sequence"] = inputs[0].detach().clone()
        captured["lengths"] = inputs[1].detach().clone()

    handle = model.critic_backbone.register_forward_pre_hook(capture)
    with torch.no_grad():
        model(factors, numeric, token_lengths=token_lengths, critic_factors=critic_factors, critic_lengths=critic_lengths)
        tokens = model.token_embedding(factors, numeric)
        public_input = tokens.new_zeros((1, factors.shape[1] + 1, model.config.d_model))
        public_input[:, :factors.shape[1]] = tokens
        public_input[0, token_lengths[0]] = model.query
        expected_public = model.public_backbone(public_input, token_lengths + 1)
    handle.remove()

    public_length = int(token_lengths[0]) + 1
    torch.testing.assert_close(captured["sequence"][0, :public_length], expected_public[0, :public_length])
    assert int(captured["lengths"][0]) == public_length + int(critic_lengths[0]) + 1


def test_critic_public_gradient_scale_only_reduces_shared_value_gradient() -> None:
    torch.manual_seed(7)
    full = KyokuTransformerActorCritic(config())
    scaled = KyokuTransformerActorCritic(config())
    scaled.load_state_dict(full.state_dict())
    factors, numeric = semantic_token_inputs(1, 3)
    critic_factors = torch.ones(1, 2, 10, dtype=torch.long)
    kwargs = {
        "token_lengths": torch.tensor([2]),
        "critic_factors": critic_factors,
        "critic_lengths": torch.tensor([2]),
    }
    full(factors, numeric, critic_public_grad_scale=1.0, **kwargs)["value"].sum().backward()
    scaled(factors, numeric, critic_public_grad_scale=0.25, **kwargs)["value"].sum().backward()

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
    factors, numeric = semantic_token_inputs(1, 3)
    short_critic = torch.ones(1, 1, 10, dtype=torch.long)
    padded_critic = torch.zeros(1, 4, 10, dtype=torch.long)
    padded_critic[:, :1] = short_critic
    kwargs = {"token_lengths": torch.tensor([2]), "critic_lengths": torch.tensor([1])}
    with torch.no_grad():
        short = model(factors, numeric, critic_factors=short_critic, **kwargs)
        padded = model(factors, numeric, critic_factors=padded_critic, **kwargs)
    torch.testing.assert_close(short["raw_policy_logits"], padded["raw_policy_logits"])
    torch.testing.assert_close(short["value"], padded["value"])


def test_critic_context_accounts_for_public_and_private_tokens() -> None:
    model = KyokuTransformerActorCritic(config(context_tokens=6))
    factors, numeric = semantic_token_inputs(1, 3)
    common = {"token_lengths": torch.tensor([3])}
    model(factors, numeric, critic_factors=torch.ones(1, 1, 10, dtype=torch.long), critic_lengths=torch.tensor([1]), **common)
    try:
        model(factors, numeric, critic_factors=torch.ones(1, 2, 10, dtype=torch.long), critic_lengths=torch.tensor([2]), **common)
    except ValueError as error:
        assert "critic context overflow" in str(error)
    else:  # pragma: no cover
        raise AssertionError("combined public and critic context overflow must be rejected")


def test_value_loss_updates_only_shared_public_and_critic_branches() -> None:
    torch.manual_seed(7)
    model = KyokuTransformerActorCritic(config())
    factors, numeric = semantic_token_inputs(1, 3)
    output = model(
        factors,
        numeric,
        torch.ones(1, 241, dtype=torch.bool),
        torch.tensor([2]),
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
        if name.startswith(("token_embedding", "query", "public_backbone", "critic_embedding", "critic_backbone", "value_query", "value_head"))
    )


def test_mid_parameter_count_matches_the_three_plus_one_plus_two_budget() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("mid"))
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_825_330
    isolated = KyokuTransformerActorCritic(ModelConfig(policy_head_type="isolated_action_query"))
    assert sum(parameter.numel() for parameter in isolated.parameters()) <= int(2_825_330 * 1.10)


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
