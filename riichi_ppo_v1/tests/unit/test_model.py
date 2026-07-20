import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig


def config(*, context_tokens: int = 64) -> ModelConfig:
    return ModelConfig(layers=2, shared_layers=1, critic_layers=1, d_model=32, query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64, context_tokens=context_tokens)


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


def test_mid_parameter_count_matches_the_v5_budget() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("mid"))
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_673_970
