import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig


def config(*, context_tokens: int = 64) -> ModelConfig:
    return ModelConfig(layers=1, d_model=32, query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64, context_tokens=context_tokens)


def v5_inputs(batch: int, length: int):
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
    factors, numeric = v5_inputs(2, 3)
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
    short = v5_inputs(1, 3)
    padded = v5_inputs(1, 7)
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
    factors, numeric = v5_inputs(1, 4)
    try:
        model(factors, numeric, token_lengths=torch.tensor([4]))
    except ValueError as error:
        assert "context overflow" in str(error)
    else:  # pragma: no cover
        raise AssertionError("V5 context overflow must be rejected")
