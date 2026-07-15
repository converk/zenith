import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig


def test_model_masks_actions_and_backpropagates() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("mid"))
    ids = torch.zeros(2, 5, 8, dtype=torch.long)
    ids[:, :3, 0] = 26
    attention = torch.tensor([[True, True, True, False, False], [True, True, True, False, False]])
    legal = torch.zeros(2, 241, dtype=torch.bool)
    legal[:, [0, 10]] = True
    output = model(ids, legal, attention, torch.tensor([3, 3]))
    assert output["policy_logits"].shape == (2, 241)
    assert torch.isneginf(output["policy_logits"][:, 1]).all()
    (output["value"].mean() + output["policy_logits"][:, 0].mean()).backward()
    assert model.policy_head.weight.grad is not None


def test_right_padding_does_not_change_decision_output() -> None:
    torch.manual_seed(3)
    model = KyokuTransformerActorCritic(ModelConfig(16, 16, 1, 2, 32)).eval()
    short = torch.zeros(1, 3, 8, dtype=torch.long)
    short[..., 0] = torch.tensor([[26, 27, 28]])
    padded = torch.zeros(1, 7, 8, dtype=torch.long)
    padded[:, :3] = short
    legal = torch.ones(1, 241, dtype=torch.bool)
    with torch.no_grad():
        a = model(short, legal, torch.ones(1, 3, dtype=torch.bool), torch.tensor([3]))
        b = model(padded, legal, torch.tensor([[True, True, True, False, False, False, False]]), torch.tensor([3]))
    torch.testing.assert_close(a["raw_policy_logits"], b["raw_policy_logits"])
    torch.testing.assert_close(a["value"], b["value"])


def test_rollout_kv_cache_matches_full_sequence_and_keeps_snapshot_temporary() -> None:
    torch.manual_seed(9)
    model = KyokuTransformerActorCritic(ModelConfig(16, 16, 2, 2, 32)).eval()
    history = torch.zeros(1, 3, 8, dtype=torch.long)
    history[..., 0] = torch.tensor([[26, 27, 28]])
    snapshot = torch.zeros(1, 2, 8, dtype=torch.long)
    snapshot[..., 0] = torch.tensor([[35, 36]])
    legal = torch.ones(1, 241, dtype=torch.bool)
    with torch.no_grad():
        full = model(torch.cat((history, snapshot), dim=1), legal, torch.ones(1, 5, dtype=torch.bool), torch.tensor([5]))
        cached, past = model.forward_cached(history, snapshot, legal)
        torch.testing.assert_close(cached["raw_policy_logits"], full["raw_policy_logits"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cached["value"], full["value"], atol=1e-5, rtol=1e-5)
        # A different temporary snapshot must not change the confirmed prefix.
        changed_snapshot = snapshot.clone()
        changed_snapshot[..., 0] = torch.tensor([[37, 38]])
        cached_changed, same_past = model.forward_cached(history[:, :0], changed_snapshot, legal, past, history_length=3)
        assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(past, same_past))
        full_changed = model(torch.cat((history, changed_snapshot), dim=1), legal, torch.ones(1, 5, dtype=torch.bool), torch.tensor([5]))
        torch.testing.assert_close(cached_changed["raw_policy_logits"], full_changed["raw_policy_logits"], atol=1e-5, rtol=1e-5)
