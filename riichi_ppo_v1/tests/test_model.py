import torch

from riichi_ppo_v1.model import BOARD_FIELDS, BOARD_TOKENS, KyokuTransformerActorCritic, ModelConfig


def v4_inputs(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kinds = torch.zeros(batch, length, dtype=torch.long)
    if length:
        kinds[:, : min(length, 2)] = torch.tensor([1, 2][: min(length, 2)])
    turn = torch.zeros(batch, length, 4, 4, dtype=torch.long)
    if length:
        turn[:, 0, 0] = torch.tensor([1, 1, 12, 1])
    meld = torch.zeros(batch, length, 8, dtype=torch.long)
    if length > 1:
        meld[:, 1] = torch.tensor([2, 4, 3, 12, 12, 12, 0, 0])
    board = torch.zeros(batch, BOARD_TOKENS, BOARD_FIELDS, dtype=torch.long)
    board[:, :, 0] = 28
    board[:, 0, 32] = 2
    return kinds, turn, meld, board


def test_model_masks_actions_and_backpropagates() -> None:
    model = KyokuTransformerActorCritic(ModelConfig.preset("mid"))
    kinds, turn, meld, board = v4_inputs(2, 3)
    legal = torch.zeros(2, 241, dtype=torch.bool)
    legal[:, [0, 10]] = True
    output = model(kinds, turn, meld, board, legal, torch.tensor([2, 2]))
    assert output["policy_logits"].shape == (2, 241)
    assert torch.isneginf(output["policy_logits"][:, 1]).all()
    (output["value"].mean() + output["policy_logits"][:, 0].mean()).backward()
    assert model.policy_head.weight.grad is not None


def test_right_padding_does_not_change_decision_output() -> None:
    torch.manual_seed(3)
    model = KyokuTransformerActorCritic(ModelConfig(16, 16, 1, 2, 32)).eval()
    short = v4_inputs(1, 3)
    padded = v4_inputs(1, 7)
    for source, target in zip(short[:3], padded[:3]):
        target[:, :3] = source
    padded = (*padded[:3], short[3])
    legal = torch.ones(1, 241, dtype=torch.bool)
    with torch.no_grad():
        a = model(*short, legal, torch.tensor([2]))
        b = model(*padded, legal, torch.tensor([2]))
    torch.testing.assert_close(a["raw_policy_logits"], b["raw_policy_logits"])
    torch.testing.assert_close(a["value"], b["value"])


def test_padded_batch_matches_individual_full_forwards() -> None:
    torch.manual_seed(9)
    model = KyokuTransformerActorCritic(ModelConfig(16, 16, 2, 2, 32)).eval()
    kinds, turn, meld, board = v4_inputs(3, 5)
    lengths = torch.tensor([1, 2, 4])
    kinds[0, 1:] = 0
    turn[0, 1:] = 0
    meld[0, 1:] = 0
    kinds[1, 2:] = 0
    turn[1, 2:] = 0
    meld[1, 2:] = 0
    kinds[2, 2:4] = torch.tensor([1, 2])
    turn[2, 2, 0] = torch.tensor([1, 2, 14, 0])
    meld[2, 3] = torch.tensor([2, 3, 4, 14, 14, 14, 0, 0])
    legal = torch.zeros(3, 241, dtype=torch.bool)
    legal[0, [0, 7]] = True
    legal[1, [3, 11]] = True
    legal[2, [9, 20]] = True
    with torch.no_grad():
        padded = model(kinds, turn, meld, board, legal, lengths)
        for row, length in enumerate(lengths.tolist()):
            single = model(
                kinds[row : row + 1, :length], turn[row : row + 1, :length], meld[row : row + 1, :length],
                board[row : row + 1], legal[row : row + 1], torch.tensor([length]),
            )
            torch.testing.assert_close(padded["raw_policy_logits"][row], single["raw_policy_logits"][0])
            torch.testing.assert_close(padded["policy_logits"][row], single["policy_logits"][0])
            torch.testing.assert_close(padded["value"][row], single["value"][0])
            assert bool(legal[row, padded["policy_logits"][row].argmax()])
