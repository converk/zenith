"""V17 GRP(Mortal 方案)模型契约测试:7 维输入、24 类排列、calc_matrix、冻结。"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch
from torch.nn import functional as F

from riichi_ppo_v1.model.grp import (
    GRPModel,
    GRP_INPUT_SIZE,
    GRP_NUM_CLASSES,
    GRP_UTILITY,
    expected_rank_utility,
)
from riichi_ppo_v1.training.grp.prepare import (
    Boundary,
    KyokuResult,
    features_from_boundaries,
    final_scores,
    grand_kyoku,
    parse_hanchan,
    rank_among,
    rank_by_player,
)


def _boundaries() -> list[Boundary]:
    start = KyokuResult(
        result_type=1, winner=0, deal_in=2, tenpai_mask=0b0101,
        deltas=(6000, -2000, -2000, -2000),
    )
    return [
        Boundary(0, 0, 0, 0, 0, (25000, 25000, 25000, 25000), None),
        Boundary(0, 1, 0, 1, 0, (31000, 23000, 23000, 23000), start),
    ]


def test_grand_kyoku_matches_mortal() -> None:
    # E1=0, E4=3, S1=4, S4=7
    assert grand_kyoku(0, 0) == 0
    assert grand_kyoku(0, 3) == 3
    assert grand_kyoku(1, 0) == 4
    assert grand_kyoku(1, 3) == 7


def test_features_from_boundaries_seven_dims() -> None:
    boundaries = _boundaries()
    features = features_from_boundaries(boundaries)
    assert features.shape == (2, 7)
    # 首局 2.5 / 25000;第二局 3.1 / 23000。
    assert np.allclose(features[0, 3:], 2.5)
    assert np.allclose(features[1, 3:], [3.1, 2.3, 2.3, 2.3])
    assert features[1, 0] == 1.0  # E2
    assert features[1, 1] == 1.0  # honba
    assert features[1, 2] == 0.0  # kyotaku


def test_rank_by_player_follows_stable_sort() -> None:
    scores = (31000, 23000, 23000, 23000)
    assert rank_by_player(scores) == (0, 1, 2, 3)
    assert rank_among(1, (23000, 31000, 23000, 23000)) == 0
    # 同分按座位号稳定:seat3 最高拿 rank0,余下 23000 中 seat0→1、seat1→2、seat2→3。
    assert rank_by_player((23000, 23000, 23000, 31000)) == (1, 2, 3, 0)


def test_prediction_is_twenty_four_classes() -> None:
    model = GRPModel()
    features = torch.randn(4, 8, GRP_INPUT_SIZE)
    lengths = torch.tensor([3, 5, 8, 4])
    logits = model(features, lengths)
    assert logits.shape == (4, GRP_NUM_CLASSES)
    assert logits.dtype == torch.float32
    assert model.perms.shape == (24, 4)


def test_calc_matrix_rows_sum_to_one() -> None:
    model = GRPModel()
    logits = torch.randn(5, 24)
    matrix = model.calc_matrix(logits)
    assert matrix.shape == (5, 4, 4)
    # 每个玩家(行)的排名概率和为 1。
    assert torch.allclose(matrix.sum(-1), torch.ones(5, 4), atol=1e-5)
    # 与 Mortal 语义等价:matrix[player][rank] = Σ_{perm[player]==rank} p(perm)。
    probs = torch.softmax(logits, dim=-1)
    for player in range(4):
        for rank in range(4):
            expected = probs[:, [
                index for index, perm in enumerate(permutations(range(4)))
                if perm[player] == rank
            ]].sum(-1)
            torch.testing.assert_close(matrix[:, player, rank], expected, rtol=1e-5, atol=1e-6)


def test_get_label_matches_permutation_mapping() -> None:
    model = GRPModel()
    rank_by_player = torch.tensor([[0, 1, 2, 3], [2, 0, 1, 3]], dtype=torch.long)
    labels = model.get_label(rank_by_player)
    perms = list(permutations(range(4)))
    assert labels[0].item() == perms.index((0, 1, 2, 3))
    assert labels[1].item() == perms.index((2, 0, 1, 3))
    assert labels[1].item() == 12


def test_expected_rank_utility() -> None:
    model = GRPModel()
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    matrix = model.calc_matrix(logits)
    # 排列 (0,1,2,3) 概率≈1 → matrix[0,0]=1。
    utility = expected_rank_utility(matrix)
    assert torch.allclose(utility[0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(utility[0, 3], torch.tensor(-1.0), atol=1e-5)
    assert GRP_UTILITY == (1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0)


def test_model_parameter_budget() -> None:
    model = GRPModel()
    total = sum(parameter.numel() for parameter in model.parameters())
    assert 30_000 <= total <= 60_000, f"total={total}"


def test_training_loss_backpropagates() -> None:
    model = GRPModel()
    features = torch.randn(3, 6, GRP_INPUT_SIZE)
    lengths = torch.tensor([4, 6, 5])
    rank_by_player = torch.tensor([[0, 1, 2, 3], [1, 0, 2, 3], [3, 2, 1, 0]])
    logits = model(features, lengths)
    labels = model.get_label(rank_by_player)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert model.rnn.weight_ih_l0.grad is not None
    assert model.fc[-1].weight.grad is not None


def test_freeze_grp_weights_do_not_change() -> None:
    model = GRPModel()
    model.freeze()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    assert all(not parameter.requires_grad for parameter in model.parameters())
    features = torch.randn(1, GRP_INPUT_SIZE + 2, GRP_INPUT_SIZE)
    with torch.no_grad():
        model(features, torch.tensor([3]))
    probe = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([probe], lr=0.1)
    optimizer.zero_grad()
    probe.backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_parse_hanchan_produces_boundaries() -> None:
    content = "\n".join([
        '{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "oya": 0, "honba": 0, "kyotaku": 0, "scores": [25000, 25000, 25000, 25000]}',
        '{"type": "hora", "actor": 0, "target": 2, "deltas": [6000, -2000, -2000, -2000]}',
        '{"type": "start_kyoku", "bakaze": "E", "kyoku": 2, "oya": 0, "honba": 1, "kyotaku": 0, "scores": [31000, 23000, 23000, 23000]}',
    ])
    boundaries = parse_hanchan(content)
    assert len(boundaries) == 2
    assert boundaries[1].honba == 1
    finals = final_scores(boundaries)
    # 终局分数 = 最后一局边界分数 + 上一局 deltas。
    assert finals == (31000 + 6000, 23000 - 2000, 23000 - 2000, 23000 - 2000)
    assert rank_by_player(finals) == (0, 1, 2, 3)


def test_prepare_grp_dataset_max_shards_limits_input_tars() -> None:
    """``max_shards`` 截断每个 split 处理的 tar 数量,并写入 dataset.json。"""
    import io
    import tarfile
    import tempfile
    from pathlib import Path

    from riichi_ppo_v1.training.grp.prepare import prepare_grp_dataset

    hanchan = "\n".join([
        '{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "oya": 0, "honba": 0, "kyotaku": 0, "scores": [25000, 25000, 25000, 25000]}',
        '{"type": "hora", "actor": 0, "target": 1, "deltas": [6000, -2000, -2000, -2000]}',
        '{"type": "end_game", "scores": [31000, 23000, 23000, 23000]}',
    ])

    def make_tar(path: Path, members: list[str]) -> None:
        with tarfile.open(path, "w") as tar:
            for index, content in enumerate(members):
                payload = content.encode("utf-8")
                # game_id 含 tar 序号与成员序号,避免跨 tar 同名聚合。
                info = tarfile.TarInfo(
                    f"2024-2024010100gm-00aa-0000-AAA{path.stem[-5:]}{index:02d}-00.mjson"
                )
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "src"
        (source / "train").mkdir(parents=True)
        (source / "validation").mkdir(parents=True)
        for shard_index in range(3):
            make_tar(
                source / "train" / f"train-{shard_index:05d}.tar",
                [hanchan, hanchan, hanchan],
            )
            make_tar(
                source / "validation" / f"validation-{shard_index:05d}.tar",
                [hanchan],
            )
        output = root / "grp"
        dataset = prepare_grp_dataset(
            source, output,
            denominator=1, remainders=(0,), kyokus_per_shard=2, max_shards=2,
        )
        # 只处理前 2 个 tar:train=6 半庄(train 3 成员 × 2 tar),validation=2。
        assert dataset["counts"]["train_hanchans"] == 6
        assert dataset["counts"]["validation_hanchans"] == 2
        assert dataset["max_shards"] == 2
        # chunk 编号只到 1(6 半庄 / kyokus_per_shard=2 → 3 chunk:0,1,2)。
        from pathlib import Path as _Path
        train_chunks = sorted((output / "train").glob("train-*.npz"))
        assert len(train_chunks) == 3