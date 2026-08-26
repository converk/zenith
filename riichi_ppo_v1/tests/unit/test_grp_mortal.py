"""V18 GRP(Mortal 方案扩展)契约测试:21 维输入、24 类排列、calc_matrix、冻结。

新增覆盖:局风类型映射、上局结果类型、各玩家累计和了/放铳/听牌流局计数、
离线(features_from_boundaries)与在线(feature_row + 累计推进)特征逐位一致。
"""

from __future__ import annotations

import json
from itertools import permutations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from riichi_ppo_v1.model.grp import (
    GAME_TYPE_HALF,
    GRPModel,
    GRP_INPUT_LAYOUT,
    GRP_INPUT_SIZE,
    GRP_NUM_CLASSES,
    GRP_UTILITY,
    expected_rank_utility,
)
from riichi_ppo_v1.training.grp.prepare import (
    Boundary,
    KyokuResult,
    feature_row,
    features_from_boundaries,
    final_scores,
    game_type_from_content,
    game_type_from_mode,
    grand_kyoku,
    parse_hanchan,
    rank_among,
    rank_by_player,
    result_increment,
)
from riichi_ppo_v1.training.worker import GrpRollout


def _boundaries() -> list[Boundary]:
    """两局:首局 2.5/2.5/2.5/2.5,然后 E1 荣和(0 和了、2 放铳)→ E2 1 本场。"""
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


def test_input_layout_has_twenty_one_names() -> None:
    assert GRP_INPUT_SIZE == 21
    assert len(GRP_INPUT_LAYOUT) == GRP_INPUT_SIZE
    assert GRP_INPUT_LAYOUT[:9] == (
        "grand_kyoku", "honba", "kyotaku",
        "s0", "s1", "s2", "s3",
        "game_type", "prev_result_type",
    )
    assert GRP_INPUT_LAYOUT[9:13] == ("wins0", "wins1", "wins2", "wins3")
    assert GRP_INPUT_LAYOUT[13:17] == ("dealins0", "dealins1", "dealins2", "dealins3")
    assert GRP_INPUT_LAYOUT[17:] == ("tenpai0", "tenpai1", "tenpai2", "tenpai3")


def test_game_type_from_content() -> None:
    east = '\n'.join([
        '{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "oya": 0}',
        '{"type": "start_kyoku", "bakaze": "E", "kyoku": 2, "oya": 1}',
    ])
    half = '\n'.join([
        '{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "oya": 0}',
        '{"type": "start_kyoku", "bakaze": "S", "kyoku": 1, "oya": 0}',
    ])
    west = '\n'.join([
        '{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "oya": 0}',
        '{"type": "start_kyoku", "bakaze": "W", "kyoku": 1, "oya": 0}',
    ])
    assert game_type_from_content(east) == 0
    assert game_type_from_content(half) == 1
    assert game_type_from_content(west) == 2
    with pytest.raises(ValueError):
        game_type_from_content('{"type": "end_game"}')


def test_game_type_from_mode() -> None:
    assert game_type_from_mode("4p-red-half") == GAME_TYPE_HALF
    assert game_type_from_mode("4p-red-east") == 0
    assert game_type_from_mode("4p-red-single") == 0
    assert game_type_from_mode("4p-red-west") == 2
    with pytest.raises(ValueError):
        game_type_from_mode("4p-red-unknown")


def test_result_increment() -> None:
    ron = KyokuResult(1, 0, 2, 0, (6000, -2000, -2000, -2000))
    tsumo = KyokuResult(2, 3, None, 0, (0, 0, 0, 8000))
    ryukyoku = KyokuResult(3, None, None, 0b0101, (0, 0, 0, 0))
    assert result_increment(ron) == ((1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0))
    assert result_increment(tsumo) == ((0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0))
    assert result_increment(ryukyoku) == ((0, 0, 0, 0), (0, 0, 0, 0), (1, 0, 1, 0))
    assert result_increment(None) == ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))


def test_features_from_boundaries_twenty_one_dims() -> None:
    boundaries = _boundaries()
    features = features_from_boundaries(boundaries, game_type=1)
    assert features.shape == (2, 21)
    # 前 7 维与 V17 完全一致:首局 2.5 / 25000;第二局 3.1 / 23000。
    assert np.allclose(features[0, 3:7], 2.5)
    assert np.allclose(features[1, 3:7], [3.1, 2.3, 2.3, 2.3])
    assert features[1, 0] == 1.0  # E2
    assert features[1, 1] == 1.0  # honba
    assert features[1, 2] == 0.0  # kyotaku
    # 新增字段:局风类型、上局结果类型(HALF / ron)。
    assert np.allclose(features[0, 7:9], [1.0, 0.0])
    assert np.allclose(features[1, 7:9], [1.0, 1.0])
    # 累计计数:首局全 0;第二局开局已含首局荣和(0 和了、2 放铳)。
    assert np.allclose(features[0, 9:21], 0.0)
    expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert np.allclose(features[1, 9:21], expected)


def test_cumulative_counts_advance_across_kyokus() -> None:
    """三局:荣和(0→2)、自摸(3)、流局(0、2 听牌)→ 第三行计数正确累计。"""
    boundaries = [
        Boundary(0, 0, 0, 0, 0, (25000, 25000, 25000, 25000), None),
        Boundary(0, 1, 0, 1, 0, (31000, 23000, 23000, 23000),
                 KyokuResult(1, 0, 2, 0, (6000, -2000, -2000, -2000))),
        Boundary(0, 2, 0, 2, 0, (31000, 23000, 23000, 31000),
                 KyokuResult(2, 3, None, 0, (-1000, -1000, -1000, 9000))),
    ]
    features = features_from_boundaries(boundaries, game_type=0)
    assert np.allclose(features[2, 9:21], [
        1.0, 0.0, 0.0, 1.0,   # wins
        0.0, 0.0, 1.0, 0.0,   # dealins
        0.0, 0.0, 0.0, 0.0,   # tenpai(本局无流局)
    ])
    # 追加一个流局小局:0、2 听牌。
    boundaries.append(Boundary(
        0, 3, 0, 2, 0, (31000, 23000, 23000, 30000),
        KyokuResult(3, None, None, 0b0101, (0, 0, 0, -1000)),
    ))
    features = features_from_boundaries(boundaries, game_type=0)
    assert np.allclose(features[3, 17:21], [1.0, 0.0, 1.0, 0.0])


def test_offline_online_feature_parity() -> None:
    """在线逐边界推进(feature_row + result_increment)与离线编码逐位一致。"""
    boundaries = _boundaries()
    offline = features_from_boundaries(boundaries, game_type=1)

    class StubGRP:
        def __call__(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            del lengths
            return torch.zeros((features.shape[0], 24))

        def calc_matrix(self, logits: torch.Tensor) -> torch.Tensor:
            matrix = torch.zeros((logits.shape[0], 4, 4))
            matrix.fill_(0.25)
            return matrix

    tracker = GrpRollout(StubGRP(), game_type=1)
    tracker.start_match(0, boundaries[0])
    tracker.boundary_reward(0, boundaries[1])
    online = np.asarray(tracker._sequences[0], dtype=np.float32)
    assert online.shape == offline.shape
    assert np.array_equal(online, offline)


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


def test_model_accepts_explicit_config_and_rejects_bad_shape() -> None:
    model = GRPModel(input_size=21, hidden_size=96, num_layers=2)
    logits = model(torch.randn(2, 5, 21), torch.tensor([3, 5]))
    assert logits.shape == (2, 24)
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 7), torch.tensor([3, 5]))


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
    assert 110_000 <= total <= 150_000, f"total={total}"


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
        '{"type": "start_kyoku", "bakaze": "E", "kyoku": 2, "oya": 1, "honba": 1, "kyotaku": 0, "scores": [31000, 23000, 23000, 23000]}',
    ])
    boundaries = parse_hanchan(content)
    assert len(boundaries) == 2
    assert boundaries[1].honba == 1
    finals = final_scores(boundaries)
    # 终局分数 = 最后一局边界分数 + 上一局 deltas。
    assert finals == (31000 + 6000, 23000 - 2000, 23000 - 2000, 23000 - 2000)
    assert rank_by_player(finals) == (0, 1, 2, 3)
    assert game_type_from_content(content) == 0


def test_prepare_grp_dataset_max_shards_limits_input_tars() -> None:
    """``max_shards`` 截断每个 split 处理的 tar 数量,dataset.json 记录 v18 契约。"""
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
        assert dataset["format"] == "riichi-grp-v18"
        assert dataset["input_size"] == 21
        assert dataset["game_types"] == {"east": 8, "half": 0, "west": 0}
        # chunk 编号只到 1(6 半庄 / kyokus_per_shard=2 → 3 chunk:0,1,2)。
        train_chunks = sorted((output / "train").glob("train-*.npz"))
        assert len(train_chunks) == 3


def test_prepare_grp_dataset_parallel_matches_serial() -> None:
    """``workers>1`` 与 ``workers=1`` 的输出(npz 内容与 dataset.json)逐位一致。"""
    import io
    import tarfile
    import tempfile
    from pathlib import Path

    from riichi_ppo_v1.training.grp.prepare import prepare_grp_dataset

    def member(year: int, game_id: str, kyoku: int, bakaze: str) -> str:
        return "\n".join([
            '{"type": "start_game", "names": ["a", "b", "c", "d"], "kyoku_first": 0, "aka_flag": true}',
            json.dumps({
                "type": "start_kyoku", "bakaze": bakaze, "kyoku": kyoku,
                "oya": (kyoku - 1) % 4, "honba": 0, "kyotaku": 0,
                "scores": [25000, 25000, 25000, 25000],
            }),
            '{"type": "tsumo", "actor": 0, "pai": "1m"}',
            '{"type": "hora", "actor": 0, "target": null, "deltas": [4000, -1000, -1000, -2000]}',
            '{"type": "end_kyoku"}',
            '{"type": "end_game", "scores": [29000, 24000, 24000, 23000]}',
        ])

    def make_tar(path: Path, members: list[tuple[str, str]]) -> None:
        with tarfile.open(path, "w") as tar:
            for name, content in members:
                payload = content.encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "src"
        (source / "train").mkdir(parents=True)
        (source / "validation").mkdir(parents=True)
        for shard_index in range(6):
            suffix = f"gm-00aa-0000-{shard_index:04d}"
            train_members: list[tuple[str, str]] = []
            # 每 shard:10 局,内含 1 局(东)、2 局(半庄,单成员 S)、
            # 3 局(半庄,双成员 E+S)与纯东风多局,覆盖聚合与局风统计。
            for game_index in range(10):
                gid = f"{suffix}-g{game_index:02d}"
                mode = game_index % 3
                if mode == 0:
                    train_members.append(
                        (f"2024-{gid}-00.mjson", member(2024, gid, 1, "E"))
                    )
                elif mode == 1:
                    train_members.append(
                        (f"2024-{gid}-00.mjson", member(2024, gid, 1, "S"))
                    )
                else:
                    train_members.append(
                        (f"2024-{gid}-00.mjson", member(2024, gid, 1, "E"))
                    )
                    train_members.append(
                        (f"2024-{gid}-01.mjson", member(2024, gid, 2, "S"))
                    )
            make_tar(
                source / "train" / f"train-{shard_index:05d}.tar", train_members,
            )
            make_tar(
                source / "validation" / f"validation-{shard_index:05d}.tar",
                train_members[:1],
            )
        serial = prepare_grp_dataset(
            source, root / "grp_serial",
            denominator=1, remainders=(0,), kyokus_per_shard=8, workers=1,
        )
        parallel = prepare_grp_dataset(
            source, root / "grp_parallel",
            denominator=1, remainders=(0,), kyokus_per_shard=8, workers=3,
        )
        assert serial["counts"] == parallel["counts"]
        assert serial["game_types"] == parallel["game_types"]
        # train:每 shard 4 东风 + 6 半庄;validation:每 shard 1 东风。
        assert serial["game_types"] == {"east": 30, "half": 36, "west": 0}
        for split in ("train", "validation"):
            serial_chunks = sorted((root / "grp_serial" / split).glob("*.npz"))
            parallel_chunks = sorted((root / "grp_parallel" / split).glob("*.npz"))
            assert [path.name for path in serial_chunks] == [
                path.name for path in parallel_chunks
            ]
            for path_serial, path_parallel in zip(
                serial_chunks, parallel_chunks, strict=True,
            ):
                with np.load(path_serial, allow_pickle=False) as a, np.load(
                    path_parallel, allow_pickle=False
                ) as b:
                    for key in ("offsets", "features", "rank_by_player", "years", "game_ids"):
                        assert np.array_equal(a[key], b[key]), (
                            split, path_serial.name, key,
                        )


def test_prepare_grp_dataset_merges_games_spanning_shards() -> None:
    """半庄被 shard 边界切断时,分组构造仍只产出一条完整记录(跨 tar 聚合)。"""
    import io
    import tarfile
    import tempfile
    from pathlib import Path

    from riichi_ppo_v1.training.grp.prepare import prepare_grp_dataset

    def kyoku(game_id: str, index: int, bakaze: str = "E") -> str:
        return "\n".join([
            '{"type": "start_game", "names": ["a", "b", "c", "d"], "kyoku_first": 0, "aka_flag": true}',
            json.dumps({
                "type": "start_kyoku", "bakaze": bakaze, "kyoku": index + 1,
                "oya": index % 4, "honba": 0, "kyotaku": 0,
                "scores": [25000, 25000, 25000, 25000],
            }),
            '{"type": "tsumo", "actor": 0, "pai": "1m"}',
            '{"type": "hora", "actor": 0, "target": null, "deltas": [4000, -1000, -1000, -2000]}',
            '{"type": "end_kyoku"}',
            '{"type": "end_game", "scores": [29000, 24000, 24000, 23000]}',
        ])

    def add_member(tar: tarfile.TarFile, name: str, content: str) -> None:
        payload = content.encode("utf-8")
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "src"
        (source / "train").mkdir(parents=True)
        (source / "validation").mkdir(parents=True)
        spanning = "gm-span-0000"
        with tarfile.open(source / "train" / "train-00000.tar", "w") as tar:
            add_member(tar, "2024-gm-a-00.mjson", kyoku("gm-a", 0))
            # 跨 shard 半庄的最后一个成员落在 shard 0 末尾。
            add_member(tar, f"2024-{spanning}-00.mjson", kyoku(spanning, 0))
        with tarfile.open(source / "train" / "train-00001.tar", "w") as tar:
            # 同一半庄其余成员落在 shard 1 开头(k 编号排序后按顺序拼接)。
            add_member(tar, f"2024-{spanning}-01.mjson", kyoku(spanning, 1, "S"))
            add_member(tar, f"2024-{spanning}-02.mjson", kyoku(spanning, 2, "S"))
            add_member(tar, "2024-gm-b-00.mjson", kyoku("gm-b", 0))
        with tarfile.open(source / "validation" / "validation-00000.tar", "w") as tar:
            add_member(tar, "2024-gm-c-00.mjson", kyoku("gm-c", 0))
        dataset = prepare_grp_dataset(
            source, root / "grp",
            denominator=1, remainders=(0,), kyokus_per_shard=64, workers=2,
        )
        assert dataset["counts"]["train_hanchans"] == 3
        with np.load(root / "grp" / "train" / "train-00000.npz", allow_pickle=False) as data:
            ids = data["game_ids"].tolist()
            # 跨 shard 半庄只出现一次,且 3 个小局完整合并在一条记录里。
            assert ids.count(spanning) == 1
            offsets = data["offsets"].tolist()
            pos = ids.index(spanning)
            assert offsets[pos + 1] - offsets[pos] == 3  # +0 在 shard0、+1/+2 在 shard1
        # train:gm-a(东) + spanning(东→南,半庄) + gm-b(东);validation:gm-c(东)。
        assert dataset["game_types"] == {"east": 3, "half": 1, "west": 0}
