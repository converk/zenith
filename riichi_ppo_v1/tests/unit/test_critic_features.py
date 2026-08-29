"""Critic 私有行新行宽编码测试。"""



from riichi_ppo_v1.model.critic_features import (
    TableState,
    encode_critic_features,
    encode_future_wall_tokens,
    encode_opponent_hand_tokens,
    tile_id_to_type,
)
from riichi_ppo_v1.model.encoding_protocol import (
    KIND_CRITIC_FUTURE,
    KIND_CRITIC_HAND,
    KIND_SEP_CRITIC,
    SEGMENT_CRITIC_FUTURE,
    SEGMENT_CRITIC_PRIVATE,
    TOKEN_ROW_WIDTH,
)


def test_tile_id_to_type() -> None:
    assert tile_id_to_type(0) == 0
    assert tile_id_to_type(16) == 4  # 红五并入五
    assert tile_id_to_type(17) == 4
    assert tile_id_to_type(200) is None


def test_opponent_hand_rows_keep_red() -> None:
    table = TableState(
        hands=(
            (),
            (16, 17, 18),  # 下家：红五 + 两张
            (52,),
            (88,),
        )
    )
    rows = encode_opponent_hand_tokens(table, 0)
    # 5m 红/普拆成两行（3 行手牌 → 4 个 (kind,red) 组合）。
    assert len(rows) == 4
    for row in rows:
        assert row[0] == SEGMENT_CRITIC_PRIVATE
        assert row[1] == KIND_CRITIC_HAND
    # 下家红五行 red=1（同一座次的 (kind=5, red=1) 行）。
    shimo_red = [row for row in rows if row[2] == 1 and row[4] == 1]
    assert shimo_red and shimo_red[0][3] == 5 and shimo_red[0][5] == 1


def test_future_wall_positions_ordered() -> None:
    rows = encode_future_wall_tokens([16, 17, 18, 19, 20])
    assert len(rows) == 5
    assert [row[2] for row in rows] == [1, 2, 3, 4, 5]
    assert [row[1] for row in rows] == [KIND_CRITIC_FUTURE] * 5
    assert all(row[0] == SEGMENT_CRITIC_FUTURE for row in rows)
    assert rows[0][4] == 1  # 红五


def test_encode_critic_features_row_width() -> None:
    table = TableState(hands=((), (1, 2, 3), (4, 5), (6, 7, 8, 9)))
    features = encode_critic_features(table, 0, future_wall_tiles=[1, 2, 3, 4, 5])
    assert features.factors.shape[1] == TOKEN_ROW_WIDTH
    assert features.factors[0, 1] == KIND_SEP_CRITIC
    # 手牌 (1,2,3)→1 行，(4,5)→1 行，(6,7,8,9)→2 行。
    assert features.length == 1 + 4 + 5
    kinds = features.factors[:, 1].astype(int).tolist()
    assert kinds[0] == KIND_SEP_CRITIC
    assert kinds[1:5] == [KIND_CRITIC_HAND] * 4
    assert kinds[5:] == [KIND_CRITIC_FUTURE] * 5
