"""V18 副露/赤牌/杠后决策模式的字段级集成测试（B9：红牌/kakan/chi 形状/实体去重证据）。"""

from __future__ import annotations

import numpy as np

from riichi_ppo_v1.tests.v18_fixtures import make_observation


def _rows(obs: object) -> np.ndarray:
    import riichienv
    return np.asarray(
        riichienv.prepare_current_state_batch([obs]).rows
    ).reshape(-1, 32)


def _meld(meld_type: str, tiles: list[int], *, from_who: int = 1,
          called: int | None = None, called_index: int | None = None,
          opened: bool = True) -> dict[str, object]:
    return {
        "meld_type": meld_type, "tiles": tiles, "opened": opened,
        "from_who": from_who, "called_tile": called, "called_tile_index": called_index,
    }


def test_red_five_fields_across_actor_categories() -> None:
    obs = make_observation(
        hands=[[16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [], [], []],
        discards=[[], [52], [], []],
        melds=[[], [], [_meld("Pon", [16, 17, 18], called=16, called_index=0)], []],
        dora_indicators=[88],
    )
    rows = _rows(obs)
    table = rows[rows[:, 1] == 2][0]
    assert int(table[18]) == 1  # dora_indicator_red_slot_1
    self_hand = rows[rows[:, 1] == 3]
    five_m = [row for row in self_hand if int(row[2]) == 5]
    assert five_m and int(five_m[0][4]) == 1  # has_red
    discard = rows[rows[:, 1] == 7][0]
    assert int(discard[5]) == 1  # RIVER_DISCARD red
    summary = rows[rows[:, 1] == 6][0]
    assert int(summary[4]) == 1  # FIRST_SIX slot1 red
    meld_rows = rows[rows[:, 1] == 8]
    assert int(meld_rows[0][3]) == 2  # meld_type_code=pon
    assert int(meld_rows[0][5]) == 1  # tile0_red


def test_meld_type_codes_and_kan_decisions() -> None:
    # 副露类型：chi/pon/daiminkan/ankan/kakan。
    for meld_type, expected_code, opened in (
        ("Chi", 1, True), ("Pon", 2, True), ("Daiminkan", 3, True),
        ("Ankan", 4, False), ("Kakan", 5, True),
    ):
        tiles = [0, 4, 8] if expected_code == 1 else [16, 17, 18, 19]
        if expected_code == 2:
            tiles = [16, 17, 18]
        obs = make_observation(melds=[[], [], [_meld(
            meld_type, tiles, from_who=1, called=(tiles[0] if expected_code != 4 else None),
            called_index=(0 if expected_code != 4 else None), opened=opened,
        )], []])
        rows = _rows(obs)
        meld_rows = rows[rows[:, 1] == 8]
        assert int(meld_rows[0][3]) == expected_code
        assert int(meld_rows[0][15]) == int(opened)
        assert int(meld_rows[0][2]) == 2  # owner_relative（观察者 0 的对面）

    # kakan 末事件 → decision_mode=2，且自身暗牌数 = 13 + pending - 4×杠。
    import json

    obs = make_observation(
        melds=[[_meld("Kakan", [16, 17, 18, 19], from_who=1, called=16, called_index=0)], [], [], []],
        events=[json.dumps({"type": "kakan", "actor": 0})],
    )
    rows = _rows(obs)
    table = rows[rows[:, 1] == 2][0]
    assert int(table[8]) == 2  # decision_mode
    player_self = rows[rows[:, 1] == 5][0]
    assert int(player_self[7]) == 10  # 13 + 1 - 4×1

    # ankan 后（无摸牌事件）静置：自身暗牌数 13 - 4×1 = 9。
    obs = make_observation(
        melds=[
            [_meld("Ankan", [16, 17, 18, 19], from_who=-1, called=None, called_index=None, opened=False)],
            [], [], [],
        ],
    )
    rows = _rows(obs)
    player_self = rows[rows[:, 1] == 5][0]
    assert int(player_self[7]) == 9


def test_three_chi_shapes_encode_consume_identity() -> None:
    # 三种吃牌形状：叫牌在低/中/高位置，MELD called_tile_type 必须对应被鸣牌。
    cases = [
        ([0, 4, 8], 0),   # 1m2m3m，吃 1m（低）
        ([0, 4, 8], 4),   # 1m2m3m，吃 2m（中）
        ([0, 4, 8], 8),   # 1m2m3m，吃 3m（高）
    ]
    for tiles, called in cases:
        obs = make_observation(
            melds=[[], [], [_meld("Chi", tiles, called=called, called_index=0)], []],
        )
        rows = _rows(obs)
        meld_row = rows[rows[:, 1] == 8][0]
        assert int(meld_row[3]) == 1  # chi
        assert int(meld_row[12]) == int(called) // 4 + 1  # called_tile_type


def test_supplied_marks_exact_claimed_index_only() -> None:
    # 同牌种两张河牌只一张被鸣：investigate river marks。
    obs = make_observation(
        discards=[[], [108, 108, 104], [], []],
        melds=[[], [], [_meld("Pon", [108, 108, 109], called=108, called_index=0)], []],
    )
    rows = _rows(obs)
    river = rows[rows[:, 1] == 7]
    marks = [(int(row[3]), int(row[8])) for row in river]
    assert marks == [(1, 1), (2, 0), (3, 0)]
