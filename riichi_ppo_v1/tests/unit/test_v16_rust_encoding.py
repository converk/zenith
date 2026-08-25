"""V16 Rust 融合 Action Query 编码回归。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from riichi_ppo_v1.model.action_query import analyze_action_queries_batch, encode_query_row
from riichi_ppo_v1.model.v16_rust_encoding import encode_action_queries_batch_rust


class _Action:
    def __init__(
        self,
        kind: str,
        *,
        tile: int | None = None,
        consume_tiles: tuple[int, ...] = (),
    ) -> None:
        self.action_type = kind
        self.tile = tile
        self.consume_tiles = consume_tiles
        self._kind = kind

    def to_mjai(self) -> str:
        kind = "hora" if self._kind in {"tsumo", "ron"} else self._kind
        return json.dumps({"type": kind})


def _physical_tiles(types: list[int]) -> list[int]:
    copies: dict[int, int] = {}
    out: list[int] = []
    for tile_type in types:
        copy = copies.get(tile_type, 0)
        if copy >= 4:
            raise ValueError(f"too many copies for tile type {tile_type}")
        out.append(4 * tile_type + copy)
        copies[tile_type] = copy + 1
    return out


def _observation(
    hand_types: list[int],
    *,
    own_melds: list[object] | None = None,
) -> SimpleNamespace:
    hand = _physical_tiles(hand_types)
    melds = own_melds or []
    return SimpleNamespace(
        player_id=0,
        hands=[hand, [], [], []],
        melds=[melds, [], [], []],
        discards=[[], [12, 40], [76], [108]],
        dora_indicators=[20],
        scores=[25_000, 25_000, 25_000, 25_000],
        oya=0,
        round_wind=0,
        honba=0,
        riichi_sticks=0,
        riichi_declared=[False, False, False, False],
        missed_agari_doujun=False,
        missed_agari_riichi=False,
        drawn_tile=hand[-1],
        last_discard=(3, 108),
    )


def test_rust_fused_rows_match_python_oracle_for_all_action_kinds() -> None:
    far_hand = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31]
    base = _observation(far_hand)
    chi_obs = _observation([0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 27, 29, 31, 33])
    pon_obs = _observation([0, 0, 4, 7, 10, 13, 16, 19, 22, 25, 27, 29, 31, 33])
    kan_obs = _observation([0, 0, 0, 0, 4, 7, 10, 13, 16, 19, 22, 27, 29, 31])
    open_pon = SimpleNamespace(tiles=_physical_tiles([0, 0, 0]), opened=True)
    kakan_obs = _observation(
        [0, 4, 7, 10, 13, 16, 19, 22, 27, 29, 31], own_melds=[open_pon],
    )

    actions = [
        (base, _Action("tsumo", tile=base.drawn_tile)),
        (base, _Action("ron", tile=108)),
        (base, _Action("reach", tile=base.hands[0][0])),
        (base, _Action("dahai", tile=base.hands[0][1])),
        (chi_obs, _Action("chi", tile=8, consume_tiles=(0, 4))),
        (pon_obs, _Action("pon", tile=2, consume_tiles=(0, 1))),
        (kan_obs, _Action("ankan", tile=0, consume_tiles=(0, 1, 2, 3))),
        (kan_obs, _Action("daiminkan", tile=0, consume_tiles=(0, 1, 2))),
        (kakan_obs, _Action("kakan", tile=0, consume_tiles=(0,))),
        (base, _Action("none")),
        (base, _Action("pass")),
        (base, _Action("ryukyoku")),
    ]
    rows = [(observation, action, index + 1) for index, (observation, action) in enumerate(actions)]

    oracle_pairs = analyze_action_queries_batch(rows)
    oracle = np.stack([
        np.stack((encode_query_row(offense), encode_query_row(defense)))
        for offense, defense in oracle_pairs
    ])
    rust = encode_action_queries_batch_rust(rows)

    np.testing.assert_array_equal(rust.query_rows, oracle)
    assert rust.query_rows.dtype == np.int32
    assert rust.query_rows.shape == (len(rows), 2, 15)


def test_rust_fused_batch_deduplicates_identical_shapes_without_dropping_rows() -> None:
    observation = _observation([0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29, 31, 33])
    rows = [
        (observation, _Action("dahai", tile=observation.hands[0][copy]), 1 + copy)
        for copy in range(2)
    ]
    rust = encode_action_queries_batch_rust(rows)

    assert rust.query_rows.shape[0] == 2
    assert rust.unique_offense_rows == 1
    assert not np.array_equal(rust.query_rows[0, :, 1], rust.query_rows[1, :, 1])
