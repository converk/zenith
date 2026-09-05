import json

from riichi_ppo_v1.model.bridge import (
    action_jsons,
    tile_id_to_mjai,
)
from riichi_ppo_v1.model.critic_features import (
    SEGMENT_CRITIC_PRIVATE,
    TableState,
    encode_critic_features,
)
from riichi_ppo_v1.model.encoding_protocol import (
    KIND_CRITIC_HAND,
    KIND_SEP_CRITIC,
)


class Action:
    def __init__(self, tile: int, pai: str) -> None:
        self.tile = tile
        self.pai = pai

    def to_mjai(self) -> str:
        return json.dumps({"type": "dahai", "pai": self.pai})


class Observation:
    drawn_tile = 16

    def legal_actions(self):
        return [Action(16, "5mr"), Action(20, "6m")]


def test_tile_conversion_and_physical_tsumogiri() -> None:
    def expected(tile: int) -> str:
        if tile == 16:
            return "5mr"
        if tile == 52:
            return "5pr"
        if tile == 88:
            return "5sr"
        if tile < 108:
            suit = "mps"[tile // 36]
            rank = (tile % 36) // 4 + 1
            return f"{rank}{suit}"
        return ("E", "S", "W", "N", "P", "F", "C")[(tile - 108) // 4]

    assert tile_id_to_mjai(None) is None
    for tile in range(136):
        assert tile_id_to_mjai(tile) == expected(tile)
    actions = [json.loads(value) for value in action_jsons(Observation())]
    assert actions[0]["tsumogiri"] is True
    assert actions[1]["tsumogiri"] is False


def test_critic_contains_three_hands_without_future() -> None:
    table = TableState(((0, 1, 2), (16, 17), (52,), (108,)))
    critic = encode_critic_features(table, observer=0)
    assert critic.length == 1 + 4
    segments = critic.factors[:, 0].tolist()
    assert segments == [SEGMENT_CRITIC_PRIVATE] * 5
    kinds = critic.factors[:, 1].tolist()
    # 三家：SEP_CRITIC + 下家两行(5m 红/普) + 对家一行 + 上家一行。
    assert kinds[0] == KIND_SEP_CRITIC
    assert len([value for value in kinds[1:] if value == KIND_CRITIC_HAND]) == 4
