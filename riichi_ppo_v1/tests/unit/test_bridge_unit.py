import json

from riichi_ppo_v1.model.bridge import action_jsons, tile_id_to_mjai


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
