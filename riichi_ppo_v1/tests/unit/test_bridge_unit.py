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
    assert tile_id_to_mjai(16) == "5mr"
    assert tile_id_to_mjai(52) == "5pr"
    assert tile_id_to_mjai(108) == "E"
    actions = [json.loads(value) for value in action_jsons(Observation())]
    assert actions[0]["tsumogiri"] is True
    assert actions[1]["tsumogiri"] is False
