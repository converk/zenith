import json
from types import SimpleNamespace

import numpy as np

from riichi_ppo_v1.model.bridge import (
    BatchedStateBridge,
    Decision,
    action_jsons,
    tile_id_to_mjai,
)
from riichi_ppo_v1.model.critic_features import (
    SEGMENT_CRITIC_FUTURE_WALL,
    SEGMENT_CRITIC_PRIVATE,
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


class StubStateMachine:
    def apply_events_batch(self, indices, events):
        return [False for _ in indices], [False for _ in indices]

    def prepare_decisions(self, batch_indices, legal_actions, snapshots):
        count = len(batch_indices)
        factors = np.zeros((count, 1, 10), dtype=np.uint8)
        numeric = np.zeros((count, 1, 8), dtype=np.float32)
        lengths = np.ones(count, dtype=np.int64)
        mask = np.ones((count, 241), dtype=np.bool_)
        generations = np.zeros(count, dtype=np.int64)
        return factors, numeric, lengths, mask, generations

    def decode_actions(self, indices, action_ids):
        return ["{\"type\":\"dahai\",\"pai\":\"1m\"}" for _ in indices]


def observation(seat: int, hand: list[int]) -> SimpleNamespace:
    hands = [[], [], [], []]
    hands[seat] = hand
    return SimpleNamespace(
        player_id=seat,
        oya=0,
        round_wind=0,
        kyoku_index=0,
        honba=0,
        riichi_sticks=0,
        scores=[25000] * 4,
        dora_indicators=[0],
        hands=hands,
        drawn_tile=None,
        riichi_declared=[False] * 4,
        discards=[[] for _ in range(4)],
        melds=[[] for _ in range(4)],
        legal_actions=lambda: [Action(16, "5mr")],
    )


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


def test_bridge_appends_ordered_future_wall_tokens_only_when_walls_given() -> None:
    bridge = BatchedStateBridge(StubStateMachine(), 1)
    bridge.observations_by_env = {
        0: {
            seat: observation(seat, hand)
            for seat, hand in enumerate(([0, 1, 2], [16, 17], [52], [108]))
        },
    }
    decisions = [Decision(0, 0, bridge.observations_by_env[0][0])]
    wall = [134, 41, 119, 67, 90]

    _factors, _numeric, _lengths, _mask, _gens, critic_factors, critic_lengths = (
        bridge.prepare(decisions, walls=[wall])
    )
    assert critic_lengths.tolist() == [4 + 5]
    segments = critic_factors[0, :9, 0].tolist()
    assert segments[:4] == [SEGMENT_CRITIC_PRIVATE] * 4
    assert segments[4:] == [SEGMENT_CRITIC_FUTURE_WALL] * 5
    positions = critic_factors[0, 4:9, 3].tolist()
    assert positions == [1, 2, 3, 4, 5]

    _factors, _numeric, _lengths, _mask, _gens, critic_factors, critic_lengths = (
        bridge.prepare(decisions, walls=None)
    )
    assert critic_lengths.tolist() == [4]
    assert np.all(critic_factors[0, :4, 0] == SEGMENT_CRITIC_PRIVATE)
    assert not np.any(critic_factors[0, :, 0] == SEGMENT_CRITIC_FUTURE_WALL)
