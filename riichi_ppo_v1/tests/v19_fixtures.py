"""V19 模型测试共用的合法张量构造器（当前局面快照协议）。"""

from __future__ import annotations

import numpy as np
import torch

from riichi_ppo_v1.model.encoding_protocol import (
    KIND_ACTION_DEFENSE_QUERY,
    KIND_ACTION_OFFENSE_QUERY,
    KIND_BOS,
    KIND_CRITIC_HAND,
    KIND_MELD,
    KIND_OPPONENT_ANALYSIS,
    KIND_PLAYER,
    KIND_RIICHI_CARD,
    KIND_RIVER_DISCARD,
    KIND_SELF_HAND,
    KIND_SELF_STATE_ANALYSIS,
    KIND_SEP_ACTIONS,
    KIND_SEP_CRITIC,
    KIND_SEP_KAMICHA_RIVER,
    KIND_SEP_MELDS,
    KIND_SEP_OPPONENT_ANALYSIS,
    KIND_SEP_PLAYERS,
    KIND_SEP_RIVERS,
    KIND_SEP_SELF_HAND,
    KIND_SEP_SHIMOCHA_RIVER,
    KIND_SEP_TILE_STATE,
    KIND_SEP_TOIMEN_RIVER,
    KIND_TABLE,
    KIND_TILE_STATE,
    SEGMENT_ACTIONS,
    SEGMENT_ANALYSIS,
    SEGMENT_CRITIC_PRIVATE,
    SEGMENT_SHARED,
    TOKEN_NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH,
)
from riichi_ppo_v1.model.schema import NUM_ACTIONS


def first_kyoku_record(path: str = "RiichiEnv/tests/data/126_204_0_mjai.jsonl") -> tuple[str, str]:
    """从真实 fixture 提取第一个 kyoku 的 JSONL 记录，返回 (record, game_id)。"""
    import json

    lines = [line for line in open(path, encoding="utf-8").read().splitlines() if line.strip()]
    start = next(
        index for index, line in enumerate(lines)
        if '"start_kyoku"' in line
    )
    end = next(
        index for index in range(start, len(lines))
        if '"end_kyoku"' in lines[index]
    )
    record = "\n".join(lines[start:end + 1]) + "\n" + json.dumps({"type": "end_game"}) + "\n"
    return record, "test-1"


def make_observation(
    *,
    hands: list[list[int]] | None = None,
    melds: list[list[dict[str, object]]] | None = None,
    discards: list[list[int]] | None = None,
    dora_indicators: list[int] | None = None,
    events: list[str] | None = None,
    drawn_tile: int | None = None,
) -> object:
    """构造带副露/赤牌/事件的合成 Observation（V19 测试夹具）。

    ``events`` 为 JSON 字符串列表（与 Observation.new_events 一致）。
    仅用于编码器字段级断言；麻将合法性由调用方负责。
    """
    import base64
    import json

    import riichienv

    data = {
        "player_id": 0,
        "hands": hands if hands is not None else [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [], [], []],
        "melds": melds if melds is not None else [[], [], [], []],
        "discards": discards if discards is not None else [[], [], [], []],
        "dora_indicators": dora_indicators if dora_indicators is not None else [],
        "scores": [25000] * 4,
        "riichi_declared": [False] * 4, "riichi_accepted": [False] * 4,
        "riichi_declaration_indices": [None] * 4,
        "missed_agari_doujun": False, "missed_agari_riichi": False,
        "tiles_left": 70, "honba": 0, "riichi_sticks": 0, "round_wind": 0,
        "oya": 0, "kyoku_index": 0, "waits": [], "is_tenpai": False,
        "tsumogiri_flags": [[], [], [], []], "riichi_sutehais": [None] * 4,
        "last_tedashis": [None] * 4, "last_discard": None,
        "drawn_tile": drawn_tile,
        "_legal_actions": [], "events": events if events is not None else [],
    }
    b64 = base64.b64encode(json.dumps(data).encode()).decode()
    return riichienv.Observation.deserialize_from_base64(b64)


def _row(segment: int, kind: int, fields: tuple[int, ...] = ()) -> np.ndarray:
    row = np.zeros(TOKEN_ROW_WIDTH, dtype=np.int64)
    row[0] = segment
    row[1] = kind
    for index, value in enumerate(fields[:30]):
        row[2 + index] = value
    return row


def shared_prefix_rows() -> list[np.ndarray]:
    """构造一个最短合法 Shared 公共前缀（空副露、三家各一张河牌、2 种自身手牌）。"""
    rows: list[np.ndarray] = [
        _row(SEGMENT_SHARED, KIND_BOS),
        _row(SEGMENT_SHARED, KIND_TABLE, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        _row(SEGMENT_SHARED, KIND_SEP_SELF_HAND),
        _row(SEGMENT_SHARED, KIND_SELF_HAND, (1, 1, 0, 0, 0)),
        _row(SEGMENT_SHARED, KIND_SELF_HAND, (5, 2, 0, 0, 0)),
        _row(SEGMENT_SHARED, KIND_SELF_STATE_ANALYSIS, (0,) * 18),
        _row(SEGMENT_SHARED, KIND_SEP_PLAYERS),
        _row(SEGMENT_SHARED, KIND_PLAYER, (0, 0, 0, 0, 1, 13, 0, 0, 1)),
        _row(SEGMENT_SHARED, KIND_PLAYER, (1, 1, 1, 0, 2, 13, 0, 0, 1)),
        _row(SEGMENT_SHARED, KIND_PLAYER, (2, 2, 2, 0, 3, 13, 0, 0, 1)),
        _row(SEGMENT_SHARED, KIND_PLAYER, (3, 3, 3, 0, 4, 13, 0, 0, 1)),
        _row(SEGMENT_SHARED, KIND_SEP_RIVERS),
    ]
    for river_sep in (KIND_SEP_SHIMOCHA_RIVER, KIND_SEP_TOIMEN_RIVER, KIND_SEP_KAMICHA_RIVER):
        rows.append(_row(SEGMENT_SHARED, river_sep))
        rows.append(_row(SEGMENT_SHARED, KIND_RIVER_DISCARD, (1, 1, 0, 0, 0, 0)))
        rows.append(_row(SEGMENT_SHARED, KIND_RIICHI_CARD, (0, 0, 0, 0, 0, 0, 0)))
    rows.append(_row(SEGMENT_SHARED, KIND_SEP_MELDS))
    # owner/type + 4 槽(type,red) + called(type,red) + supplier/open/meld_index/yakuhai/dora/meld_turn/called_tsumogiri
    rows.append(_row(
        SEGMENT_SHARED, KIND_MELD,
        (0, 1, 1, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0),
    ))
    rows.append(_row(SEGMENT_SHARED, KIND_SEP_TILE_STATE))
    for kind in range(1, 35):
        rows.append(_row(
            SEGMENT_SHARED, KIND_TILE_STATE,
            (kind, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ))
    rows.append(_row(SEGMENT_ANALYSIS, KIND_SEP_OPPONENT_ANALYSIS))
    rows.append(_row(SEGMENT_ANALYSIS, KIND_OPPONENT_ANALYSIS, (1, 0, 0, 0, 0, 0, 0)))
    rows.append(_row(SEGMENT_ANALYSIS, KIND_OPPONENT_ANALYSIS, (2, 0, 0, 0, 0, 0, 0)))
    rows.append(_row(SEGMENT_ANALYSIS, KIND_OPPONENT_ANALYSIS, (3, 0, 0, 0, 0, 0, 0)))
    return rows


def _action_row(query_type: int, action_id: int) -> np.ndarray:
    # query 行特征：action_type=1, primary=0, source=0, tsumogiri=0, action_id, answers=0
    fields = (1, 0, 0, 0, action_id) + (0,) * 10
    kind = KIND_ACTION_OFFENSE_QUERY if query_type == 1 else KIND_ACTION_DEFENSE_QUERY
    return _row(SEGMENT_ACTIONS, kind, fields)


def actor_inputs(*, batch: int = 2, action_ids: tuple[int, ...] = (1, 7, 12)) -> dict[str, torch.Tensor]:
    """构造合成 Actor 输入（含 SEP_ACTIONS 与 O/D 对）。"""
    shared = np.stack(shared_prefix_rows())
    count = len(action_ids)
    action_rows = [_row(SEGMENT_ACTIONS, KIND_SEP_ACTIONS)]
    for action_id in action_ids:
        action_rows.append(_action_row(1, action_id))
        action_rows.append(_action_row(2, action_id))
    actions = np.stack(action_rows)
    sequence = np.vstack([shared, actions])
    numeric = np.zeros((sequence.shape[0], TOKEN_NUMERIC_WIDTH), dtype=np.float32)
    ids = torch.tensor(action_ids, dtype=torch.long)[None].expand(batch, -1).clone()
    legal_mask = torch.zeros(batch, NUM_ACTIONS, dtype=torch.bool)
    legal_mask.scatter_(1, ids, True)
    return {
        "actor_factors": torch.from_numpy(np.tile(sequence[None], (batch, 1, 1))).to(torch.long),
        "actor_numeric": torch.from_numpy(np.tile(numeric[None], (batch, 1, 1))),
        "actor_lengths": torch.full((batch,), sequence.shape[0], dtype=torch.long),
        "action_ids": ids,
        "query_pair_counts": torch.full((batch,), count, dtype=torch.long),
        "legal_mask": legal_mask,
    }


def critic_inputs(*, batch: int = 2, observer: int = 0) -> dict[str, torch.Tensor]:
    """构造合成 Critic 私有行：SEP_CRITIC + 三家闭手各 1 行（无 future）。"""
    rows: list[np.ndarray] = [_row(SEGMENT_CRITIC_PRIVATE, KIND_SEP_CRITIC)]
    for relative in (1, 2, 3):
        rows.append(_row(SEGMENT_CRITIC_PRIVATE, KIND_CRITIC_HAND, (relative, 1, 0, 1)))
    factors = np.stack(rows)
    lengths = torch.full((batch,), factors.shape[0], dtype=torch.long)
    return {
        "critic_factors": torch.from_numpy(np.tile(factors[None], (batch, 1, 1))).to(torch.long),
        "critic_lengths": lengths,
    }
