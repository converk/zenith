"""V18 模型测试共用的合法张量构造器。"""

from __future__ import annotations

import torch

from riichi_ppo_v1.model.encoding_protocol import SNAPSHOT_FIELDS
from riichi_ppo_v1.model.schema import NUM_ACTIONS


def actor_inputs(*, batch: int = 2, action_ids: tuple[int, ...] = (1, 7, 12)) -> dict[str, torch.Tensor]:
    history_factors = torch.zeros(batch, 3, 10, dtype=torch.long)
    history_factors[:, :, 0] = 1
    history_numeric = torch.zeros(batch, 3, 8)
    history_lengths = torch.full((batch,), 3, dtype=torch.long)
    snapshot_factors = torch.zeros(batch, 29, 4, dtype=torch.long)
    for index, field in enumerate(SNAPSHOT_FIELDS):
        snapshot_factors[:, index, 0] = field.field_id
        snapshot_factors[:, index, 1] = field.relative_seat
    snapshot_numeric = torch.zeros(batch, 29, 1)
    snapshot_lengths = torch.full((batch,), 29, dtype=torch.long)
    count = len(action_ids)
    ids = torch.tensor(action_ids, dtype=torch.long)[None].expand(batch, -1).clone()
    query_rows = torch.zeros(batch, 2 * count, 15, dtype=torch.long)
    query_rows[:, 0::2, 0] = 1
    query_rows[:, 1::2, 0] = 2
    query_rows[:, 0::2, 1] = ids
    query_rows[:, 1::2, 1] = ids
    query_rows[:, :, 2] = 1
    legal_mask = torch.zeros(batch, NUM_ACTIONS, dtype=torch.bool)
    legal_mask.scatter_(1, ids, True)
    return {
        "history_factors": history_factors,
        "history_numeric": history_numeric,
        "history_lengths": history_lengths,
        "snapshot_factors": snapshot_factors,
        "snapshot_numeric": snapshot_numeric,
        "snapshot_lengths": snapshot_lengths,
        "query_rows": query_rows,
        "query_action_ids": ids,
        "query_pair_counts": torch.full((batch,), count, dtype=torch.long),
        "legal_mask": legal_mask,
    }
