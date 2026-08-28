"""推理批按长度排序(B4):排序性质、路由不变式与行间独立性测试。"""

from __future__ import annotations

import numpy as np
import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic
from riichi_ppo_v1.model.encoding_protocol import TOKEN_NUMERIC_WIDTH, TOKEN_ROW_WIDTH
from riichi_ppo_v1.model.schema import NUM_ACTIONS
from riichi_ppo_v1.training.inference import collate_request_rows, sort_rows_by_sequence_length

from .test_v18_architecture import _tiny_config


def _fake_request(rng: np.random.Generator, actor_length: int, pair_count: int, critic_length: int) -> dict:
    """单行请求:长度可控,内容随机(collate/排序只依赖长度与数组形状)。"""
    return {
        "actor_factors": rng.integers(0, 4, size=(1, actor_length, TOKEN_ROW_WIDTH)).astype(np.int64),
        "actor_numeric": rng.random(size=(1, actor_length, TOKEN_NUMERIC_WIDTH)).astype(np.float32),
        "actor_lengths": np.asarray([actor_length], dtype=np.int64),
        "query_rows": np.zeros((1, 2 * pair_count, 15), dtype=np.int32),
        "query_action_ids": rng.integers(1, NUM_ACTIONS, size=(1, pair_count)).astype(np.int64),
        "query_pair_counts": np.asarray([pair_count], dtype=np.int64),
        "critic_factors": rng.integers(0, 4, size=(1, critic_length, TOKEN_ROW_WIDTH)).astype(np.int64),
        "critic_lengths": np.asarray([critic_length], dtype=np.int64),
        "legal_mask": rng.random(size=(1, NUM_ACTIONS)) > 0.5,
    }


def test_sort_rows_by_sequence_length_orders_descending_and_stable() -> None:
    rng = np.random.default_rng(0)
    # (actor_length, pair_count) 交错的到达顺序;同长度对(70,*)验证稳定性。
    plan = [(70, 5), (200, 3), (70, 9), (128, 1), (200, 8), (64, 0)]
    requests = [
        _fake_request(rng, actor_length, max(1, pair), 4)
        for actor_length, pair in plan
    ]
    rows = [(index, 0) for index in range(len(requests))]
    sort_rows_by_sequence_length(requests, rows)
    ordered = [int(requests[i]["actor_lengths"][0]) for i, _ in rows]
    assert ordered == sorted(ordered, reverse=True)
    # 稳定排序:同长度(70,5)与(70,9)保持原相对顺序(request 0 在 2 之前)。
    assert rows.index((0, 0)) < rows.index((2, 0))
    # 排列性质:不增不减。
    assert sorted(rows) == [(index, 0) for index in range(len(requests))]


def test_collate_request_rows_maps_group_order_to_batch_rows() -> None:
    """路由不变式:collate 后第 i 行必须严格来自 group[i] 指向的请求行。"""
    rng = np.random.default_rng(1)
    requests = [_fake_request(rng, 30 + 7 * r, 1 + r, 3 + r) for r in range(4)]
    group = [(2, 0), (0, 0), (3, 0), (1, 0)]
    (
        actor_factors, actor_numeric, actor_lengths, _query_rows, query_action_ids,
        pair_counts, critic_factors, critic_lengths, legal,
    ) = collate_request_rows(requests, group)
    for batch_row, (request_index, row) in enumerate(group):
        request = requests[request_index]
        np.testing.assert_array_equal(
            actor_factors[batch_row, : actor_lengths[batch_row]],
            request["actor_factors"][row, : actor_lengths[batch_row]],
        )
        assert actor_lengths[batch_row] == request["actor_lengths"][row]
        assert pair_counts[batch_row] == request["query_pair_counts"][row]
        np.testing.assert_array_equal(legal[batch_row], request["legal_mask"][row])
        np.testing.assert_array_equal(
            query_action_ids[batch_row, : pair_counts[batch_row]],
            request["query_action_ids"][row, : pair_counts[batch_row]],
        )
        assert critic_lengths[batch_row] == request["critic_lengths"][row]
        np.testing.assert_array_equal(
            critic_factors[batch_row, : critic_lengths[batch_row]],
            request["critic_factors"][row, : critic_lengths[batch_row]],
        )


def test_row_outputs_independent_of_batch_order() -> None:
    """行间独立:同一行在不同批组装顺序(排序前后)下输出一致(≤1e-4,T2)。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2)
    model = KyokuTransformerActorCritic(_tiny_config()).to(device).eval()
    rng = np.random.default_rng(3)
    lengths = [68, 90, 68, 121, 75, 90, 68, 105]
    requests = [_fake_request(rng, length, 1 + r % 3, 3 + r % 2) for r, length in enumerate(lengths)]
    identity = [(index, 0) for index in range(len(requests))]
    rows_sorted = list(identity)
    sort_rows_by_sequence_length(requests, rows_sorted)

    outputs: dict[str, dict[tuple[int, int], tuple[np.ndarray, float]]] = {}
    with torch.no_grad():
        for pass_name, group in (("identity", identity), ("sorted", rows_sorted)):
            (
                actor_factors, actor_numeric, actor_lengths, _query_rows, query_action_ids,
                pair_counts, critic_factors, critic_lengths, legal,
            ) = collate_request_rows(requests, group)
            tensors = dict(
                actor_factors=torch.from_numpy(actor_factors).to(device),
                actor_numeric=torch.from_numpy(actor_numeric).to(device),
                actor_lengths=torch.from_numpy(actor_lengths).to(device),
                query_action_ids=torch.from_numpy(query_action_ids).to(device),
                query_pair_counts=torch.from_numpy(pair_counts).to(device),
                legal_mask=torch.from_numpy(legal).to(device),
                critic_factors=torch.from_numpy(critic_factors).to(device),
                critic_lengths=torch.from_numpy(critic_lengths).to(device),
            )
            output = model(validate_structure=False, **tensors)
            logits = output["policy_logits"].float().cpu().numpy()
            values = output["value"].float().cpu().numpy()
            for position, key in enumerate(group):
                outputs.setdefault(pass_name, {})[key] = (
                    logits[position], float(values[position]),
                )
    for key in identity:
        first_logits, first_value = outputs["identity"][key]
        second_logits, second_value = outputs["sorted"][key]
        # 排序只改变批组装顺序:同一行在两种顺序下的输出须一致(T2 ≤1e-4)。
        assert abs(first_value - second_value) <= 1e-4
        finite = np.isfinite(first_logits) & np.isfinite(second_logits)
        assert np.array_equal(np.isfinite(first_logits), np.isfinite(second_logits))
        if finite.any():
            assert float(np.abs(first_logits[finite] - second_logits[finite]).max()) <= 1e-4
