"""V16 跨请求推理批的 collate 与响应路由测试。"""

from __future__ import annotations

import numpy as np
import pytest

from riichi_ppo_v1.training.inference import (
    assign_batch_outputs,
    collate_request_rows,
    dispatch_reason,
    parse_history_namespace,
)


def request(worker_id: int, lengths: list[int], marker: int) -> dict[str, object]:
    width, batch = max(lengths), len(lengths)
    history_factors = np.zeros((batch, width, 10), dtype=np.uint8)
    history_numeric = np.zeros((batch, width, 8), dtype=np.float32)
    snapshot_kinds = np.zeros((batch, 2), dtype=np.uint8)
    snapshot_cat = np.zeros((batch, 2, 4), dtype=np.uint8)
    snapshot_num = np.zeros((batch, 2, 7), dtype=np.float32)
    query_rows = np.zeros((batch, 4, 15), dtype=np.int32)
    query_action_ids = np.zeros((batch, 2), dtype=np.int32)
    critic_factors = np.zeros((batch, 2, 10), dtype=np.uint8)
    legal = np.zeros((batch, 241), dtype=np.bool_)
    for row, length in enumerate(lengths):
        history_factors[row, :length, 0] = marker + row
        history_numeric[row, :length, 0] = marker + row
        query_rows[row, 0, 1] = marker + row
        critic_factors[row, :, 0] = marker + row
        legal[row, marker + row] = True
    return {
        "worker_id": worker_id,
        "batch_indices": np.arange(batch),
        "history_factors": history_factors,
        "history_numeric": history_numeric,
        "history_lengths": np.asarray(lengths, dtype=np.int64),
        "snapshot_kinds": snapshot_kinds,
        "snapshot_cat": snapshot_cat,
        "snapshot_num": snapshot_num,
        "snapshot_lengths": np.full(batch, 2, dtype=np.int64),
        "query_rows": query_rows,
        "query_action_ids": query_action_ids,
        "query_pair_counts": np.full(batch, 2, dtype=np.int64),
        "legal_mask": legal,
        "critic_factors": critic_factors,
        "critic_lengths": np.full(batch, 2, dtype=np.int64),
    }


def test_cross_request_collation_preserves_row_order_and_pads_segments() -> None:
    requests = [request(0, [1, 3], 1), request(1, [2], 3)]
    group = [(0, 1), (1, 0), (0, 0)]
    (
        history_factors,
        history_numeric,
        history_lengths,
        snapshot_kinds,
        snapshot_cat,
        snapshot_num,
        snapshot_lengths,
        query_rows,
        query_action_ids,
        pair_counts,
        critic_factors,
        critic_lengths,
        legal,
    ) = collate_request_rows(requests, group)
    np.testing.assert_array_equal(history_lengths, [3, 2, 1])
    np.testing.assert_array_equal(snapshot_lengths, [2, 2, 2])
    np.testing.assert_array_equal(pair_counts, [2, 2, 2])
    np.testing.assert_array_equal(critic_lengths, [2, 2, 2])
    np.testing.assert_array_equal(history_factors[:, 0, 0], [2, 3, 1])
    np.testing.assert_array_equal(history_numeric[:, 0, 0], [2, 3, 1])
    np.testing.assert_array_equal(critic_factors[:, 0, 0], [2, 3, 1])
    np.testing.assert_array_equal(query_rows[:, 0, 1], [2, 3, 1])
    assert history_factors.shape == (3, 3, 10)
    assert snapshot_cat.shape == (3, 2, 4)
    assert query_rows.shape == (3, 4, 15)
    assert critic_factors.shape == (3, 2, 10)
    assert history_factors[1, 2].sum() == 0 and history_factors[2, 1].sum() == 0
    assert legal[0, 2] and legal[1, 3] and legal[2, 1]


def test_batch_output_routing_preserves_request_row_mapping() -> None:
    responses = [
        {
            "action_ids": [0, 0],
            "logprobs": [0.0, 0.0],
            "values": [0.0, 0.0],
            "q_taken": [0.0, 0.0],
            "expected_q": [0.0, 0.0],
            "top3_ids": [np.zeros(3, np.int32), np.zeros(3, np.int32)],
        },
        {
            "action_ids": [0],
            "logprobs": [0.0],
            "values": [0.0],
            "q_taken": [0.0],
            "expected_q": [0.0],
            "top3_ids": [np.zeros(3, np.int32)],
        },
    ]
    top3 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    assign_batch_outputs(
        responses,
        [(0, 1), (1, 0), (0, 0)],
        [101, 202, 203],
        [-1.0, -2.0, -3.0],
        [0.1, 0.2, 0.3],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        top3,
    )
    assert responses[0]["action_ids"] == [203, 101]
    assert responses[1]["action_ids"] == [202]
    assert responses[0]["q_taken"] == [3.0, 1.0]
    assert responses[0]["expected_q"] == [6.0, 4.0]
    assert responses[0]["values"] == [0.3, 0.1]
    np.testing.assert_array_equal(responses[0]["top3_ids"][1], [1, 2, 3])
    np.testing.assert_array_equal(responses[1]["top3_ids"][0], [4, 5, 6])


def test_dispatch_reason_prefers_full_worker_batch_then_timeout() -> None:
    assert dispatch_reason([0, 1], 4, deadline=10.0, now=1.0, row_count=512, target_rows=512) == "rows"
    assert dispatch_reason([0, 1, 2, 3], 4, deadline=10.0, now=1.0) == "target"
    assert dispatch_reason([0, 1], 4, deadline=10.0, now=9.9) is None
    assert dispatch_reason([0, 1], 4, deadline=10.0, now=10.0) == "timeout"


def test_history_namespace_parses_update_with_u_prefix() -> None:
    assert parse_history_namespace("history:u060") == 60
    assert parse_history_namespace("history:u780") == 780
    assert parse_history_namespace("history:0060") == 60
    with pytest.raises(RuntimeError, match="malformed history namespace"):
        parse_history_namespace("history:")
    with pytest.raises(RuntimeError, match="malformed history namespace"):
        parse_history_namespace("sft")
