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
    factors = np.zeros((batch, width, 10), dtype=np.uint8)
    numeric = np.zeros((batch, width, 8), dtype=np.float32)
    critic_factors = np.zeros((batch, 2, 10), dtype=np.uint8)
    critic_lengths = np.full(batch, 2, dtype=np.int64)
    legal = np.zeros((batch, 241), dtype=np.bool_)
    for row, length in enumerate(lengths):
        factors[row, :length, 0] = marker + row
        numeric[row, :length, 0] = marker + row
        critic_factors[row, :, 0] = marker + row
        legal[row, marker + row] = True
    return {"worker_id": worker_id, "batch_indices": list(range(batch)), "token_factors": factors,
            "token_numeric": numeric, "critic_factors": critic_factors, "critic_lengths": critic_lengths,
            "legal_mask": legal, "token_lengths": np.asarray(lengths, dtype=np.int64)}


def test_cross_request_collation_and_response_routing_preserve_row_order() -> None:
    requests = [request(0, [1, 3], 1), request(1, [2], 3)]
    group = [(0, 1), (1, 0), (0, 0)]
    factors, numeric, critic_factors, critic_lengths, legal, lengths = collate_request_rows(requests, group)
    np.testing.assert_array_equal(lengths, [3, 2, 1])
    np.testing.assert_array_equal(critic_lengths, [2, 2, 2])
    np.testing.assert_array_equal(factors[:, 0, 0], [2, 3, 1])
    np.testing.assert_array_equal(numeric[:, 0, 0], [2, 3, 1])
    np.testing.assert_array_equal(critic_factors[:, 0, 0], [2, 3, 1])
    assert factors.shape == (3, 3, 10)
    assert critic_factors.shape == (3, 2, 10)
    assert factors[1, 2].sum() == 0 and factors[2, 1].sum() == 0
    assert legal[0, 2] and legal[1, 3] and legal[2, 1]
    responses = [
        {"action_ids": [0, 0], "logprobs": [0.0, 0.0], "q_taken": [0.0, 0.0], "expected_q": [0.0, 0.0]},
        {"action_ids": [0], "logprobs": [0.0], "q_taken": [0.0], "expected_q": [0.0]},
    ]
    assign_batch_outputs(
        responses, group, [101, 202, 203], [-1.0, -2.0, -3.0],
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
    )
    assert responses[0]["action_ids"] == [203, 101]
    assert responses[1]["action_ids"] == [202]
    assert responses[0]["q_taken"] == [3.0, 1.0]
    assert responses[0]["expected_q"] == [6.0, 4.0]


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
