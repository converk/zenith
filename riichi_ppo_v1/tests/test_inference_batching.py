import numpy as np

from riichi_ppo_v1.inference import assign_batch_outputs, collate_request_rows, dispatch_reason


def request(worker_id: int, lengths: list[int], marker: int) -> dict[str, object]:
    width, batch = max(lengths), len(lengths)
    factors = np.zeros((batch, width, 10), dtype=np.uint8)
    numeric = np.zeros((batch, width, 8), dtype=np.float32)
    legal = np.zeros((batch, 241), dtype=np.bool_)
    for row, length in enumerate(lengths):
        factors[row, :length, 0] = marker + row
        numeric[row, :length, 0] = marker + row
        legal[row, marker + row] = True
    return {"worker_id": worker_id, "batch_indices": list(range(batch)), "token_factors": factors,
            "token_numeric": numeric, "legal_mask": legal, "token_lengths": np.asarray(lengths, dtype=np.int64)}


def test_cross_request_collation_and_response_routing_preserve_row_order() -> None:
    requests = [request(0, [1, 3], 1), request(1, [2], 3)]
    group = [(0, 1), (1, 0), (0, 0)]
    factors, numeric, legal, lengths = collate_request_rows(requests, group)
    np.testing.assert_array_equal(lengths, [3, 2, 1])
    np.testing.assert_array_equal(factors[:, 0, 0], [2, 3, 1])
    np.testing.assert_array_equal(numeric[:, 0, 0], [2, 3, 1])
    assert factors.shape == (3, 3, 10)
    assert factors[1, 2].sum() == 0 and factors[2, 1].sum() == 0
    assert legal[0, 2] and legal[1, 3] and legal[2, 1]
    responses = [{"action_ids": [0, 0], "logprobs": [0.0, 0.0], "values": [0.0, 0.0]},
                 {"action_ids": [0], "logprobs": [0.0], "values": [0.0]}]
    assign_batch_outputs(responses, group, [101, 202, 203], [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
    assert responses[0]["action_ids"] == [203, 101]
    assert responses[1]["action_ids"] == [202]


def test_dispatch_reason_prefers_full_worker_batch_then_timeout() -> None:
    assert dispatch_reason([0, 1], 4, deadline=10.0, now=1.0, row_count=512, target_rows=512) == "rows"
    assert dispatch_reason([0, 1, 2, 3], 4, deadline=10.0, now=1.0) == "target"
    assert dispatch_reason([0, 1], 4, deadline=10.0, now=9.9) is None
    assert dispatch_reason([0, 1], 4, deadline=10.0, now=10.0) == "timeout"
