import numpy as np

from riichi_ppo_v1.inference import assign_batch_outputs, collate_request_rows, dispatch_reason


def request(worker_id: int, lengths: list[int], marker: int) -> dict[str, object]:
    width = max(lengths)
    batch = len(lengths)
    kinds = np.zeros((batch, width), dtype=np.uint8)
    turn = np.zeros((batch, width, 4, 4), dtype=np.uint8)
    meld = np.zeros((batch, width, 8), dtype=np.uint8)
    board = np.zeros((batch, 12, 160), dtype=np.uint8)
    legal = np.zeros((batch, 241), dtype=np.bool_)
    for row, length in enumerate(lengths):
        kinds[row, :length] = marker + row
        turn[row, :length, 0, 0] = marker + row
        meld[row, :length, 0] = marker + row
        board[row, :, 0] = marker + row
        legal[row, marker + row] = True
    return {
        "worker_id": worker_id,
        "batch_indices": list(range(batch)),
        "block_kinds": kinds,
        "turn_fields": turn,
        "meld_fields": meld,
        "board_state": board,
        "legal_mask": legal,
        "block_lengths": np.asarray(lengths, dtype=np.int64),
    }


def test_cross_request_collation_and_response_routing_preserve_row_order() -> None:
    requests = [request(0, [1, 3], 10), request(1, [2], 20)]
    group = [(0, 1), (1, 0), (0, 0)]

    kinds, turn, meld, board, legal, lengths = collate_request_rows(requests, group)

    np.testing.assert_array_equal(lengths, [3, 2, 1])
    np.testing.assert_array_equal(kinds[:, 0], [11, 20, 10])
    assert kinds.shape == (3, 3)
    assert kinds[1, 2] == 0 and kinds[2, 1] == 0
    np.testing.assert_array_equal(turn[:, 0, 0, 0], [11, 20, 10])
    np.testing.assert_array_equal(meld[:, 0, 0], [11, 20, 10])
    np.testing.assert_array_equal(board[:, 0, 0], [11, 20, 10])
    assert legal[0, 11] and legal[1, 20] and legal[2, 10]

    responses = [
        {"action_ids": [0, 0], "logprobs": [0.0, 0.0], "values": [0.0, 0.0]},
        {"action_ids": [0], "logprobs": [0.0], "values": [0.0]},
    ]
    assign_batch_outputs(responses, group, [101, 202, 303], [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])

    assert responses == [
        {"action_ids": [303, 101], "logprobs": [-3.0, -1.0], "values": [3.0, 1.0]},
        {"action_ids": [202], "logprobs": [-2.0], "values": [2.0]},
    ]


def test_dispatch_reason_prefers_full_worker_batch_then_timeout() -> None:
    assert dispatch_reason([0, 1, 2, 3], 4, deadline=10.0, now=1.0) == "target"
    assert dispatch_reason([0, 1], 4, deadline=10.0, now=9.9) is None
    assert dispatch_reason([0, 1], 4, deadline=10.0, now=10.0) == "timeout"
    assert dispatch_reason([0], 1, deadline=10.0, now=1.0) == "target"
