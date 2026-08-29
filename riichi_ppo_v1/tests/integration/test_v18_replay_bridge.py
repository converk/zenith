"""真实 MJAI replay → 当前局面编码一致性。"""

from __future__ import annotations

import numpy as np

from riichi_ppo_v1.model.semantic_validation import assert_actor_input_semantics
from riichi_ppo_v1.sft.data import encode_kyoku
from riichi_ppo_v1.tests.v18_fixtures import first_kyoku_record


def test_real_mjai_replay_decisions_encode() -> None:
    record, _game_id = first_kyoku_record()
    samples = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    assert len(samples) > 0
    for sample in samples[:8]:
        assert sample.token_length <= 256
        assert sample.query_pair_count > 0
        assert 0 <= sample.action < 241
        assert sample.legal_mask[sample.action]
        assert_actor_input_semantics(
            sample.actor_factors[None],
            sample.actor_numeric[None],
            np.asarray([sample.token_length]),
            sample.query_rows[None],
            sample.action_ids[None],
            np.asarray([sample.query_pair_count]),
            sample.legal_mask[None],
        )


def test_replay_samples_have_no_critic_or_history_fields() -> None:
    record, _game_id = first_kyoku_record()
    samples = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    for sample in samples[:8]:
        segments = sample.actor_factors[:, 0].astype(int)
        kinds = sample.actor_factors[:, 1].astype(int)
        assert not np.any(np.isin(segments, (4, 5)))
        assert not np.any((kinds >= 20) & (kinds < 100))


def test_single_self_hand_row_all_meld_pair_accepted() -> None:
    """回归：四副露+对子（闭手仅一种牌 → SELF_HAND 恰好 1 行）必须通过语义校验。

    真实数据（如 2024021011gm-00a9-0000-07d2a4be kyoku0 seat3 的和牌决策）中，
    合法闭手可以只有一对（3 碰 + 1 吃 + 对子），此时 SELF_HAND 只有 1 行。
    旧实现校验 SEP_SELF_HAND 后 cursor += 2 跳过了第一行，恰好 1 行被判为空并
    抛 "SELF_HAND requires at least one nonzero kind"。
    """
    record, _game_id = first_kyoku_record()
    samples = encode_kyoku(record, year=2024, game_id="test-1", kyoku_index=0)
    sample = next(value for value in samples if (value.actor_factors[:, 1] == 3).sum() > 1)
    rows = sample.actor_factors
    self_hand_positions = np.flatnonzero(rows[:, 1] == 3)
    assert len(self_hand_positions) > 1
    reduced_rows = np.delete(rows, self_hand_positions[1:], axis=0)
    reduced_numeric = np.delete(sample.actor_numeric, self_hand_positions[1:], axis=0)
    assert_actor_input_semantics(
        reduced_rows[None],
        reduced_numeric[None],
        np.asarray([reduced_rows.shape[0]]),
        sample.query_rows[None],
        sample.action_ids[None],
        np.asarray([sample.query_pair_count]),
        sample.legal_mask[None],
    )
