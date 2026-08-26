"""真实 MJAI replay → 当前局面编码一致性。"""

from __future__ import annotations

import numpy as np

from riichi_ppo_v1.sft.data import encode_kyoku
from riichi_ppo_v1.model.semantic_validation import assert_actor_input_semantics
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
