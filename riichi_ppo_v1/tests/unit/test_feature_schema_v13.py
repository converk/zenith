from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

import riichi
from riichienv import calculate_shanten

from riichi_ppo_v1.model.feature_schema import (
    DECISION_ANALYSIS_VERSION,
    ENCODED_FORMAT,
    FEATURE_SCHEMA_VERSION,
    RUST_ANALYSIS_VERSION,
    feature_schema_sha256,
    legacy_encoder_component_sha256,
    legacy_encoder_sha256,
)
from riichi_ppo_v1.model.actor_features import (
    _event_threat_rows,
    _observation_threat_rows,
    encode_actor_state_summary,
)
from riichi_ppo_v1.training.rewards.efficiency import EfficiencyAnalyzer
from riichi_ppo_v1.training.rewards.decision import _defense_feature_reference


def _physical(counts: np.ndarray) -> list[int]:
    return [tile * 4 + copy for tile, count in enumerate(counts) for copy in range(int(count))]


def _random_counts(seed: int, totals: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for total in totals:
        counts = np.zeros(34, dtype=np.uint8)
        while int(counts.sum()) < total:
            tile = int(rng.integers(0, 34))
            if counts[tile] < 4:
                counts[tile] += 1
        rows.append(counts)
    return np.ascontiguousarray(rows)


def test_v13_contract_is_stable_and_versioned() -> None:
    assert FEATURE_SCHEMA_VERSION == 13
    assert ENCODED_FORMAT == "riichi-sft-encoded-v3"
    assert RUST_ANALYSIS_VERSION == 4 == riichi.ANALYSIS_VERSION
    assert DECISION_ANALYSIS_VERSION == 16
    assert len(feature_schema_sha256()) == 64
    components = legacy_encoder_component_sha256()
    assert "RiichiEnv/riichienv-core/src/replay/mod.rs" in components
    assert "riichi_ppo_v1/sft/data.py" in components
    assert len(legacy_encoder_sha256()) == 64


def test_rust_batch_matches_scalar_python_reference_for_random_13_and_14_tile_hands() -> None:
    counts = _random_counts(20260801, (13, 14, 13, 14))
    rows = len(counts)
    remaining = np.maximum(4 - counts, 0).astype(np.uint8)
    result = riichi.analyze_features(
        np.asarray((0, 239, 75, 240), dtype=np.uint16), counts, np.zeros(rows, dtype=np.uint8), remaining,
        np.full(rows, -1, dtype=np.int16), np.zeros((rows, 3), dtype=np.uint64),
        np.zeros((rows, 3), dtype=np.uint64),
    )
    for row, hand in enumerate(counts):
        base = int(calculate_shanten(_physical(hand)))
        improving_mask = 0
        if int(hand.sum()) == 13:
            for tile in range(34):
                if hand[tile] >= 4:
                    continue
                next_hand = hand.copy()
                next_hand[tile] += 1
                if int(calculate_shanten(_physical(next_hand))) < base:
                    improving_mask |= 1 << tile
        assert int(result.shanten[row, 0]) == base
        assert int(result.improving_type_mask[row]) == improving_mask
        assert int(result.wait_count[row]) == improving_mask.bit_count()
        assert int(result.ukeire[row]) == sum(
            int(remaining[row, tile]) for tile in range(34) if improving_mask & (1 << tile)
        )


def test_rust_defense_uses_public_masks_and_never_applies_suji_to_honors() -> None:
    counts = _random_counts(7, (13, 13))
    remaining = np.maximum(4 - counts, 0).astype(np.uint8)
    result = riichi.analyze_features(
        np.array([1, 2], dtype=np.uint16), counts, np.zeros(2, dtype=np.uint8), remaining,
        np.array([27, 4], dtype=np.int16),
        np.array([[1 << 24, 1 << 30, 0], [(1 << 1) | (1 << 7), 0, 0]], dtype=np.uint64),
        np.array([[1 << 27, 0, 0], [0, 1 << 4, 0]], dtype=np.uint64),
    )
    # 1z is an honor: suited tiles at +/-3 can never make it suji.
    assert int(result.defense[0, 1]) == 0
    assert int(result.defense[0, 4]) == 1
    # 5m is fully suji-safe only when both 2m and 8m are present.
    assert int(result.defense[1, 1]) == 1
    assert int(result.defense[1, 4]) == 2
    np.testing.assert_array_equal(result.categorical[:, (0, 1, 2, 9)], [[7, 2, 2, 2], [7, 2, 3, 2]])
    np.testing.assert_allclose(result.numeric[:, :2], [[0.0, 0.0], [0.0, 1 / 3]])


def test_middle_tile_requires_both_suji_and_wall_reads_related_tiles() -> None:
    base = _random_counts(17, (13,))[0]
    counts = np.repeat(base[None], 2, axis=0)
    remaining = np.full((2, 34), 4, dtype=np.uint8)
    remaining[0, 1] = 1  # three 2m visible
    remaining[1, 1] = 0  # four 2m visible
    result = riichi.analyze_features(
        np.asarray((1, 2), dtype=np.uint16), counts, np.zeros(2, dtype=np.uint8), remaining,
        np.asarray((3, 3), dtype=np.int16),
        np.asarray(((1 << 0, 0, 0), ((1 << 0) | (1 << 6), 0, 0)), dtype=np.uint64),
        np.zeros((2, 3), dtype=np.uint64),
    )
    # 1m alone blocks 23m but 4m can still hit a 56m wait; 1m+7m is complete suji.
    assert int(result.defense[0, 1]) == 0
    assert int(result.defense[1, 1]) == 1
    # The wall strength changes with the related 2m count, not the 4m count.
    assert tuple(int(value) for value in result.defense[:, 2]) == (1, 2)


def test_rust_defense_rows_match_full_python_reference() -> None:
    base = _random_counts(81, (13,))[0]
    rows = 8
    counts = np.repeat(base[None], rows, axis=0)
    remaining = np.repeat(np.maximum(4 - base, 0)[None], rows, axis=0).astype(np.uint8)
    action_ids = np.arange(1, rows + 1, dtype=np.uint16)
    discard_types = np.asarray((27, 4, 8, 13, 22, 31, 0, 26), dtype=np.int16)
    river_masks = np.asarray([
        [1 << 24, 1 << 27, 0], [1 << 1, 1 << 7, 0],
        [1 << 5, 0, 1 << 11], [1 << 10, 1 << 16, 0],
        [1 << 19, 0, 1 << 25], [1 << 31, 0, 0],
        [1 << 3, 0, 0], [1 << 23, 0, 0],
    ], dtype=np.uint64)
    passed_masks = np.asarray([
        [1 << int(tile), 0, 1 << int(tile)] for tile in discard_types
    ], dtype=np.uint64)
    result = riichi.analyze_features(
        action_ids, counts, np.zeros(rows, dtype=np.uint8), remaining,
        discard_types, river_masks, passed_masks,
    )
    expected = _defense_feature_reference(
        action_ids, remaining[0], discard_types, river_masks, passed_masks, base,
    )
    np.testing.assert_array_equal(result.defense, expected[0])
    np.testing.assert_array_equal(result.categorical, expected[1])
    np.testing.assert_allclose(result.numeric, expected[2])


def test_rust_feature_api_rejects_duplicate_actions_and_invalid_hand_size() -> None:
    counts = _random_counts(9, (13, 13))
    args = (
        np.array([1, 1], dtype=np.uint16), counts, np.zeros(2, dtype=np.uint8),
        np.full((2, 34), 4, dtype=np.uint8), np.zeros(2, dtype=np.int16),
        np.zeros((2, 3), dtype=np.uint64), np.zeros((2, 3), dtype=np.uint64),
    )
    with pytest.raises(ValueError, match="unique"):
        riichi.analyze_features(*args)
    broken = counts.copy()
    broken[0, np.flatnonzero(broken[0])[0]] -= 1
    with pytest.raises(ValueError, match="expected 13 or 14"):
        riichi.analyze_features(args[0] + np.array([0, 1], dtype=np.uint16), broken, *args[2:])


def test_reach_declaration_and_acceptance_are_separate_public_states() -> None:
    declared = SimpleNamespace(events=[
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": False},
    ])
    accepted = SimpleNamespace(events=[*declared.events, {"type": "reach_accepted", "actor": 1}])
    assert _event_threat_rows(declared)[1]["declared"] is True
    assert _event_threat_rows(declared)[1]["accepted"] is False
    assert _event_threat_rows(accepted)[1]["accepted"] is True


def test_snapshot_threat_requires_and_separates_declaration_fields() -> None:
    observation = SimpleNamespace(
        riichi_declared=[False, True, False, False],
        riichi_accepted=[False, False, False, False],
        riichi_declaration_indices=[None, 5, None, None],
        discards=[[], [0, 4, 8, 12, 16, 20], [], []],
        melds=[[], [], [], []], last_tedashis=[None, 20, None, None],
        tsumogiri_flags=[[], [False, True, True], [], []],
        riichi_sutehais=[None, 20, None, None],
    )
    row = _observation_threat_rows(observation)[1]
    assert row["declared"] is True and row["accepted"] is False
    assert row["reach_turn"] == 5 and row["reach_tile"] == 5
    observation.riichi_accepted[1] = True
    assert _observation_threat_rows(observation)[1]["accepted"] is True
    observation.riichi_sutehais[1] = None
    with pytest.raises(RuntimeError, match="declaration tile"):
        _observation_threat_rows(observation)


def test_state_rows_do_not_depend_on_candidate_analysis_or_order() -> None:
    hand = [0, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76, 80, 108]
    observation = SimpleNamespace(
        player_id=0, hands=[hand, [], [], []], hand=hand,
        melds=[[], [], [], []], discards=[[], [], [], []], dora_indicators=[],
        riichi_declared=[False] * 4, scores=[25000] * 4, oya=0, round_wind=0,
        kyoku_index=0, honba=0, riichi_sticks=0, tiles_left=70,
        missed_agari_doujun=False, missed_agari_riichi=False,
        riichi_accepted=[False] * 4, riichi_declaration_indices=[None] * 4,
        riichi_sutehais=[None] * 4, last_tedashis=[None] * 4,
        tsumogiri_flags=[[], [], [], []], events=[{"type": "start_kyoku"}],
    )
    first = encode_actor_state_summary(observation, object(), EfficiencyAnalyzer())
    second = encode_actor_state_summary(observation, {"reversed_candidates": True}, EfficiencyAnalyzer())
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    observation.missed_agari_doujun = True
    furiten = encode_actor_state_summary(observation, object(), EfficiencyAnalyzer())
    assert int(furiten[0][0, 7]) == 2
    del observation.riichi_accepted
    with pytest.raises(RuntimeError, match="observation.riichi_accepted"):
        encode_actor_state_summary(observation, object(), EfficiencyAnalyzer())


def test_state_progress_ukeire_na_and_value_tiles_follow_strict_contract() -> None:
    hand = [0, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76, 80, 108, 112]
    observation = SimpleNamespace(
        player_id=0, hands=[hand, [], [], []], hand=hand,
        melds=[[], [], [], []], discards=[[], [], [], []], dora_indicators=[0],
        riichi_declared=[False] * 4, scores=[90_000, 25_000, 0, -15_000], oya=0,
        round_wind=0, kyoku_index=0, honba=30, riichi_sticks=12, tiles_left=90,
        missed_agari_doujun=False, missed_agari_riichi=False,
        riichi_accepted=[False] * 4, riichi_declaration_indices=[None] * 4,
        riichi_sutehais=[None] * 4, last_tedashis=[None] * 4,
        tsumogiri_flags=[[], [], [], []], events=[{"type": "start_kyoku"}],
    )
    east_one = encode_actor_state_summary(observation, object(), EfficiencyAnalyzer())
    assert int(east_one[0][0, 8]) == 0  # 14-tile current hand: ukeire is N/A
    assert float(east_one[1][0, 4]) == 0.0
    assert np.max(np.abs(east_one[1])) <= 1.0
    observation.kyoku_index = 2
    east_three = encode_actor_state_summary(observation, object(), EfficiencyAnalyzer())
    assert not np.array_equal(east_one[0][2], east_three[0][2])


def test_state_dora_counts_deduplicate_called_discard_and_include_own_meld() -> None:
    meld = SimpleNamespace(tiles=[4, 5, 6], opened=True)
    hand = [0, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    base = dict(
        player_id=0, hands=[hand, [], [], []], hand=hand,
        dora_indicators=[0], riichi_declared=[False] * 4, scores=[25000] * 4,
        oya=0, round_wind=0, kyoku_index=0, honba=0, riichi_sticks=0, tiles_left=60,
        missed_agari_doujun=False, missed_agari_riichi=False,
        riichi_accepted=[False] * 4, riichi_declaration_indices=[None] * 4,
        riichi_sutehais=[None] * 4, last_tedashis=[None] * 4,
        tsumogiri_flags=[[], [], [], []], events=[{"type": "start_kyoku"}],
    )
    own = SimpleNamespace(**base, melds=[[meld], [], [], []], discards=[[], [], [], []])
    own_rows = encode_actor_state_summary(own, object(), EfficiencyAnalyzer())
    assert float(own_rows[1][1, 5]) == pytest.approx(3 / 8)
    filler = SimpleNamespace(tiles=[44, 45, 46], opened=True)
    opponent = SimpleNamespace(
        **base, melds=[[filler], [meld], [], []], discards=[[], [4], [], []],
    )
    opponent_rows = encode_actor_state_summary(opponent, object(), EfficiencyAnalyzer())
    assert float(opponent_rows[1][3, 4]) == pytest.approx(3 / 8)


def test_equal_dora_indicators_preserve_multiplicity_in_state_rows() -> None:
    meld = SimpleNamespace(tiles=[4, 5, 6], opened=True)
    hand = [0, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    observation = SimpleNamespace(
        player_id=0, hands=[hand, [], [], []], hand=hand,
        melds=[[meld], [], [], []], discards=[[], [], [], []],
        dora_indicators=[0, 0], riichi_declared=[False] * 4,
        scores=[25000] * 4, oya=0, round_wind=0, kyoku_index=0,
        honba=0, riichi_sticks=0, tiles_left=60,
        missed_agari_doujun=False, missed_agari_riichi=False,
        riichi_accepted=[False] * 4, riichi_declaration_indices=[None] * 4,
        riichi_sutehais=[None] * 4, last_tedashis=[None] * 4,
        tsumogiri_flags=[[], [], [], []], events=[{"type": "start_kyoku"}],
    )
    rows = encode_actor_state_summary(observation, object(), EfficiencyAnalyzer())
    assert float(rows[1][1, 5]) == pytest.approx(6 / 8)
