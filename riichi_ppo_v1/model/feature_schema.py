"""Frozen semantic field contract for actor feature schema 13.

Categorical zero is N/A unless the row's explicit applicability marker is set.
For an applicable row it may represent a known all-false bit mask; terminal
rows leave the marker unset, so N/A never aliases a known numerical zero.
"""

from __future__ import annotations

import hashlib
import json

FEATURE_SCHEMA_VERSION = 13
ENCODED_FORMAT = "riichi-sft-encoded-v3"
RUST_ANALYSIS_VERSION = 4
DECISION_ANALYSIS_VERSION = 16

STATE_SUMMARY_SEGMENT = 6
ACTION_QUERY_SEGMENT = 7
ACTION_QUERY_OFFENSE = 1
ACTION_QUERY_DEFENSE = 2

STATE_HAND = 1
STATE_VALUE = 2
STATE_PLACEMENT = 3
STATE_THREAT = 4

# Hand/value bit masks. The row kind or discard action id is the independent
# applicability marker, so an all-false known mask may remain numeric zero.
HAND_CLOSED = 1 << 0
HAND_RIICHI = 1 << 1
VALUE_DAMATEN_YAKU = 1 << 0
VALUE_RIICHI_ROUTE = 1 << 1

# Candidate structure flags are tri-state: the known-mask says which facts
# were analyzable and the value-mask says which of those facts are true.
BREAK_MELD = 1 << 0
BREAK_PAIR = 1 << 1
BREAK_RYANMEN = 1 << 2

FEATURE_CONTRACT = {
    "version": FEATURE_SCHEMA_VERSION,
    "categorical_na": "0 unless the row-kind/applicability marker is set",
    "shanten_encoding": "value_plus_2; -1=agari, 0=tenpai, 0=N/A",
    "state_rows": ["hand", "value", "placement", "threat_right", "threat_across", "threat_left"],
    "query_rows": ["offense", "defense"],
    "categorical_slots": {
        "all": {"0": "segment", "1": "row_or_action_kind", "2": "row_or_action_id_plus_1", "9": "query_role"},
        "state.hand": {
            "2": "overall_shanten_plus_2", "3": "standard_shanten_plus_2",
            "4": "seven_pairs_shanten_plus_2", "5": "thirteen_orphans_shanten_plus_2",
            "6": "closed|riichi mask", "7": "1+furiten (rule-analysis applicability)",
            "8": "ukeire_available (13-tile post-action shape only)",
        },
        "state.value": {"2": "analysis_available", "3": "damaten_yaku|riichi_route mask"},
        "state.placement": {
            "2": "rank", "3": "round_wind_plus_1", "4": "kyoku_index_plus_1",
            "5": "continuous_match_progress_plus_1", "6": "1+all_last",
        },
        "state.threat": {
            "2": "relative_seat", "3": "1+declared+2*accepted", "4": "meld_count_plus_1",
            "5": "river_count_plus_1", "6": "tsumogiri_streak_clipped_3",
            "8": "reach_declaration_tile_type_plus_1",
        },
        "query.offense": {
            "3": "preserved_dora_clipped_6_plus_1", "4": "structural_shanten_plus_2",
            "5": "effective_shanten_plus_2", "6": "yaku+2*riichi_route",
            "7": "open_no_yaku+2*furiten+4*closed",
            "8": "packed_ron_tsumo_thousands",
        },
        "query.defense.discard": {
            "3": "threat_count", "4": "genbutsu_opponent_mask",
            "5": "complete_suji_opponent_mask", "6": "adjacent_wall_class_0_to_2",
            "7": "post_reach_pass_mask", "8": "1+structure_known_mask+8*structure_break_mask",
        },
        "query.defense.non_discard": {
            "4": "non_discard_variant", "5": "1+open_call", "6": "1+open_no_yaku",
            "7": "1+kan_under_threat",
        },
    },
    "numeric_slots": {
        "state.hand": ["overall_shanten/6", "standard_shanten/6", "seven_pairs_shanten/6", "orphans_shanten/13", "ukeire/40", "improving_types/34", "reserved", "reserved"],
        "state.value": ["wait_types/13", "live_ron/16", "live_tsumo/16", "ron_points/12000", "tsumo_points/12000", "dora/8", "red/3", "reserved"],
        "state.placement": ["score/50000", "gap_rank1/50000", "gap_rank2/50000", "gap_rank3/50000", "gap_rank4/50000", "honba/10", "sticks/10", "tiles_left/70"],
        "state.threat": ["river/24", "melds/4", "last_tedashi_type_plus_1/35", "tsumogiri_streak/12", "exposed_dora/8", "reach_turn/24", "reserved", "reserved"],
        "query.offense": ["standard_shanten/6", "seven_pairs_shanten/6", "orphans_shanten/13", "ukeire/40", "wait_types/13", "live_ron/16", "live_tsumo/16", "preserved_red/3"],
        "query.defense.discard": ["genbutsu_count/3", "suji_count/3", "visible_count/4", "honor_visible/4", "is_dora", "is_red", "threat_count/3", "known_structure_mask/7"],
        "query.defense.non_discard": ["shanten/6", "ukeire/40", "threat_count/3", "riichi_discard_count/14", "legal_post_call_discards/14", "foregone_shanten/6", "foregone_ukeire/40", "foregone_win"],
    },
    "clipping": {
        "all_numeric": "nonnegative_[0,1];signed_score_and_gaps_[-1,1]",
        "state.value.dora_count": 8,
        "state.threat.exposed_dora_count": 8,
        "query.offense.preserved_dora_count": 6,
        "points_thousands_packed": 255,
        "tsumogiri_streak": 3, "river_count": 14,
    },
    "numeric_scaling": {
        "shanten": 6,
        "orphans_shanten": 13,
        "ukeire": 40,
        "wait_types": 13,
        "live_tiles": 16,
        "points": 12000,
        "score": 50000,
        "river": 24,
    },
    "bit_masks": {
        "hand": {"closed": HAND_CLOSED, "riichi": HAND_RIICHI},
        "value": {
            "damaten_yaku": VALUE_DAMATEN_YAKU,
            "riichi_route": VALUE_RIICHI_ROUTE,
        },
        "structure": {
            "meld": BREAK_MELD,
            "pair": BREAK_PAIR,
            "ryanmen": BREAK_RYANMEN,
        },
    },
}


def feature_schema_sha256() -> str:
    payload = json.dumps(FEATURE_CONTRACT, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def encode_shanten(value: int | None, *, maximum: int) -> int:
    if value is None:
        return 0
    return min(max(int(value) + 2, 1), int(maximum))
