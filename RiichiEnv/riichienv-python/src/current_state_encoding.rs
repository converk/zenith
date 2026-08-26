//! V18 当前局面快照的 Rust/PyO3 批编码器。
//!
//! 直接以原生 `Observation` 当前字段构造共享公共前缀 + 三个 Opponent Analysis 的
//! 扁平行；Action Query 行由 Python 侧沿用 `riichi.encode_query_batch` 生成并拼接。
//! 行布局与 `riichi_ppo_v1/model/encoding_protocol.py` 镜像（见 specs/010 契约 §3）。

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

use riichi::analysis;
use riichi::shanten;
use riichienv_core::observation::Observation;
use riichienv_core::types::{Meld, MeldType};

use crate::encoding_facts::{
    count_dora_aka, decompose_melds, dora_kind, kernel_shape, open_meld_yakuhai_han,
    tile_counts, visible_meld_dora_aka_han,
};

const TILE_KINDS: usize = 34;
const ROW_WIDTH: usize = 32;
const NUMERIC_WIDTH: usize = 8;
const RED_FIVE_TILE_IDS: [u8; 3] = [16, 52, 88];
const MAX_DORA_INDICATORS: usize = 5;

// segment
const SEGMENT_SHARED: u8 = 1;
const SEGMENT_ANALYSIS: u8 = 2;
// kind
const KIND_BOS: u8 = 1;
const KIND_TABLE: u8 = 2;
const KIND_SELF_HAND: u8 = 3;
const KIND_SELF_STATE: u8 = 4;
const KIND_PLAYER: u8 = 5;
const KIND_RIVER_SUMMARY: u8 = 6;
const KIND_RIVER_DISCARD: u8 = 7;
const KIND_MELD: u8 = 8;
const KIND_TILE_STATE: u8 = 9;
const KIND_OPPONENT_ANALYSIS: u8 = 10;
const KIND_SEP_SELF_HAND: u8 = 101;
const KIND_SEP_PLAYERS: u8 = 102;
const KIND_SEP_RIVERS: u8 = 103;
const KIND_SEP_SHIMOCHA_RIVER: u8 = 104;
const KIND_SEP_TOIMEN_RIVER: u8 = 105;
const KIND_SEP_KAMICHA_RIVER: u8 = 106;
const KIND_SEP_MELDS: u8 = 107;
const KIND_SEP_TILE_STATE: u8 = 108;
const KIND_SEP_OPPONENT_ANALYSIS: u8 = 109;

#[pyclass(name = "CurrentStateBatch", frozen)]
pub struct CurrentStateBatch {
    #[pyo3(get)]
    rows: Py<PyArray2<i32>>,
    #[pyo3(get)]
    numeric: Py<PyArray2<f32>>,
    #[pyo3(get)]
    offsets: Py<PyArray1<i64>>,
}

fn push_row(rows: &mut Vec<i32>, segment: u8, kind: u8, fields: &[i32]) {
    rows.push(i32::from(segment));
    rows.push(i32::from(kind));
    for index in 0..30 {
        rows.push(fields.get(index).copied().unwrap_or(0));
    }
}

fn push_numeric(numerics: &mut Vec<f32>, values: &[f32]) {
    for index in 0..NUMERIC_WIDTH {
        numerics.push(values.get(index).copied().unwrap_or(0.0));
    }
}

fn red_flag(tile: u32) -> i32 {
    i32::from(RED_FIVE_TILE_IDS.contains(&(tile as u8)))
}

fn tile_type_code(tile: u32) -> i32 {
    (tile as usize / 4) as i32 + 1
}

fn kind_of(tile: u32) -> usize {
    tile as usize / 4
}

fn is_red(tile: u32) -> bool {
    RED_FIVE_TILE_IDS.contains(&(tile as u8))
}

fn bucket_turn(turn: u8) -> i32 {
    if turn == 0 {
        0
    } else if turn <= 25 {
        i32::from(turn)
    } else {
        26
    }
}

fn bucket_post_riichi(value: usize) -> i32 {
    if value <= 15 {
        value as i32
    } else {
        16
    }
}

fn bucket_count6(value: usize) -> i32 {
    value.min(6) as i32
}

fn bucket_kind_count(value: usize) -> i32 {
    if value <= 33 {
        value as i32
    } else {
        34
    }
}

fn bucket_entity_count(value: usize) -> i32 {
    if value <= 99 {
        value as i32
    } else {
        100
    }
}

fn bucket_yakuhai(value: u8) -> i32 {
    if value <= 5 {
        i32::from(value)
    } else {
        6
    }
}

fn bucket_dora_aka(value: u8) -> i32 {
    if value <= 7 {
        i32::from(value)
    } else {
        8
    }
}

fn bucket_base_han(value: u8) -> i32 {
    if value <= 9 {
        i32::from(value)
    } else {
        10
    }
}

fn bucket_honba(value: u8) -> i32 {
    if value <= 19 {
        i32::from(value)
    } else {
        20
    }
}

fn bucket_sticks(value: u32) -> i32 {
    if value <= 3 {
        value as i32
    } else {
        4
    }
}

/// 排名：分数降序，同分按绝对座次稳定排序。
fn ranks(scores: &[i32; 4]) -> [i32; 4] {
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&a, &b| scores[b].cmp(&scores[a]).then(a.cmp(&b)));
    let mut out = [0_i32; 4];
    for (index, seat) in order.iter().enumerate() {
        out[*seat] = index as i32 + 1;
    }
    out
}

fn normalize_score(value: i32) -> f32 {
    (value as f32 / 100_000.0).clamp(-1.0, 1.0)
}

fn shanten_code(value: i8) -> i32 {
    if value < 0 {
        0
    } else {
        i32::from(value.min(7)) + 1
    }
}

fn self_riichi_status(observation: &Observation, player: usize) -> i32 {
    if observation.riichi_accepted[player] {
        2
    } else if observation.riichi_declared[player] {
        1
    } else {
        0
    }
}

fn decision_mode(observation: &Observation) -> i32 {
    if observation.drawn_tile.is_some() {
        return 0;
    }
    let last = observation.new_events().into_iter().rev().find_map(|raw| {
        serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .and_then(|value| value["type"].as_str().map(str::to_string))
    });
    if last.as_deref() == Some("kakan") {
        2
    } else {
        1
    }
}

fn river_mask(observation: &Observation, player: usize) -> u64 {
    let mut mask = 0_u64;
    for &tile in &observation.discards[player] {
        mask |= 1_u64 << kind_of(tile);
    }
    mask
}

fn relative_code(observer: usize, source: usize) -> i32 {
    if source == observer {
        0
    } else {
        ((source as i32 - observer as i32 + 3) % 4) as i32
    }
}

fn rel_order(seat: usize) -> [usize; 4] {
    [seat, (seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]
}

fn seat_wind(seat: usize, oya: u8) -> u8 {
    ((seat as i32 + 4 - i32::from(oya)) % 4) as u8
}

fn meld_type_code(meld: &Meld) -> i32 {
    match meld.meld_type {
        MeldType::Chi => 1,
        MeldType::Pon => 2,
        MeldType::Daiminkan => 3,
        MeldType::Ankan => 4,
        MeldType::Kakan => 5,
    }
}

fn wind_kind(wind: u8) -> usize {
    27 + usize::from(wind)
}

/// 归一 13 张形状的进张/听牌掩码。
fn progress_masks(
    own_hand: &[u8],
    melds: &[Meld],
) -> Result<(u64, u64, u8), String> {
    let (three_melds, kans) = decompose_melds(melds);
    let (counts, open_count) = kernel_shape(own_hand, three_melds, &kans, 13);
    let value = shanten::calculate(&counts, open_count);
    let after = shanten::calculate_after_draws(&counts, open_count);
    let mut advance = 0_u64;
    let mut wait = 0_u64;
    for tile in 0..TILE_KINDS {
        if counts[tile] >= 4 {
            continue;
        }
        if value.overall > 0 && after[tile] < value.overall {
            advance |= 1_u64 << tile;
        }
        if value.overall == 0 && after[tile] < 0 {
            wait |= 1_u64 << tile;
        }
    }
    Ok((advance, wait, three_melds))
}

/// 剩余实体数（4 - 公开可见 - 自己暗手）。
fn remaining_counts(own_counts: &[u8; TILE_KINDS], public_counts: &[u8; TILE_KINDS]) -> [u8; TILE_KINDS] {
    let mut remaining = [4_u8; TILE_KINDS];
    for kind in 0..TILE_KINDS {
        remaining[kind] = 4_u8
            .saturating_sub(own_counts[kind])
            .saturating_sub(public_counts[kind]);
    }
    remaining
}

/// 当前握着刚摸到的牌的玩家（用于公开暗牌数）。
fn pending_draw_actor(observation: &Observation) -> Option<u8> {
    if observation.drawn_tile.is_some() {
        return Some(observation.player_id);
    }
    let last = observation.new_events().into_iter().rev().find_map(|raw| {
        serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .and_then(|value| match value["type"].as_str() {
                Some("tsumo") => value["actor"].as_u64().map(|actor| actor as u8),
                Some("dahai") | Some("kakan") | Some("pon") | Some("chi") | Some("daiminkan") => {
                    Some(u8::MAX)
                }
                _ => None,
            })
    });
    match last {
        Some(actor) if actor != u8::MAX => Some(actor),
        _ => None,
    }
}

fn concealed_count(observation: &Observation, player: usize, pending: Option<u8>) -> i32 {
    let mut total = 13_i32 + i32::from(pending == Some(player as u8));
    for meld in &observation.melds[player] {
        match meld.meld_type {
            MeldType::Chi | MeldType::Pon => total -= 2,
            MeldType::Daiminkan => total -= 3,
            MeldType::Ankan => total -= 4,
            MeldType::Kakan => total -= 1,
        }
    }
    total.max(0)
}

fn is_supplied(observation: &Observation, discard_seat: usize, tile: u32) -> bool {
    observation.melds.iter().flatten().any(|meld| {
        meld.from_who >= 0
            && meld.from_who as usize == discard_seat
            && meld.called_tile.map(|value| u32::from(value)) == Some(tile)
    })
}

fn riichi_stage(declared: bool, declaration: Option<u8>, index: usize) -> i32 {
    if !declared {
        return 0;
    }
    match declaration {
        None => 0,
        Some(value) => {
            let value = value as usize;
            if index < value {
                0
            } else if index == value {
                1
            } else {
                2
            }
        }
    }
}

fn summary_fields(observation: &Observation, player: usize, recent: bool) -> Vec<i32> {
    let discards = &observation.discards[player];
    let flags = &observation.tsumogiri_flags[player];
    let declared = observation.riichi_declared[player];
    let declaration = observation.riichi_declaration_indices[player];
    let count = discards.len();
    let selected: Vec<(usize, u32)> = if recent {
        let start = count.saturating_sub(6);
        (start..count).map(|index| (index, discards[index])).collect()
    } else {
        (0..count.min(6)).map(|index| (index, discards[index])).collect()
    };
    let mut fields = vec![selected.len() as i32];
    for slot in 0..6 {
        if let Some((index, tile)) = selected.get(slot) {
            let flag = flags.get(*index).copied().unwrap_or(false);
            fields.push(tile_type_code(*tile));
            fields.push(red_flag(*tile));
            fields.push(i32::from(flag));
            fields.push(riichi_stage(declared, declaration, *index));
        } else {
            fields.extend([0, 0, 0, 0]);
        }
    }
    fields
}

fn suji_category(tile: usize, river: u64) -> i32 {
    if tile >= 27 {
        return 3;
    }
    let rank = tile % 9;
    let lower = (rank >= 3).then_some(tile - 3);
    let upper = (rank <= 5).then_some(tile + 3);
    let present = |anchor: Option<usize>| anchor.is_some_and(|value| river & (1_u64 << value) != 0);
    match (present(lower), present(upper)) {
        (true, true) => 2,
        (true, false) | (false, true) => 1,
        (false, false) => 0,
    }
}

fn encode_one(observation: &Observation) -> Result<(Vec<i32>, Vec<f32>), String> {
    let seat = usize::from(observation.player_id);
    if seat >= 4 {
        return Err("observation player_id must be in 0..4".to_string());
    }
    let own_hand: Vec<u8> = observation.hands[seat]
        .iter()
        .map(|tile| *tile as u8)
        .collect();
    let mut rows: Vec<i32> = Vec::new();
    let mut numerics: Vec<f32> = Vec::new();
    let order = rel_order(seat);

    // 公开可见计数：四家牌河 + 全部副露 + 宝牌指示牌。
    let mut public_counts = [0_u8; TILE_KINDS];
    for river in &observation.discards {
        for &tile in river {
            let kind = kind_of(tile);
            if kind < TILE_KINDS {
                public_counts[kind] = public_counts[kind].min(3) + 1;
            }
        }
    }
    for meld_rows in &observation.melds {
        for meld in meld_rows {
            for &tile in &meld.tiles {
                let kind = kind_of(u32::from(tile));
                if kind < TILE_KINDS {
                    public_counts[kind] = public_counts[kind].min(3) + 1;
                }
            }
        }
    }
    for &tile in &observation.dora_indicators {
        let kind = kind_of(tile);
        if kind < TILE_KINDS {
            public_counts[kind] = public_counts[kind].min(3) + 1;
        }
    }
    let own_counts = tile_counts(&own_hand);
    let remaining = remaining_counts(&own_counts, &public_counts);
    let own_river_u8: Vec<u8> = observation.discards[seat]
        .iter()
        .map(|tile| *tile as u8)
        .collect();
    let own_river_counts = tile_counts(&own_river_u8);

    let mut dora_multiplicity = [0_u8; TILE_KINDS];
    for &indicator in &observation.dora_indicators {
        let dora = dora_kind(indicator)?;
        dora_multiplicity[dora] = dora_multiplicity[dora].saturating_add(1);
    }
    let dora_types: Vec<usize> = (0..TILE_KINDS)
        .filter(|kind| dora_multiplicity[*kind] > 0)
        .collect();

    let melds = &observation.melds[seat];
    let (advance_mask, wait_mask, three_melds) = progress_masks(&own_hand, melds)?;
    let shanten_value = shanten::calculate(
        &kernel_shape(&own_hand, three_melds, &decompose_melds(melds).1, 13).0,
        three_melds,
    );
    let advance_kinds = advance_mask.count_ones() as usize;
    let wait_kinds = wait_mask.count_ones() as usize;
    let advance_remaining: usize = (0..TILE_KINDS)
        .filter(|tile| advance_mask & (1_u64 << tile) != 0)
        .map(|tile| usize::from(remaining[tile]))
        .sum();
    let wait_remaining: usize = (0..TILE_KINDS)
        .filter(|tile| wait_mask & (1_u64 << tile) != 0)
        .map(|tile| usize::from(remaining[tile]))
        .sum();
    let pending = pending_draw_actor(observation);
    let self_riichi_accepted = observation.riichi_accepted[seat];
    let drawn_kind = observation.drawn_tile.map(|tile| kind_of(u32::from(tile)));

    // 1) BOS
    push_row(&mut rows, SEGMENT_SHARED, KIND_BOS, &[]);
    push_numeric(&mut numerics, &[]);

    // 2) TABLE
    let mut table = vec![
        i32::from(observation.round_wind),
        i32::from(observation.kyoku_index),
        bucket_honba(observation.honba),
        bucket_sticks(observation.riichi_sticks),
        i32::from(observation.oya),
        seat as i32,
        decision_mode(observation),
        observation.drawn_tile.map(|tile| tile_type_code(u32::from(tile))).unwrap_or(0),
        observation.drawn_tile.map(|tile| red_flag(u32::from(tile))).unwrap_or(0),
        i32::from(observation.drawn_tile.is_some()),
        self_riichi_status(observation, seat),
    ];
    for slot in 0..MAX_DORA_INDICATORS {
        let value = observation.dora_indicators.get(slot).copied();
        table.push(value.map(tile_type_code).unwrap_or(0));
    }
    for slot in 0..MAX_DORA_INDICATORS {
        let value = observation.dora_indicators.get(slot).copied();
        table.push(value.map(red_flag).unwrap_or(0));
    }
    table.push(ranks(&observation.scores)[seat]);
    push_row(&mut rows, SEGMENT_SHARED, KIND_TABLE, &table);
    let score_num: Vec<f32> = observation
        .scores
        .iter()
        .map(|value| normalize_score(*value))
        .collect();
    let diff_num: Vec<f32> = (1..=3)
        .map(|relative| {
            let opponent = (seat + relative) % 4;
            normalize_score(observation.scores[seat] - observation.scores[opponent])
        })
        .collect();
    push_numeric(
        &mut numerics,
        &[score_num[0], score_num[1], score_num[2], score_num[3], diff_num[0], diff_num[1], diff_num[2], 0.0],
    );

    // 3) SEP_SELF_HAND + 4) SELF_HAND
    push_row(&mut rows, SEGMENT_SHARED, KIND_SEP_SELF_HAND, &[]);
    push_numeric(&mut numerics, &[]);
    for kind in 0..TILE_KINDS {
        let count = own_counts[kind];
        if count == 0 {
            continue;
        }
        let has_red = observation.hands[seat]
            .iter()
            .any(|tile| kind_of(*tile) == kind && is_red(*tile));
        let is_drawn = drawn_kind == Some(kind);
        let locked = self_riichi_accepted && (drawn_kind.is_none() || drawn_kind != Some(kind));
        push_row(
            &mut rows,
            SEGMENT_SHARED,
            KIND_SELF_HAND,
            &[
                kind as i32 + 1,
                i32::from(count),
                i32::from(has_red),
                i32::from(is_drawn),
                i32::from(locked),
            ],
        );
        push_numeric(&mut numerics, &[]);
    }

    // 5) SELF_STATE_ANALYSIS
    let self_meld_yakuhai =
        open_meld_yakuhai_han(melds, seat_wind(seat, observation.oya), observation.round_wind)?;
    let all_tiles: Vec<u8> = {
        let mut values = own_hand.clone();
        for meld in melds {
            values.extend(meld.tiles.iter().copied());
        }
        values
    };
    let dora_aka = count_dora_aka(&all_tiles, &dora_multiplicity);
    let aka_count = all_tiles.iter().filter(|tile| is_red(u32::from(**tile))).count() as u8;
    let chiitoi = shanten_value.seven_pairs;
    let kokushi = shanten_value.thirteen_orphans;
    push_row(
        &mut rows,
        SEGMENT_SHARED,
        KIND_SELF_STATE,
        &[
            i32::from(melds.iter().all(|meld| !meld.opened)),
            own_hand.len() as i32,
            melds.len() as i32,
            shanten_code(shanten_value.overall),
            shanten_code(shanten_value.standard),
            if chiitoi == 127 { 9 } else { shanten_code(chiitoi) },
            if kokushi == 127 { 9 } else { shanten_code(kokushi) },
            bucket_kind_count(advance_kinds),
            bucket_entity_count(advance_remaining),
            if wait_kinds == 0 { 0 } else { bucket_kind_count(wait_kinds) },
            if wait_kinds == 0 { 0 } else { bucket_entity_count(wait_remaining) },
            i32::from(observation.missed_agari_riichi),
            i32::from(observation.missed_agari_doujun),
            i32::from(observation.missed_agari_riichi && self_riichi_accepted),
            bucket_dora_aka(dora_aka),
            if aka_count <= 5 { i32::from(aka_count) } else { 5 },
            bucket_yakuhai(self_meld_yakuhai),
            bucket_base_han(dora_aka + aka_count + self_meld_yakuhai),
        ],
    );
    push_numeric(&mut numerics, &[]);

    // 6) SEP_PLAYERS + 7) PLAYER × 4
    push_row(&mut rows, SEGMENT_SHARED, KIND_SEP_PLAYERS, &[]);
    push_numeric(&mut numerics, &[]);
    let rank_values = ranks(&observation.scores);
    for (relative, player) in order.iter().enumerate() {
        let player = *player;
        let flags = &observation.tsumogiri_flags[player];
        let river_len = observation.discards[player].len();
        let declaration = observation.riichi_declaration_indices[player];
        let status = self_riichi_status(observation, player);
        let riichi_turn = declaration.map(|value| value + 1).unwrap_or(0);
        let decl_tile = declaration
            .and_then(|index| observation.discards[player].get(index as usize).copied());
        let post_riichi = declaration.map(|index| {
            let start = usize::from(index) + 1;
            let mut cut = 0usize;
            let mut tsumo = 0usize;
            for flag in flags.iter().skip(start) {
                if *flag {
                    tsumo += 1;
                } else {
                    cut += 1;
                }
            }
            (cut, tsumo)
        });
        let kan_count = observation.melds[player]
            .iter()
            .filter(|meld| matches!(meld.meld_type, MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan))
            .count() as i32;
        let player_yakuhai = open_meld_yakuhai_han(
            &observation.melds[player],
            seat_wind(player, observation.oya),
            observation.round_wind,
        )?;
        let player_dora_aka = visible_meld_dora_aka_han(&observation.melds[player], &dora_multiplicity)?;
        push_row(
            &mut rows,
            SEGMENT_SHARED,
            KIND_PLAYER,
            &[
                relative as i32,
                player as i32,
                i32::from(seat_wind(player, observation.oya)),
                i32::from(player == observation.oya as usize),
                rank_values[player],
                concealed_count(observation, player, pending),
                observation.melds[player].len() as i32,
                kan_count,
                i32::from(observation.melds[player].iter().all(|meld| !meld.opened)),
                river_len.min(24) as i32,
                status,
                bucket_turn(riichi_turn),
                decl_tile.map(tile_type_code).unwrap_or(0),
                decl_tile.map(red_flag).unwrap_or(0),
                post_riichi.map(|(cut, tsumo)| bucket_post_riichi(cut + tsumo)).unwrap_or(0),
                bucket_yakuhai(player_yakuhai),
                bucket_dora_aka(player_dora_aka),
            ],
        );
        push_numeric(
            &mut numerics,
            &[
                normalize_score(observation.scores[player]),
                normalize_score(observation.scores[player] - observation.scores[seat]),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );
    }

    // 8) SEP_RIVERS + 三家牌河
    push_row(&mut rows, SEGMENT_SHARED, KIND_SEP_RIVERS, &[]);
    push_numeric(&mut numerics, &[]);
    for river_index in 0..3 {
        let player = order[river_index + 1];
        let river_sep = match river_index {
            0 => KIND_SEP_SHIMOCHA_RIVER,
            1 => KIND_SEP_TOIMEN_RIVER,
            _ => KIND_SEP_KAMICHA_RIVER,
        };
        push_row(&mut rows, SEGMENT_SHARED, river_sep, &[]);
        push_numeric(&mut numerics, &[]);
        push_row(
            &mut rows,
            SEGMENT_SHARED,
            KIND_RIVER_SUMMARY,
            &summary_fields(observation, player, false),
        );
        push_numeric(&mut numerics, &[]);
        let discards = &observation.discards[player];
        let flags = &observation.tsumogiri_flags[player];
        let declared = observation.riichi_declared[player];
        let declaration = observation.riichi_declaration_indices[player];
        let count = discards.len();
        for (index, tile) in discards.iter().enumerate() {
            let age = if count <= 1 {
                0
            } else {
                let distance = count - 1 - index;
                if distance == 0 {
                    0
                } else if distance <= 2 {
                    1
                } else if distance <= 5 {
                    2
                } else {
                    3
                }
            };
            push_row(
                &mut rows,
                SEGMENT_SHARED,
                KIND_RIVER_DISCARD,
                &[
                    (river_index + 1) as i32,
                    (index + 1) as i32,
                    tile_type_code(*tile),
                    red_flag(*tile),
                    i32::from(flags.get(index).copied().unwrap_or(false)),
                    riichi_stage(declared, declaration, index),
                    i32::from(is_supplied(observation, player, *tile)),
                    age,
                ],
            );
            push_numeric(&mut numerics, &[]);
        }
        push_row(
            &mut rows,
            SEGMENT_SHARED,
            KIND_RIVER_SUMMARY,
            &summary_fields(observation, player, true),
        );
        push_numeric(&mut numerics, &[]);
    }

    // 15) SEP_MELDS + 16) MELD × M
    push_row(&mut rows, SEGMENT_SHARED, KIND_SEP_MELDS, &[]);
    push_numeric(&mut numerics, &[]);
    for (owner_relative, player) in order.iter().enumerate() {
        let player = *player;
        for (meld_index, meld) in observation.melds[player].iter().enumerate() {
            let mut fields = vec![owner_relative as i32, meld_type_code(meld)];
            for slot in 0..4 {
                if let Some(tile) = meld.tiles.get(slot) {
                    fields.push(tile_type_code(u32::from(*tile)));
                    fields.push(red_flag(u32::from(*tile)));
                } else {
                    fields.extend([0, 0]);
                }
            }
            fields.push(
                meld.called_tile
                    .map(|tile| tile_type_code(u32::from(tile)))
                    .unwrap_or(0),
            );
            fields.push(
                meld.called_tile
                    .map(|tile| red_flag(u32::from(tile)))
                    .unwrap_or(0),
            );
            fields.push(if meld.from_who >= 0 {
                relative_code(seat, meld.from_who as usize)
            } else {
                0
            });
            fields.push(i32::from(meld.opened));
            fields.push((meld_index + 1) as i32);
            let meld_yakuhai = open_meld_yakuhai_han(
                &[meld.clone()],
                seat_wind(player, observation.oya),
                observation.round_wind,
            )?;
            let meld_dora_aka = visible_meld_dora_aka_han(&[meld.clone()], &dora_multiplicity)?;
            fields.push(bucket_yakuhai(meld_yakuhai));
            fields.push(bucket_dora_aka(meld_dora_aka));
            push_row(&mut rows, SEGMENT_SHARED, KIND_MELD, &fields);
            push_numeric(&mut numerics, &[]);
        }
    }

    // 17) SEP_TILE_STATE + 18) TILE_STATE × 34
    push_row(&mut rows, SEGMENT_SHARED, KIND_SEP_TILE_STATE, &[]);
    push_numeric(&mut numerics, &[]);
    for kind in 0..TILE_KINDS {
        let public_count = i32::from(public_counts[kind]);
        let own_concealed = i32::from(own_counts[kind]);
        let known = (public_count + own_concealed).min(4);
        let unknown = 4 - known;
        let mut genbutsu = [0_i32; 3];
        let mut suji = [0_i32; 3];
        for river_index in 0..3 {
            let player = order[river_index + 1];
            let river = river_mask(observation, player);
            genbutsu[river_index] = i32::from(river & (1_u64 << kind) != 0);
            suji[river_index] = suji_category(kind, river);
        }
        let wall = analysis::wall_class(kind, &remaining);
        let dora_neighbor = kind < 27
            && dora_types.iter().any(|dora| {
                *dora < 27
                    && dora / 9 == kind / 9
                    && (dora % 9).abs_diff(kind % 9) == 1
            });
        push_row(
            &mut rows,
            SEGMENT_SHARED,
            KIND_TILE_STATE,
            &[
                kind as i32 + 1,
                own_concealed,
                i32::from(own_river_counts[kind]),
                i32::from(own_river_counts[kind] > 0),
                public_count,
                known,
                unknown,
                i32::from(unknown == 0),
                i32::from(dora_multiplicity[kind]),
                i32::from(dora_multiplicity[kind] > 0),
                i32::from(kind == wind_kind(observation.round_wind)),
                i32::from(kind == wind_kind(seat_wind(seat, observation.oya))),
                i32::from(kind == 4 || kind == 13 || kind == 22),
                i32::from(advance_mask & (1_u64 << kind) != 0),
                i32::from(wait_mask & (1_u64 << kind) != 0),
                genbutsu[0],
                genbutsu[1],
                genbutsu[2],
                suji[0],
                suji[1],
                suji[2],
                wall.into(),
                i32::from(dora_neighbor),
            ],
        );
        push_numeric(&mut numerics, &[]);
    }

    // 19) SEP_OPPONENT_ANALYSIS + 20) OPPONENT_ANALYSIS × 3
    push_row(&mut rows, SEGMENT_ANALYSIS, KIND_SEP_OPPONENT_ANALYSIS, &[]);
    push_numeric(&mut numerics, &[]);
    for river_index in 0..3 {
        let player = order[river_index + 1];
        let relative = (river_index + 1) as i32;
        let declaration = observation.riichi_declaration_indices[player];
        let flags = &observation.tsumogiri_flags[player];
        let count = observation.discards[player].len();
        let (post_cut, post_tsumo) = declaration
            .map(|index| {
                let start = usize::from(index) + 1;
                let mut cut = 0usize;
                let mut tsumo = 0usize;
                for flag in flags.iter().skip(start) {
                    if *flag {
                        tsumo += 1;
                    } else {
                        cut += 1;
                    }
                }
                (cut, tsumo)
            })
            .unwrap_or((0, 0));
        let recent_start = count.saturating_sub(6);
        let mut recent_cut = 0usize;
        let mut recent_tsumo = 0usize;
        for flag in flags.iter().skip(recent_start) {
            if *flag {
                recent_tsumo += 1;
            } else {
                recent_cut += 1;
            }
        }
        let river_mask_value = river_mask(observation, player);
        let mut own_genbutsu_kinds = 0usize;
        let mut own_genbutsu_entities = 0usize;
        for kind in 0..TILE_KINDS {
            if river_mask_value & (1_u64 << kind) != 0 && own_counts[kind] > 0 {
                own_genbutsu_kinds += 1;
                own_genbutsu_entities += usize::from(own_counts[kind]);
            }
        }
        let decl_tile = declaration
            .and_then(|index| observation.discards[player].get(index as usize).copied());
        let opp_yakuhai = open_meld_yakuhai_han(
            &observation.melds[player],
            seat_wind(player, observation.oya),
            observation.round_wind,
        )?;
        let opp_dora_aka = visible_meld_dora_aka_han(&observation.melds[player], &dora_multiplicity)?;
        push_row(
            &mut rows,
            SEGMENT_ANALYSIS,
            KIND_OPPONENT_ANALYSIS,
            &[
                relative,
                self_riichi_status(observation, player),
                bucket_turn(declaration.map(|value| value + 1).unwrap_or(0)),
                decl_tile.map(tile_type_code).unwrap_or(0),
                decl_tile.map(red_flag).unwrap_or(0),
                i32::from(observation.melds[player].iter().all(|meld| !meld.opened)),
                concealed_count(observation, player, pending),
                observation.melds[player].len() as i32,
                observation.melds[player]
                    .iter()
                    .filter(|meld| matches!(meld.meld_type, MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan))
                    .count() as i32,
                bucket_yakuhai(opp_yakuhai),
                bucket_dora_aka(opp_dora_aka),
                bucket_post_riichi(post_cut),
                bucket_post_riichi(post_tsumo),
                bucket_count6(recent_cut),
                bucket_count6(recent_tsumo),
                bucket_kind_count(own_genbutsu_kinds),
                bucket_entity_count(own_genbutsu_entities),
                count.min(24) as i32,
            ],
        );
        push_numeric(&mut numerics, &[]);
    }
    Ok((rows, numerics))
}

/// 从原生 Observation 列表批量派生当前局面行。
#[pyfunction]
pub fn prepare_current_state_batch(
    py: Python<'_>,
    observations: Vec<Observation>,
) -> PyResult<CurrentStateBatch> {
    if observations.is_empty() {
        return Err(PyValueError::new_err("current-state 批次不能为空"));
    }
    let mut all_rows: Vec<i32> = Vec::new();
    let mut all_numeric: Vec<f32> = Vec::new();
    let mut offsets: Vec<i64> = vec![0];
    for observation in &observations {
        let (rows, numerics) = encode_one(observation).map_err(PyValueError::new_err)?;
        if rows.len() % ROW_WIDTH != 0 || numerics.len() % NUMERIC_WIDTH != 0 {
            return Err(PyValueError::new_err("编码器输出了错位的行"));
        }
        all_rows.extend(rows);
        all_numeric.extend(numerics);
        offsets.push((all_rows.len() / ROW_WIDTH) as i64);
    }
    let total = all_rows.len() / ROW_WIDTH;
    let rows_array = Array2::from_shape_vec((total, ROW_WIDTH), all_rows)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_pyarray(py);
    let numeric_array = Array2::from_shape_vec((total, NUMERIC_WIDTH), all_numeric)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_pyarray(py);
    let offsets_array = PyArray1::from_vec(py, offsets);
    Ok(CurrentStateBatch {
        rows: rows_array.unbind(),
        numeric: numeric_array.unbind(),
        offsets: offsets_array.unbind(),
    })
}
