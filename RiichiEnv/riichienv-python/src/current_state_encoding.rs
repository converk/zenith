//! V18 当前局面快照的 Rust/PyO3 批编码器。
//!
//! 直接以原生 `Observation` 当前字段构造共享公共前缀 + 三个 Opponent Analysis 的
//! 扁平行；Action Query 行由 Python 侧沿用 `riichi.encode_query_batch` 生成并拼接。
//! 行布局与 `riichi_ppo_v1/model/encoding_protocol.py` 镜像。

use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
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
        (source as i32 - observer as i32 + 3) % 4
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
                Some("tsumo") | Some("pon") | Some("chi") | Some("daiminkan")
                | Some("ankan") | Some("kakan") => value["actor"].as_u64().map(|actor| actor as u8),
                Some("dahai") => Some(u8::MAX),
                _ => None,
            })
    });
    match last {
        Some(actor) if actor != u8::MAX => Some(actor),
        _ => None,
    }
}

fn concealed_count_from(melds: &[Meld], pending: Option<u8>, player: usize) -> i32 {
    let three = melds
        .iter()
        .filter(|meld| matches!(meld.meld_type, MeldType::Chi | MeldType::Pon))
        .count();
    let kans = melds
        .iter()
        .filter(|meld| matches!(meld.meld_type, MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan))
        .count();
    (13_i32 + i32::from(pending == Some(player as u8)) - 3 * three as i32 - 4 * kans as i32).max(0)
}

fn concealed_count(observation: &Observation, player: usize, pending: Option<u8>) -> i32 {
    concealed_count_from(&observation.melds[player], pending, player)
}

/// 收集每座牌河中被鸣的下标集合（0 基，实体去重用）。
fn claimed_river_indices(melds: &[Vec<Meld>], river_lengths: &[usize]) -> Vec<Vec<bool>> {
    let mut claimed: Vec<Vec<bool>> = river_lengths.iter().map(|&len| vec![false; len]).collect();
    for meld_rows in melds {
        for meld in meld_rows {
            if meld.from_who >= 0
                && let Some(index) = meld.called_tile_index
            {
                let from = meld.from_who as usize;
                if from < claimed.len()
                    && let Some(slot) = claimed[from].get_mut(index as usize)
                {
                    *slot = true;
                }
            }
        }
    }
    claimed
}

/// 实体口径公开计数：副露全部 + 未被鸣的河牌 + 宝牌指示牌，被鸣河牌只出现一次。
fn entity_public_counts(
    melds: &[Vec<Meld>],
    discards: &[Vec<u32>],
    dora_indicators: &[u32],
) -> [u8; TILE_KINDS] {
    let river_lengths: Vec<usize> = discards.iter().map(|river| river.len()).collect();
    let claimed = claimed_river_indices(melds, &river_lengths);
    let mut counts = [0_u8; TILE_KINDS];
    for (player, river) in discards.iter().enumerate() {
        for (index, &tile) in river.iter().enumerate() {
            if claimed[player].get(index).copied().unwrap_or(false) {
                continue;
            }
            let kind = kind_of(tile);
            if kind < TILE_KINDS {
                counts[kind] = counts[kind].min(3) + 1;
            }
        }
    }
    for meld_rows in melds {
        for meld in meld_rows {
            for &tile in &meld.tiles {
                let kind = kind_of(u32::from(tile));
                if kind < TILE_KINDS {
                    counts[kind] = counts[kind].min(3) + 1;
                }
            }
        }
    }
    for &tile in dora_indicators {
        let kind = kind_of(tile);
        if kind < TILE_KINDS {
            counts[kind] = counts[kind].min(3) + 1;
        }
    }
    counts
}

/// 判断某条河牌（0 基下标）是否恰好被某副露鸣走。
fn is_supplied_by(melds: &[Vec<Meld>], discard_seat: usize, river_index: usize) -> bool {
    melds.iter().flatten().any(|meld| {
        meld.from_who >= 0
            && meld.from_who as usize == discard_seat
            && meld.called_tile_index == Some(river_index as u8)
    })
}

fn is_supplied(observation: &Observation, discard_seat: usize, river_index: usize) -> bool {
    is_supplied_by(&observation.melds, discard_seat, river_index)
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

    // 公开可见计数（实体口径：被鸣河牌不再双计，只出现在副露里一次）。
    let public_counts = entity_public_counts(
        &observation.melds,
        &observation.discards,
        &observation.dora_indicators,
    );
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
    for (kind, &count) in own_counts.iter().enumerate() {
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
                    i32::from(is_supplied(observation, player, index)),
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
                std::slice::from_ref(meld),
                seat_wind(player, observation.oya),
                observation.round_wind,
            )?;
            let meld_dora_aka =
                visible_meld_dora_aka_han(std::slice::from_ref(meld), &dora_multiplicity)?;
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
        for (kind, &count) in own_counts.iter().enumerate() {
            if river_mask_value & (1_u64 << kind) != 0 && count > 0 {
                own_genbutsu_kinds += 1;
                own_genbutsu_entities += usize::from(count);
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
    // 计算段释放 GIL(observations 已在参数绑定时深拷贝为 Vec,encode_one
    // 纯 Rust 计算):供多线程 worker 并行调用本函数时真正并发。
    let encode_result: Result<(), PyErr> = py.detach(|| {
        for observation in &observations {
            let (rows, numerics) =
                encode_one(observation).map_err(|error| PyValueError::new_err(error.to_string()))?;
            if rows.len() % ROW_WIDTH != 0 || numerics.len() % NUMERIC_WIDTH != 0 {
                return Err(PyValueError::new_err("编码器输出了错位的行"));
            }
            all_rows.extend(rows);
            all_numeric.extend(numerics);
            offsets.push((all_rows.len() / ROW_WIDTH) as i64);
        }
        Ok(())
    });
    encode_result?;
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

// ---- V18 完整 Actor 批装配（Rust 侧一次性物化，替代 Python 逐决策 numpy 拼接）----

/// Action Query 行宽（与 ``encoding_protocol.QUERY_ROW_WIDTH`` 镜像）。
const QUERY_ROW_WIDTH: usize = 15;
/// Query 行中答案特征起始下标（与 ``encoding_protocol.QUERY_ROW_ANSWER_START`` 镜像）。
const QUERY_ROW_ANSWER_START: usize = 5;
/// Query 行中 action id 下标。
const QUERY_ROW_ACTION_ID: usize = 1;
/// 动作段与 O/D Query kind（与 ``encoding_protocol`` 镜像）。
const SEGMENT_ACTIONS: u8 = 3;
const KIND_ACTION_OFFENSE_QUERY: i32 = 11;
const KIND_ACTION_DEFENSE_QUERY: i32 = 12;
const KIND_SEP_ACTIONS: i32 = 110;
/// 固定 241 维动作空间（断言用）。
const NUM_ACTIONS: usize = 241;

/// Rust 装配后的完整 V18 Actor 批（含动作 token/分隔符、query 元数据与合法掩码）。
#[pyclass(name = "AssembledCurrentStateBatch", frozen)]
pub struct AssembledCurrentStateBatch {
    #[pyo3(get)]
    actor_factors: Py<PyArray3<i32>>,
    #[pyo3(get)]
    actor_numeric: Py<PyArray3<f32>>,
    #[pyo3(get)]
    actor_lengths: Py<PyArray1<i64>>,
    #[pyo3(get)]
    query_rows: Py<PyArray3<i32>>,
    #[pyo3(get)]
    action_ids: Py<PyArray2<i32>>,
    #[pyo3(get)]
    query_pair_counts: Py<PyArray1<i64>>,
    #[pyo3(get)]
    legal_mask: Py<PyArray2<bool>>,
}

fn tsumogiri_mode(action_id: i32) -> i32 {
    if (1..75).contains(&action_id) {
        (action_id - 1) % 2
    } else {
        0
    }
}

/// 把一条 15 宽 query 行展开成 32 宽 action token 行（段/kind + 15 个嵌入特征）。
fn write_action_row(
    out: &mut [i32],
    offset: usize,
    query_row: &[i32],
    kind: i32,
    action_id: i32,
) {
    debug_assert!(query_row.len() >= QUERY_ROW_WIDTH);
    out[offset] = i32::from(SEGMENT_ACTIONS);
    out[offset + 1] = kind;
    out[offset + 2] = query_row[2];
    out[offset + 3] = query_row[3];
    out[offset + 4] = query_row[4];
    out[offset + 5] = tsumogiri_mode(action_id);
    out[offset + 6] = query_row[QUERY_ROW_ACTION_ID];
    for (index, value) in query_row
        .iter()
        .enumerate()
        .skip(QUERY_ROW_ANSWER_START)
        .take(QUERY_ROW_WIDTH - QUERY_ROW_ANSWER_START)
    {
        out[offset + 7 + (index - QUERY_ROW_ANSWER_START)] = *value;
    }
}

/// 一次装配完整 Actor 批：Shared/Analysis 行 + SEP_ACTIONS + O/D Query 行。
///
/// 输入均为 ``encode_batch`` 已有中间产物（Rust 编码行与 query 行），输出布局与
/// 旧 Python 装配逐位一致：``actor_factors`` 容量 =
/// ``max(最大共享/分析行数 + 1 + 2×最大动作对数, 1)``。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn assemble_current_state_batch<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray2<'py, i32>,
    numerics: PyReadonlyArray2<'py, f32>,
    offsets: PyReadonlyArray1<'py, i64>,
    query_rows: PyReadonlyArray2<'py, i32>,
    action_ids: PyReadonlyArray1<'py, i32>,
    pair_counts: PyReadonlyArray1<'py, i64>,
    legal_mask: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<AssembledCurrentStateBatch> {
    let rows_shape = rows.shape();
    let numerics_shape = numerics.shape();
    let query_shape = query_rows.shape();
    if rows_shape[1] != ROW_WIDTH || numerics_shape[1] != NUMERIC_WIDTH {
        return Err(PyValueError::new_err(
            "rows/numerics row width must be 32/8",
        ));
    }
    if query_shape[1] != QUERY_ROW_WIDTH {
        return Err(PyValueError::new_err("query_rows row width must be 15"));
    }
    let batch = pair_counts.len();
    if batch == 0 {
        return Err(PyValueError::new_err("assemble batch cannot be empty"));
    }
    if offsets.len() != batch + 1 {
        return Err(PyValueError::new_err(
            "offsets must have batch+1 entries",
        ));
    }
    if let Some(legal) = &legal_mask
        && legal.shape() != [batch, NUM_ACTIONS]
    {
        return Err(PyValueError::new_err("legal_mask must be [batch, 241]"));
    }
    let rows_slice = rows
        .as_slice()
        .map_err(|_| PyValueError::new_err("rows must be contiguous"))?;
    let numerics_slice = numerics
        .as_slice()
        .map_err(|_| PyValueError::new_err("numerics must be contiguous"))?;
    let offsets_slice = offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("offsets must be contiguous"))?;
    let query_slice = query_rows
        .as_slice()
        .map_err(|_| PyValueError::new_err("query_rows must be contiguous"))?;
    let ids_slice = action_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("action_ids must be contiguous"))?;
    let counts_slice = pair_counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("pair_counts must be contiguous"))?;
    let provided_legal: Option<Vec<bool>> = match &legal_mask {
        Some(array) => Some(
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("legal_mask must be contiguous"))?
                .to_vec(),
        ),
        None => None,
    };
    let total_pairs: usize = counts_slice.iter().map(|&count| count as usize).sum();
    if query_rows.shape()[0] < total_pairs * 2 {
        return Err(PyValueError::new_err(
            "query_rows row count must cover 2×total action pairs",
        ));
    }
    if action_ids.len() != total_pairs {
        return Err(PyValueError::new_err(
            "action_ids length must equal total action pairs",
        ));
    }
    let native_total = rows_shape[0];
    if offsets_slice[batch] as usize != native_total {
        return Err(PyValueError::new_err(
            "offsets must cover exactly the native rows",
        ));
    }

    // 兼容旧 Python 容量公式：max_native + 1 + 2×max_pairs。
    let native_max = counts_slice
        .iter()
        .zip(offsets_slice.windows(2))
        .map(|(_count, window)| (window[1] - window[0]) as usize)
        .max()
        .unwrap_or(0);
    let pair_max = counts_slice.iter().map(|&count| count as usize).max().unwrap_or(0);
    let capacity = std::cmp::max(native_max + 1 + 2 * pair_max, 1);

    let mut actor_factors = vec![0_i32; batch * capacity * ROW_WIDTH];
    let mut actor_numeric = vec![0_f32; batch * capacity * NUMERIC_WIDTH];
    let mut query_out = vec![0_i32; batch * 2 * pair_max * QUERY_ROW_WIDTH];
    let mut ids_out = vec![0_i32; batch * pair_max];
    let mut lengths_out = vec![0_i64; batch];
    let mut pair_cursor = 0usize;
    for batch_index in 0..batch {
        let native_length = (offsets_slice[batch_index + 1] - offsets_slice[batch_index]) as usize;
        let pair_count = counts_slice[batch_index] as usize;
        let assembled_length = native_length + 1 + 2 * pair_count;
        lengths_out[batch_index] = assembled_length as i64;
        let base = batch_index * capacity * ROW_WIDTH;
        let numeric_base = batch_index * capacity * NUMERIC_WIDTH;
        let native_start = offsets_slice[batch_index] as usize;
        for row_index in 0..native_length {
            let src = (native_start + row_index) * ROW_WIDTH;
            let dst = base + row_index * ROW_WIDTH;
            actor_factors[dst..dst + ROW_WIDTH]
                .copy_from_slice(&rows_slice[src..src + ROW_WIDTH]);
            let numeric_src = (native_start + row_index) * NUMERIC_WIDTH;
            let numeric_dst = numeric_base + row_index * NUMERIC_WIDTH;
            actor_numeric[numeric_dst..numeric_dst + NUMERIC_WIDTH].copy_from_slice(
                &numerics_slice[numeric_src..numeric_src + NUMERIC_WIDTH],
            );
        }
        let sep_offset = base + native_length * ROW_WIDTH;
        actor_factors[sep_offset] = i32::from(SEGMENT_ACTIONS);
        actor_factors[sep_offset + 1] = KIND_SEP_ACTIONS;
        let query_base = batch_index * 2 * pair_max * QUERY_ROW_WIDTH;
        let ids_base = batch_index * pair_max;
        for pair_index in 0..pair_count {
            let query_src = (pair_cursor + pair_index) * 2 * QUERY_ROW_WIDTH;
            let action_id = ids_slice[pair_cursor + pair_index];
            let offense_offset = sep_offset + (1 + 2 * pair_index) * ROW_WIDTH;
            write_action_row(
                &mut actor_factors,
                offense_offset,
                &query_slice[query_src..query_src + QUERY_ROW_WIDTH],
                KIND_ACTION_OFFENSE_QUERY,
                action_id,
            );
            let defense_offset = offense_offset + ROW_WIDTH;
            write_action_row(
                &mut actor_factors,
                defense_offset,
                &query_slice[query_src + QUERY_ROW_WIDTH..query_src + 2 * QUERY_ROW_WIDTH],
                KIND_ACTION_DEFENSE_QUERY,
                action_id,
            );
            let dst = query_base + 2 * pair_index * QUERY_ROW_WIDTH;
            query_out[dst..dst + QUERY_ROW_WIDTH]
                .copy_from_slice(&query_slice[query_src..query_src + QUERY_ROW_WIDTH]);
            query_out[dst + QUERY_ROW_WIDTH..dst + 2 * QUERY_ROW_WIDTH].copy_from_slice(
                &query_slice[query_src + QUERY_ROW_WIDTH..query_src + 2 * QUERY_ROW_WIDTH],
            );
            ids_out[ids_base + pair_index] = action_id;
        }
        pair_cursor += pair_count;
    }

    let legal_out: Vec<bool> = match provided_legal {
        Some(values) => values,
        None => {
            let mut values = vec![false; batch * NUM_ACTIONS];
            for batch_index in 0..batch {
                for pair_index in 0..counts_slice[batch_index] as usize {
                    let action_id = ids_out[batch_index * pair_max + pair_index] as usize;
                    if action_id < NUM_ACTIONS {
                        values[batch_index * NUM_ACTIONS + action_id] = true;
                    }
                }
            }
            values
        }
    };

    let actor_factors = Array3::from_shape_vec((batch, capacity, ROW_WIDTH), actor_factors)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_pyarray(py);
    let actor_numeric = Array3::from_shape_vec((batch, capacity, NUMERIC_WIDTH), actor_numeric)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_pyarray(py);
    let actor_lengths = PyArray1::from_vec(py, lengths_out);
    let query_rows = Array3::from_shape_vec(
        (batch, 2 * pair_max, QUERY_ROW_WIDTH),
        query_out,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?
    .into_pyarray(py);
    let action_ids = Array2::from_shape_vec((batch, pair_max), ids_out)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_pyarray(py);
    let query_pair_counts = PyArray1::from_vec(py, counts_slice.to_vec());
    let legal_mask = Array2::from_shape_vec((batch, NUM_ACTIONS), legal_out)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .into_pyarray(py);
    Ok(AssembledCurrentStateBatch {
        actor_factors: actor_factors.unbind(),
        actor_numeric: actor_numeric.unbind(),
        actor_lengths: actor_lengths.unbind(),
        query_rows: query_rows.unbind(),
        action_ids: action_ids.unbind(),
        query_pair_counts: query_pair_counts.unbind(),
        legal_mask: legal_mask.unbind(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meld(meld_type: MeldType, from_who: i8, called_tile: Option<u8>, index: Option<u8>) -> Meld {
        Meld::new_with_index(
            meld_type,
            match meld_type {
                MeldType::Pon => vec![108, 108, 109],
                MeldType::Chi => vec![96, 100, 104],
                MeldType::Daiminkan => vec![108, 108, 109, 110],
                MeldType::Ankan => vec![16, 17, 18, 19],
                MeldType::Kakan => vec![108, 108, 109, 110],
            },
            meld_type != MeldType::Ankan,
            from_who,
            called_tile,
            index,
        )
    }

    #[test]
    fn supplied_marks_only_exact_claimed_river_index() {
        // 同牌种两张河牌：只有下标 0 被鸣，is_supplied 只标 0，不标 1。
        let melds = vec![vec![meld(MeldType::Pon, 1, Some(108), Some(0))]];
        assert!(is_supplied_by(&melds, 1, 0));
        assert!(!is_supplied_by(&melds, 1, 1));
        assert!(!is_supplied_by(&melds, 1, 2));
        // 下标 1 被鸣时只标 1。
        let melds2 = vec![vec![meld(MeldType::Pon, 1, Some(108), Some(1))]];
        assert!(!is_supplied_by(&melds2, 1, 0));
        assert!(is_supplied_by(&melds2, 1, 1));
    }

    #[test]
    fn entity_counts_deduplicate_claimed_river_tile() {
        // 河 1 有 [E,E,9s]（108,108,104），副露为 p2 的 Pon E（from p1, index 0）。
        let melds = vec![
            vec![],
            vec![],
            vec![meld(MeldType::Pon, 1, Some(108), Some(0))],
            vec![],
        ];
        let discards = vec![vec![], vec![108, 108, 104], vec![], vec![]];
        let dora: Vec<u32> = vec![];
        let counts = entity_public_counts(&melds, &discards, &dora);
        assert_eq!(counts[27], 4); // 河 1 张 E（另一张被鸣）+ 副露 3 张 E = 4 实体
        assert_eq!(counts[26], 1); // 9s
    }

    #[test]
    fn entity_counts_do_not_double_count_with_cap() {
        // 河 2×E + 副露 3×E = 5 出现次数 → 实体 public=4（E 共 4 张），不被 5 掩盖。
        let melds = vec![
            vec![],
            vec![],
            vec![meld(MeldType::Pon, 1, Some(108), Some(0))],
            vec![],
        ];
        let discards = vec![vec![], vec![108, 108], vec![], vec![]];
        let counts = entity_public_counts(&melds, &discards, &[]);
        assert_eq!(counts[27], 4);
    }

    #[test]
    fn concealed_count_matches_contract_formula() {
        // 13 + pending - 3×三张副露 - 4×杠。
        let no_meld: Vec<Meld> = vec![];
        assert_eq!(concealed_count_from(&no_meld, Some(0), 0), 14);
        assert_eq!(concealed_count_from(&no_meld, None, 0), 13);
        let pon = vec![meld(MeldType::Pon, 1, Some(108), Some(0))];
        assert_eq!(concealed_count_from(&pon, Some(0), 0), 11);
        assert_eq!(concealed_count_from(&pon, None, 0), 10);
        let chi = vec![meld(MeldType::Chi, 1, Some(104), Some(0))];
        assert_eq!(concealed_count_from(&chi, None, 0), 10);
        let daiminkan = vec![meld(MeldType::Daiminkan, 1, Some(108), Some(0))];
        assert_eq!(concealed_count_from(&daiminkan, None, 0), 9);
        let ankan = vec![meld(MeldType::Ankan, -1, None, None)];
        assert_eq!(concealed_count_from(&ankan, None, 0), 9);
        let kakan = vec![meld(MeldType::Kakan, 1, Some(108), Some(0))];
        assert_eq!(concealed_count_from(&kakan, None, 0), 9);
    }

    #[test]
    fn bucket_boundaries_are_exact() {
        assert_eq!(bucket_honba(19), 19);
        assert_eq!(bucket_honba(20), 20);
        assert_eq!(bucket_sticks(3), 3);
        assert_eq!(bucket_sticks(4), 4);
        assert_eq!(bucket_turn(0), 0);
        assert_eq!(bucket_turn(25), 25);
        assert_eq!(bucket_turn(26), 26);
        assert_eq!(bucket_post_riichi(15), 15);
        assert_eq!(bucket_post_riichi(16), 16);
        assert_eq!(bucket_count6(5), 5);
        assert_eq!(bucket_count6(6), 6);
        assert_eq!(bucket_kind_count(33), 33);
        assert_eq!(bucket_kind_count(34), 34);
        assert_eq!(bucket_entity_count(99), 99);
        assert_eq!(bucket_entity_count(100), 100);
        assert_eq!(bucket_yakuhai(5), 5);
        assert_eq!(bucket_yakuhai(6), 6);
        assert_eq!(bucket_dora_aka(7), 7);
        assert_eq!(bucket_dora_aka(8), 8);
        assert_eq!(bucket_base_han(9), 9);
        assert_eq!(bucket_base_han(10), 10);
    }
}
