use std::collections::{HashMap, HashSet};

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyUntypedArrayMethods,
    ndarray::Array2,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use riichi::{
    OPEN_MELD_YAKUHAI_HAN_OVERFLOW_BUCKET, VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET,
};
use riichienv_core::{
    action::{Action, ActionType},
    observation::Observation,
    offense_analysis::analyze_offense_rows,
    types::Meld,
};

const TILE_KINDS: usize = 34;
const PHYSICAL_TILE_COUNT: u32 = 136;
const RED_FIVE_TILE_IDS: [u8; 3] = [16, 52, 88];
const HONOR_TILE_START: usize = 27;
const WIND_TILE_COUNT: usize = 4;
const MODE_FULL_OFFENSE: u8 = 0;
const MODE_SIMPLE_SHANTEN: u8 = 1;
const MODE_WIN: u8 = 2;
const MODE_MIN_DROP: u8 = 3;
const O8_FROM_ANALYSIS: u8 = u8::MAX;

struct ObservationFacts {
    hand: Vec<u8>,
    melds: Vec<Meld>,
    remaining: [u8; TILE_KINDS],
    rivers: [u64; 4],
    public_visible: [u8; TILE_KINDS],
    dora_multiplicity: [u8; TILE_KINDS],
    own_river: u64,
    score: i32,
    declared: bool,
    menzen: bool,
}

/// 当前宝牌指示牌所对应的牌种,只依赖已翻开的指示牌。
pub(crate) fn dora_kind(indicator: u32) -> Result<usize, String> {
    if indicator >= PHYSICAL_TILE_COUNT {
        return Err("宝牌指示牌含非法实体牌 ID".to_string());
    }
    let tile = indicator as usize / 4;
    if tile < HONOR_TILE_START {
        let suit_start = tile / 9 * 9;
        Ok(suit_start + (tile - suit_start + 1) % 9)
    } else if tile < HONOR_TILE_START + WIND_TILE_COUNT {
        Ok(HONOR_TILE_START + (tile - HONOR_TILE_START + 1) % WIND_TILE_COUNT)
    } else {
        Ok(
            HONOR_TILE_START
                + WIND_TILE_COUNT
                + (tile - HONOR_TILE_START - WIND_TILE_COUNT + 1) % 3,
        )
    }
}

/// 开放副露中已确认的役牌番数;连风刻按场风与自风分别累计。
pub(crate) fn open_meld_yakuhai_han(melds: &[Meld], player_wind: u8, round_wind: u8) -> Result<u8, String> {
    if player_wind >= WIND_TILE_COUNT as u8 || round_wind >= WIND_TILE_COUNT as u8 {
        return Err("场风或自风超出范围".to_string());
    }
    let mut han = 0_u8;
    for meld in melds.iter().filter(|meld| meld.opened) {
        let Some(&first) = meld.tiles.first() else {
            return Err("副露不应为空".to_string());
        };
        if first as u32 >= PHYSICAL_TILE_COUNT {
            return Err("副露含非法实体牌 ID".to_string());
        }
        let kind = usize::from(first) / 4;
        if kind < HONOR_TILE_START {
            continue;
        }
        if meld.tiles.iter().any(|&tile| tile / 4 != first / 4) {
            return Err("役牌副露含不一致牌种".to_string());
        }
        if kind >= HONOR_TILE_START + WIND_TILE_COUNT {
            han = han.saturating_add(1);
        } else {
            han = han
                .saturating_add(u8::from(
                    kind == HONOR_TILE_START + usize::from(player_wind),
                ))
                .saturating_add(u8::from(kind == HONOR_TILE_START + usize::from(round_wind)));
        }
    }
    Ok(han.min(OPEN_MELD_YAKUHAI_HAN_OVERFLOW_BUCKET))
}

/// 已表示于副露的宝牌与赤宝牌番数;暗杠不改变门清,但其牌面可在本统计中出现。
pub(crate) fn visible_meld_dora_aka_han(
    melds: &[Meld],
    dora_multiplicity: &[u8; TILE_KINDS],
) -> Result<u8, String> {
    let mut han = 0_u8;
    for meld in melds {
        for &tile in &meld.tiles {
            if tile as u32 >= PHYSICAL_TILE_COUNT {
                return Err("副露含非法实体牌 ID".to_string());
            }
            let kind = usize::from(tile) / 4;
            han = han.saturating_add(dora_multiplicity[kind]);
            han = han.saturating_add(u8::from(RED_FIVE_TILE_IDS.contains(&tile)));
        }
    }
    Ok(han.min(VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET))
}

#[pyclass(name = "CompactEncodingFacts", frozen)]
pub struct CompactEncodingFacts {
    #[pyo3(get)]
    action_ids: Py<PyArray1<u16>>,
    #[pyo3(get)]
    action_types: Py<PyArray1<u8>>,
    #[pyo3(get)]
    primary_types: Py<PyArray1<i16>>,
    #[pyo3(get)]
    source_seats: Py<PyArray1<i8>>,
    #[pyo3(get)]
    modes: Py<PyArray1<u8>>,
    #[pyo3(get)]
    shape_counts: Py<PyArray2<u8>>,
    #[pyo3(get)]
    open_melds: Py<PyArray1<u8>>,
    #[pyo3(get)]
    remaining: Py<PyArray2<u8>>,
    #[pyo3(get)]
    own_rivers: Py<PyArray1<u64>>,
    #[pyo3(get)]
    opponent_rivers: Py<PyArray2<u64>>,
    #[pyo3(get)]
    defense_counts: Py<PyArray2<u8>>,
    #[pyo3(get)]
    discard_types: Py<PyArray1<i16>>,
    #[pyo3(get)]
    defense_visible: Py<PyArray1<u8>>,
    #[pyo3(get)]
    missed_doujun: Py<PyArray1<bool>>,
    #[pyo3(get)]
    missed_riichi: Py<PyArray1<bool>>,
    #[pyo3(get)]
    riichi_declared: Py<PyArray1<bool>>,
    #[pyo3(get)]
    scores: Py<PyArray1<i32>>,
    #[pyo3(get)]
    o7_values: Py<PyArray1<u8>>,
    #[pyo3(get)]
    o8_values: Py<PyArray1<u8>>,
    #[pyo3(get)]
    o9_values: Py<PyArray1<u8>>,
}

#[pyclass(name = "EncodingYakuValues", frozen)]
pub struct EncodingYakuValues {
    #[pyo3(get)]
    yaku_class: Py<PyArray1<u8>>,
    #[pyo3(get)]
    base_han: Py<PyArray1<u8>>,
}

fn dora_type(indicator: u8) -> usize {
    let tile = usize::from(indicator) / 4;
    if tile < 27 {
        let base = tile / 9 * 9;
        base + (tile - base + 1) % 9
    } else if tile <= 30 {
        27 + (tile - 27 + 1) % 4
    } else {
        31 + (tile - 31 + 1) % 3
    }
}

pub(crate) fn tile_counts(tiles: &[u8]) -> [u8; TILE_KINDS] {
    let mut counts = [0_u8; TILE_KINDS];
    for &tile in tiles {
        counts[usize::from(tile) / 4] += 1;
    }
    counts
}

fn observation_facts(
    observation: &Observation,
    declared: bool,
) -> Result<ObservationFacts, String> {
    let seat = usize::from(observation.player_id);
    if seat >= 4 {
        return Err("observation player_id must be in 0..4".to_string());
    }
    let hand = observation.hands[seat]
        .iter()
        .map(|&tile| u8::try_from(tile).map_err(|_| "physical tile exceeds u8".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let melds = observation.melds[seat].clone();
    let mut visible = HashSet::new();
    for river in &observation.discards {
        visible.extend(river.iter().copied());
    }
    for meld_rows in &observation.melds {
        for meld in meld_rows {
            visible.extend(meld.tiles.iter().map(|&tile| u32::from(tile)));
        }
    }
    visible.extend(observation.dora_indicators.iter().copied());
    let own: HashSet<u32> = hand.iter().map(|&tile| u32::from(tile)).collect();
    let mut remaining = [4_u8; TILE_KINDS];
    for &tile in &hand {
        remaining[usize::from(tile) / 4] = remaining[usize::from(tile) / 4].saturating_sub(1);
    }
    for tile in visible.difference(&own) {
        let kind = usize::try_from(*tile).map_err(|_| "physical tile conversion failed")? / 4;
        if kind < TILE_KINDS {
            remaining[kind] = remaining[kind].saturating_sub(1);
        }
    }
    let mut rivers = [0_u64; 4];
    for (river_seat, river) in observation.discards.iter().enumerate() {
        for &tile in river {
            rivers[river_seat] |= 1_u64 << (tile / 4);
        }
    }
    let mut public_visible = [0_u8; TILE_KINDS];
    for river in &observation.discards {
        for &tile in river {
            let kind = tile as usize / 4;
            public_visible[kind] = public_visible[kind].saturating_add(1).min(4);
        }
    }
    for meld_rows in &observation.melds {
        for meld in meld_rows {
            for &tile in &meld.tiles {
                let kind = usize::from(tile) / 4;
                public_visible[kind] = public_visible[kind].saturating_add(1).min(4);
            }
        }
    }
    for &tile in &observation.dora_indicators {
        let kind = tile as usize / 4;
        public_visible[kind] = public_visible[kind].saturating_add(1).min(4);
    }
    let mut dora_multiplicity = [0_u8; TILE_KINDS];
    for &indicator in &observation.dora_indicators {
        let indicator = u8::try_from(indicator).map_err(|_| "dora indicator exceeds u8")?;
        dora_multiplicity[dora_type(indicator)] += 1;
    }
    Ok(ObservationFacts {
        hand,
        melds,
        remaining,
        rivers,
        public_visible,
        dora_multiplicity,
        own_river: rivers[seat],
        score: observation.scores[seat],
        declared,
        menzen: observation.melds[seat].iter().all(|meld| !meld.opened),
    })
}

pub(crate) fn decompose_melds(melds: &[Meld]) -> (u8, Vec<usize>) {
    let mut three_melds = 0_u8;
    let mut kans = Vec::new();
    for meld in melds {
        if meld.tiles.len() == 4 && !meld.tiles.is_empty() {
            kans.push(usize::from(meld.tiles[0]) / 4);
        } else if !meld.tiles.is_empty() {
            three_melds += 1;
        }
    }
    (three_melds, kans)
}

pub(crate) fn kernel_shape(
    concealed: &[u8],
    three_melds: u8,
    kan_types: &[usize],
    target: usize,
) -> ([u8; TILE_KINDS], u8) {
    let mut counts = tile_counts(concealed);
    for &kind in kan_types {
        counts[kind] += 4;
    }
    let total = counts
        .iter()
        .map(|&value| usize::from(value))
        .sum::<usize>()
        + 3 * usize::from(three_melds);
    for _ in 0..total.saturating_sub(target) {
        if let Some(value) = counts.iter_mut().find(|value| **value > 0) {
            *value -= 1;
        }
    }
    (counts, three_melds)
}

fn remove_by_type(hand: &mut Vec<u8>, tile_type: usize, count: usize) {
    for _ in 0..count {
        if let Some(index) = hand
            .iter()
            .position(|&tile| usize::from(tile) / 4 == tile_type)
        {
            hand.remove(index);
        }
    }
}

fn physical_tiles(hand: &[u8], melds: &[Meld]) -> Vec<u8> {
    let mut out = hand.to_vec();
    for meld in melds {
        out.extend(meld.tiles.iter().copied());
    }
    out
}

pub(crate) fn count_dora_aka(tiles: &[u8], multiplicity: &[u8; TILE_KINDS]) -> u8 {
    tiles
        .iter()
        .map(|&tile| multiplicity[usize::from(tile) / 4] + u8::from(matches!(tile, 16 | 52 | 88)))
        .sum::<u8>()
        .min(5)
}

fn action_type_code(action_type: ActionType) -> u8 {
    match action_type {
        ActionType::Pass => 1,
        ActionType::Discard => 2,
        ActionType::Riichi => 3,
        ActionType::Chi => 4,
        ActionType::Pon => 5,
        ActionType::Daiminkan => 6,
        ActionType::Ankan => 7,
        ActionType::Kakan => 8,
        ActionType::Tsumo => 9,
        ActionType::Ron => 10,
        ActionType::KyushuKyuhai => 11,
    }
}

fn relative_source_code(observer: u8, source: u8) -> Result<i8, &'static str> {
    if observer >= 4 || source >= 4 || observer == source {
        return Err("供牌 actor 不是有效对手座次");
    }
    Ok(((i16::from(source) - i16::from(observer) + 3) % 4) as i8)
}

fn action_requires_supplier(action_type: ActionType) -> bool {
    matches!(
        action_type,
        ActionType::Chi | ActionType::Pon | ActionType::Daiminkan | ActionType::Ron
    )
}

/// 从唯一 Observation 与逐动作索引一次生成 Rust 融合编码器所需的连续事实。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn prepare_encoding_facts(
    py: Python<'_>,
    observations: Vec<Observation>,
    observation_indices: PyReadonlyArray1<'_, u32>,
    actions: Vec<Action>,
    action_ids: PyReadonlyArray1<'_, u16>,
    missed_doujun_overrides: PyReadonlyArray1<'_, bool>,
    missed_riichi_overrides: PyReadonlyArray1<'_, bool>,
    riichi_declared_overrides: PyReadonlyArray1<'_, bool>,
    drawn_tile_overrides: PyReadonlyArray1<'_, i16>,
) -> PyResult<CompactEncodingFacts> {
    let rows = actions.len();
    if observation_indices.len() != rows || action_ids.len() != rows {
        return Err(PyValueError::new_err(
            "observation_indices/actions/action_ids must have the same length",
        ));
    }
    let observation_indices = observation_indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("observation_indices must be contiguous"))?;
    let action_ids_slice = action_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("action_ids must be contiguous"))?;
    let observation_count = observations.len();
    for (name, length) in [
        ("missed_doujun_overrides", missed_doujun_overrides.len()),
        ("missed_riichi_overrides", missed_riichi_overrides.len()),
        ("riichi_declared_overrides", riichi_declared_overrides.len()),
        ("drawn_tile_overrides", drawn_tile_overrides.len()),
    ] {
        if length != observation_count {
            return Err(PyValueError::new_err(format!(
                "{name} must have length {observation_count}, got {length}"
            )));
        }
    }
    let missed_doujun_overrides = missed_doujun_overrides
        .as_slice()
        .map_err(|_| PyValueError::new_err("missed_doujun_overrides must be contiguous"))?;
    let missed_riichi_overrides = missed_riichi_overrides
        .as_slice()
        .map_err(|_| PyValueError::new_err("missed_riichi_overrides must be contiguous"))?;
    let riichi_declared_overrides = riichi_declared_overrides
        .as_slice()
        .map_err(|_| PyValueError::new_err("riichi_declared_overrides must be contiguous"))?;
    let drawn_tile_overrides = drawn_tile_overrides
        .as_slice()
        .map_err(|_| PyValueError::new_err("drawn_tile_overrides must be contiguous"))?;
    if drawn_tile_overrides
        .iter()
        .any(|&tile| !(-1..136).contains(&tile))
    {
        return Err(PyValueError::new_err(
            "drawn_tile_overrides values must be -1 or physical tile ids in 0..136",
        ));
    }
    let facts = observations
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            observation_facts(observation, riichi_declared_overrides[index])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyValueError::new_err)?;

    let mut action_types = vec![0_u8; rows];
    let mut primary_types = vec![-1_i16; rows];
    let mut source_seats = vec![-1_i8; rows];
    let mut modes = vec![0_u8; rows];
    let mut shape_counts = vec![0_u8; rows * TILE_KINDS];
    let mut open_melds = vec![0_u8; rows];
    let mut remaining = vec![0_u8; rows * TILE_KINDS];
    let mut own_rivers = vec![0_u64; rows];
    let mut opponent_rivers = vec![0_u64; rows * 3];
    let mut defense_counts = vec![0_u8; rows * TILE_KINDS];
    let mut discard_types = vec![-1_i16; rows];
    let mut defense_visible = vec![5_u8; rows];
    let mut missed_doujun = vec![false; rows];
    let mut missed_riichi = vec![false; rows];
    let mut riichi_declared = vec![false; rows];
    let mut scores = vec![0_i32; rows];
    let mut o7_values = vec![0_u8; rows];
    let mut o8_values = vec![2_u8; rows];
    let mut o9_values = vec![0_u8; rows];

    for (row, action) in actions.iter().enumerate() {
        let observation_index = usize::try_from(observation_indices[row])
            .map_err(|_| PyValueError::new_err("observation index conversion failed"))?;
        let observation = observations
            .get(observation_index)
            .ok_or_else(|| PyValueError::new_err("observation index is out of range"))?;
        let fact = &facts[observation_index];
        let seat = usize::from(observation.player_id);
        let mut tile = action.tile;
        if tile.is_none() && matches!(action.action_type, ActionType::Riichi | ActionType::Discard)
        {
            let override_tile = drawn_tile_overrides[observation_index];
            tile = (override_tile >= 0).then_some(override_tile as u8);
        }
        let primary = tile.map(|value| usize::from(value) / 4);
        action_types[row] = action_type_code(action.action_type);
        if action_types[row] == 0 {
            return Err(PyValueError::new_err("unsupported four-player action type"));
        }
        if action_requires_supplier(action.action_type) {
            let source = observation
                .last_offer_actor()
                .ok_or_else(|| PyValueError::new_err("供牌动作缺少最近的 dahai/kakan actor"))?;
            source_seats[row] = relative_source_code(observation.player_id, source)
                .map_err(PyValueError::new_err)?;
        }
        primary_types[row] = primary.map_or(-1, |value| value as i16);
        remaining[row * TILE_KINDS..(row + 1) * TILE_KINDS].copy_from_slice(&fact.remaining);
        own_rivers[row] = fact.own_river;
        for opponent in 0..3 {
            opponent_rivers[row * 3 + opponent] = fact.rivers[(seat + opponent + 1) % 4];
        }
        missed_doujun[row] = missed_doujun_overrides[observation_index];
        missed_riichi[row] = missed_riichi_overrides[observation_index];
        riichi_declared[row] = fact.declared;
        scores[row] = fact.score;
        o7_values[row] = u8::from(!fact.menzen);

        if matches!(action.action_type, ActionType::Tsumo | ActionType::Ron) {
            modes[row] = MODE_WIN;
            let mut full = physical_tiles(&fact.hand, &fact.melds);
            if let Some(tile) = tile {
                full.push(tile);
            }
            o9_values[row] = count_dora_aka(&full, &fact.dora_multiplicity);
            defense_visible[row] = primary.map_or(5, |kind| fact.public_visible[kind].min(4));
            continue;
        }

        if matches!(action.action_type, ActionType::Riichi | ActionType::Discard) {
            let mut post = fact.hand.clone();
            if let Some(tile) = tile
                && let Some(index) = post.iter().position(|&value| value == tile)
            {
                post.remove(index);
            }
            let (three_melds, kans) = decompose_melds(&fact.melds);
            let (shape, meld_count) = kernel_shape(&post, three_melds, &kans, 13);
            shape_counts[row * TILE_KINDS..(row + 1) * TILE_KINDS].copy_from_slice(&shape);
            defense_counts[row * TILE_KINDS..(row + 1) * TILE_KINDS]
                .copy_from_slice(&tile_counts(&post));
            modes[row] = MODE_FULL_OFFENSE;
            open_melds[row] = meld_count;
            discard_types[row] = primary.map_or(-1, |kind| kind as i16);
            riichi_declared[row] = action.action_type == ActionType::Riichi || fact.declared;
            o8_values[row] = if action.action_type == ActionType::Riichi {
                2
            } else {
                O8_FROM_ANALYSIS
            };
            o9_values[row] =
                count_dora_aka(&physical_tiles(&post, &fact.melds), &fact.dora_multiplicity);
            defense_visible[row] = primary.map_or(5, |kind| fact.public_visible[kind].min(4));
            continue;
        }

        if matches!(
            action.action_type,
            ActionType::Chi
                | ActionType::Pon
                | ActionType::Ankan
                | ActionType::Daiminkan
                | ActionType::Kakan
        ) {
            let mut post = fact.hand.clone();
            let called = tile;
            if action.action_type == ActionType::Kakan {
                let added = called.or_else(|| action.consume_tiles.last().copied());
                if let Some(added) = added {
                    remove_by_type(&mut post, usize::from(added) / 4, 1);
                }
            } else {
                let mut consumed = HashMap::new();
                for &value in &action.consume_tiles {
                    *consumed.entry(usize::from(value) / 4).or_insert(0_usize) += 1;
                }
                let mut consumed = consumed.into_iter().collect::<Vec<_>>();
                consumed.sort_unstable();
                for (kind, count) in consumed {
                    remove_by_type(&mut post, kind, count);
                }
            }
            let (three_melds, kans) = decompose_melds(&fact.melds);
            let kan_type = called
                .or_else(|| action.consume_tiles.first().copied())
                .map(|value| usize::from(value) / 4);
            o7_values[row] = u8::from(!(fact.menzen && action.action_type == ActionType::Ankan));
            let (shape, meld_count, full) = match action.action_type {
                ActionType::Chi | ActionType::Pon => {
                    let (shape, meld_count) = kernel_shape(&post, three_melds + 1, &kans, 14);
                    let mut new_meld = action.consume_tiles.clone();
                    if let Some(called) = called
                        && !new_meld.contains(&called)
                    {
                        new_meld.push(called);
                    }
                    let mut full = physical_tiles(&post, &fact.melds);
                    full.extend(new_meld);
                    (shape, meld_count, full)
                }
                ActionType::Daiminkan => {
                    let mut extra = kans.clone();
                    if let Some(kind) = kan_type {
                        extra.push(kind);
                    }
                    let (shape, meld_count) = kernel_shape(&post, three_melds, &extra, 13);
                    let mut full = post.clone();
                    if let Some(called) = called {
                        full.extend([called; 3]);
                    } else {
                        full.extend(action.consume_tiles.iter().copied());
                    }
                    (shape, meld_count, full)
                }
                ActionType::Ankan => {
                    let mut extra = kans.clone();
                    if let Some(kind) = kan_type {
                        extra.push(kind);
                    }
                    let (shape, meld_count) = kernel_shape(&post, three_melds, &extra, 13);
                    let mut full = post.clone();
                    full.extend(action.consume_tiles.iter().copied());
                    (shape, meld_count, full)
                }
                ActionType::Kakan => {
                    let added = called.or_else(|| action.consume_tiles.last().copied());
                    if let Some(added) = added
                        && let Some(index) = post.iter().position(|&value| value == added)
                    {
                        post.remove(index);
                    }
                    let mut extra = kans.clone();
                    if let Some(kind) = kan_type {
                        extra.push(kind);
                    }
                    let (shape, meld_count) =
                        kernel_shape(&post, three_melds.saturating_sub(1), &extra, 13);
                    let mut full = physical_tiles(&post, &fact.melds);
                    if let Some(added) = added {
                        full.push(added);
                    }
                    (shape, meld_count, full)
                }
                _ => unreachable!(),
            };
            modes[row] = MODE_SIMPLE_SHANTEN;
            shape_counts[row * TILE_KINDS..(row + 1) * TILE_KINDS].copy_from_slice(&shape);
            open_melds[row] = meld_count;
            defense_counts[row * TILE_KINDS..(row + 1) * TILE_KINDS]
                .copy_from_slice(&tile_counts(&post));
            o8_values[row] = u8::from(!matches!(
                action.action_type,
                ActionType::Chi | ActionType::Pon
            )) + 1;
            o9_values[row] = count_dora_aka(&full, &fact.dora_multiplicity);
            defense_visible[row] = primary.map_or(5, |kind| fact.public_visible[kind].min(4));
            continue;
        }

        let full = physical_tiles(&fact.hand, &fact.melds);
        defense_counts[row * TILE_KINDS..(row + 1) * TILE_KINDS]
            .copy_from_slice(&tile_counts(&fact.hand));
        o9_values[row] = count_dora_aka(&full, &fact.dora_multiplicity);
        if full.len() == 13 {
            let (three_melds, kans) = decompose_melds(&fact.melds);
            let (shape, meld_count) = kernel_shape(&fact.hand, three_melds, &kans, 13);
            modes[row] = MODE_FULL_OFFENSE;
            shape_counts[row * TILE_KINDS..(row + 1) * TILE_KINDS].copy_from_slice(&shape);
            open_melds[row] = meld_count;
        } else if fact.melds.is_empty() {
            modes[row] = MODE_MIN_DROP;
            shape_counts[row * TILE_KINDS..(row + 1) * TILE_KINDS]
                .copy_from_slice(&tile_counts(&fact.hand));
        } else {
            let (three_melds, kans) = decompose_melds(&fact.melds);
            let (shape, meld_count) = kernel_shape(&fact.hand, three_melds, &kans, 13);
            modes[row] = MODE_SIMPLE_SHANTEN;
            shape_counts[row * TILE_KINDS..(row + 1) * TILE_KINDS].copy_from_slice(&shape);
            open_melds[row] = meld_count;
        }
    }

    let make_u8_2d = |values, width| {
        Array2::from_shape_vec((rows, width), values)
            .expect("compact fact shape")
            .into_pyarray(py)
            .unbind()
    };
    let opponent_rivers = Array2::from_shape_vec((rows, 3), opponent_rivers)
        .expect("opponent river shape")
        .into_pyarray(py)
        .unbind();
    Ok(CompactEncodingFacts {
        action_ids: PyArray1::from_vec(py, action_ids_slice.to_vec()).unbind(),
        action_types: PyArray1::from_vec(py, action_types).unbind(),
        primary_types: PyArray1::from_vec(py, primary_types).unbind(),
        source_seats: PyArray1::from_vec(py, source_seats).unbind(),
        modes: PyArray1::from_vec(py, modes).unbind(),
        shape_counts: make_u8_2d(shape_counts, TILE_KINDS),
        open_melds: PyArray1::from_vec(py, open_melds).unbind(),
        remaining: make_u8_2d(remaining, TILE_KINDS),
        own_rivers: PyArray1::from_vec(py, own_rivers).unbind(),
        opponent_rivers,
        defense_counts: make_u8_2d(defense_counts, TILE_KINDS),
        discard_types: PyArray1::from_vec(py, discard_types).unbind(),
        defense_visible: PyArray1::from_vec(py, defense_visible).unbind(),
        missed_doujun: PyArray1::from_vec(py, missed_doujun).unbind(),
        missed_riichi: PyArray1::from_vec(py, missed_riichi).unbind(),
        riichi_declared: PyArray1::from_vec(py, riichi_declared).unbind(),
        scores: PyArray1::from_vec(py, scores).unbind(),
        o7_values: PyArray1::from_vec(py, o7_values).unbind(),
        o8_values: PyArray1::from_vec(py, o8_values).unbind(),
        o9_values: PyArray1::from_vec(py, o9_values).unbind(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        action_requires_supplier, open_meld_yakuhai_han, relative_source_code,
        visible_meld_dora_aka_han,
    };
    use riichienv_core::{
        action::ActionType,
        types::{Meld, MeldType},
    };

    #[test]
    fn supplier_seats_cover_all_three_relative_opponents() {
        assert_eq!(relative_source_code(0, 1), Ok(0));
        assert_eq!(relative_source_code(0, 2), Ok(1));
        assert_eq!(relative_source_code(0, 3), Ok(2));
        assert_eq!(relative_source_code(3, 0), Ok(0));
    }

    #[test]
    fn supplier_seat_rejects_self_and_invalid_seats() {
        assert!(relative_source_code(2, 2).is_err());
        assert!(relative_source_code(0, 4).is_err());
    }

    #[test]
    fn exactly_four_action_types_require_a_supplier() {
        for action_type in [
            ActionType::Chi,
            ActionType::Pon,
            ActionType::Daiminkan,
            ActionType::Ron,
        ] {
            assert!(action_requires_supplier(action_type));
        }
        for action_type in [
            ActionType::Discard,
            ActionType::Riichi,
            ActionType::Ankan,
            ActionType::Kakan,
            ActionType::Tsumo,
            ActionType::Pass,
            ActionType::KyushuKyuhai,
        ] {
            assert!(!action_requires_supplier(action_type));
        }
    }

    #[test]
    fn tsumo_and_ron_have_distinct_action_codes() {
        assert_eq!(super::action_type_code(ActionType::Tsumo), 9);
        assert_eq!(super::action_type_code(ActionType::Ron), 10);
        assert_eq!(super::action_type_code(ActionType::KyushuKyuhai), 11);
    }

    #[test]
    fn yakuhai_wind_and_ankan_dora_follow_public_meld_rules() {
        let east_pon = Meld::new(MeldType::Pon, vec![108, 109, 110], true, 1, Some(108));
        let red_five_ankan = Meld::new(MeldType::Ankan, vec![16, 17, 18, 19], false, -1, None);
        assert_eq!(open_meld_yakuhai_han(&[east_pon], 0, 0), Ok(2));
        let mut dora = [0_u8; 34];
        dora[4] = 1;
        assert_eq!(visible_meld_dora_aka_han(&[red_five_ankan], &dora), Ok(5));
    }
}

/// 用 Rust Observation/Action 直接计算融合编码等待行的 O4/O5。
#[pyfunction]
pub fn analyze_encoding_yaku_batch(
    py: Python<'_>,
    observations: Vec<Observation>,
    observation_indices: PyReadonlyArray1<'_, u32>,
    actions: Vec<Action>,
    wait_masks: PyReadonlyArray1<'_, u64>,
    drawn_tile_overrides: PyReadonlyArray1<'_, i16>,
) -> PyResult<EncodingYakuValues> {
    let rows = actions.len();
    if observation_indices.len() != rows || wait_masks.len() != rows {
        return Err(PyValueError::new_err(
            "observation_indices/actions/wait_masks must have the same length",
        ));
    }
    if drawn_tile_overrides.len() != observations.len() {
        return Err(PyValueError::new_err(
            "drawn_tile_overrides must have one value per observation",
        ));
    }
    let observation_indices = observation_indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("observation_indices must be contiguous"))?;
    let wait_masks = wait_masks
        .as_slice()
        .map_err(|_| PyValueError::new_err("wait_masks must be contiguous"))?;
    let drawn_tile_overrides = drawn_tile_overrides
        .as_slice()
        .map_err(|_| PyValueError::new_err("drawn_tile_overrides must be contiguous"))?;
    if drawn_tile_overrides
        .iter()
        .any(|&tile| !(-1..136).contains(&tile))
    {
        return Err(PyValueError::new_err(
            "drawn_tile_overrides values must be -1 or physical tile ids in 0..136",
        ));
    }

    let mut selected_rows = Vec::new();
    let mut concealed_tiles = Vec::new();
    let mut melds = Vec::new();
    let mut selected_wait_masks = Vec::new();
    let mut dora_indicators = Vec::new();
    let mut player_wind = Vec::new();
    let mut round_wind = Vec::new();
    let mut honba = Vec::new();
    let mut riichi_sticks = Vec::new();
    for row in 0..rows {
        if wait_masks[row] == 0 {
            continue;
        }
        let observation_index = usize::try_from(observation_indices[row])
            .map_err(|_| PyValueError::new_err("observation index conversion failed"))?;
        let observation = observations
            .get(observation_index)
            .ok_or_else(|| PyValueError::new_err("observation index is out of range"))?;
        let action = &actions[row];
        if !matches!(
            action.action_type,
            ActionType::Riichi | ActionType::Discard | ActionType::Pass | ActionType::KyushuKyuhai
        ) {
            return Err(PyValueError::new_err(format!(
                "row {row} has waits for an unsupported action type"
            )));
        }
        let seat = usize::from(observation.player_id);
        let mut post = observation.hands[seat]
            .iter()
            .map(|&tile| u8::try_from(tile).map_err(|_| "physical tile exceeds u8"))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        if matches!(action.action_type, ActionType::Riichi | ActionType::Discard) {
            let override_tile = drawn_tile_overrides[observation_index];
            let tile = action
                .tile
                .or_else(|| (override_tile >= 0).then_some(override_tile as u8));
            if let Some(tile) = tile
                && let Some(index) = post.iter().position(|&value| value == tile)
            {
                post.remove(index);
            }
        }
        selected_rows.push(row);
        concealed_tiles.push(post);
        melds.push(observation.melds[seat].clone());
        selected_wait_masks.push(wait_masks[row]);
        dora_indicators.push(
            observation
                .dora_indicators
                .iter()
                .map(|&tile| u8::try_from(tile).map_err(|_| "dora indicator exceeds u8"))
                .collect::<Result<Vec<_>, _>>()
                .map_err(PyValueError::new_err)?,
        );
        player_wind.push(((seat + 4 - usize::from(observation.oya)) % 4) as u8);
        round_wind.push(observation.round_wind);
        honba.push(observation.honba);
        riichi_sticks.push(observation.riichi_sticks.min(u32::from(u8::MAX)) as u8);
    }

    let selected = py
        .detach(|| {
            analyze_offense_rows(
                &concealed_tiles,
                &melds,
                &selected_wait_masks,
                &dora_indicators,
                &player_wind,
                &round_wind,
                &honba,
                &riichi_sticks,
            )
        })
        .map_err(PyValueError::new_err)?;
    let mut yaku_class = vec![0_u8; rows];
    let mut base_han = vec![0_u8; rows];
    for (source, row) in selected.into_iter().zip(selected_rows) {
        yaku_class[row] = source.yaku_class;
        base_han[row] = source.base_han;
    }
    let yaku_class = PyArray1::from_vec(py, yaku_class);
    let base_han = PyArray1::from_vec(py, base_han);
    for array in [yaku_class.as_any(), base_han.as_any()] {
        array.call_method1("setflags", (false,))?;
    }
    Ok(EncodingYakuValues {
        yaku_class: yaku_class.unbind(),
        base_han: base_han.unbind(),
    })
}
