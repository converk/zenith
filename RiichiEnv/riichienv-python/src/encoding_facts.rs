use std::collections::{HashMap, HashSet};

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyUntypedArrayMethods,
    ndarray::{Array2, Array3},
};
use pyo3::{exceptions::PyValueError, prelude::*};

use riichi::atomic_snapshot::{
    AtomicSnapshotInput, SNAPSHOT_FACTOR_WIDTH, SNAPSHOT_FIELD_COUNT, SNAPSHOT_FIRST_DISCARD_LIMIT,
    SNAPSHOT_FULLY_VISIBLE_KIND_OVERFLOW_BUCKET, SNAPSHOT_NUMERIC_WIDTH, SNAPSHOT_OPPONENT_COUNT,
    SNAPSHOT_POST_RIICHI_TSUMOGIRI_OVERFLOW_BUCKET, SNAPSHOT_PROGRESS_TILE_OVERFLOW_BUCKET,
    SNAPSHOT_UNKNOWN_DORA_COPY_OVERFLOW_BUCKET, SNAPSHOT_VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET,
    SNAPSHOT_YAKUHAI_HAN_OVERFLOW_BUCKET, encode as encode_atomic_snapshot,
};
use riichi::shanten;
use riichienv_core::{
    action::{Action, ActionType},
    observation::Observation,
    offense_analysis::analyze_offense_rows,
    types::{Meld, MeldType},
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

/// 前六张舍牌按花色与幺九字牌归类;不足六张自然以零补足。
pub(crate) fn first_six_discard_counts(discards: &[u32]) -> Result<[u8; 4], String> {
    let mut result = [0_u8; 4];
    for &tile in discards.iter().take(SNAPSHOT_FIRST_DISCARD_LIMIT) {
        if tile >= PHYSICAL_TILE_COUNT {
            return Err("牌河含非法实体牌 ID".to_string());
        }
        let kind = tile as usize / 4;
        let category = if kind < 9 {
            0
        } else if kind < 18 {
            1
        } else if kind < HONOR_TILE_START {
            2
        } else {
            3
        };
        result[category] += 1;
        if kind < HONOR_TILE_START && (kind % 9 == 0 || kind % 9 == 8) {
            result[3] += 1;
        }
    }
    Ok(result)
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
    Ok(han.min(SNAPSHOT_YAKUHAI_HAN_OVERFLOW_BUCKET))
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
    Ok(han.min(SNAPSHOT_VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET))
}

/// 观察者合法已知区域的四张可见牌种与不同宝牌种未知实体牌数。
pub(crate) fn global_visible_facts(observation: &Observation) -> Result<(u8, u8, [u8; TILE_KINDS]), String> {
    let observer = usize::from(observation.player_id);
    if observer >= 4 {
        return Err("观察者座次必须位于 0..3".to_string());
    }
    let mut known_tiles = HashSet::new();
    known_tiles.extend(observation.hands[observer].iter().copied());
    for river in &observation.discards {
        known_tiles.extend(river.iter().copied());
    }
    for player_melds in &observation.melds {
        for meld in player_melds {
            known_tiles.extend(meld.tiles.iter().map(|&tile| u32::from(tile)));
        }
    }
    known_tiles.extend(observation.dora_indicators.iter().copied());
    let mut visible = [0_u8; TILE_KINDS];
    for tile in known_tiles {
        if tile >= PHYSICAL_TILE_COUNT {
            return Err("合法已知区域含非法实体牌 ID".to_string());
        }
        let kind = tile as usize / 4;
        visible[kind] = visible[kind].saturating_add(1);
    }
    let fully_visible = visible.iter().filter(|&&count| count >= 4).count() as u8;
    let mut distinct_dora = [false; TILE_KINDS];
    for &indicator in &observation.dora_indicators {
        distinct_dora[dora_kind(indicator)?] = true;
    }
    let unknown_dora_copies = distinct_dora
        .iter()
        .enumerate()
        .filter(|(_, is_dora)| **is_dora)
        .map(|(kind, _)| 4_u8.saturating_sub(visible[kind]))
        .sum::<u8>();
    Ok((
        fully_visible.min(SNAPSHOT_FULLY_VISIBLE_KIND_OVERFLOW_BUCKET),
        unknown_dora_copies.min(SNAPSHOT_UNKNOWN_DORA_COPY_OVERFLOW_BUCKET),
        visible,
    ))
}

/// 自身进张与听牌和牌张数:把当前手牌归一为十三张形,分别摸入 34 类后只统计
/// 能降低综合向听的剩余实体牌;听牌(综合向听为 0)时统计摸入即和牌的剩余张数。
/// 只依赖观察者合法已知区域(自身手牌/副露、四家牌河、公开副露牌、宝牌指示牌)。
pub(crate) fn self_progress_facts(
    observation: &Observation,
    visible: &[u8; TILE_KINDS],
) -> Result<(u8, u8), String> {
    let seat = usize::from(observation.player_id);
    if seat >= 4 {
        return Err("观察者座次必须位于 0..3".to_string());
    }
    let hand = observation.hands[seat]
        .iter()
        .map(|&tile| u8::try_from(tile).map_err(|_| "physical tile exceeds u8".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let (three_melds, kans) = decompose_melds(&observation.melds[seat]);
    let (counts, meld_count) = kernel_shape(&hand, three_melds, &kans, 13);
    let current = shanten::calculate(&counts, meld_count).overall;
    let after_draws = shanten::calculate_after_draws(&counts, meld_count);
    let mut improve = 0_u32;
    let mut win = 0_u32;
    for kind in 0..TILE_KINDS {
        let remaining = 4_u32.saturating_sub(u32::from(visible[kind]));
        if remaining == 0 {
            continue;
        }
        if after_draws[kind] < current {
            improve += remaining;
        }
        if current == 0 && after_draws[kind] == -1 {
            win += remaining;
        }
    }
    Ok((
        improve.min(u32::from(SNAPSHOT_PROGRESS_TILE_OVERFLOW_BUCKET)) as u8,
        win.min(u32::from(SNAPSHOT_PROGRESS_TILE_OVERFLOW_BUCKET)) as u8,
    ))
}

/// 立直宣言牌与立直后摸切数:宣言牌取河中的实体牌,立直后摸切数从宣言下标
/// 之后逐张统计(宣言牌本身必为手切,天然不计入)。
pub(crate) fn opponent_riichi_traits(
    observation: &Observation,
    opponent: usize,
) -> Result<(u8, u8), String> {
    let declaration_tile = match observation.riichi_sutehais[opponent] {
        Some(tile) => {
            riichienv_core::observation::sequence_features::tile_id_to_kan37(u32::from(tile)) + 1
        }
        None => 0,
    };
    let post_riichi_tsumogiri = match observation.riichi_declaration_indices[opponent] {
        Some(index) => {
            let flags = &observation.tsumogiri_flags[opponent];
            if flags.len() != observation.discards[opponent].len() {
                return Err("牌河与摸切标记长度不一致".to_string());
            }
            let start = usize::from(index) + 1;
            let count = flags
                .iter()
                .skip(start)
                .filter(|&&flag| flag)
                .count()
                .min(usize::from(SNAPSHOT_POST_RIICHI_TSUMOGIRI_OVERFLOW_BUCKET)) as u8;
            count
        }
        None => 0,
    };
    Ok((declaration_tile, post_riichi_tsumogiri))
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

#[pyclass(name = "AtomicSnapshotBatch", frozen)]
pub struct AtomicSnapshotBatch {
    #[pyo3(get)]
    factors: Py<PyArray3<u8>>,
    #[pyo3(get)]
    numeric: Py<PyArray3<f32>>,
    #[pyo3(get)]
    lengths: Py<PyArray1<u8>>,
}

/// 从原生 Observation 批量派生固定 49 行原子 Snapshot。
/// 注意：这是 V16/V17 旧输入契约（PPO/rollout 待迁移项）；V18 活跃编码入口为
/// `prepare_current_state_batch`（当前局面快照），不经过本函数。
#[pyfunction]
pub fn prepare_atomic_snapshots(
    py: Python<'_>,
    observations: Vec<Observation>,
) -> PyResult<AtomicSnapshotBatch> {
    if observations.is_empty() {
        return Err(PyValueError::new_err("Snapshot 批次不能为空"));
    }
    let mut factor_values =
        Vec::with_capacity(observations.len() * SNAPSHOT_FIELD_COUNT * SNAPSHOT_FACTOR_WIDTH);
    let mut numeric_values =
        Vec::with_capacity(observations.len() * SNAPSHOT_FIELD_COUNT * SNAPSHOT_NUMERIC_WIDTH);
    for observation in &observations {
        let observer = usize::from(observation.player_id);
        if observer >= 4 {
            return Err(PyValueError::new_err("观察者座次必须位于 0..3"));
        }
        let mut hand_counts = [0_u8; TILE_KINDS];
        for &tile in &observation.hands[observer] {
            if tile >= 136 {
                return Err(PyValueError::new_err("手牌含非法实体牌 ID"));
            }
            hand_counts[tile as usize / 4] += 1;
        }
        let mut special_counts = hand_counts;
        for meld in &observation.melds[observer] {
            if meld.meld_type == MeldType::Ankan {
                for &tile in &meld.tiles {
                    special_counts[usize::from(tile) / 4] += 1;
                }
            }
        }
        let hand_is_open = observation.melds[observer].iter().any(|meld| meld.opened);
        let meld_count = u8::try_from(observation.melds[observer].len())
            .map_err(|_| PyValueError::new_err("面子数转换失败"))?;
        let mut riichi_status = [1_u8; 3];
        let mut riichi_turn = [0_u8; 3];
        let mut open_meld_count = [0_u8; 3];
        let mut tedashi_count = [0_u8; 3];
        let mut tsumogiri_count = [0_u8; 3];
        let mut post_riichi_tsumogiri_values = [0_u8; SNAPSHOT_OPPONENT_COUNT];
        let mut riichi_declaration_tile = [0_u8; SNAPSHOT_OPPONENT_COUNT];
        let mut tsumogiri_streak = [0_u8; 3];
        let mut first_six_discard_values = [[0_u8; 4]; SNAPSHOT_OPPONENT_COUNT];
        let mut open_meld_yakuhai_values = [0_u8; SNAPSHOT_OPPONENT_COUNT];
        let mut visible_meld_dora_aka_values = [0_u8; SNAPSHOT_OPPONENT_COUNT];
        let mut dora_multiplicity = [0_u8; TILE_KINDS];
        for &indicator in &observation.dora_indicators {
            let kind = dora_kind(indicator).map_err(PyValueError::new_err)?;
            dora_multiplicity[kind] = dora_multiplicity[kind].saturating_add(1);
        }
        let (fully_visible_tile_kind_count, unknown_distinct_dora_copy_count, visible_counts) =
            global_visible_facts(observation).map_err(PyValueError::new_err)?;
        let (self_improve_tile_count, self_win_tile_count) =
            self_progress_facts(observation, &visible_counts).map_err(PyValueError::new_err)?;
        for (relative, opponent) in (1..=3).map(|relative| (relative, (observer + relative) % 4)) {
            let index = relative - 1;
            riichi_status[index] = if observation.riichi_accepted[opponent] {
                3
            } else if observation.riichi_declared[opponent] {
                2
            } else {
                1
            };
            if riichi_status[index] != 1 {
                let declaration = observation.riichi_declaration_indices[opponent]
                    .ok_or_else(|| PyValueError::new_err("立直玩家缺少宣言河牌索引"))?;
                riichi_turn[index] = declaration.saturating_add(1).min(25);
            }
            open_meld_count[index] = observation.melds[opponent]
                .iter()
                .filter(|meld| meld.opened)
                .count()
                .try_into()
                .map_err(|_| PyValueError::new_err("副露数转换失败"))?;
            let flags = &observation.tsumogiri_flags[opponent];
            if flags.len() != observation.discards[opponent].len() {
                return Err(PyValueError::new_err("牌河与摸切标记长度不一致"));
            }
            tsumogiri_count[index] = flags.iter().filter(|&&flag| flag).count().min(255) as u8;
            tedashi_count[index] = flags.iter().filter(|&&flag| !flag).count().min(255) as u8;
            let (declaration_tile, post_riichi_tsumogiri) =
                opponent_riichi_traits(observation, opponent).map_err(PyValueError::new_err)?;
            riichi_declaration_tile[index] = declaration_tile;
            post_riichi_tsumogiri_values[index] = post_riichi_tsumogiri;
            tsumogiri_streak[index] =
                flags.iter().rev().take_while(|&&flag| flag).count().min(4) as u8;
            first_six_discard_values[index] =
                first_six_discard_counts(&observation.discards[opponent])
                    .map_err(PyValueError::new_err)?;
            let player_wind = ((opponent + 4 - usize::from(observation.oya)) % 4) as u8;
            open_meld_yakuhai_values[index] = open_meld_yakuhai_han(
                &observation.melds[opponent],
                player_wind,
                observation.round_wind,
            )
            .map_err(PyValueError::new_err)?;
            visible_meld_dora_aka_values[index] =
                visible_meld_dora_aka_han(&observation.melds[opponent], &dora_multiplicity)
                    .map_err(PyValueError::new_err)?;
        }
        let encoded = encode_atomic_snapshot(&AtomicSnapshotInput {
            observer: observation.player_id,
            scores: observation.scores,
            hand_counts,
            special_counts,
            meld_count,
            hand_is_open,
            riichi_status,
            riichi_turn,
            open_meld_count,
            tedashi_count,
            tsumogiri_count,
            post_riichi_tsumogiri_count: post_riichi_tsumogiri_values,
            riichi_declaration_tile,
            tsumogiri_streak,
            first_six_discard_counts: first_six_discard_values,
            open_meld_yakuhai_han: open_meld_yakuhai_values,
            visible_meld_dora_aka_han: visible_meld_dora_aka_values,
            fully_visible_tile_kind_count,
            unknown_distinct_dora_copy_count,
            self_improve_tile_count,
            self_win_tile_count,
        })
        .map_err(PyValueError::new_err)?;
        factor_values.extend(encoded.factors.into_iter().flatten());
        numeric_values.extend(encoded.numeric.into_iter().flatten());
    }
    let batch = observations.len();
    Ok(AtomicSnapshotBatch {
        factors: Array3::from_shape_vec(
            (batch, SNAPSHOT_FIELD_COUNT, SNAPSHOT_FACTOR_WIDTH),
            factor_values,
        )
        .expect("原子 Snapshot factor 形状")
        .into_pyarray(py)
        .unbind(),
        numeric: Array3::from_shape_vec(
            (batch, SNAPSHOT_FIELD_COUNT, SNAPSHOT_NUMERIC_WIDTH),
            numeric_values,
        )
        .expect("原子 Snapshot numeric 形状")
        .into_pyarray(py)
        .unbind(),
        lengths: PyArray1::from_vec(py, vec![SNAPSHOT_FIELD_COUNT as u8; batch]).unbind(),
    })
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
        ActionType::Kita => 0,
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
        action_requires_supplier, first_six_discard_counts, global_visible_facts,
        opponent_riichi_traits, open_meld_yakuhai_han, relative_source_code,
        self_progress_facts, visible_meld_dora_aka_han,
    };
    use riichienv_core::{
        action::ActionType,
        observation::Observation,
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
            ActionType::Kita,
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
    fn first_six_discards_keep_overlapping_terminal_and_suit_counts() {
        let counts = first_six_discard_counts(&[0, 4, 36, 72, 108, 132, 8]).expect("合法牌河");
        assert_eq!(counts, [2, 1, 1, 5]);
        assert_eq!(
            first_six_discard_counts(&[0, 36]).expect("不足六张舍牌"),
            [1, 1, 0, 2]
        );
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

    #[test]
    fn global_facts_ignore_opponent_concealed_hands_and_deduplicate_dora_kinds() {
        let mut observation = Observation::new(
            0,
            [vec![16], vec![20], vec![24], vec![28]],
            [vec![], vec![], vec![], vec![]],
            [vec![0, 1, 2, 3], vec![], vec![], vec![]],
            vec![12, 12],
            [25_000; 4],
            [false; 4],
            [false; 4],
            [None; 4],
            false,
            false,
            70,
            vec![],
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
        );
        let baseline = global_visible_facts(&observation).expect("合法公开状态");
        observation.hands[1] = vec![40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88];
        let changed = global_visible_facts(&observation).expect("合法公开状态");
        assert_eq!(baseline.0, 1);
        assert_eq!(baseline.1, 3);
        assert_eq!(baseline.0, changed.0);
        assert_eq!(baseline.1, changed.1);
    }


    /// 构造只有给定手牌、无公开信息的观察者(座次 0)视图。
    fn base_observation(hand: Vec<u8>) -> Observation {
        Observation::new(
            0,
            [hand, vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25_000; 4],
            [false; 4],
            [false; 4],
            [None; 4],
            false,
            false,
            70,
            vec![],
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
        )
    }

    #[test]
    fn self_progress_counts_win_tiles_when_tenpai() {
        // 三组顺子 + 双对(E/N) = 0 向听,只听 E/N;各两张实体在手中,各剩 2 张。
        let observation =
            base_observation(vec![0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 112, 113]);
        let (_, _, visible) = global_visible_facts(&observation).expect("合法公开状态");
        let (improve, win) = self_progress_facts(&observation, &visible).expect("合法手牌");
        assert_eq!(improve, 4);
        assert_eq!(win, 4);
    }

    #[test]
    fn self_progress_one_shanten_has_no_win_tiles() {
        // 三组顺子 + E 对 + N/W 单张:1 向听,改善张为 E/N/W 的剩余实体
        // (2/3/3),非听牌故和牌张为 0。
        let observation =
            base_observation(vec![0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 112, 116]);
        let (_, _, visible) = global_visible_facts(&observation).expect("合法公开状态");
        let (improve, win) = self_progress_facts(&observation, &visible).expect("合法手牌");
        assert_eq!(improve, 8);
        assert_eq!(win, 0);
    }

    #[test]
    fn riichi_traits_skip_declaration_discard_and_encode_red_five() {
        let mut observation = base_observation(vec![]);
        observation.discards[1] = vec![0, 4, 8, 36, 40];
        observation.tsumogiri_flags[1] = vec![false, true, true, false, true];
        observation.riichi_declaration_indices[1] = Some(3);
        observation.riichi_sutehais[1] = Some(16);
        let (declaration_tile, post_riichi_tsumogiri) =
            opponent_riichi_traits(&observation, 1).expect("合法立直状态");
        assert_eq!(declaration_tile, 1); // 红 5m → 1..37 规范码 1
        assert_eq!(post_riichi_tsumogiri, 1); // 宣言之后只有第 5 张为摸切
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
