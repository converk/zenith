//! 现行输入协议的原子 Snapshot 单一来源。

use pyo3::prelude::*;

use crate::{shanten, SHANTEN_UNAVAILABLE};

pub const SNAPSHOT_FIELD_COUNT: usize = 54;
pub const SNAPSHOT_FACTOR_WIDTH: usize = 4;
pub const SNAPSHOT_NUMERIC_WIDTH: usize = 1;
pub const SCORE_PRESSURE_SCALE: f32 = 100_000.0;
pub const SNAPSHOT_OPPONENT_COUNT: usize = 3;
pub const SNAPSHOT_OPPONENT_SUMMARY_WIDTH: usize = 13;
pub const SNAPSHOT_FIRST_DISCARD_LIMIT: usize = 6;
pub const SNAPSHOT_FIRST_DISCARD_COUNT_MAX: u8 = SNAPSHOT_FIRST_DISCARD_LIMIT as u8;
pub const SNAPSHOT_YAKUHAI_HAN_OVERFLOW_BUCKET: u8 = 6;
pub const SNAPSHOT_VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET: u8 = 8;
pub const SNAPSHOT_FULLY_VISIBLE_KIND_OVERFLOW_BUCKET: u8 = 25;
pub const SNAPSHOT_UNKNOWN_DORA_COPY_OVERFLOW_BUCKET: u8 = 16;
/// 立直后摸切数的溢出桶:0..15 精确计数,16=16+。
pub const SNAPSHOT_POST_RIICHI_TSUMOGIRI_OVERFLOW_BUCKET: u8 = 16;
/// 自身进张/和牌张数的溢出桶:0..39 精确计数,40=40+。
pub const SNAPSHOT_PROGRESS_TILE_OVERFLOW_BUCKET: u8 = 40;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotFieldSpec {
    pub field_id: u8,
    pub name: &'static str,
    pub relative_seat: u8,
    pub categorical_max: u8,
    pub tile_max: u8,
    pub numeric: bool,
}

const fn field(
    field_id: u8,
    name: &'static str,
    relative_seat: u8,
    categorical_max: u8,
    tile_max: u8,
    numeric: bool,
) -> SnapshotFieldSpec {
    SnapshotFieldSpec {
        field_id,
        name,
        relative_seat,
        categorical_max,
        tile_max,
        numeric,
    }
}

pub const SNAPSHOT_SCHEMA: [SnapshotFieldSpec; SNAPSHOT_FIELD_COUNT] = [
    field(1, "own_rank", 0, 4, 0, false),
    field(2, "score_pressure_1", 1, 0, 0, true),
    field(3, "score_pressure_2", 2, 0, 0, true),
    field(4, "score_pressure_3", 3, 0, 0, true),
    field(5, "opponent_1_riichi_status", 1, 3, 0, false),
    field(6, "opponent_1_riichi_turn", 1, 25, 0, false),
    field(7, "opponent_1_open_meld_count", 1, 4, 0, false),
    field(8, "opponent_1_tedashi_count", 1, 25, 0, false),
    field(9, "opponent_1_tsumogiri_count", 1, 25, 0, false),
    field(
        10,
        "opponent_1_post_riichi_tsumogiri_count",
        1,
        SNAPSHOT_POST_RIICHI_TSUMOGIRI_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(
        11,
        "opponent_1_first_six_man_count",
        1,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        12,
        "opponent_1_first_six_pin_count",
        1,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        13,
        "opponent_1_first_six_sou_count",
        1,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        14,
        "opponent_1_first_six_terminal_honor_count",
        1,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        15,
        "opponent_1_open_meld_yakuhai_han",
        1,
        SNAPSHOT_YAKUHAI_HAN_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(
        16,
        "opponent_1_visible_meld_dora_aka_han",
        1,
        SNAPSHOT_VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(17, "opponent_1_riichi_declaration_tile", 1, 0, 37, false),
    field(18, "opponent_2_riichi_status", 2, 3, 0, false),
    field(19, "opponent_2_riichi_turn", 2, 25, 0, false),
    field(20, "opponent_2_open_meld_count", 2, 4, 0, false),
    field(21, "opponent_2_tedashi_count", 2, 25, 0, false),
    field(22, "opponent_2_tsumogiri_count", 2, 25, 0, false),
    field(
        23,
        "opponent_2_post_riichi_tsumogiri_count",
        2,
        SNAPSHOT_POST_RIICHI_TSUMOGIRI_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(
        24,
        "opponent_2_first_six_man_count",
        2,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        25,
        "opponent_2_first_six_pin_count",
        2,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        26,
        "opponent_2_first_six_sou_count",
        2,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        27,
        "opponent_2_first_six_terminal_honor_count",
        2,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        28,
        "opponent_2_open_meld_yakuhai_han",
        2,
        SNAPSHOT_YAKUHAI_HAN_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(
        29,
        "opponent_2_visible_meld_dora_aka_han",
        2,
        SNAPSHOT_VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(30, "opponent_2_riichi_declaration_tile", 2, 0, 37, false),
    field(31, "opponent_3_riichi_status", 3, 3, 0, false),
    field(32, "opponent_3_riichi_turn", 3, 25, 0, false),
    field(33, "opponent_3_open_meld_count", 3, 4, 0, false),
    field(34, "opponent_3_tedashi_count", 3, 25, 0, false),
    field(35, "opponent_3_tsumogiri_count", 3, 25, 0, false),
    field(
        36,
        "opponent_3_post_riichi_tsumogiri_count",
        3,
        SNAPSHOT_POST_RIICHI_TSUMOGIRI_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(
        37,
        "opponent_3_first_six_man_count",
        3,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        38,
        "opponent_3_first_six_pin_count",
        3,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        39,
        "opponent_3_first_six_sou_count",
        3,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        40,
        "opponent_3_first_six_terminal_honor_count",
        3,
        SNAPSHOT_FIRST_DISCARD_COUNT_MAX,
        0,
        false,
    ),
    field(
        41,
        "opponent_3_open_meld_yakuhai_han",
        3,
        SNAPSHOT_YAKUHAI_HAN_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(
        42,
        "opponent_3_visible_meld_dora_aka_han",
        3,
        SNAPSHOT_VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(43, "opponent_3_riichi_declaration_tile", 3, 0, 37, false),
    field(44, "overall_shanten", 0, 8, 0, false),
    field(45, "standard_shanten", 0, 8, 0, false),
    field(46, "chiitoitsu_shanten", 0, 8, 0, false),
    field(47, "kokushi_shanten", 0, 15, 0, false),
    field(
        48,
        "fully_visible_tile_kind_count",
        0,
        SNAPSHOT_FULLY_VISIBLE_KIND_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(
        49,
        "unknown_distinct_dora_copy_count",
        0,
        SNAPSHOT_UNKNOWN_DORA_COPY_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(
        50,
        "self_improve_tile_count",
        0,
        SNAPSHOT_PROGRESS_TILE_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(
        51,
        "self_win_tile_count",
        0,
        SNAPSHOT_PROGRESS_TILE_OVERFLOW_BUCKET,
        0,
        false,
    ),
    field(52, "opponent_1_tsumogiri_streak", 1, 4, 0, false),
    field(53, "opponent_2_tsumogiri_streak", 2, 4, 0, false),
    field(54, "opponent_3_tsumogiri_streak", 3, 4, 0, false),
];

#[derive(Clone, Debug)]
pub struct AtomicSnapshotInput {
    pub observer: u8,
    pub scores: [i32; 4],
    pub hand_counts: [u8; 34],
    pub special_counts: [u8; 34],
    pub meld_count: u8,
    pub hand_is_open: bool,
    pub riichi_status: [u8; 3],
    pub riichi_turn: [u8; 3],
    pub open_meld_count: [u8; 3],
    pub tedashi_count: [u8; 3],
    pub tsumogiri_count: [u8; 3],
    pub post_riichi_tsumogiri_count: [u8; 3],
    pub riichi_declaration_tile: [u8; 3],
    pub tsumogiri_streak: [u8; 3],
    pub first_six_discard_counts: [[u8; 4]; SNAPSHOT_OPPONENT_COUNT],
    pub open_meld_yakuhai_han: [u8; SNAPSHOT_OPPONENT_COUNT],
    pub visible_meld_dora_aka_han: [u8; SNAPSHOT_OPPONENT_COUNT],
    pub fully_visible_tile_kind_count: u8,
    pub unknown_distinct_dora_copy_count: u8,
    pub self_improve_tile_count: u8,
    pub self_win_tile_count: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AtomicSnapshot {
    pub factors: [[u8; SNAPSHOT_FACTOR_WIDTH]; SNAPSHOT_FIELD_COUNT],
    pub numeric: [[f32; SNAPSHOT_NUMERIC_WIDTH]; SNAPSHOT_FIELD_COUNT],
}

fn count_bucket(value: u8) -> u8 {
    value.min(25)
}

fn shanten_code(value: i8, maximum: i8) -> Result<u8, String> {
    if value == SHANTEN_UNAVAILABLE {
        return Ok(0);
    }
    if value == -1 {
        return Ok(1);
    }
    if value >= 0 {
        return Ok(value.min(maximum) as u8 + 2);
    }
    Err(format!("向听值 {value} 超出协议范围 -1..{maximum}"))
}

pub fn encode(input: &AtomicSnapshotInput) -> Result<AtomicSnapshot, String> {
    if input.observer >= 4 {
        return Err("观察者座次必须位于 0..3".to_string());
    }
    if input.meld_count > 4 {
        return Err("面子数不得超过 4".to_string());
    }
    if input.hand_counts.iter().any(|&value| value > 4)
        || input.special_counts.iter().any(|&value| value > 4)
    {
        return Err("同种牌计数不得超过 4".to_string());
    }
    for index in 0..SNAPSHOT_OPPONENT_COUNT {
        if !(1..=3).contains(&input.riichi_status[index])
            || input.riichi_turn[index] > 25
            || input.open_meld_count[index] > 4
            || input.post_riichi_tsumogiri_count[index]
                > SNAPSHOT_POST_RIICHI_TSUMOGIRI_OVERFLOW_BUCKET
            || input.riichi_declaration_tile[index] > 37
            || input.tsumogiri_streak[index] > 4
            || input.first_six_discard_counts[index]
                .iter()
                .any(|&value| value > SNAPSHOT_FIRST_DISCARD_COUNT_MAX)
            || input.open_meld_yakuhai_han[index] > SNAPSHOT_YAKUHAI_HAN_OVERFLOW_BUCKET
            || input.visible_meld_dora_aka_han[index]
                > SNAPSHOT_VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET
        {
            return Err(format!("对手 {index} 的原子字段超出协议域"));
        }
        if input.riichi_status[index] == 1 && input.riichi_turn[index] != 0 {
            return Err(format!("对手 {index} 未立直时巡目必须为 N/A"));
        }
        if input.riichi_status[index] != 1 && input.riichi_turn[index] == 0 {
            return Err(format!("对手 {index} 已立直时巡目不得为 N/A"));
        }
    }
    if input.fully_visible_tile_kind_count > SNAPSHOT_FULLY_VISIBLE_KIND_OVERFLOW_BUCKET
        || input.unknown_distinct_dora_copy_count > SNAPSHOT_UNKNOWN_DORA_COPY_OVERFLOW_BUCKET
        || input.self_improve_tile_count > SNAPSHOT_PROGRESS_TILE_OVERFLOW_BUCKET
        || input.self_win_tile_count > SNAPSHOT_PROGRESS_TILE_OVERFLOW_BUCKET
    {
        return Err("全局原子字段超出协议域".to_string());
    }

    let mut factors = [[0_u8; SNAPSHOT_FACTOR_WIDTH]; SNAPSHOT_FIELD_COUNT];
    let mut numeric = [[0_f32; SNAPSHOT_NUMERIC_WIDTH]; SNAPSHOT_FIELD_COUNT];
    for (row, spec) in SNAPSHOT_SCHEMA.iter().enumerate() {
        factors[row][0] = spec.field_id;
        factors[row][1] = spec.relative_seat;
    }

    let mut order = [0_u8, 1, 2, 3];
    order.sort_by_key(|&seat| (-input.scores[seat as usize], seat));
    factors[0][2] = order
        .iter()
        .position(|&seat| seat == input.observer)
        .expect("观察者必在四家座次中") as u8
        + 1;
    for relative in 0..3 {
        let opponent = (usize::from(input.observer) + relative + 1) % 4;
        let delta = input.scores[usize::from(input.observer)] - input.scores[opponent];
        numeric[relative + 1][0] = (delta as f32 / SCORE_PRESSURE_SCALE).clamp(-1.0, 1.0);
    }

    for opponent in 0..SNAPSHOT_OPPONENT_COUNT {
        let start = 4 + opponent * SNAPSHOT_OPPONENT_SUMMARY_WIDTH;
        factors[start][2] = input.riichi_status[opponent];
        factors[start + 1][2] = input.riichi_turn[opponent];
        factors[start + 2][2] = input.open_meld_count[opponent];
        factors[start + 3][2] = count_bucket(input.tedashi_count[opponent]);
        factors[start + 4][2] = count_bucket(input.tsumogiri_count[opponent]);
        factors[start + 5][2] = input.post_riichi_tsumogiri_count[opponent];
        for category in 0..4 {
            factors[start + 6 + category][2] = input.first_six_discard_counts[opponent][category];
        }
        factors[start + 10][2] = input.open_meld_yakuhai_han[opponent];
        factors[start + 11][2] = input.visible_meld_dora_aka_han[opponent];
        factors[start + 12][3] = input.riichi_declaration_tile[opponent];
    }

    let standard = shanten::standard(&input.hand_counts, input.meld_count);
    let chiitoitsu = if input.hand_is_open {
        SHANTEN_UNAVAILABLE
    } else {
        shanten::seven_pairs(&input.special_counts)
    };
    let kokushi = if input.hand_is_open {
        SHANTEN_UNAVAILABLE
    } else {
        shanten::thirteen_orphans(&input.special_counts)
    };
    let overall = standard.min(chiitoitsu).min(kokushi);
    factors[43][2] = shanten_code(overall, 6)?;
    factors[44][2] = shanten_code(standard, 6)?;
    factors[45][2] = shanten_code(chiitoitsu, 6)?;
    factors[46][2] = shanten_code(kokushi, 13)?;
    factors[47][2] = input.fully_visible_tile_kind_count;
    factors[48][2] = input.unknown_distinct_dora_copy_count;
    factors[49][2] = input.self_improve_tile_count;
    factors[50][2] = input.self_win_tile_count;
    for opponent in 0..SNAPSHOT_OPPONENT_COUNT {
        factors[51 + opponent][2] = input.tsumogiri_streak[opponent];
    }

    validate(&factors, &numeric)?;
    Ok(AtomicSnapshot { factors, numeric })
}

pub fn validate(
    factors: &[[u8; SNAPSHOT_FACTOR_WIDTH]; SNAPSHOT_FIELD_COUNT],
    numeric: &[[f32; SNAPSHOT_NUMERIC_WIDTH]; SNAPSHOT_FIELD_COUNT],
) -> Result<(), String> {
    for (row, spec) in SNAPSHOT_SCHEMA.iter().enumerate() {
        let value = factors[row];
        if value[0] != spec.field_id || value[1] != spec.relative_seat {
            return Err(format!("Snapshot 第 {row} 行的字段或座次顺序错误"));
        }
        if value[2] > spec.categorical_max || value[3] > spec.tile_max {
            return Err(format!("Snapshot 第 {row} 行超出离散域"));
        }
        if !numeric[row][0].is_finite()
            || (!spec.numeric && numeric[row][0] != 0.0)
            || (spec.numeric && !(-1.0..=1.0).contains(&numeric[row][0]))
        {
            return Err(format!("Snapshot 第 {row} 行的连续值无效"));
        }
    }
    Ok(())
}

#[pyfunction]
pub fn atomic_snapshot_schema() -> Vec<(u8, String, u8, u8, u8, bool)> {
    SNAPSHOT_SCHEMA
        .iter()
        .map(|spec| {
            (
                spec.field_id,
                spec.name.to_string(),
                spec.relative_seat,
                spec.categorical_max,
                spec.tile_max,
                spec.numeric,
            )
        })
        .collect()
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(atomic_snapshot_schema, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> AtomicSnapshotInput {
        let mut hand_counts = [0_u8; 34];
        for tile in [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28] {
            hand_counts[tile] += 1;
        }
        AtomicSnapshotInput {
            observer: 0,
            scores: [25_000, 30_000, 20_000, 25_000],
            hand_counts,
            special_counts: hand_counts,
            meld_count: 0,
            hand_is_open: false,
            riichi_status: [1, 2, 3],
            riichi_turn: [0, 24, 25],
            open_meld_count: [0, 1, 4],
            tedashi_count: [0, 24, 30],
            tsumogiri_count: [30, 2, 0],
            post_riichi_tsumogiri_count: [0, 15, 16],
            riichi_declaration_tile: [0, 1, 37],
            tsumogiri_streak: [0, 3, 4],
            first_six_discard_counts: [[1, 2, 3, 0], [6, 0, 0, 0], [0, 0, 0, 6]],
            open_meld_yakuhai_han: [0, 5, 6],
            visible_meld_dora_aka_han: [0, 7, 8],
            fully_visible_tile_kind_count: 25,
            unknown_distinct_dora_copy_count: 16,
            self_improve_tile_count: 40,
            self_win_tile_count: 12,
        }
    }

    #[test]
    fn schema_is_fixed_and_contiguous() {
        assert_eq!(SNAPSHOT_SCHEMA.len(), 54);
        for (index, spec) in SNAPSHOT_SCHEMA.iter().enumerate() {
            assert_eq!(usize::from(spec.field_id), index + 1);
        }
    }

    #[test]
    fn encodes_domains_and_overflow_buckets() {
        let snapshot = encode(&input()).expect("合法 Snapshot");
        assert_eq!(snapshot.factors[0][2], 2);
        // 对手 1 摘要(start=4):状态/巡目/副露/手切/摸切/立直后摸切/四类前六/
        // 役番/宝番/宣言牌。
        assert_eq!(snapshot.factors[4][2], 1);
        assert_eq!(snapshot.factors[7][2], 0);
        assert_eq!(snapshot.factors[8][2], 25);
        assert_eq!(snapshot.factors[9][2], 0);
        assert_eq!(snapshot.factors[10][2], 1);
        assert_eq!(snapshot.factors[12][2], 3);
        assert_eq!(snapshot.factors[13][2], 0);
        assert_eq!(snapshot.factors[14][2], 0);
        assert_eq!(snapshot.factors[15][2], 0);
        assert_eq!(snapshot.factors[16][3], 0);
        // 对手 2(start=17):立直后摸切 15、宣言牌 1。
        assert_eq!(snapshot.factors[22][2], 15);
        assert_eq!(snapshot.factors[29][3], 1);
        // 对手 3(start=30):立直后摸切 16=16+、宣言牌 37、新六项溢出桶截至。
        assert_eq!(snapshot.factors[35][2], 16);
        assert_eq!(snapshot.factors[39][2], 6);
        assert_eq!(snapshot.factors[40][2], 6);
        assert_eq!(snapshot.factors[41][2], 8);
        assert_eq!(snapshot.factors[42][3], 37);
        // 全局与自身进展:完全可见 25=25+、未知宝牌 16=16+、进张 40=40+、
        // 和牌 12、三条摸切连打。
        assert_eq!(snapshot.factors[47][2], 25);
        assert_eq!(snapshot.factors[48][2], 16);
        assert_eq!(snapshot.factors[49][2], 40);
        assert_eq!(snapshot.factors[50][2], 12);
        assert_eq!(snapshot.factors[51][2], 0);
        assert_eq!(snapshot.factors[52][2], 3);
        assert_eq!(snapshot.factors[53][2], 4);
        assert!((snapshot.numeric[1][0] + 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn open_hand_marks_special_shanten_unavailable() {
        let mut value = input();
        value.hand_is_open = true;
        value.meld_count = 1;
        let snapshot = encode(&value).expect("合法开手 Snapshot");
        assert_eq!(snapshot.factors[45][2], 0);
        assert_eq!(snapshot.factors[46][2], 0);
    }

    #[test]
    fn rejects_inconsistent_riichi_turn() {
        let mut value = input();
        value.riichi_status[0] = 1;
        value.riichi_turn[0] = 1;
        assert!(encode(&value).is_err());
    }

    #[test]
    fn rejects_out_of_domain_progress_and_post_riichi_values() {
        let mut value = input();
        value.self_improve_tile_count = 41;
        assert!(encode(&value).is_err());
        let mut value = input();
        value.post_riichi_tsumogiri_count[0] = 17;
        assert!(encode(&value).is_err());
        let mut value = input();
        value.riichi_declaration_tile[0] = 38;
        assert!(encode(&value).is_err());
    }

    #[test]
    fn shanten_overflow_uses_the_maximum_bucket() {
        assert_eq!(shanten_code(7, 6), Ok(8));
        assert_eq!(shanten_code(20, 13), Ok(15));
    }
}
