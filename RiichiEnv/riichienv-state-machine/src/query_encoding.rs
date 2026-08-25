use std::collections::HashMap;

use numpy::{
    ndarray::Array3, IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::analysis::{defense_row, offense_row, OffenseRow};
use crate::mjai_kyoku_state_machine::TILE_KINDS;
use crate::shanten;

const QUERY_ROWS_PER_ACTION: usize = 2;
const QUERY_ROW_WIDTH: usize = 15;
const ANSWER_START: usize = 5;

const MODE_FULL_OFFENSE: u8 = 0;
const MODE_SIMPLE_SHANTEN: u8 = 1;
const MODE_WIN: u8 = 2;
const MODE_MIN_DROP: u8 = 3;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct OffenseKey {
    counts: [u8; TILE_KINDS],
    open_melds: u8,
    remaining: [u8; TILE_KINDS],
    own_river: u64,
    missed_doujun: bool,
    missed_riichi: bool,
    riichi_declared: bool,
    score: i32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ShantenKey {
    counts: [u8; TILE_KINDS],
    open_melds: u8,
    mode: u8,
}

#[pyclass(name = "QueryBatchEncoding", frozen)]
pub struct QueryBatchEncoding {
    query_rows: Py<PyArray3<i32>>,
    wait_masks: Py<PyArray1<u64>>,
    #[pyo3(get)]
    row_count: usize,
    #[pyo3(get)]
    unique_offense_rows: usize,
    #[pyo3(get)]
    unique_shanten_rows: usize,
}

#[pymethods]
impl QueryBatchEncoding {
    #[getter]
    fn query_rows(&self, py: Python<'_>) -> Py<PyArray3<i32>> {
        self.query_rows.clone_ref(py)
    }

    #[getter]
    fn wait_masks(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.wait_masks.clone_ref(py)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<QueryBatchEncoding>()?;
    module.add_function(wrap_pyfunction!(encode_query_batch, module)?)?;
    Ok(())
}

fn bucket_o0(shanten_value: i8) -> i32 {
    if shanten_value < 0 {
        0
    } else {
        i32::from(shanten_value.min(5)) + 1
    }
}

fn bucket_o2(value: u16) -> i32 {
    if value == 0 {
        0
    } else {
        i32::from(((value - 1) / 4 + 1).min(6))
    }
}

fn encode_can_riichi(value: u8) -> i32 {
    match value {
        0 => 2,
        1 => 0,
        2 => 1,
        _ => 2,
    }
}

fn simple_shanten(counts: &[u8; TILE_KINDS], open_melds: u8, mode: u8) -> Result<i8, String> {
    if counts.iter().any(|&value| value > 4) || open_melds > 4 {
        return Err("invalid hand counts or open meld count".to_string());
    }
    let total = counts
        .iter()
        .map(|&value| usize::from(value))
        .sum::<usize>()
        + 3 * usize::from(open_melds);
    match mode {
        MODE_SIMPLE_SHANTEN if total == 13 || total == 14 => {
            Ok(shanten::calculate(counts, open_melds).overall)
        }
        MODE_MIN_DROP if total == 14 => {
            let mut best = i8::MAX;
            for tile in 0..TILE_KINDS {
                if counts[tile] == 0 {
                    continue;
                }
                let mut post = *counts;
                post[tile] -= 1;
                best = best.min(shanten::calculate(&post, open_melds).overall);
            }
            if best == i8::MAX {
                Err("14-tile min-drop shape has no removable tile".to_string())
            } else {
                Ok(best)
            }
        }
        MODE_SIMPLE_SHANTEN => Err(format!(
            "simple-shanten shape represents {total} tiles; expected 13 or 14"
        )),
        MODE_MIN_DROP => Err(format!(
            "min-drop shape represents {total} tiles; expected 14"
        )),
        _ => Err(format!("unsupported shanten mode {mode}")),
    }
}

/// 把紧凑的逐动作事实融合编码成 V18 O0..O9 / D0..D9 连续行。
///
/// `modes`:0=完整进攻分析,1=仅向听,2=和牌固定值,3=14 张逐牌型弃牌取最小向听。
/// `o8_values` 的 255 表示从完整进攻分析读取可立直状态,其余值直接写入 O8。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn encode_query_batch(
    py: Python<'_>,
    action_ids: PyReadonlyArray1<'_, u16>,
    action_types: PyReadonlyArray1<'_, u8>,
    primary_types: PyReadonlyArray1<'_, i16>,
    source_seats: PyReadonlyArray1<'_, i8>,
    modes: PyReadonlyArray1<'_, u8>,
    shape_counts: PyReadonlyArray2<'_, u8>,
    open_melds: PyReadonlyArray1<'_, u8>,
    remaining: PyReadonlyArray2<'_, u8>,
    own_rivers: PyReadonlyArray1<'_, u64>,
    opponent_rivers: PyReadonlyArray2<'_, u64>,
    defense_counts: PyReadonlyArray2<'_, u8>,
    discard_types: PyReadonlyArray1<'_, i16>,
    defense_visible: PyReadonlyArray1<'_, u8>,
    missed_doujun: PyReadonlyArray1<'_, bool>,
    missed_riichi: PyReadonlyArray1<'_, bool>,
    riichi_declared: PyReadonlyArray1<'_, bool>,
    scores: PyReadonlyArray1<'_, i32>,
    o7_values: PyReadonlyArray1<'_, u8>,
    o8_values: PyReadonlyArray1<'_, u8>,
    o9_values: PyReadonlyArray1<'_, u8>,
) -> PyResult<QueryBatchEncoding> {
    let rows = action_ids.len();
    for (name, length) in [
        ("action_types", action_types.len()),
        ("primary_types", primary_types.len()),
        ("source_seats", source_seats.len()),
        ("modes", modes.len()),
        ("open_melds", open_melds.len()),
        ("own_rivers", own_rivers.len()),
        ("discard_types", discard_types.len()),
        ("defense_visible", defense_visible.len()),
        ("missed_doujun", missed_doujun.len()),
        ("missed_riichi", missed_riichi.len()),
        ("riichi_declared", riichi_declared.len()),
        ("scores", scores.len()),
        ("o7_values", o7_values.len()),
        ("o8_values", o8_values.len()),
        ("o9_values", o9_values.len()),
    ] {
        if length != rows {
            return Err(PyValueError::new_err(format!(
                "{name} must have length {rows}, got {length}"
            )));
        }
    }
    for (name, shape) in [
        ("shape_counts", shape_counts.shape()),
        ("remaining", remaining.shape()),
        ("defense_counts", defense_counts.shape()),
    ] {
        if shape != [rows, TILE_KINDS] {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape uint8[N,{TILE_KINDS}]"
            )));
        }
    }
    if opponent_rivers.shape() != [rows, 3] {
        return Err(PyValueError::new_err(
            "opponent_rivers must have shape uint64[N,3]",
        ));
    }

    let action_ids = action_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("action_ids must be contiguous"))?;
    let action_types = action_types
        .as_slice()
        .map_err(|_| PyValueError::new_err("action_types must be contiguous"))?;
    let primary_types = primary_types
        .as_slice()
        .map_err(|_| PyValueError::new_err("primary_types must be contiguous"))?;
    let source_seats = source_seats
        .as_slice()
        .map_err(|_| PyValueError::new_err("source_seats must be contiguous"))?;
    let modes = modes
        .as_slice()
        .map_err(|_| PyValueError::new_err("modes must be contiguous"))?;
    let shape_counts = shape_counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("shape_counts must be contiguous"))?;
    let open_melds = open_melds
        .as_slice()
        .map_err(|_| PyValueError::new_err("open_melds must be contiguous"))?;
    let remaining = remaining
        .as_slice()
        .map_err(|_| PyValueError::new_err("remaining must be contiguous"))?;
    let own_rivers = own_rivers
        .as_slice()
        .map_err(|_| PyValueError::new_err("own_rivers must be contiguous"))?;
    let opponent_rivers = opponent_rivers
        .as_slice()
        .map_err(|_| PyValueError::new_err("opponent_rivers must be contiguous"))?;
    let defense_counts = defense_counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("defense_counts must be contiguous"))?;
    let discard_types = discard_types
        .as_slice()
        .map_err(|_| PyValueError::new_err("discard_types must be contiguous"))?;
    let defense_visible = defense_visible
        .as_slice()
        .map_err(|_| PyValueError::new_err("defense_visible must be contiguous"))?;
    let missed_doujun = missed_doujun
        .as_slice()
        .map_err(|_| PyValueError::new_err("missed_doujun must be contiguous"))?;
    let missed_riichi = missed_riichi
        .as_slice()
        .map_err(|_| PyValueError::new_err("missed_riichi must be contiguous"))?;
    let riichi_declared = riichi_declared
        .as_slice()
        .map_err(|_| PyValueError::new_err("riichi_declared must be contiguous"))?;
    let scores = scores
        .as_slice()
        .map_err(|_| PyValueError::new_err("scores must be contiguous"))?;
    let o7_values = o7_values
        .as_slice()
        .map_err(|_| PyValueError::new_err("o7_values must be contiguous"))?;
    let o8_values = o8_values
        .as_slice()
        .map_err(|_| PyValueError::new_err("o8_values must be contiguous"))?;
    let o9_values = o9_values
        .as_slice()
        .map_err(|_| PyValueError::new_err("o9_values must be contiguous"))?;

    let mut query_rows = vec![0_i32; rows * QUERY_ROWS_PER_ACTION * QUERY_ROW_WIDTH];
    let mut wait_masks = vec![0_u64; rows];
    let mut offense_cache: HashMap<OffenseKey, OffenseRow> = HashMap::new();
    let mut shanten_cache: HashMap<ShantenKey, i8> = HashMap::new();

    let encoding_result: Result<(), String> = py.detach(|| {
        for row in 0..rows {
            let start = row * TILE_KINDS;
            let counts: [u8; TILE_KINDS] = shape_counts[start..start + TILE_KINDS]
                .try_into()
                .expect("shape row width");
            let remain: [u8; TILE_KINDS] = remaining[start..start + TILE_KINDS]
                .try_into()
                .expect("remaining row width");
            let defense_hand: [u8; TILE_KINDS] = defense_counts[start..start + TILE_KINDS]
                .try_into()
                .expect("defense row width");
            let rivers: [u64; 3] = opponent_rivers[row * 3..row * 3 + 3]
                .try_into()
                .expect("opponent river row width");
            if !(1..=11).contains(&action_types[row]) {
                return Err(format!("row {row} action type is out of range"));
            }
            if !(-1..TILE_KINDS as i16).contains(&primary_types[row]) {
                return Err(format!("row {row} primary type is out of range"));
            }
            if !(-1..=3).contains(&source_seats[row]) {
                return Err(format!("row {row} source seat is out of range"));
            }
            if defense_visible[row] > 5 || o7_values[row] > 1 || o9_values[row] > 5 {
                return Err(format!(
                    "row {row} contains an out-of-range categorical value"
                ));
            }

            let offense_base = row * QUERY_ROWS_PER_ACTION * QUERY_ROW_WIDTH;
            let defense_base = offense_base + QUERY_ROW_WIDTH;
            for (base, query_type) in [(offense_base, 1_i32), (defense_base, 2_i32)] {
                query_rows[base] = query_type;
                query_rows[base + 1] = i32::from(action_ids[row]);
                query_rows[base + 2] = i32::from(action_types[row]);
                query_rows[base + 3] = if primary_types[row] < 0 {
                    0
                } else {
                    i32::from(primary_types[row]) + 1
                };
                query_rows[base + 4] = if source_seats[row] < 0 {
                    0
                } else {
                    i32::from(source_seats[row]) + 1
                };
            }

            let answers =
                &mut query_rows[offense_base + ANSWER_START..offense_base + QUERY_ROW_WIDTH];
            match modes[row] {
                MODE_FULL_OFFENSE => {
                    let key = OffenseKey {
                        counts,
                        open_melds: open_melds[row],
                        remaining: remain,
                        own_river: own_rivers[row],
                        missed_doujun: missed_doujun[row],
                        missed_riichi: missed_riichi[row],
                        riichi_declared: riichi_declared[row],
                        score: scores[row],
                    };
                    let value = if let Some(value) = offense_cache.get(&key) {
                        *value
                    } else {
                        let value = offense_row(
                            &key.counts,
                            key.open_melds,
                            &key.remaining,
                            key.own_river,
                            key.missed_doujun,
                            key.missed_riichi,
                            key.riichi_declared,
                            key.score,
                        )
                        .map_err(|error| format!("row {row} {error}"))?;
                        offense_cache.insert(key, value);
                        value
                    };
                    answers[0] = bucket_o0(value.shanten);
                    answers[1] = i32::from(value.effective_kinds.min(10));
                    answers[2] = bucket_o2(value.effective_remaining);
                    answers[3] = if value.wait_kinds == 0 {
                        0
                    } else {
                        i32::from(value.wait_kinds.min(13))
                    };
                    answers[4] = 0;
                    answers[5] = 0;
                    answers[6] = i32::from(value.furiten);
                    answers[7] = i32::from(o7_values[row]);
                    answers[8] = if o8_values[row] == u8::MAX {
                        encode_can_riichi(value.can_riichi)
                    } else {
                        i32::from(o8_values[row])
                    };
                    answers[9] = i32::from(o9_values[row]);
                    wait_masks[row] = value.wait_mask;
                }
                MODE_SIMPLE_SHANTEN | MODE_MIN_DROP => {
                    let key = ShantenKey {
                        counts,
                        open_melds: open_melds[row],
                        mode: modes[row],
                    };
                    let value = if let Some(value) = shanten_cache.get(&key) {
                        *value
                    } else {
                        let value = simple_shanten(&key.counts, key.open_melds, key.mode)
                            .map_err(|error| format!("row {row} {error}"))?;
                        shanten_cache.insert(key, value);
                        value
                    };
                    answers[0] = bucket_o0(value);
                    answers[7] = i32::from(o7_values[row]);
                    answers[8] = i32::from(o8_values[row]);
                    answers[9] = i32::from(o9_values[row]);
                }
                MODE_WIN => {
                    answers[7] = i32::from(o7_values[row]);
                    answers[8] = i32::from(o8_values[row]);
                    answers[9] = i32::from(o9_values[row]);
                }
                mode => return Err(format!("row {row} uses unsupported mode {mode}")),
            }

            let defense_answers =
                &mut query_rows[defense_base + ANSWER_START..defense_base + QUERY_ROW_WIDTH];
            if modes[row] == MODE_WIN {
                defense_answers[0..6].fill(2);
                defense_answers[6..9].fill(0);
            } else {
                let (genbutsu, suji, stock, _visible) =
                    defense_row(discard_types[row], &rivers, &defense_hand, &remain);
                for opponent in 0..3 {
                    defense_answers[opponent] = if genbutsu[opponent] == 2 {
                        2
                    } else {
                        1 - i32::from(genbutsu[opponent])
                    };
                    defense_answers[3 + opponent] = if suji[opponent] == 2 {
                        2
                    } else {
                        1 - i32::from(suji[opponent])
                    };
                    defense_answers[6 + opponent] = i32::from(stock[opponent].min(4));
                }
            }
            defense_answers[9] = i32::from(defense_visible[row]);
        }
        Ok(())
    });
    encoding_result.map_err(PyValueError::new_err)?;

    let query_rows =
        Array3::from_shape_vec((rows, QUERY_ROWS_PER_ACTION, QUERY_ROW_WIDTH), query_rows)
            .expect("query row shape")
            .into_pyarray(py);
    let wait_masks = PyArray1::from_vec(py, wait_masks);
    wait_masks.call_method1("setflags", (false,))?;
    Ok(QueryBatchEncoding {
        query_rows: query_rows.unbind(),
        wait_masks: wait_masks.unbind(),
        row_count: rows,
        unique_offense_rows: offense_cache.len(),
        unique_shanten_rows: shanten_cache.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_drop_uses_best_13_tile_shape() {
        let mut counts = [0_u8; TILE_KINDS];
        for tile in [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28, 29] {
            counts[tile] += 1;
        }
        let expected = (0..TILE_KINDS)
            .filter(|&tile| counts[tile] > 0)
            .map(|tile| {
                let mut post = counts;
                post[tile] -= 1;
                shanten::calculate(&post, 0).overall
            })
            .min()
            .unwrap();
        assert_eq!(simple_shanten(&counts, 0, MODE_MIN_DROP).unwrap(), expected);
    }
}
