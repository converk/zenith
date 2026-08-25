//! V16 进攻「役/基础番」内核(Offense Query O4/O5)。
//!
//! 向听、有效牌与等待牌由 state-machine 的 `analyze_offense_v16` 计算(其 shanten
//! 是既有生产路径);本模块只复用 `HandEvaluator` 对每个等待牌逐张计算是否有役
//! (O4)与无立直荣和路线的基础番(O5,不含一发/里宝/海底)。

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::hand_evaluator::HandEvaluator;
use crate::types::{Conditions, Meld, Wind};

#[cfg(feature = "python")]
#[pyclass(name = "YakuAnalysisV16", frozen)]
pub struct YakuAnalysisV16 {
    yaku_class: Py<PyArray1<u8>>,
    base_han: Py<PyArray1<u8>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct YakuRowV16 {
    pub yaku_class: u8,
    pub base_han: u8,
}

/// 对已经确定的等待牌批量计算 V16 O4/O5。
#[allow(clippy::too_many_arguments)]
pub fn analyze_offense_v16_rows(
    concealed_tiles: &[Vec<u8>],
    melds: &[Vec<Meld>],
    wait_masks: &[u64],
    dora_indicators: &[Vec<u8>],
    player_wind: &[u8],
    round_wind: &[u8],
    honba: &[u8],
    riichi_sticks: &[u8],
) -> Result<Vec<YakuRowV16>, String> {
    let rows = concealed_tiles.len();
    for (name, length) in [
        ("melds", melds.len()),
        ("wait_masks", wait_masks.len()),
        ("dora_indicators", dora_indicators.len()),
        ("player_wind", player_wind.len()),
        ("round_wind", round_wind.len()),
        ("honba", honba.len()),
        ("riichi_sticks", riichi_sticks.len()),
    ] {
        if length != rows {
            return Err(format!("{name} must have length {rows}, got {length}"));
        }
    }

    let mut output = vec![YakuRowV16::default(); rows];
    for row in 0..rows {
        let mask = wait_masks[row];
        if mask >> 34 != 0 {
            return Err(format!("row {row} wait mask exceeds 34 bits"));
        }
        let evaluator = HandEvaluator::new(concealed_tiles[row].clone(), melds[row].clone());
        let conditions = Conditions {
            tsumo: false,
            riichi: false,
            player_wind: Wind::from(player_wind[row]),
            round_wind: Wind::from(round_wind[row]),
            honba: u32::from(honba[row]),
            riichi_sticks: u32::from(riichi_sticks[row]),
            ..Default::default()
        };
        let waits: Vec<usize> = (0..34)
            .filter(|&tile| mask & (1_u64 << tile) != 0)
            .collect();
        let mut win_count = 0_u8;
        let mut max_han = 0_u8;
        for &wait in &waits {
            // 0 号拷贝是赤五;基础番用普通五(与决策分析约定一致)。
            let mut win = (wait * 4) as u8;
            if win == 16 || win == 52 || win == 88 {
                win += 1;
            }
            let result = evaluator.calc(
                win,
                dora_indicators[row].clone(),
                Vec::new(),
                Some(conditions.clone()),
            );
            if result.is_win {
                win_count += 1;
                max_han = max_han.max(result.han.min(255) as u8);
            }
        }
        output[row].yaku_class = if win_count == 0 {
            1
        } else if usize::from(win_count) == waits.len() {
            3
        } else {
            2
        };
        if win_count > 0 {
            output[row].base_han = max_han.clamp(1, 5);
        }
    }
    Ok(output)
}

#[cfg(feature = "python")]
#[pymethods]
impl YakuAnalysisV16 {
    #[getter]
    fn yaku_class(&self, py: Python<'_>) -> Py<PyArray1<u8>> {
        self.yaku_class.clone_ref(py)
    }
    #[getter]
    fn base_han(&self, py: Python<'_>) -> Py<PyArray1<u8>> {
        self.base_han.clone_ref(py)
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn analyze_offense_v16(
    py: Python<'_>,
    concealed_tiles: Vec<Vec<u8>>,
    melds: Vec<Vec<Meld>>,
    wait_masks: PyReadonlyArray1<'_, u64>,
    dora_indicators: Vec<Vec<u8>>,
    player_wind: PyReadonlyArray1<'_, u8>,
    round_wind: PyReadonlyArray1<'_, u8>,
    honba: PyReadonlyArray1<'_, u8>,
    riichi_sticks: PyReadonlyArray1<'_, u8>,
) -> PyResult<YakuAnalysisV16> {
    let rows = concealed_tiles.len();
    if melds.len() != rows || dora_indicators.len() != rows {
        return Err(PyValueError::new_err(
            "concealed_tiles/melds/dora_indicators must have one entry per row",
        ));
    }
    for (name, length) in [
        ("wait_masks", wait_masks.len()),
        ("player_wind", player_wind.len()),
        ("round_wind", round_wind.len()),
        ("honba", honba.len()),
        ("riichi_sticks", riichi_sticks.len()),
    ] {
        if length != rows {
            return Err(PyValueError::new_err(format!(
                "{name} must have length {rows}, got {length}"
            )));
        }
    }
    let wait_values = wait_masks
        .as_slice()
        .map_err(|_| PyValueError::new_err("wait_masks must be contiguous"))?;
    let wind_values = player_wind
        .as_slice()
        .map_err(|_| PyValueError::new_err("player_wind must be contiguous"))?;
    let round_values = round_wind
        .as_slice()
        .map_err(|_| PyValueError::new_err("round_wind must be contiguous"))?;
    let honba_values = honba
        .as_slice()
        .map_err(|_| PyValueError::new_err("honba must be contiguous"))?;
    let sticks_values = riichi_sticks
        .as_slice()
        .map_err(|_| PyValueError::new_err("riichi_sticks must be contiguous"))?;

    let output = py
        .detach(|| {
            analyze_offense_v16_rows(
                &concealed_tiles,
                &melds,
                wait_values,
                &dora_indicators,
                wind_values,
                round_values,
                honba_values,
                sticks_values,
            )
        })
        .map_err(PyValueError::new_err)?;
    let out_class = output.iter().map(|row| row.yaku_class).collect();
    let out_han = output.iter().map(|row| row.base_han).collect();

    let class_array = PyArray1::from_vec(py, out_class);
    let han_array = PyArray1::from_vec(py, out_han);
    for array in [class_array.as_any(), han_array.as_any()] {
        array.call_method1("setflags", (false,))?;
    }
    Ok(YakuAnalysisV16 {
        yaku_class: class_array.unbind(),
        base_han: han_array.unbind(),
    })
}

#[cfg(test)]
mod tests {
    use crate::hand_evaluator::HandEvaluator;
    use crate::types::Conditions;

    #[test]
    fn pinfu_damaten_ron_has_yaku() {
        // 123m 456p 789s 55m 67s(13 张,听 5s/8s,平和 1 番;普通五,无赤)。
        let tiles = vec![0, 4, 8, 48, 53, 56, 96, 100, 104, 17, 18, 92, 96];
        let evaluator = HandEvaluator::new(tiles, Vec::new());
        let result = evaluator.calc(
            89, // 普通 5s
            Vec::new(),
            Vec::new(),
            Some(Conditions {
                tsumo: false,
                riichi: false,
                ..Default::default()
            }),
        );
        assert!(result.is_win);
        assert_eq!(result.han, 1);
    }
}
