use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{shanten, HAND_ANALYSIS_VERSION, SHANTEN_UNAVAILABLE};

fn suji_safe(tile: usize, river: u64) -> bool {
    if tile >= 27 {
        return false;
    }
    let rank = tile % 9;
    let lower = (rank >= 3).then_some(tile - 3);
    let upper = (rank <= 5).then_some(tile + 3);
    // Middle tiles have two independent ryanmen routes.  Both suji anchors
    // must be present; e.g. a discarded 1m alone does not make 4m safe from
    // a 56m wait.  Edge tiles have only one applicable anchor.
    lower
        .into_iter()
        .chain(upper)
        .all(|anchor| river & (1_u64 << anchor) != 0)
}

fn wall_class(tile: usize, remaining: &[u8]) -> u8 {
    if tile >= 27 {
        return 0;
    }
    let rank = tile % 9;
    let base = tile - rank;
    // A kabe is made by related tiles needed alongside the candidate in a
    // sequence, not by copies of the candidate itself.  Preserve the compact
    // contract: 0=no wall, 1=three-visible wall, 2=four-visible wall.
    let strongest = (0..9)
        .filter(|&other| other != rank && other.abs_diff(rank) <= 2)
        .map(|other| 4_u8.saturating_sub(remaining[base + other]))
        .max()
        .unwrap_or(0);
    if strongest >= 4 {
        2
    } else if strongest >= 3 {
        1
    } else {
        0
    }
}

#[pyclass(name = "HandAnalysis", frozen)]
pub struct HandAnalysis {
    shanten: Py<PyArray2<i8>>,
    improving_type_mask: Py<PyArray1<u64>>,
    #[pyo3(get)]
    row_count: usize,
    #[pyo3(get)]
    analysis_version: u32,
}

#[pyclass(name = "FeatureAnalysis", frozen)]
pub struct FeatureAnalysis {
    shanten: Py<PyArray2<i8>>,
    improving_type_mask: Py<PyArray1<u64>>,
    ukeire: Py<PyArray1<u16>>,
    wait_count: Py<PyArray1<u8>>,
    defense: Py<PyArray2<u8>>,
    categorical: Py<PyArray2<u8>>,
    numeric: Py<PyArray2<f32>>,
    #[pyo3(get)]
    row_count: usize,
    #[pyo3(get)]
    analysis_version: u32,
}

#[pymethods]
impl FeatureAnalysis {
    #[getter]
    fn shanten(&self, py: Python<'_>) -> Py<PyArray2<i8>> {
        self.shanten.clone_ref(py)
    }
    #[getter]
    fn improving_type_mask(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.improving_type_mask.clone_ref(py)
    }
    #[getter]
    fn ukeire(&self, py: Python<'_>) -> Py<PyArray1<u16>> {
        self.ukeire.clone_ref(py)
    }
    #[getter]
    fn wait_count(&self, py: Python<'_>) -> Py<PyArray1<u8>> {
        self.wait_count.clone_ref(py)
    }
    #[getter]
    fn defense(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.defense.clone_ref(py)
    }
    #[getter]
    fn categorical(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.categorical.clone_ref(py)
    }
    #[getter]
    fn numeric(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.numeric.clone_ref(py)
    }
}

#[pymethods]
impl HandAnalysis {
    #[getter]
    fn shanten(&self, py: Python<'_>) -> Py<PyArray2<i8>> {
        self.shanten.clone_ref(py)
    }

    #[getter]
    fn improving_type_mask(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.improving_type_mask.clone_ref(py)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<HandAnalysis>()?;
    module.add_class::<FeatureAnalysis>()?;
    module.add_function(wrap_pyfunction!(analyze_hands, module)?)?;
    module.add_function(wrap_pyfunction!(analyze_features, module)?)?;
    Ok(())
}

#[pyfunction]
fn analyze_features(
    py: Python<'_>,
    action_ids: PyReadonlyArray1<'_, u16>,
    counts: PyReadonlyArray2<'_, u8>,
    open_melds: PyReadonlyArray1<'_, u8>,
    remaining: PyReadonlyArray2<'_, u8>,
    discard_types: PyReadonlyArray1<'_, i16>,
    river_masks: PyReadonlyArray2<'_, u64>,
    passed_masks: PyReadonlyArray2<'_, u64>,
) -> PyResult<FeatureAnalysis> {
    let shape = counts.shape();
    if shape.len() != 2 || shape[1] != 34 {
        return Err(PyValueError::new_err("counts must have shape uint8[N,34]"));
    }
    let rows = shape[0];
    if action_ids.shape() != [rows]
        || open_melds.shape() != [rows]
        || remaining.shape() != [rows, 34]
        || discard_types.shape() != [rows]
        || river_masks.shape() != [rows, 3]
        || passed_masks.shape() != [rows, 3]
    {
        return Err(PyValueError::new_err(
            "feature-analysis arrays have incompatible shapes",
        ));
    }
    let action_ids = action_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("action_ids must be contiguous"))?;
    let mut sorted_ids = action_ids.to_vec();
    sorted_ids.sort_unstable();
    if sorted_ids.windows(2).any(|pair| pair[0] == pair[1])
        || sorted_ids.iter().any(|&id| id >= 241)
    {
        return Err(PyValueError::new_err(
            "action ids must be unique and inside [0,241)",
        ));
    }
    let counts = counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("counts must be contiguous"))?;
    let open_melds = open_melds
        .as_slice()
        .map_err(|_| PyValueError::new_err("open_melds must be contiguous"))?;
    let remaining = remaining
        .as_slice()
        .map_err(|_| PyValueError::new_err("remaining must be contiguous"))?;
    let discard_types = discard_types
        .as_slice()
        .map_err(|_| PyValueError::new_err("discard_types must be contiguous"))?;
    let rivers = river_masks
        .as_slice()
        .map_err(|_| PyValueError::new_err("river_masks must be contiguous"))?;
    let passed = passed_masks
        .as_slice()
        .map_err(|_| PyValueError::new_err("passed_masks must be contiguous"))?;
    if remaining.iter().any(|&value| value > 4) {
        return Err(PyValueError::new_err(
            "remaining counts must be inside [0,4]",
        ));
    }
    if discard_types.iter().any(|&tile| !(-1..=33).contains(&tile)) {
        return Err(PyValueError::new_err(
            "discard types must be N/A (-1) or inside [0,34)",
        ));
    }
    if action_ids
        .iter()
        .zip(discard_types)
        .any(|(&id, &tile)| (1..=74).contains(&id) != (tile >= 0))
    {
        return Err(PyValueError::new_err(
            "discard action ids and discard-type N/A markers disagree",
        ));
    }
    if rivers
        .iter()
        .chain(passed.iter())
        .any(|&mask| mask >> 34 != 0)
    {
        return Err(PyValueError::new_err(
            "public tile masks may use only the low 34 bits",
        ));
    }

    let values = py.detach(|| {
        (0..rows).map(|row| {
            let hand_slice = &counts[row * 34..(row + 1) * 34];
            let melds = open_melds[row];
            if melds > 4 || hand_slice.iter().any(|&count| count > 4) {
                return Err(format!("invalid hand row {row}"));
            }
            let total = hand_slice.iter().map(|&v| usize::from(v)).sum::<usize>() + 3 * usize::from(melds);
            if total != 13 && total != 14 {
                return Err(format!("row {row} represents {total} tiles; expected 13 or 14"));
            }
            let action_id = action_ids[row];
            if action_id < 239 && total != 13 {
                return Err(format!(
                    "row {row} action {action_id} must describe a normalized 13-tile post-action shape"
                ));
            }
            let mut hand = [0_u8; 34];
            hand.copy_from_slice(hand_slice);
            let sh = shanten::calculate(&hand, melds);
            let mut improving = 0_u64;
            // Ukeire is defined for a 13-tile post-discard shape.  A 14-tile
            // row has no single well-defined improving draw until a discard
            // is chosen, so return the explicit empty/N/A mask.
            if total == 13 && sh.overall != SHANTEN_UNAVAILABLE {
                for tile in 0..34 {
                    if hand[tile] >= 4 { continue; }
                    let mut next = hand;
                    next[tile] += 1;
                    if shanten::calculate(&next, melds).overall < sh.overall {
                        improving |= 1_u64 << tile;
                    }
                }
            }
            let remain = &remaining[row * 34..(row + 1) * 34];
            let ukeire = (0..34).filter(|&tile| improving & (1_u64 << tile) != 0)
                .map(|tile| u16::from(remain[tile])).sum::<u16>();
            let tile = discard_types[row];
            let mut defense = [0_u8; 5]; // genbutsu, suji, wall, honor-visible, passed
            if tile >= 0 {
                let tile = usize::try_from(tile).map_err(|_| format!("invalid discard type at row {row}"))?;
                if tile >= 34 { return Err(format!("invalid discard type at row {row}")); }
                for opponent in 0..3 {
                    let river = rivers[row * 3 + opponent];
                    if river & (1_u64 << tile) != 0 { defense[0] |= 1 << opponent; }
                    if suji_safe(tile, river) { defense[1] |= 1 << opponent; }
                    if passed[row * 3 + opponent] & (1_u64 << tile) != 0 { defense[4] |= 1 << opponent; }
                }
                let visible = 4_u8.saturating_sub(remain[tile]);
                defense[2] = wall_class(tile, remain);
                let public_visible = visible.saturating_sub(hand_slice[tile]);
                defense[3] = if tile >= 27 { public_visible } else { 0 };
            }
            Ok(([
                sh.overall, sh.standard, sh.seven_pairs, sh.thirteen_orphans,
            ], improving, ukeire, improving.count_ones() as u8, defense))
        }).collect::<Result<Vec<_>, String>>()
    }).map_err(PyValueError::new_err)?;

    let shanten_values = values.iter().flat_map(|value| value.0).collect::<Vec<_>>();
    let masks = values.iter().map(|value| value.1).collect::<Vec<_>>();
    let ukeire = values.iter().map(|value| value.2).collect::<Vec<_>>();
    let waits = values.iter().map(|value| value.3).collect::<Vec<_>>();
    let defense_values = values.iter().flat_map(|value| value.4).collect::<Vec<_>>();
    let mut categorical_values = vec![0_u8; rows * 10];
    let mut numeric_values = vec![0_f32; rows * 8];
    for row in 0..rows {
        let aid = action_ids[row];
        let action_kind = match aid {
            0 => 1,
            1..=74 => 2,
            75 => 3,
            76..=132 => 4,
            133..=169 => 5,
            170 => 6,
            171..=204 => 7,
            205..=238 => 8,
            239 => 9,
            _ => 10,
        };
        let defense = values[row].4;
        let base = row * 10;
        categorical_values[base] = 7;
        categorical_values[base + 1] = action_kind;
        categorical_values[base + 2] = u8::try_from((aid + 1).min(255)).unwrap_or(255);
        categorical_values[base + 4] = defense[0];
        categorical_values[base + 5] = defense[1];
        categorical_values[base + 6] = defense[2];
        categorical_values[base + 7] = defense[4];
        categorical_values[base + 9] = 2;
        let numeric_base = row * 8;
        numeric_values[numeric_base] = defense[0].count_ones() as f32 / 3.0;
        numeric_values[numeric_base + 1] = defense[1].count_ones() as f32 / 3.0;
        if discard_types[row] >= 0 {
            let tile = discard_types[row] as usize;
            numeric_values[numeric_base + 2] =
                (4_u8.saturating_sub(remaining[row * 34 + tile])) as f32 / 4.0;
            numeric_values[numeric_base + 3] = defense[3] as f32 / 4.0;
        }
    }
    let shanten = Array2::from_shape_vec((rows, 4), shanten_values)
        .expect("analysis shape")
        .into_pyarray(py);
    let masks = PyArray1::from_vec(py, masks);
    let ukeire = PyArray1::from_vec(py, ukeire);
    let waits = PyArray1::from_vec(py, waits);
    let defense = Array2::from_shape_vec((rows, 5), defense_values)
        .expect("defense shape")
        .into_pyarray(py);
    let categorical = Array2::from_shape_vec((rows, 10), categorical_values)
        .expect("categorical shape")
        .into_pyarray(py);
    let numeric = Array2::from_shape_vec((rows, 8), numeric_values)
        .expect("numeric shape")
        .into_pyarray(py);
    for array in [
        shanten.as_any(),
        masks.as_any(),
        ukeire.as_any(),
        waits.as_any(),
        defense.as_any(),
        categorical.as_any(),
        numeric.as_any(),
    ] {
        array.call_method1("setflags", (false,))?;
    }
    Ok(FeatureAnalysis {
        shanten: shanten.unbind(),
        improving_type_mask: masks.unbind(),
        ukeire: ukeire.unbind(),
        wait_count: waits.unbind(),
        defense: defense.unbind(),
        categorical: categorical.unbind(),
        numeric: numeric.unbind(),
        row_count: rows,
        analysis_version: HAND_ANALYSIS_VERSION,
    })
}

#[pyfunction]
fn analyze_hands(
    py: Python<'_>,
    counts: PyReadonlyArray2<'_, u8>,
    open_melds: PyReadonlyArray1<'_, u8>,
) -> PyResult<HandAnalysis> {
    let shape = counts.shape();
    if shape.len() != 2 || shape[1] != 34 {
        return Err(PyValueError::new_err("counts must have shape uint8[N,34]"));
    }
    let rows = shape[0];
    if open_melds.shape() != [rows] {
        return Err(PyValueError::new_err("open_melds must have shape uint8[N]"));
    }
    let counts = counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("counts must be C-contiguous"))?;
    let open_melds = open_melds
        .as_slice()
        .map_err(|_| PyValueError::new_err("open_melds must be C-contiguous"))?;
    let mut hands = Vec::with_capacity(rows);
    for (row, (&melds, values)) in open_melds.iter().zip(counts.chunks_exact(34)).enumerate() {
        if melds > 4 || values.iter().any(|&count| count > 4) {
            return Err(PyValueError::new_err(format!(
                "invalid count or open_melds in row {row}"
            )));
        }
        let total = values
            .iter()
            .map(|&value| usize::from(value))
            .sum::<usize>()
            + 3 * usize::from(melds);
        if total != 13 && total != 14 {
            return Err(PyValueError::new_err(format!(
                "row {row} represents {total} tiles; expected 13 or 14"
            )));
        }
        let mut hand = [0_u8; 34];
        hand.copy_from_slice(values);
        hands.push((hand, melds));
    }

    let values = py.detach(|| {
        hands
            .iter()
            .map(|(hand, melds)| {
                let value = shanten::calculate(hand, *melds);
                let mut mask = 0_u64;
                let total = hand.iter().map(|&value| usize::from(value)).sum::<usize>()
                    + 3 * usize::from(*melds);
                if total == 13 && value.overall != SHANTEN_UNAVAILABLE {
                    for tile in 0..34 {
                        if hand[tile] >= 4 {
                            continue;
                        }
                        let mut next = *hand;
                        next[tile] += 1;
                        if shanten::calculate(&next, *melds).overall < value.overall {
                            mask |= 1_u64 << tile;
                        }
                    }
                }
                (
                    [
                        value.overall,
                        value.standard,
                        value.seven_pairs,
                        value.thirteen_orphans,
                    ],
                    mask,
                )
            })
            .collect::<Vec<_>>()
    });
    let shanten_values = values
        .iter()
        .flat_map(|(family, _)| *family)
        .collect::<Vec<_>>();
    let mask_values = values.into_iter().map(|(_, mask)| mask).collect::<Vec<_>>();
    let shanten = Array2::from_shape_vec((rows, 4), shanten_values)
        .expect("analysis shape")
        .into_pyarray(py);
    let masks = PyArray1::from_vec(py, mask_values);
    shanten.call_method1("setflags", (false,))?;
    masks.call_method1("setflags", (false,))?;
    Ok(HandAnalysis {
        shanten: shanten.unbind(),
        improving_type_mask: masks.unbind(),
        row_count: rows,
        analysis_version: HAND_ANALYSIS_VERSION,
    })
}
