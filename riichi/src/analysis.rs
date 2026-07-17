use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{shanten, HAND_ANALYSIS_VERSION, SHANTEN_UNAVAILABLE};

#[pyclass(name = "HandAnalysis", frozen)]
pub struct HandAnalysis {
    shanten: Py<PyArray2<i8>>,
    improving_type_mask: Py<PyArray1<u64>>,
    #[pyo3(get)]
    row_count: usize,
    #[pyo3(get)]
    analysis_version: u32,
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
    module.add_function(wrap_pyfunction!(analyze_hands, module)?)?;
    Ok(())
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
                if value.overall != SHANTEN_UNAVAILABLE {
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
