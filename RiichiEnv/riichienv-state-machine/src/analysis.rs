use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::mjai_kyoku_state_machine::{NUM_ACTIONS, TILE_KINDS};
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
    module.add_class::<DefenseAnalysisV16>()?;
    module.add_class::<OffenseAnalysisV16>()?;
    module.add_function(wrap_pyfunction!(analyze_hands, module)?)?;
    module.add_function(wrap_pyfunction!(analyze_features, module)?)?;
    module.add_function(wrap_pyfunction!(analyze_defense_v16, module)?)?;
    module.add_function(wrap_pyfunction!(analyze_offense_v16, module)?)?;
    module.add_function(wrap_pyfunction!(public_opponent_summary, module)?)?;
    Ok(())
}

/// V16 进攻事实内核(O0–O3、O6、O8)。
///
/// 只评价「动作后形状」(暗牌计数 + 3×副露数 = 13);役/基础番(O4/O5)由 core 的
/// `riichienv.analyze_offense_v16` 按等待牌计算。向听复用本 crate 的既有
/// shanten 实现(生产路径)。
#[pyclass(name = "OffenseAnalysisV16", frozen)]
pub struct OffenseAnalysisV16 {
    shanten: Py<PyArray1<i8>>,
    effective_kinds: Py<PyArray1<u8>>,
    effective_remaining: Py<PyArray1<u16>>,
    wait_kinds: Py<PyArray1<u8>>,
    wait_mask: Py<PyArray1<u64>>,
    furiten: Py<PyArray1<u8>>,
    can_riichi: Py<PyArray1<u8>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OffenseRowV16 {
    pub shanten: i8,
    pub effective_kinds: u8,
    pub effective_remaining: u16,
    pub wait_kinds: u8,
    pub wait_mask: u64,
    pub furiten: u8,
    pub can_riichi: u8,
}

/// 计算单个 13 张动作后形状的 V16 进攻事实。
///
/// Python 批接口与融合编码器共用这一实现,避免两条热路径的语义漂移。
#[allow(clippy::too_many_arguments)]
pub(crate) fn offense_row_v16(
    counts: &[u8; TILE_KINDS],
    open_melds: u8,
    remaining: &[u8; TILE_KINDS],
    own_river: u64,
    missed_doujun: bool,
    missed_riichi: bool,
    riichi_declared: bool,
    score: i32,
) -> Result<OffenseRowV16, String> {
    let total: u16 =
        counts.iter().map(|&value| u16::from(value)).sum::<u16>() + 3 * u16::from(open_melds);
    if total != 13 {
        return Err(format!(
            "shape represents {total} tiles (counts + 3×melds); expected 13"
        ));
    }
    if counts.iter().any(|&value| value > 4) {
        return Err("shape contains a count above 4".to_string());
    }
    if own_river >> TILE_KINDS != 0 {
        return Err(format!("own river mask exceeds {TILE_KINDS} bits"));
    }

    let shanten_value = shanten::calculate(counts, open_melds).overall;
    let after_draws = shanten::calculate_after_draws(counts, open_melds);
    let mut improving = 0_u64;
    if shanten_value > 0 {
        for tile in 0..TILE_KINDS {
            if counts[tile] >= 4 {
                continue;
            }
            if after_draws[tile] < shanten_value {
                improving |= 1_u64 << tile;
            }
        }
    }
    let mut wait_mask = 0_u64;
    if shanten_value == 0 {
        for tile in 0..TILE_KINDS {
            if counts[tile] >= 4 {
                continue;
            }
            if after_draws[tile] < 0 {
                wait_mask |= 1_u64 << tile;
            }
        }
    }
    let furiten = if shanten_value != 0 || wait_mask == 0 {
        0
    } else if (0..TILE_KINDS)
        .any(|tile| wait_mask & (1_u64 << tile) != 0 && own_river & (1_u64 << tile) != 0)
    {
        2
    } else if missed_doujun || missed_riichi {
        3
    } else {
        1
    };
    let can_riichi = if riichi_declared {
        0
    } else if open_melds > 0 || score < 1000 || wait_mask == 0 {
        2
    } else {
        1
    };
    let effective_remaining = (0..TILE_KINDS)
        .filter(|&tile| improving & (1_u64 << tile) != 0)
        .map(|tile| u16::from(remaining[tile]))
        .sum::<u16>();

    Ok(OffenseRowV16 {
        shanten: shanten_value,
        effective_kinds: improving.count_ones() as u8,
        effective_remaining,
        wait_kinds: u8::try_from(wait_mask.count_ones()).unwrap_or(u8::MAX),
        wait_mask,
        furiten,
        can_riichi,
    })
}

#[pymethods]
impl OffenseAnalysisV16 {
    #[getter]
    fn shanten(&self, py: Python<'_>) -> Py<PyArray1<i8>> {
        self.shanten.clone_ref(py)
    }
    #[getter]
    fn effective_kinds(&self, py: Python<'_>) -> Py<PyArray1<u8>> {
        self.effective_kinds.clone_ref(py)
    }
    #[getter]
    fn effective_remaining(&self, py: Python<'_>) -> Py<PyArray1<u16>> {
        self.effective_remaining.clone_ref(py)
    }
    #[getter]
    fn wait_kinds(&self, py: Python<'_>) -> Py<PyArray1<u8>> {
        self.wait_kinds.clone_ref(py)
    }
    #[getter]
    fn wait_mask(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.wait_mask.clone_ref(py)
    }
    #[getter]
    fn furiten(&self, py: Python<'_>) -> Py<PyArray1<u8>> {
        self.furiten.clone_ref(py)
    }
    #[getter]
    fn can_riichi(&self, py: Python<'_>) -> Py<PyArray1<u8>> {
        self.can_riichi.clone_ref(py)
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn analyze_offense_v16(
    py: Python<'_>,
    shape_counts: PyReadonlyArray2<'_, u8>,
    open_melds: PyReadonlyArray1<'_, u8>,
    remaining: PyReadonlyArray2<'_, u8>,
    own_rivers: PyReadonlyArray1<'_, u64>,
    missed_doujun: PyReadonlyArray1<'_, bool>,
    missed_riichi: PyReadonlyArray1<'_, bool>,
    riichi_declared: PyReadonlyArray1<'_, bool>,
    scores: PyReadonlyArray1<'_, i32>,
) -> PyResult<OffenseAnalysisV16> {
    let shape = shape_counts.shape();
    if shape.len() != 2 || shape[1] != TILE_KINDS {
        return Err(PyValueError::new_err(format!(
            "shape_counts must have shape uint8[N,{TILE_KINDS}]"
        )));
    }
    let rows = shape[0];
    for (name, length) in [
        ("open_melds", open_melds.len()),
        ("own_rivers", own_rivers.len()),
        ("missed_doujun", missed_doujun.len()),
        ("missed_riichi", missed_riichi.len()),
        ("riichi_declared", riichi_declared.len()),
        ("scores", scores.len()),
    ] {
        if length != rows {
            return Err(PyValueError::new_err(format!(
                "{name} must have length {rows}, got {length}"
            )));
        }
    }
    if remaining.shape() != [rows, TILE_KINDS] {
        return Err(PyValueError::new_err(
            "remaining must have shape uint8[N,34]",
        ));
    }
    let shape_values = shape_counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("shape_counts must be contiguous"))?;
    let meld_values = open_melds
        .as_slice()
        .map_err(|_| PyValueError::new_err("open_melds must be contiguous"))?;
    let remaining_values = remaining
        .as_slice()
        .map_err(|_| PyValueError::new_err("remaining must be contiguous"))?;
    let river_values = own_rivers
        .as_slice()
        .map_err(|_| PyValueError::new_err("own_rivers must be contiguous"))?;
    let doujun_values = missed_doujun
        .as_slice()
        .map_err(|_| PyValueError::new_err("missed_doujun must be contiguous"))?;
    let riichi_missed_values = missed_riichi
        .as_slice()
        .map_err(|_| PyValueError::new_err("missed_riichi must be contiguous"))?;
    let riichi_values = riichi_declared
        .as_slice()
        .map_err(|_| PyValueError::new_err("riichi_declared must be contiguous"))?;
    let score_values = scores
        .as_slice()
        .map_err(|_| PyValueError::new_err("scores must be contiguous"))?;

    let mut out_shanten = vec![0_i8; rows];
    let mut out_kinds = vec![0_u8; rows];
    let mut out_remaining = vec![0_u16; rows];
    let mut out_waits = vec![0_u8; rows];
    let mut out_mask = vec![0_u64; rows];
    let mut out_furiten = vec![0_u8; rows];
    let mut out_riichi = vec![0_u8; rows];

    let analysis_result: Result<(), String> = py.detach(|| {
        for row in 0..rows {
            let start = row * TILE_KINDS;
            let mut counts = [0_u8; TILE_KINDS];
            counts.copy_from_slice(&shape_values[start..start + TILE_KINDS]);
            let remaining_row = &remaining_values[start..start + TILE_KINDS];
            let remaining_row: [u8; TILE_KINDS] =
                remaining_row.try_into().expect("remaining row width");
            let river = river_values[row];
            let value = offense_row_v16(
                &counts,
                meld_values[row],
                &remaining_row,
                river,
                doujun_values[row],
                riichi_missed_values[row],
                riichi_values[row],
                score_values[row],
            )
            .map_err(|error| format!("row {row} {error}"))?;

            out_shanten[row] = value.shanten;
            out_kinds[row] = value.effective_kinds;
            out_remaining[row] = value.effective_remaining;
            out_waits[row] = value.wait_kinds;
            out_mask[row] = value.wait_mask;
            out_furiten[row] = value.furiten;
            out_riichi[row] = value.can_riichi;
        }
        Ok(())
    });
    analysis_result.map_err(PyValueError::new_err)?;

    let shanten_array = PyArray1::from_vec(py, out_shanten);
    let kinds_array = PyArray1::from_vec(py, out_kinds);
    let remaining_array = PyArray1::from_vec(py, out_remaining);
    let waits_array = PyArray1::from_vec(py, out_waits);
    let mask_array = PyArray1::from_vec(py, out_mask);
    let furiten_array = PyArray1::from_vec(py, out_furiten);
    let riichi_array = PyArray1::from_vec(py, out_riichi);
    for array in [
        shanten_array.as_any(),
        kinds_array.as_any(),
        remaining_array.as_any(),
        waits_array.as_any(),
        mask_array.as_any(),
        furiten_array.as_any(),
        riichi_array.as_any(),
    ] {
        array.call_method1("setflags", (false,))?;
    }
    Ok(OffenseAnalysisV16 {
        shanten: shanten_array.unbind(),
        effective_kinds: kinds_array.unbind(),
        effective_remaining: remaining_array.unbind(),
        wait_kinds: waits_array.unbind(),
        wait_mask: mask_array.unbind(),
        furiten: furiten_array.unbind(),
        can_riichi: riichi_array.unbind(),
    })
}

/// V16 对手 7 项摘要:是否立直、立直巡目、副露数、是否门清、舍牌数、手切次数、
/// 摸切次数。输入为公开 MJAI 状态逐座位事实,输出每对手一行的归一化编码。
///
/// 立直巡目 N/A(-1)编码为 255;计数按契约边界截断(副露 0..4,其余 0..24)。
#[pyfunction]
pub fn public_opponent_summary(
    py: Python<'_>,
    declared: PyReadonlyArray1<'_, u8>,
    reach_turn: PyReadonlyArray1<'_, i16>,
    meld_count: PyReadonlyArray1<'_, u8>,
    river_count: PyReadonlyArray1<'_, u8>,
    tedashi_count: PyReadonlyArray1<'_, u8>,
    tsumogiri_count: PyReadonlyArray1<'_, u8>,
) -> PyResult<Py<PyArray2<u8>>> {
    let rows = declared.len();
    for (name, length) in [
        ("reach_turn", reach_turn.len()),
        ("meld_count", meld_count.len()),
        ("river_count", river_count.len()),
        ("tedashi_count", tedashi_count.len()),
        ("tsumogiri_count", tsumogiri_count.len()),
    ] {
        if length != rows {
            return Err(PyValueError::new_err(format!(
                "{name} must have length {rows}, got {length}"
            )));
        }
    }
    let declared_values = declared
        .as_slice()
        .map_err(|_| PyValueError::new_err("declared must be contiguous"))?;
    let reach_values = reach_turn
        .as_slice()
        .map_err(|_| PyValueError::new_err("reach_turn must be contiguous"))?;
    let meld_values = meld_count
        .as_slice()
        .map_err(|_| PyValueError::new_err("meld_count must be contiguous"))?;
    let river_values = river_count
        .as_slice()
        .map_err(|_| PyValueError::new_err("river_count must be contiguous"))?;
    let tedashi_values = tedashi_count
        .as_slice()
        .map_err(|_| PyValueError::new_err("tedashi_count must be contiguous"))?;
    let tsumogiri_values = tsumogiri_count
        .as_slice()
        .map_err(|_| PyValueError::new_err("tsumogiri_count must be contiguous"))?;

    let mut out = vec![0_u8; rows * 7];
    let summary: Result<(), String> = py.detach(|| {
        for row in 0..rows {
            let melds = meld_values[row].min(4);
            out[row * 7] = declared_values[row].min(1);
            out[row * 7 + 1] = if reach_values[row] < 0 {
                255
            } else {
                u8::try_from(reach_values[row]).unwrap_or(255).min(24)
            };
            out[row * 7 + 2] = melds;
            out[row * 7 + 3] = u8::from(melds == 0);
            out[row * 7 + 4] = river_values[row].min(24);
            out[row * 7 + 5] = tedashi_values[row].min(24);
            out[row * 7 + 6] = tsumogiri_values[row].min(24);
        }
        Ok(())
    });
    drop(summary);
    let array = Array2::from_shape_vec((rows, 7), out)
        .expect("opponent summary shape")
        .into_pyarray(py);
    array.call_method1("setflags", (false,))?;
    Ok(array.unbind())
}

#[pyclass(name = "DefenseAnalysisV16", frozen)]
pub struct DefenseAnalysisV16 {
    genbutsu: Py<PyArray2<u8>>,
    suji: Py<PyArray2<u8>>,
    stock: Py<PyArray2<u8>>,
    visible: Py<PyArray1<u8>>,
}

#[pymethods]
impl DefenseAnalysisV16 {
    #[getter]
    fn genbutsu(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.genbutsu.clone_ref(py)
    }
    #[getter]
    fn suji(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.suji.clone_ref(py)
    }
    #[getter]
    fn stock(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.stock.clone_ref(py)
    }
    #[getter]
    fn visible(&self, py: Python<'_>) -> Py<PyArray1<u8>> {
        self.visible.clone_ref(py)
    }
}

/// V16 防守事实内核(D0–D9)。
///
/// 编码值与 `model/encoding_protocol.py` 的 DEFENSE_SLOT_LABELS 下标一致:
/// GENBUTSU=0/NOT_GENBUTSU=1/N/A=2;SUJI=0/NOT_SUJI=1/N/A=2;库存 0..4(4+);
/// D9 公开出现数 0..4,N/A=5。
#[pyfunction]
pub fn analyze_defense_v16(
    py: Python<'_>,
    discard_types: PyReadonlyArray1<'_, i16>,
    river_masks: PyReadonlyArray2<'_, u64>,
    hand_counts: PyReadonlyArray2<'_, u8>,
    remaining: PyReadonlyArray2<'_, u8>,
) -> PyResult<DefenseAnalysisV16> {
    let rows = discard_types.len();
    if river_masks.shape() != [rows, 3] {
        return Err(PyValueError::new_err(
            "river_masks must have shape uint64[N,3]",
        ));
    }
    if hand_counts.shape() != [rows, TILE_KINDS] {
        return Err(PyValueError::new_err(
            "hand_counts must have shape uint8[N,34]",
        ));
    }
    if remaining.shape() != [rows, TILE_KINDS] {
        return Err(PyValueError::new_err(
            "remaining must have shape uint8[N,34]",
        ));
    }
    let discard_values = discard_types
        .as_slice()
        .map_err(|_| PyValueError::new_err("discard_types must be contiguous"))?;
    let river_values = river_masks
        .as_slice()
        .map_err(|_| PyValueError::new_err("river_masks must be contiguous"))?;
    let hand_values = hand_counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("hand_counts must be contiguous"))?;
    let remaining_values = remaining
        .as_slice()
        .map_err(|_| PyValueError::new_err("remaining must be contiguous"))?;

    let mut genbutsu = vec![0_u8; rows * 3];
    let mut suji = vec![0_u8; rows * 3];
    let mut stock = vec![0_u8; rows * 3];
    let mut visible = vec![0_u8; rows];

    let values = py.detach(|| {
        for row in 0..rows {
            if river_values[row * 3..row * 3 + 3]
                .iter()
                .any(|&mask| mask >> TILE_KINDS != 0)
            {
                return Err(format!("row {row} river mask exceeds {TILE_KINDS} bits"));
            }
            let tile = discard_values[row];
            if tile < -1 || tile >= TILE_KINDS as i16 {
                return Err(format!("row {row} discard type {tile} is out of range"));
            }
            let hand_row: [u8; TILE_KINDS] = hand_values[row * TILE_KINDS..(row + 1) * TILE_KINDS]
                .try_into()
                .expect("hand row width");
            let remaining_row: [u8; TILE_KINDS] = remaining_values
                [row * TILE_KINDS..(row + 1) * TILE_KINDS]
                .try_into()
                .expect("remaining row width");
            let rivers: [u64; 3] = river_values[row * 3..row * 3 + 3]
                .try_into()
                .expect("river row width");
            let (row_genbutsu, row_suji, row_stock, row_visible) =
                defense_row(tile, &rivers, &hand_row, &remaining_row);
            for opponent in 0..3 {
                genbutsu[row * 3 + opponent] = row_genbutsu[opponent];
                suji[row * 3 + opponent] = row_suji[opponent];
                stock[row * 3 + opponent] = row_stock[opponent];
            }
            visible[row] = row_visible;
        }
        Ok(())
    });
    drop(values);

    let genbutsu_array = Array2::from_shape_vec((rows, 3), genbutsu)
        .expect("defense genbutsu shape")
        .into_pyarray(py);
    let suji_array = Array2::from_shape_vec((rows, 3), suji)
        .expect("defense suji shape")
        .into_pyarray(py);
    let stock_array = Array2::from_shape_vec((rows, 3), stock)
        .expect("defense stock shape")
        .into_pyarray(py);
    let visible_array = PyArray1::from_vec(py, visible);
    for array in [
        genbutsu_array.as_any(),
        suji_array.as_any(),
        stock_array.as_any(),
        visible_array.as_any(),
    ] {
        array.call_method1("setflags", (false,))?;
    }
    Ok(DefenseAnalysisV16 {
        genbutsu: genbutsu_array.unbind(),
        suji: suji_array.unbind(),
        stock: stock_array.unbind(),
        visible: visible_array.unbind(),
    })
}

/// 单行防守事实(D0–D9),与 Python 侧 `encoding_protocol.py` 的编码下标一致。
pub(crate) fn defense_row(
    tile: i16,
    rivers: &[u64; 3],
    hand: &[u8; TILE_KINDS],
    remaining: &[u8; TILE_KINDS],
) -> ([u8; 3], [u8; 3], [u8; 3], u8) {
    let mut genbutsu = [2_u8; 3];
    let mut suji = [2_u8; 3];
    let mut stock = [0_u8; 3];
    for opponent in 0..3 {
        let river = rivers[opponent];
        let mut count = 0_u8;
        for kind in 0..TILE_KINDS {
            if hand[kind] > 0 && river & (1_u64 << kind) != 0 {
                count += 1;
            }
        }
        stock[opponent] = count.min(4);
    }
    if tile < 0 {
        return (genbutsu, suji, stock, 5);
    }
    let tile = tile as usize;
    let visible = (4_usize.saturating_sub(usize::from(remaining[tile]))).min(4) as u8;
    for opponent in 0..3 {
        let river = rivers[opponent];
        genbutsu[opponent] = u8::from(river & (1_u64 << tile) != 0);
        suji[opponent] = u8::from(suji_safe(tile, river));
    }
    (genbutsu, suji, stock, visible)
}

#[cfg(test)]
mod defense_tests {
    use super::*;

    #[test]
    fn test_genbutsu_suji_and_stock() {
        // 对手 0 河含 1m/4m/7m:4m 是现物,且两侧筋锚(1m、7m)齐全故成筋。
        let rivers = [1_u64 << 0 | 1_u64 << 3 | 1_u64 << 6, 0, 1_u64 << 8];
        let mut hand = [0_u8; TILE_KINDS];
        hand[2] = 1; // 3m
        hand[3] = 1; // 4m
        let remaining = [4_u8; TILE_KINDS];
        let (genbutsu, suji, stock, visible) = defense_row(3, &rivers, &hand, &remaining);
        assert_eq!(genbutsu, [1, 0, 0]);
        assert_eq!(suji, [1, 0, 0]);
        // 现物库存只数「手中牌种 ∩ 对手河」:手中 3m/4m,仅 4m 在对手 0 河内。
        assert_eq!(stock, [1, 0, 0]);
        assert_eq!(visible, 0);
    }

    #[test]
    fn test_non_discard_na_and_visible() {
        let rivers = [1_u64 << 3, 0, 0];
        let mut hand = [0_u8; TILE_KINDS];
        hand[3] = 2;
        let mut remaining = [4_u8; TILE_KINDS];
        remaining[5] = 1;
        let (genbutsu, suji, stock, visible) = defense_row(-1, &rivers, &hand, &remaining);
        assert_eq!(genbutsu, [2, 2, 2]);
        assert_eq!(suji, [2, 2, 2]);
        // 非打牌动作仍计算安全牌库存:手中 4m 是对手 0 的现物。
        assert_eq!(stock, [1, 0, 0]);
        assert_eq!(visible, 5);
        let (_g, _s, _k, visible_discard) = defense_row(5, &rivers, &hand, &remaining);
        assert_eq!(visible_discard, 3);
    }
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
    if shape.len() != 2 || shape[1] != TILE_KINDS {
        return Err(PyValueError::new_err(format!(
            "counts must have shape uint8[N,{TILE_KINDS}]"
        )));
    }
    let rows = shape[0];
    if action_ids.shape() != [rows]
        || open_melds.shape() != [rows]
        || remaining.shape() != [rows, TILE_KINDS]
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
        || sorted_ids.iter().any(|&id| usize::from(id) >= NUM_ACTIONS)
    {
        return Err(PyValueError::new_err(format!(
            "action ids must be unique and inside [0,{NUM_ACTIONS})"
        )));
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
        return Err(PyValueError::new_err(format!(
            "discard types must be N/A (-1) or inside [0,{TILE_KINDS})"
        )));
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
        .any(|&mask| mask >> TILE_KINDS != 0)
    {
        return Err(PyValueError::new_err(format!(
            "public tile masks may use only the low {TILE_KINDS} bits"
        )));
    }

    let values = py.detach(|| {
        (0..rows).map(|row| {
            let hand_slice = &counts[row * TILE_KINDS..(row + 1) * TILE_KINDS];
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
            let mut hand = [0_u8; TILE_KINDS];
            hand.copy_from_slice(hand_slice);
            let sh = shanten::calculate(&hand, melds);
            let mut improving = 0_u64;
            // Ukeire is defined for a 13-tile post-discard shape.  A 14-tile
            // row has no single well-defined improving draw until a discard
            // is chosen, so return the explicit empty/N/A mask.
            if total == 13 && sh.overall != SHANTEN_UNAVAILABLE {
                for tile in 0..TILE_KINDS {
                    if hand[tile] >= 4 { continue; }
                    let mut next = hand;
                    next[tile] += 1;
                    if shanten::calculate(&next, melds).overall < sh.overall {
                        improving |= 1_u64 << tile;
                    }
                }
            }
            let remain = &remaining[row * TILE_KINDS..(row + 1) * TILE_KINDS];
            let ukeire = (0..TILE_KINDS).filter(|&tile| improving & (1_u64 << tile) != 0)
                .map(|tile| u16::from(remain[tile])).sum::<u16>();
            let tile = discard_types[row];
            let mut defense = [0_u8; 5]; // genbutsu, suji, wall, honor-visible, passed
            if tile >= 0 {
                let tile = usize::try_from(tile).map_err(|_| format!("invalid discard type at row {row}"))?;
                if tile >= TILE_KINDS { return Err(format!("invalid discard type at row {row}")); }
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
                (4_u8.saturating_sub(remaining[row * TILE_KINDS + tile])) as f32 / 4.0;
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
    if shape.len() != 2 || shape[1] != TILE_KINDS {
        return Err(PyValueError::new_err(format!(
            "counts must have shape uint8[N,{TILE_KINDS}]"
        )));
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
    for (row, (&melds, values)) in open_melds
        .iter()
        .zip(counts.chunks_exact(TILE_KINDS))
        .enumerate()
    {
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
        let mut hand = [0_u8; TILE_KINDS];
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
                    for tile in 0..TILE_KINDS {
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
