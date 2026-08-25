//! Python entry point for the MJAI kyoku state-machine extension.
//!
//! The previous experimental vector environment is deliberately not part of
//! this crate any more.  Keeping this surface small makes the extension
//! reproducible for the RiichiEnv PPO integration.

mod analysis;
pub mod atomic_snapshot;
#[path = "MjaiKyokuStateMachine/mod.rs"]
mod mjai_kyoku_state_machine;
mod query_encoding;
/// 向听与"摸入后向听"算法接口;Python 侧进张/和牌张数统计复用同一算法。
pub mod shanten;
mod shanten_table;

use pyo3::prelude::*;

use crate::mjai_kyoku_state_machine::MjaiKyokuStateMachineManager;

const HAND_ANALYSIS_VERSION: u32 = 4;
const SHANTEN_UNAVAILABLE: i8 = 127;

#[pymodule]
fn riichi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MjaiKyokuStateMachineManager>()?;
    analysis::register(m)?;
    atomic_snapshot::register(m)?;
    query_encoding::register(m)?;
    m.add("ANALYSIS_VERSION", HAND_ANALYSIS_VERSION)?;
    Ok(())
}
