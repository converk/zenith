//! Python entry point for the MJAI kyoku state-machine extension.
//!
//! The previous experimental vector environment is deliberately not part of
//! this crate any more.  Keeping this surface small makes the extension
//! reproducible for the RiichiEnv PPO integration.

mod analysis;
#[path = "MjaiKyokuStateMachine/mod.rs"]
mod mjai_kyoku_state_machine;
mod shanten;
mod shanten_table;

use pyo3::prelude::*;

use crate::mjai_kyoku_state_machine::MjaiKyokuStateMachineManager;

const HAND_ANALYSIS_VERSION: u32 = 4;
const SHANTEN_UNAVAILABLE: i8 = 127;

#[pymodule]
fn riichi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MjaiKyokuStateMachineManager>()?;
    analysis::register(m)?;
    m.add("ANALYSIS_VERSION", HAND_ANALYSIS_VERSION)?;
    Ok(())
}
