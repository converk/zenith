//! Python entry point for the MJAI kyoku state-machine extension.
//!
//! The previous experimental vector environment is deliberately not part of
//! this crate any more.  Keeping this surface small makes the extension
//! reproducible for the RiichiEnv PPO integration.

#[path = "MjaiKyokuStateMachine/mod.rs"]
mod mjai_kyoku_state_machine;

use pyo3::prelude::*;

use crate::mjai_kyoku_state_machine::MjaiKyokuStateMachineManager;

#[pymodule]
fn riichi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MjaiKyokuStateMachineManager>()?;
    Ok(())
}
