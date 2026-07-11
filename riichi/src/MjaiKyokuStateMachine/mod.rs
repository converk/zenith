//! Append-only MJAI-to-model state machines for one four-player kyoku per table.
//!
//! `TableStateMachine` owns the hanchan-level context for one environment and
//! broadcasts each MJAI event to four independent `PlayerKyokuStateMachine`s.
//! A player machine keeps its own visible hand state and an immutable-prefix
//! token vector for the future Transformer. The event schema mirrors
//! `libriichi::mjai::Event` without requiring the upstream crate as a build
//! dependency, because this repository currently builds with Rust 1.75.

use std::thread;

use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};
use serde::Deserialize;
use serde_json::{json, Value};

include!("types.rs");
include!("player.rs");
include!("table.rs");
include!("manager.rs");
include!("protocol.rs");

#[cfg(test)]
mod tests;
