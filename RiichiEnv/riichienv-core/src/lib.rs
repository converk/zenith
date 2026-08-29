pub mod agari;
pub mod errors;
pub mod hand_evaluator;
pub mod score;
mod tests;
pub mod types;
pub mod yaku;

pub mod action;
pub mod observation;
pub mod parser;
pub mod replay;
pub mod rule;
pub mod shanten;
pub mod state;
#[cfg(feature = "python")]
mod yaku_checker;
pub mod offense_analysis;

pub use hand_evaluator::check_riichi_candidates;

/// Bump whenever replay decisions or their emitted observation semantics change.
pub const REPLAY_SEMANTICS_VERSION: u32 = 1;
