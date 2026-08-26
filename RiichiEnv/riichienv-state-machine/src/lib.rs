//! Python entry point for the MJAI kyoku state-machine extension.
//!
//! The previous experimental vector environment is deliberately not part of
//! this crate any more.  Keeping this surface small makes the extension
//! reproducible for the RiichiEnv PPO integration.

pub mod analysis;
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
/// 现行输入编码协议版本(单源;Python 侧 `encoding_protocol.ENCODING_PROTOCOL_VERSION` 从此镜像)。
const ENCODING_PROTOCOL_VERSION: u8 = 18;

/// 开放副露役牌番数的溢出桶:0..6 精确计数,超出截断到 6。
/// 当前编码器(open_meld_yakuhai_han/current_state)的活跃依赖;Rust/Python 镜像由
/// `test_v18_encoding_protocol.py::test_bucket_constants_mirror_rust` 交叉验证。
pub const OPEN_MELD_YAKUHAI_HAN_OVERFLOW_BUCKET: u8 = 6;
/// 已表示于副露的宝牌/赤牌番数溢出桶:0..8 精确计数,超出截断到 8。
/// 当前编码器(visible_meld_dora_aka_han/current_state)的活跃依赖;镜像交叉验证同上。
pub const VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET: u8 = 8;

#[pymodule]
fn riichi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MjaiKyokuStateMachineManager>()?;
    analysis::register(m)?;
    atomic_snapshot::register(m)?;
    query_encoding::register(m)?;
    m.add("ANALYSIS_VERSION", HAND_ANALYSIS_VERSION)?;
    m.add("ENCODING_PROTOCOL_VERSION", ENCODING_PROTOCOL_VERSION)?;
    m.add(
        "OPEN_MELD_YAKUHAI_HAN_OVERFLOW_BUCKET",
        OPEN_MELD_YAKUHAI_HAN_OVERFLOW_BUCKET,
    )?;
    m.add(
        "VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET",
        VISIBLE_MELD_DORA_AKA_OVERFLOW_BUCKET,
    )?;
    Ok(())
}
