//! MJAI 事件同步与固定 241 动作空间的表级状态机(每张桌子一个容器)。
//!
//! 旧 per-player semantic-token 历史(事件→令牌前缀 + 快照→状态后缀)已删除:
//! V18 当前局面输入由 `prepare_current_state_batch`/`encode_query_batch` 独立装配。
//! 本模块仅保留:事件解析与边界同步(`apply_events_batch`)、合法动作登记与
//! 241 维掩码(`prepare_decisions`)、动作解码(`decode_actions`/`action_ids_with_source_indices`)。

use std::thread;

use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};
use serde::Deserialize;
use serde_json::Value;

include!("types.rs");
include!("table.rs");
include!("manager.rs");
include!("protocol.rs");
