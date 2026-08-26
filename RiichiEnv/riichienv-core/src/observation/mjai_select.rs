//! Shared helpers for mapping Mjai messages to a legal `Action`.
//!
//! Used by both `Observation::select_action_from_mjai` (4P) and
//! `Observation3P::select_action_from_mjai` (3P).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::action::{Action, ActionType};
use crate::parser::tid_to_mjai;

pub(crate) struct ParsedMjai {
    pub type_str: String,
    pub tile_str: String,
    pub tsumogiri: Option<bool>,
    pub consumed: Option<Vec<String>>,
}

pub(crate) fn parse_mjai_message(mjai_data: &Bound<'_, PyAny>) -> Option<ParsedMjai> {
    if let Ok(s) = mjai_data.extract::<String>() {
        let v: serde_json::Value = serde_json::from_str(&s).ok()?;
        let type_str = v["type"].as_str()?.to_string();
        let tile_str = v["pai"].as_str().unwrap_or("").to_string();
        let tsumogiri = v.get("tsumogiri").and_then(|x| x.as_bool());
        let consumed = v.get("consumed").and_then(|x| x.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|e| e.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        });
        Some(ParsedMjai {
            type_str,
            tile_str,
            tsumogiri,
            consumed,
        })
    } else if let Ok(dict) = mjai_data.cast::<PyDict>() {
        let type_str: String = dict
            .get_item("type")
            .ok()
            .flatten()
            .and_then(|x| x.extract::<String>().ok())
            .unwrap_or_default();
        let tile_str: String = dict
            .get_item("pai")
            .ok()
            .flatten()
            .or_else(|| dict.get_item("tile").ok().flatten())
            .and_then(|x| x.extract::<String>().ok())
            .unwrap_or_default();
        let tsumogiri = dict
            .get_item("tsumogiri")
            .ok()
            .flatten()
            .and_then(|x| x.extract::<bool>().ok());
        let consumed = dict
            .get_item("consumed")
            .ok()
            .flatten()
            .and_then(|x| x.extract::<Vec<String>>().ok());
        Some(ParsedMjai {
            type_str,
            tile_str,
            tsumogiri,
            consumed,
        })
    } else {
        None
    }
}

fn consumed_matches(action_consume: &[u8], expected: &[String]) -> bool {
    if action_consume.len() != expected.len() {
        return false;
    }
    let mut a: Vec<String> = action_consume.iter().map(|&t| tid_to_mjai(t)).collect();
    let mut b: Vec<String> = expected.to_vec();
    a.sort();
    b.sort();
    a == b
}

/// 兼容「Action.consume_tiles 含被鸣牌」表示：MJAI 协议的 chi/pon/daiminkan
/// consumed 只含手牌侧（2/2/3 张），而部分回放构造的合法动作把被鸣牌也放进
/// consume_tiles（3/3/4 张）。该函数在精确等长匹配失败后，去掉与 `pai`
/// 相同的一张再比较，仅当剩余张数恰为手牌侧数量时生效。
fn hand_only_consumed_matches(action: &Action, tile_str: &str, expected: &[String]) -> bool {
    if tile_str.is_empty() {
        return false;
    }
    let expected_hand = match action.action_type {
        ActionType::Chi | ActionType::Pon => 2,
        ActionType::Daiminkan => 3,
        _ => return false,
    };
    let mut tiles: Vec<String> = action.consume_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
    if tiles.len() != expected_hand + 1 {
        return false;
    }
    let Some(position) = tiles.iter().position(|tile| tile == tile_str) else {
        return false;
    };
    tiles.remove(position);
    let mut a = tiles;
    let mut b = expected.to_vec();
    a.sort();
    b.sort();
    a == b
}

/// Select a matching `Action` from a slice of legal actions for a parsed Mjai
/// message.
///
/// `three_player` controls whether 3P-only types (`kita`) are recognized; chi
/// is rejected when set.
pub(crate) fn select_action<'a>(
    legal_actions: &'a [Action],
    parsed: &ParsedMjai,
    drawn_tile: Option<u8>,
    three_player: bool,
) -> Option<&'a Action> {
    let atype = parsed.type_str.as_str();

    if atype == "hora" {
        return legal_actions
            .iter()
            .find(|a| matches!(a.action_type, ActionType::Tsumo | ActionType::Ron));
    }

    if atype == "none" {
        return legal_actions
            .iter()
            .find(|a| a.action_type == ActionType::Pass);
    }

    let target_type = match atype {
        "dahai" => Some(ActionType::Discard),
        "chi" if !three_player => Some(ActionType::Chi),
        "pon" => Some(ActionType::Pon),
        "kakan" => Some(ActionType::Kakan),
        "daiminkan" => Some(ActionType::Daiminkan),
        "ankan" => Some(ActionType::Ankan),
        "kita" if three_player => Some(ActionType::Kita),
        "reach" => Some(ActionType::Riichi),
        "ryukyoku" => Some(ActionType::KyushuKyuhai),
        _ => None,
    };

    let tt = target_type?;

    // Special-case Discard: filter by mjai pai (or any Discard if pai is
    // omitted) then disambiguate via tsumogiri.
    //
    // NOTE: An mjai `dahai` message without a `pai` field is malformed per
    // the protocol, but we still return a non-empty Action (the first
    // legal Discard) instead of `None` to preserve backward compatibility
    // with the previous implementation; bailing out here would silently
    // break callers that rely on the old lenient behavior.
    if tt == ActionType::Discard {
        let candidates: Vec<&Action> = legal_actions
            .iter()
            .filter(|a| {
                a.action_type == ActionType::Discard
                    && (parsed.tile_str.is_empty()
                        || a.tile.is_some_and(|t| tid_to_mjai(t) == parsed.tile_str))
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        if let (Some(tsumogiri), Some(drawn)) = (parsed.tsumogiri, drawn_tile) {
            let preferred = candidates.iter().find(|a| {
                let is_drawn = a.tile == Some(drawn);
                if tsumogiri { is_drawn } else { !is_drawn }
            });
            if let Some(a) = preferred {
                return Some(*a);
            }
        }

        return Some(candidates[0]);
    }

    legal_actions.iter().find(|a| {
        if a.action_type != tt {
            return false;
        }

        if let Some(consumed) = parsed.consumed.as_ref() {
            let exact = consumed_matches(&a.consume_tiles, consumed);
            let fallback = !exact
                && hand_only_consumed_matches(a, &parsed.tile_str, consumed);
            if !exact && !fallback {
                return false;
            }
            // If pai is also given, double-check tile match for actions that
            // carry a meaningful tile (chi/pon/daiminkan/kakan).
            if !parsed.tile_str.is_empty()
                && matches!(
                    tt,
                    ActionType::Chi | ActionType::Pon | ActionType::Daiminkan | ActionType::Kakan
                )
            {
                if let Some(t) = a.tile {
                    if tid_to_mjai(t) != parsed.tile_str {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        }

        // No consumed field: fall back to pai-based match.
        if !parsed.tile_str.is_empty() {
            if let Some(t) = a.tile {
                return tid_to_mjai(t) == parsed.tile_str;
            }
            return false;
        }
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed_pon(consumed: &[&str], pai: &str) -> ParsedMjai {
        ParsedMjai {
            type_str: "pon".to_string(),
            tile_str: pai.to_string(),
            tsumogiri: None,
            consumed: Some(consumed.iter().map(|value| value.to_string()).collect()),
        }
    }

    fn parsed_chi(consumed: &[&str], pai: &str) -> ParsedMjai {
        ParsedMjai {
            type_str: "chi".to_string(),
            tile_str: pai.to_string(),
            tsumogiri: None,
            consumed: Some(consumed.iter().map(|value| value.to_string()).collect()),
        }
    }

    #[test]
    fn full_consume_pon_accepts_canonical_hand_only_consumed() {
        // 回放构造：consume_tiles 含被鸣牌（3 张 E），MJAI 消息只带手牌侧 2 张。
        let actions = [Action::new(ActionType::Pon, Some(108), vec![108, 108, 108], None)];
        let parsed = parsed_pon(&["E", "E"], "E");
        let selected = select_action(&actions, &parsed, None, false);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().consume_tiles, vec![108, 108, 108]);
    }

    #[test]
    fn full_consume_pon_keeps_full_form_exact_match() {
        // 原始 3 张形式仍走精确匹配（向后兼容）。
        let actions = [Action::new(ActionType::Pon, Some(108), vec![108, 108, 108], None)];
        let parsed = parsed_pon(&["E", "E", "E"], "E");
        let selected = select_action(&actions, &parsed, None, false);
        assert!(selected.is_some());
    }

    #[test]
    fn hand_only_consume_pon_still_matches_canonical() {
        // env 表示：consume_tiles 只含手牌侧 2 张，匹配不变。
        let actions = [Action::new(ActionType::Pon, Some(108), vec![108, 109], None)];
        let parsed = parsed_pon(&["E", "E"], "E");
        let selected = select_action(&actions, &parsed, None, false);
        assert!(selected.is_some());
    }

    #[test]
    fn full_consume_pon_wrong_pai_rejected() {
        let actions = [Action::new(ActionType::Pon, Some(108), vec![108, 108, 108], None)];
        let parsed = parsed_pon(&["E", "E"], "S");
        let selected = select_action(&actions, &parsed, None, false);
        assert!(selected.is_none());
    }

    #[test]
    fn full_consume_chi_accepts_canonical_hand_only_consumed() {
        // 红五被鸣：consume_tiles=[5sr,6s,7s]，canonical consumed=["6s","7s"]，pai="5sr"。
        let actions = [Action::new(ActionType::Chi, Some(88), vec![88, 92, 96], None)];
        let parsed = parsed_chi(&["6s", "7s"], "5sr");
        let selected = select_action(&actions, &parsed, None, false);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().consume_tiles, vec![88, 92, 96]);
    }

    #[test]
    fn full_consume_chi_keeps_full_form_exact_match() {
        let actions = [Action::new(ActionType::Chi, Some(88), vec![88, 92, 96], None)];
        let parsed = parsed_chi(&["5sr", "6s", "7s"], "5sr");
        let selected = select_action(&actions, &parsed, None, false);
        assert!(selected.is_some());
    }

    #[test]
    fn full_consume_chi_red_disambiguates() {
        // 手牌 [5mr,5m,5m]（16,17,18），被鸣普通 5m（19）：canonical ["5m","5m"] 只匹配非红动作。
        let actions = [
            Action::new(ActionType::Pon, Some(19), vec![16, 17, 19], None),
            Action::new(ActionType::Pon, Some(19), vec![17, 18, 19], None),
        ];
        let parsed = parsed_pon(&["5m", "5m"], "5m");
        let selected = select_action(&actions, &parsed, None, false);
        assert!(selected.is_some());
        assert!(!selected.unwrap().consume_tiles.contains(&16));
    }
}
