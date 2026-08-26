struct TableStateMachine {
    /// 每个座位最近一次 prepare_decisions 登记的合法动作请求。
    pending_requests: [Option<ActionRequest>; NUM_PLAYERS],
}

#[derive(Clone)]
struct ActionRequest {
    /// The exact MJAI strings supplied by RiichiEnv, indexed by the fixed
    /// 241-action protocol.  Keeping the original string avoids a parse /
    /// serialize round-trip before `Observation.select_action_from_mjai`.
    possible_actions: [Option<String>; NUM_ACTIONS],
    /// 每个固定 action id 对应调用方合法动作列表中的首个下标。
    source_indices: [Option<usize>; NUM_ACTIONS],
}

impl TableStateMachine {
    fn new() -> Self {
        Self {
            pending_requests: std::array::from_fn(|_| None),
        }
    }

    fn apply_player_events(
        &mut self,
        player_index: u8,
        _events: Vec<MjaiEvent>,
    ) -> Result<(), String> {
        if player_index >= NUM_PLAYERS as u8 {
            return Err("player_index must be in 0..4".to_owned());
        }
        // 事件到达即作废该座位上一次登记的合法动作请求(旧 history 已删除,
        // 本方法仅保留请求生命周期职责)。
        self.pending_requests[player_index as usize] = None;
        Ok(())
    }

    fn set_legal_action_jsons(
        &mut self,
        player_index: u8,
        action_jsons: Vec<String>,
    ) -> Result<(), String> {
        if player_index >= NUM_PLAYERS as u8 {
            return Err("player_index must be in 0..4".to_owned());
        }
        let mut possible_actions: [Option<String>; NUM_ACTIONS] = std::array::from_fn(|_| None);
        let mut source_indices: [Option<usize>; NUM_ACTIONS] = std::array::from_fn(|_| None);
        for (source_index, action_json) in action_jsons.into_iter().enumerate() {
            let action: Value = serde_json::from_str(&action_json)
                .map_err(|error| format!("invalid legal action JSON: {error}"))?;
            let action_id = action_id_from_mjai_value(&action)?;
            if let Some(existing) = &possible_actions[action_id] {
                let existing_value: Value = serde_json::from_str(existing)
                    .map_err(|error| format!("stored legal action JSON became invalid: {error}"))?;
                if existing_value != action {
                    return Err(format!(
                        "distinct RiichiEnv actions map to the same fixed action id {action_id}"
                    ));
                }
                continue;
            }
            possible_actions[action_id] = Some(action_json);
            source_indices[action_id] = Some(source_index);
        }
        self.pending_requests[player_index as usize] = Some(ActionRequest {
            possible_actions,
            source_indices,
        });
        Ok(())
    }

    fn action_source_indices(&self, player_index: u8) -> Result<Vec<(usize, usize)>, String> {
        if player_index >= NUM_PLAYERS as u8 {
            return Err("player_index must be in 0..4".to_owned());
        }
        let Some(request) = &self.pending_requests[player_index as usize] else {
            return Ok(Vec::new());
        };
        Ok(request
            .source_indices
            .iter()
            .enumerate()
            .filter_map(|(action_id, source)| source.map(|index| (action_id, index)))
            .collect())
    }

    fn action_mask(&self, player_index: u8) -> [bool; NUM_ACTIONS] {
        let mut mask = [false; NUM_ACTIONS];
        if let Some(request) = &self.pending_requests[player_index as usize] {
            for (action_id, action) in request.possible_actions.iter().enumerate() {
                mask[action_id] = action.is_some();
            }
        }
        mask
    }

    fn requested_action_to_env(
        &self,
        player_index: u8,
        action_id: usize,
    ) -> Result<Option<String>, String> {
        if player_index >= NUM_PLAYERS as u8 {
            return Err("player_index must be in 0..4".to_owned());
        }
        if action_id >= NUM_ACTIONS {
            return Err(format!("action_id must be in 0..{NUM_ACTIONS}"));
        }

        let Some(request) = &self.pending_requests[player_index as usize] else {
            return Ok(None);
        };
        let Some(action_json) = &request.possible_actions[action_id] else {
            return Err(format!(
                "action_id {action_id} is not in this request's possible_actions"
            ));
        };

        Ok(Some(action_json.clone()))
    }
}
