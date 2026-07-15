struct TableStateMachine {
    players: [PlayerKyokuStateMachine; NUM_PLAYERS],
    /// Monotonic cache epoch for each player history.  A new kyoku resets
    /// tokens, so a length alone is not sufficient to validate a rollout KV
    /// prefix.
    history_generations: [u64; NUM_PLAYERS],
    pending_requests: [Option<ActionRequest>; NUM_PLAYERS],
}

#[derive(Clone)]
struct ActionRequest {
    /// The exact MJAI strings supplied by RiichiEnv, indexed by the fixed
    /// 241-action protocol.  Keeping the original string avoids a parse /
    /// serialize round-trip before `Observation.select_action_from_mjai`.
    possible_actions: [Option<String>; NUM_ACTIONS],
}

impl TableStateMachine {
    fn new() -> Self {
        Self {
            players: std::array::from_fn(|seat| PlayerKyokuStateMachine::new(seat as u8)),
            history_generations: [0; NUM_PLAYERS],
            pending_requests: std::array::from_fn(|_| None),
        }
    }

    fn apply_player_events(
        &mut self,
        player_index: u8,
        events: Vec<MjaiEvent>,
    ) -> Result<(), String> {
        if player_index >= NUM_PLAYERS as u8 {
            return Err("player_index must be in 0..4".to_owned());
        }
        self.pending_requests[player_index as usize] = None;
        for event in events {
            if matches!(event, MjaiEvent::StartKyoku { .. }) {
                self.history_generations[player_index as usize] = self.history_generations[player_index as usize].wrapping_add(1);
            }
            self.players[player_index as usize].apply_player_event(&event)?;
        }
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
        for action_json in action_jsons {
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
        }
        self.pending_requests[player_index as usize] = Some(ActionRequest { possible_actions });
        Ok(())
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

#[cfg(test)]
impl TableStateMachine {
    fn set_legal_actions(&mut self, player_index: u8, action_jsons: Vec<String>) -> Result<(), String> {
        self.set_legal_action_jsons(player_index, action_jsons)
    }
}
