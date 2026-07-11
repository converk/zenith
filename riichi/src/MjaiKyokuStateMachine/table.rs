#[derive(Clone, Copy)]
enum ReactionKind {
    Discard,
    Kakan,
}

#[derive(Clone, Copy)]
enum DecisionWindow {
    None,
    SelfTurn {
        actor: u8,
        drawn_tile: Option<MjaiTile>,
    },
    Reaction {
        target: u8,
        pai: MjaiTile,
        kind: ReactionKind,
    },
}

struct TableStateMachine {
    mode: GameMode,
    reveal_opponent_initial_hands: bool,
    game_context: GameContext,
    players: [PlayerKyokuStateMachine; NUM_PLAYERS],
    kyoku_active: bool,
    draw_count: u8,
    step: i64,
    discard_counts: [u8; NUM_PLAYERS],
    has_open_call: bool,
    decision_window: DecisionWindow,
    pending_requests: [Option<ActionRequest>; NUM_PLAYERS],
}

#[derive(Default)]
struct GameContext {
    started: bool,
    // `start_game` metadata belongs to the table, not to a kyoku model input.
    #[allow(dead_code)]
    names: [String; NUM_PLAYERS],
    #[allow(dead_code)]
    seed: Option<(u64, u64)>,
}

#[derive(Clone)]
struct ActionRequest {
    request_id: i64,
    possible_actions: Vec<Option<String>>,
}

#[derive(Clone, Deserialize)]
struct RiichiEnvRequestAction {
    request_id: i64,
    possible_actions: Vec<Value>,
}

impl TableStateMachine {
    fn new(mode: GameMode, reveal_opponent_initial_hands: bool) -> Self {
        Self {
            mode,
            reveal_opponent_initial_hands,
            game_context: GameContext::default(),
            players: std::array::from_fn(|seat| PlayerKyokuStateMachine::new(seat as u8)),
            kyoku_active: false,
            draw_count: 0,
            step: 0,
            discard_counts: [0; NUM_PLAYERS],
            has_open_call: false,
            decision_window: DecisionWindow::None,
            pending_requests: std::array::from_fn(|_| None),
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.mode, self.reveal_opponent_initial_hands);
    }

    fn apply_event(&mut self, event: MjaiEvent) -> Result<(), String> {
        self.pending_requests.fill(None);
        match &event {
            MjaiEvent::StartGame { names, seed } => {
                self.game_context = GameContext {
                    started: true,
                    names: names.clone(),
                    seed: *seed,
                };
                self.kyoku_active = false;
                return Ok(());
            }
            MjaiEvent::StartKyoku { .. } => {
                self.game_context.started = true;
                self.kyoku_active = true;
                self.draw_count = 0;
                self.step = 0;
                self.discard_counts.fill(0);
                self.has_open_call = false;
                self.decision_window = DecisionWindow::None;
                for player in &mut self.players {
                    player.start_kyoku(&event, self.mode, self.reveal_opponent_initial_hands)?;
                }
                return Ok(());
            }
            MjaiEvent::EndKyoku => {
                self.kyoku_active = false;
                self.decision_window = DecisionWindow::None;
                return Ok(());
            }
            MjaiEvent::EndGame => {
                self.kyoku_active = false;
                self.game_context.started = false;
                self.decision_window = DecisionWindow::None;
                return Ok(());
            }
            _ => {}
        }

        if !self.game_context.started || !self.kyoku_active {
            return Err("received a kyoku event before start_kyoku".to_owned());
        }

        let double_reach = match &event {
            MjaiEvent::Reach { actor } => {
                !self.has_open_call && self.discard_counts[*actor as usize] == 0
            }
            _ => false,
        };
        for player in &mut self.players {
            player.apply_event(&event, self.step, double_reach)?;
        }
        self.advance_clock(&event);
        self.update_decision_window(&event);
        Ok(())
    }

    fn apply_request(
        &mut self,
        player_index: u8,
        request: RiichiEnvRequestAction,
    ) -> Result<(), String> {
        if player_index >= NUM_PLAYERS as u8 {
            return Err("player_index must be in 0..4".to_owned());
        }

        let mut possible_actions = vec![None; NUM_ACTIONS];
        for action in request.possible_actions {
            let action_id = action_id_from_mjai_value(&action)?;
            let serialized = serialize_mjai(action)?;
            possible_actions[action_id] = Some(serialized);
        }

        self.pending_requests[player_index as usize] = Some(ActionRequest {
            request_id: request.request_id,
            possible_actions,
        });
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

        let mut action = serde_json::from_str::<Value>(action_json)
            .map_err(|error| format!("stored possible action is invalid JSON: {error}"))?;
        prepare_riichi_env_response(&mut action, request.request_id, player_index, self)?;
        serialize_mjai(action).map(Some)
    }

    fn advance_clock(&mut self, event: &MjaiEvent) {
        match event {
            MjaiEvent::Tsumo { .. } => {
                self.draw_count = self.draw_count.saturating_add(1);
                self.step = i64::from((self.draw_count / NUM_PLAYERS as u8).min(17));
            }
            MjaiEvent::Dahai { actor, .. } => {
                self.discard_counts[*actor as usize] =
                    self.discard_counts[*actor as usize].saturating_add(1);
            }
            MjaiEvent::Chi { .. } | MjaiEvent::Pon { .. } | MjaiEvent::Daiminkan { .. } => {
                self.has_open_call = true;
            }
            _ => {}
        }
    }

    fn update_decision_window(&mut self, event: &MjaiEvent) {
        self.decision_window = match event {
            MjaiEvent::Tsumo { actor, pai } => DecisionWindow::SelfTurn {
                actor: *actor,
                drawn_tile: Some(*pai),
            },
            MjaiEvent::Dahai { actor, pai, .. } => DecisionWindow::Reaction {
                target: *actor,
                pai: *pai,
                kind: ReactionKind::Discard,
            },
            MjaiEvent::Chi { actor, .. } | MjaiEvent::Pon { actor, .. } => {
                DecisionWindow::SelfTurn {
                    actor: *actor,
                    drawn_tile: None,
                }
            }
            MjaiEvent::Kakan { actor, pai, .. } => DecisionWindow::Reaction {
                target: *actor,
                pai: *pai,
                kind: ReactionKind::Kakan,
            },
            MjaiEvent::Daiminkan { .. }
            | MjaiEvent::Ankan { .. }
            | MjaiEvent::Hora { .. }
            | MjaiEvent::Ryukyoku { .. }
            | MjaiEvent::EndKyoku
            | MjaiEvent::EndGame => DecisionWindow::None,
            MjaiEvent::None
            | MjaiEvent::StartGame { .. }
            | MjaiEvent::StartKyoku { .. }
            | MjaiEvent::Dora { .. }
            | MjaiEvent::Reach { .. }
            | MjaiEvent::ReachAccepted { .. } => self.decision_window,
        };
    }

    fn player_can_act(&self, player_index: u8) -> bool {
        match self.decision_window {
            DecisionWindow::SelfTurn { actor, .. } => actor == player_index,
            DecisionWindow::Reaction { target, .. } => target != player_index,
            DecisionWindow::None => false,
        }
    }

    fn self_turn_for(&self, player_index: u8) -> Result<Option<MjaiTile>, String> {
        match self.decision_window {
            DecisionWindow::SelfTurn { actor, drawn_tile } if actor == player_index => {
                Ok(drawn_tile)
            }
            _ => Err("action requires this player to be in a self-turn decision window".to_owned()),
        }
    }

    fn drawn_self_turn_for(&self, player_index: u8) -> Result<MjaiTile, String> {
        self.self_turn_for(player_index)?.ok_or_else(|| {
            "ankan, kakan, reach, hora, and ryukyoku require a draw decision window".to_owned()
        })
    }

    fn reaction_for(&self, player_index: u8) -> Result<(u8, MjaiTile, ReactionKind), String> {
        match self.decision_window {
            DecisionWindow::Reaction { target, pai, kind } if target != player_index => {
                Ok((target, pai, kind))
            }
            _ => Err("action requires this player to be in a reaction decision window".to_owned()),
        }
    }

    fn action_to_mjai(&self, player_index: u8, action_id: usize) -> Result<String, String> {
        if player_index >= NUM_PLAYERS as u8 {
            return Err("player_index must be in 0..4".to_owned());
        }
        if action_id >= NUM_ACTIONS {
            return Err(format!("action_id must be in 0..{NUM_ACTIONS}"));
        }
        if !self.player_can_act(player_index) {
            return Err("this player has no active MJAI decision window".to_owned());
        }

        let player = &self.players[player_index as usize];
        match action_id {
            0 => {
                self.reaction_for(player_index)?;
                serialize_mjai(json!({"type": "none"}))
            }
            1..=74 => {
                let drawn_tile = self.self_turn_for(player_index)?;
                let tile = action_tile((action_id - 1) / 2)?;
                let tsumogiri = (action_id - 1) % 2 == 1;
                if !player.holds_tiles(&[tile]) {
                    return Err(format!("cannot discard absent tile {}", tile_name(tile)?));
                }
                if tsumogiri && drawn_tile != Some(tile) {
                    return Err("tsumogiri action does not match this turn's drawn tile".to_owned());
                }
                serialize_mjai(json!({
                    "type": "dahai",
                    "actor": player_index,
                    "pai": tile_name(tile)?,
                    "tsumogiri": tsumogiri,
                }))
            }
            75 => {
                self.drawn_self_turn_for(player_index)?;
                serialize_mjai(json!({"type": "reach", "actor": player_index}))
            }
            76..=132 => {
                let (target, pai, kind) = self.reaction_for(player_index)?;
                if !matches!(kind, ReactionKind::Discard)
                    || player_index != (target + 1) % NUM_PLAYERS as u8
                {
                    return Err("chi is only available on the previous player's discard".to_owned());
                }
                let consumed = chi_consumed_pair(action_id - 76)?;
                if !is_sequence(pai, consumed[0], consumed[1]) {
                    return Err(
                        "chi consumed pair does not form a sequence with the current discard"
                            .to_owned(),
                    );
                }
                if !player.holds_tiles(&consumed) {
                    return Err("chi consumed tiles are absent from the player's hand".to_owned());
                }
                serialize_mjai(json!({
                    "type": "chi",
                    "actor": player_index,
                    "target": target,
                    "pai": tile_name(pai)?,
                    "consumed": [tile_name(consumed[0])?, tile_name(consumed[1])?],
                }))
            }
            133..=169 => {
                let (target, pai, kind) = self.reaction_for(player_index)?;
                if !matches!(kind, ReactionKind::Discard) {
                    return Err("pon is only available on a discard reaction window".to_owned());
                }
                let consumed = pon_consumed_pair(action_id - 133)?;
                if pai.deaka() != consumed[0].deaka() || pai.deaka() != consumed[1].deaka() {
                    return Err("pon tile kind does not match the current discard".to_owned());
                }
                if !player.holds_tiles(&consumed) {
                    return Err("pon consumed tiles are absent from the player's hand".to_owned());
                }
                serialize_mjai(json!({
                    "type": "pon",
                    "actor": player_index,
                    "target": target,
                    "pai": tile_name(pai)?,
                    "consumed": [tile_name(consumed[0])?, tile_name(consumed[1])?],
                }))
            }
            170 => {
                let (target, pai, kind) = self.reaction_for(player_index)?;
                if !matches!(kind, ReactionKind::Discard) {
                    return Err("daiminkan is only available on a discard reaction window".to_owned());
                }
                let consumed = player.tiles_of_kind(pai.deaka());
                let consumed: [MjaiTile; 3] = consumed.try_into().map_err(|_| {
                    "daiminkan requires exactly three matching tiles in the player's hand".to_owned()
                })?;
                serialize_mjai(json!({
                    "type": "daiminkan",
                    "actor": player_index,
                    "target": target,
                    "pai": tile_name(pai)?,
                    "consumed": [
                        tile_name(consumed[0])?,
                        tile_name(consumed[1])?,
                        tile_name(consumed[2])?,
                    ],
                }))
            }
            171..=204 => {
                self.drawn_self_turn_for(player_index)?;
                let tile34 = tile34(action_id - 171)?;
                let consumed = player.tiles_of_kind(tile34);
                let consumed: [MjaiTile; 4] = consumed.try_into().map_err(|_| {
                    "ankan requires exactly four matching tiles in the player's hand".to_owned()
                })?;
                serialize_mjai(json!({
                    "type": "ankan",
                    "actor": player_index,
                    "consumed": [
                        tile_name(consumed[0])?,
                        tile_name(consumed[1])?,
                        tile_name(consumed[2])?,
                        tile_name(consumed[3])?,
                    ],
                }))
            }
            205..=238 => {
                self.drawn_self_turn_for(player_index)?;
                let tile34 = tile34(action_id - 205)?;
                let consumed = player.pon_tiles(tile34).ok_or_else(|| {
                    "kakan requires an existing matching pon".to_owned()
                })?;
                let added_tiles = player.tiles_of_kind(tile34);
                let added_tile = match added_tiles.as_slice() {
                    [tile] => *tile,
                    _ => {
                        return Err(
                            "kakan requires exactly one matching tile in the player's hand"
                                .to_owned(),
                        )
                    }
                };
                serialize_mjai(json!({
                    "type": "kakan",
                    "actor": player_index,
                    "pai": tile_name(added_tile)?,
                    "consumed": [
                        tile_name(consumed[0])?,
                        tile_name(consumed[1])?,
                        tile_name(consumed[2])?,
                    ],
                }))
            }
            239 => match self.decision_window {
                DecisionWindow::SelfTurn { actor, .. } if actor == player_index => {
                    self.drawn_self_turn_for(player_index)?;
                    serialize_mjai(json!({"type": "hora", "actor": player_index, "target": player_index}))
                }
                DecisionWindow::Reaction { target, .. } if target != player_index => {
                    serialize_mjai(json!({"type": "hora", "actor": player_index, "target": target}))
                }
                _ => Err("hora requires a self-turn or reaction decision window".to_owned()),
            },
            240 => {
                self.drawn_self_turn_for(player_index)?;
                serialize_mjai(json!({"type": "ryukyoku"}))
            }
            _ => unreachable!("action_id was checked against NUM_ACTIONS"),
        }
    }
}
