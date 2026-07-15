struct PlayerKyokuStateMachine {
    absolute_seat: u8,
    tokens: Vec<Token>,
}

impl PlayerKyokuStateMachine {
    fn new(absolute_seat: u8) -> Self {
        Self {
            absolute_seat,
            tokens: Vec::new(),
        }
    }

    fn apply_player_event(
        &mut self,
        event: &MjaiEvent,
        reveal_opponent_initial_hands: bool,
    ) -> Result<(), String> {
        match event {
            MjaiEvent::StartGame { id, .. } => {
                if let Some(id) = id {
                    if *id >= NUM_PLAYERS as u8 {
                        return Err("start_game.id must be in 0..4".to_owned());
                    }
                    self.absolute_seat = *id;
                }
                Ok(())
            }
            MjaiEvent::StartKyoku { .. } => {
                self.start_kyoku(event, reveal_opponent_initial_hands)
            }
            MjaiEvent::EndKyoku | MjaiEvent::EndGame => Ok(()),
            _ => {
                if self.tokens.is_empty() {
                    return Ok(());
                }
                self.apply_event(event, 0)?;
                Ok(())
            }
        }
    }

    #[cfg(test)]
    fn tokens_with_snapshot(&self, snapshot: &DecisionSnapshot) -> Result<Vec<Token>, String> {
        if snapshot.player_id >= NUM_PLAYERS as u8 {
            return Err("snapshot.player_id must be in 0..4".to_owned());
        }
        if snapshot.oya >= NUM_PLAYERS as u8 {
            return Err("snapshot.oya must be in 0..4".to_owned());
        }
        let mut tokens = self.tokens.clone();
        tokens.extend(self.snapshot_tokens(snapshot)?);
        Ok(tokens)
    }

    fn snapshot_tokens(&self, snapshot: &DecisionSnapshot) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        let relative = |absolute_actor: u8| -> Result<i64, String> {
            if absolute_actor >= NUM_PLAYERS as u8 {
                return Err("snapshot actor must be in 0..4".to_owned());
            }
            Ok(self.relative(absolute_actor))
        };

        tokens.push(token(
            TYPE_STATE_SNAPSHOT_BEGIN,
            ACTOR_SELF,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        tokens.push(token(
            TYPE_STATE_SELF_ID,
            ACTOR_SELF,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(u32::from(snapshot.player_id)),
            FLAG_NONE,
        ));
        tokens.push(token(
            TYPE_STATE_OYA,
            relative(snapshot.oya)?,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        tokens.push(token(
            TYPE_STATE_JIKAZE,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(jikaze_for(snapshot.player_id, snapshot.oya)),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        tokens.push(token(
            TYPE_STATE_BAKAZE,
            ACTOR_NONE,
            ACTOR_NONE,
            protocol_tile(round_wind_tile(snapshot.round_wind)?),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        tokens.push(token(
            TYPE_STATE_KYOKU_INDEX,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(u32::from(snapshot.kyoku_index) + 1),
            FLAG_NONE,
        ));
        tokens.push(token(
            TYPE_STATE_HONBA,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(u32::from(snapshot.honba)),
            FLAG_NONE,
        ));
        tokens.push(token(
            TYPE_STATE_RIICHI_STICKS,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(snapshot.riichi_sticks),
            FLAG_NONE,
        ));
        for absolute_actor in 0..NUM_PLAYERS as u8 {
            let score = snapshot.scores[absolute_actor as usize].max(0) as u32;
            tokens.push(token(
                TYPE_STATE_SCORE,
                relative(absolute_actor)?,
                ACTOR_NONE,
                TILE_NONE,
                TILE_NONE,
                TILE_NONE,
                encode_value(score / 5_000),
                FLAG_NONE,
                ));
        }
        for (index, tile_name) in snapshot.dora_indicators.iter().enumerate() {
            let tile = snapshot_tile(tile_name)?;
            tokens.push(token(
                TYPE_STATE_DORA,
                ACTOR_NONE,
                ACTOR_NONE,
                protocol_tile(tile),
                TILE_NONE,
                TILE_NONE,
                encode_value(index as u32),
                FLAG_NONE,
                ));
        }

        let mut hand_counts = [0u8; 37];
        for tile_name in &snapshot.hand {
            let tile = snapshot_tile(tile_name)?;
            if tile.as_usize() >= 37 {
                return Err("snapshot hand cannot contain unknown tile".to_owned());
            }
            hand_counts[tile.as_usize()] = hand_counts[tile.as_usize()].saturating_add(1);
        }
        for (tile_id, count) in hand_counts.into_iter().enumerate() {
            if count > 0 {
                tokens.push(token(
                    TYPE_STATE_HAND,
                    ACTOR_SELF,
                    ACTOR_NONE,
                    tile_id as i64 + 1,
                    TILE_NONE,
                    TILE_NONE,
                    encode_value(u32::from(count)),
                    FLAG_NONE,
                        ));
            }
        }
        tokens.push(token(
            TYPE_STATE_DRAWN_TILE,
            ACTOR_SELF,
            ACTOR_NONE,
            snapshot
                .drawn_tile
                .as_deref()
                .map(snapshot_tile)
                .transpose()?
                .map(protocol_tile)
                .unwrap_or(TILE_NONE),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        for absolute_actor in 0..NUM_PLAYERS as u8 {
            tokens.push(token(
                TYPE_STATE_RIICHI_DECLARED,
                relative(absolute_actor)?,
                ACTOR_NONE,
                TILE_NONE,
                TILE_NONE,
                TILE_NONE,
                if snapshot.riichi_declared[absolute_actor as usize] {
                    encode_value(1)
                } else {
                    VALUE_NONE
                },
                FLAG_NONE,
                ));
        }
        tokens.push(token(
            TYPE_STATE_LAST_DISCARD,
            ACTOR_NONE,
            ACTOR_NONE,
            snapshot
                .last_discard
                .as_deref()
                .map(snapshot_tile)
                .transpose()?
                .map(protocol_tile)
                .unwrap_or(TILE_NONE),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        for absolute_actor in 0..NUM_PLAYERS as u8 {
            tokens.push(token(
                TYPE_STATE_LAST_TEDASHI,
                relative(absolute_actor)?,
                ACTOR_NONE,
                snapshot.last_tedashis[absolute_actor as usize]
                    .as_deref()
                    .map(snapshot_tile)
                    .transpose()?
                    .map(protocol_tile)
                    .unwrap_or(TILE_NONE),
                TILE_NONE,
                TILE_NONE,
                VALUE_NONE,
                FLAG_NONE,
                ));
        }
        tokens.push(token(
            TYPE_STATE_SNAPSHOT_END,
            ACTOR_SELF,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        Ok(tokens)
    }

    fn start_kyoku(
        &mut self,
        event: &MjaiEvent,
        reveal_opponent_initial_hands: bool,
    ) -> Result<(), String> {
        let MjaiEvent::StartKyoku {
            bakaze,
            dora_marker,
            kyoku,
            honba,
            kyotaku,
            oya,
            scores,
            tehais,
        } = event
        else {
            return Err("start_kyoku must receive MjaiEvent::StartKyoku".to_owned());
        };

        self.tokens.clear();

        self.push(token(
            TYPE_EVENT_START_KYOKU,
            ACTOR_NONE,
            self.relative(*oya),
            protocol_tile(*bakaze),
            protocol_tile(*dora_marker),
            TILE_NONE,
            encode_value(u32::from(*kyoku)),
            FLAG_NONE,
        ));
        self.push(token(
            TYPE_STATE_BAKAZE,
            ACTOR_NONE,
            ACTOR_NONE,
            protocol_tile(*bakaze),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        self.push(token(
            TYPE_STATE_JIKAZE,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(jikaze_for(self.absolute_seat, *oya)),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        self.push(token(
            TYPE_STATE_OYA,
            self.relative(*oya),
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        self.push(token(
            TYPE_STATE_KYOKU_INDEX,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(u32::from(*kyoku)),
            FLAG_NONE,
        ));
        self.push(token(
            TYPE_STATE_HONBA,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(u32::from(*honba)),
            FLAG_NONE,
        ));
        self.push(token(
            TYPE_STATE_KYOTAKU,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(u32::from(*kyotaku)),
            FLAG_NONE,
        ));
        for absolute_actor in 0..NUM_PLAYERS as u8 {
            let score = scores[absolute_actor as usize].max(0) as u32;
            self.push(token(
                TYPE_STATE_SCORE,
                self.relative(absolute_actor),
                ACTOR_NONE,
                TILE_NONE,
                TILE_NONE,
                TILE_NONE,
                encode_value(score / 5_000),
                FLAG_NONE,
                ));
        }
        self.push(token(
            TYPE_STATE_LEFT_TILES,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(70 / 4),
            FLAG_NONE,
        ));
        self.push(token(
            TYPE_STATE_DORA,
            ACTOR_NONE,
            ACTOR_NONE,
            protocol_tile(*dora_marker),
            TILE_NONE,
            TILE_NONE,
            encode_value(0),
            FLAG_NONE,
        ));

        for absolute_actor in 0..NUM_PLAYERS as u8 {
            let actor = self.relative(absolute_actor);
            if absolute_actor != self.absolute_seat && !reveal_opponent_initial_hands {
                self.push(token(
                    TYPE_STATE_HAND,
                    actor,
                    ACTOR_NONE,
                    TILE_UNKNOWN,
                    TILE_NONE,
                    TILE_NONE,
                    encode_value(13),
                    FLAG_NONE,
                        ));
                continue;
            }

            let mut initial_hand_counts = [0u8; 38];
            for tile in tehais[absolute_actor as usize] {
                initial_hand_counts[tile.as_usize()] += 1;
            }
            for (tile_id, count) in initial_hand_counts.into_iter().enumerate() {
                if count > 0 {
                    self.push(token(
                        TYPE_STATE_HAND,
                        actor,
                        ACTOR_NONE,
                        tile_id as i64 + 1,
                        TILE_NONE,
                        TILE_NONE,
                        encode_value(u32::from(count)),
                        FLAG_NONE,
                                ));
                }
            }
        }
        self.push(token(
            TYPE_SEP,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ));
        Ok(())
    }

    fn apply_event(&mut self, event: &MjaiEvent, _step: i64) -> Result<(), String> {
        match event {
            MjaiEvent::None
            | MjaiEvent::StartGame { .. }
            | MjaiEvent::StartKyoku { .. }
            | MjaiEvent::EndKyoku
            | MjaiEvent::EndGame => {}
            MjaiEvent::Tsumo { actor, pai } => {
                let tile = if *actor == self.absolute_seat {
                    protocol_tile(*pai)
                } else {
                    TILE_UNKNOWN
                };
                self.push(token(
                    TYPE_EVENT_DRAW,
                    self.relative(*actor),
                    ACTOR_NONE,
                    tile,
                    TILE_NONE,
                    TILE_NONE,
                    VALUE_NONE,
                    FLAG_NONE,
                        ));
            }
            MjaiEvent::Dahai {
                actor,
                pai,
                tsumogiri,
            } => {
                self.push(token(
                    TYPE_EVENT_DISCARD,
                    self.relative(*actor),
                    ACTOR_NONE,
                    protocol_tile(*pai),
                    TILE_NONE,
                    TILE_NONE,
                    VALUE_NONE,
                    if *tsumogiri { FLAG_TSUMOGIRI } else { FLAG_TEDASHI },
                        ));
            }
            MjaiEvent::Chi {
                actor,
                target,
                pai,
                consumed,
            } => {
                self.push(token(
                    TYPE_EVENT_CHI,
                    self.relative(*actor),
                    self.relative(*target),
                    protocol_tile(*pai),
                    protocol_tile(consumed[0]),
                    protocol_tile(consumed[1]),
                    VALUE_NONE,
                    chi_flag(*pai, *consumed),
                        ));
            }
            MjaiEvent::Pon {
                actor,
                target,
                pai,
                consumed,
            } => {
                self.push(token(
                    TYPE_EVENT_PON,
                    self.relative(*actor),
                    self.relative(*target),
                    protocol_tile(*pai),
                    protocol_tile(consumed[0]),
                    protocol_tile(consumed[1]),
                    VALUE_NONE,
                    FLAG_NONE,
                        ));
            }
            MjaiEvent::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => {
                self.push(token(
                    TYPE_EVENT_DAIMINKAN,
                    self.relative(*actor),
                    self.relative(*target),
                    protocol_tile(*pai),
                    protocol_tile(consumed[0]),
                    protocol_tile(consumed[1]),
                    VALUE_NONE,
                    FLAG_MELD_DAIMINKAN,
                        ));
                self.push(token(
                    TYPE_EVENT_MELD_CONT,
                    self.relative(*actor),
                    self.relative(*target),
                    protocol_tile(consumed[2]),
                    TILE_NONE,
                    TILE_NONE,
                    encode_value(0),
                    FLAG_MELD_DAIMINKAN,
                        ));
            }
            MjaiEvent::Kakan {
                actor,
                pai,
                consumed,
            } => {
                self.push(token(
                    TYPE_EVENT_KAKAN,
                    self.relative(*actor),
                    ACTOR_NONE,
                    protocol_tile(*pai),
                    protocol_tile(consumed[0]),
                    protocol_tile(consumed[1]),
                    VALUE_NONE,
                    FLAG_MELD_KAKAN,
                        ));
                self.push(token(
                    TYPE_EVENT_MELD_CONT,
                    self.relative(*actor),
                    ACTOR_NONE,
                    protocol_tile(consumed[2]),
                    TILE_NONE,
                    TILE_NONE,
                    encode_value(0),
                    FLAG_MELD_KAKAN,
                        ));
            }
            MjaiEvent::Ankan { actor, consumed } => {
                self.push(token(
                    TYPE_EVENT_ANKAN,
                    self.relative(*actor),
                    ACTOR_NONE,
                    protocol_tile(consumed[0]),
                    protocol_tile(consumed[1]),
                    protocol_tile(consumed[2]),
                    VALUE_NONE,
                    FLAG_MELD_ANKAN,
                        ));
                self.push(token(
                    TYPE_EVENT_MELD_CONT,
                    self.relative(*actor),
                    ACTOR_NONE,
                    protocol_tile(consumed[3]),
                    TILE_NONE,
                    TILE_NONE,
                    encode_value(0),
                    FLAG_MELD_ANKAN,
                        ));
            }
            MjaiEvent::Dora { dora_marker } => {
                self.push(token(
                    TYPE_EVENT_DORA,
                    ACTOR_NONE,
                    ACTOR_NONE,
                    protocol_tile(*dora_marker),
                    TILE_NONE,
                    TILE_NONE,
                    VALUE_NONE,
                    FLAG_NONE,
                        ));
            }
            MjaiEvent::Reach { actor } => {
                self.push(token(
                    TYPE_EVENT_REACH,
                    self.relative(*actor),
                    ACTOR_NONE,
                    TILE_NONE,
                    TILE_NONE,
                    TILE_NONE,
                    VALUE_NONE,
                    FLAG_REACH_DECLARE,
                        ));
            }
            MjaiEvent::ReachAccepted { actor } => {
                self.push(token(
                    TYPE_EVENT_REACH_ACCEPTED,
                    self.relative(*actor),
                    ACTOR_NONE,
                    TILE_NONE,
                    TILE_NONE,
                    TILE_NONE,
                    VALUE_NONE,
                    FLAG_NONE,
                        ));
            }
            MjaiEvent::Hora {
                actor,
                target,
                deltas,
                ura_markers,
            } => {
                let is_tsumo = actor == target;
                self.push(token(
                    TYPE_EVENT_HORA,
                    self.relative(*actor),
                    if is_tsumo {
                        ACTOR_NONE
                    } else {
                        self.relative(*target)
                    },
                    TILE_NONE,
                    TILE_NONE,
                    TILE_NONE,
                    VALUE_NONE,
                    if is_tsumo { FLAG_TSUMO } else { FLAG_RON },
                        ));
                self.push_score_deltas(deltas.as_ref());
                if *actor == self.absolute_seat {
                    for marker in ura_markers.iter().flatten() {
                        self.push(token(
                            TYPE_EVENT_URA_DORA,
                            ACTOR_SELF,
                            ACTOR_NONE,
                            protocol_tile(*marker),
                            TILE_NONE,
                            TILE_NONE,
                            VALUE_NONE,
                            FLAG_NONE,
                                        ));
                    }
                }
            }
            MjaiEvent::Ryukyoku { deltas } => {
                self.push(token(
                    TYPE_EVENT_RYUKYOKU,
                    ACTOR_NONE,
                    ACTOR_NONE,
                    TILE_NONE,
                    TILE_NONE,
                    TILE_NONE,
                    VALUE_NONE,
                    FLAG_NONE,
                        ));
                self.push_score_deltas(deltas.as_ref());
            }
        }

        Ok(())
    }

    fn push_score_deltas(&mut self, deltas: Option<&[i32; NUM_PLAYERS]>) {
        let Some(deltas) = deltas else {
            return;
        };
        for (absolute_actor, delta) in deltas.iter().enumerate() {
            let flag = if *delta > 0 {
                FLAG_DELTA_POSITIVE
            } else if *delta < 0 {
                FLAG_DELTA_NEGATIVE
            } else {
                FLAG_DELTA_ZERO
            };
            self.push(token(
                TYPE_EVENT_SCORE_DELTA,
                self.relative(absolute_actor as u8),
                ACTOR_NONE,
                TILE_NONE,
                TILE_NONE,
                TILE_NONE,
                encode_value(delta.unsigned_abs() / 1_000),
                flag,
                ));
        }
    }

    fn relative(&self, absolute_actor: u8) -> i64 {
        match (absolute_actor + NUM_PLAYERS as u8 - self.absolute_seat) % NUM_PLAYERS as u8 {
            0 => ACTOR_SELF,
            1 => ACTOR_SHIMOCHA,
            2 => ACTOR_TOIMEN,
            3 => ACTOR_KAMICHA,
            _ => unreachable!("four-player relative seat must be in 0..4"),
        }
    }

    fn push(&mut self, token: Token) {
        self.tokens.push(token);
    }
}
