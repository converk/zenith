/// Per-observer semantic token state for one kyoku.
///
/// History is append-only within a kyoku and consists only of public MJAI
/// events.  The current state suffix is rebuilt from the decision snapshot;
/// it deliberately contains no river or meld projection because the event
/// prefix already represents those facts without duplication.
const TOKEN_WIDTH: usize = 10;
const NUMERIC_WIDTH: usize = 8;
const MAX_CONTEXT_TOKENS: usize = 4096;

const SEGMENT_EVENT: u8 = 1;
const SEGMENT_ACTOR_STATE: u8 = 2;
const KIND_EVENT: u8 = 1;
const KIND_SCORE: u8 = 2;
const KIND_COUNTER: u8 = 3;
const KIND_TILE_COUNT: u8 = 4;

const VIS_PUBLIC: u8 = 1;

#[derive(Clone, Copy)]
struct SemanticToken {
    factors: [u8; TOKEN_WIDTH],
    numeric: [f32; NUMERIC_WIDTH],
}

impl SemanticToken {
    const fn categorical(factors: [u8; TOKEN_WIDTH]) -> Self {
        Self { factors, numeric: [0.0; NUMERIC_WIDTH] }
    }

    fn number(mut self, field: u8, value: f32) -> Self {
        self.numeric = numeric_features(field, value);
        self
    }
}

struct PlayerKyokuStateMachine {
    absolute_seat: u8,
    history: Vec<SemanticToken>,
    live_wall: u8,
}

impl PlayerKyokuStateMachine {
    fn new(absolute_seat: u8) -> Self {
        Self { absolute_seat, history: Vec::new(), live_wall: 70 }
    }

    fn apply_player_event(&mut self, event: &MjaiEvent) -> Result<(), String> {
        match event {
            MjaiEvent::StartGame { id, .. } => {
                if let Some(id) = id {
                    if *id >= NUM_PLAYERS as u8 {
                        return Err("start_game.id must be in 0..4".to_owned());
                    }
                    self.absolute_seat = *id;
                }
            }
            MjaiEvent::StartKyoku { bakaze, kyoku, .. } => self.start_kyoku(*bakaze, *kyoku),
            // These events delimit rewards/lifecycles. No same-kyoku decision
            // follows, therefore putting settlement information into actor
            // input would be both redundant and a leakage risk.
            MjaiEvent::Hora { .. } | MjaiEvent::Ryukyoku { .. }
            | MjaiEvent::EndKyoku | MjaiEvent::EndGame | MjaiEvent::None => {}
            _ => self.apply_event(event),
        }
        Ok(())
    }

    fn start_kyoku(&mut self, bakaze: MjaiTile, kyoku: u8) {
        self.history.clear();
        self.live_wall = 70;
        let wind = bakaze.deaka().as_u8().saturating_sub(27).min(3);
        let hand = kyoku.saturating_sub(1).min(3);
        self.history.push(self.event_token(2, ACTOR_NONE, ACTOR_NONE, None, 1 + wind * 4 + hand));
    }

    fn apply_event(&mut self, event: &MjaiEvent) {
        match event {
            // A draw changes the active wall and own snapshot hand. It has no
            // standalone semantic token, matching the encoder in exp.
            MjaiEvent::Tsumo { .. } => self.live_wall = self.live_wall.saturating_sub(1),
            MjaiEvent::Dahai { actor, pai, tsumogiri } => {
                self.history.push(self.event_token(4, self.relative(*actor), ACTOR_NONE, Some(*pai), 1 + u8::from(*tsumogiri)));
            }
            MjaiEvent::Reach { actor } => {
                self.history.push(self.event_token(11, self.relative(*actor), ACTOR_NONE, None, 0));
            }
            MjaiEvent::ReachAccepted { actor } => {
                self.history.push(self.event_token(12, self.relative(*actor), ACTOR_NONE, None, 0));
            }
            MjaiEvent::Dora { dora_marker } => {
                self.history.push(self.event_token(10, ACTOR_NONE, ACTOR_NONE, Some(*dora_marker), 0));
            }
            MjaiEvent::Chi { actor, target, pai, consumed } => {
                self.history.push(self.event_token(5, self.relative(*actor), self.relative(*target), Some(*pai), chi_detail(*pai, consumed)));
            }
            MjaiEvent::Pon { actor, target, pai, consumed } => {
                self.history.push(self.event_token(6, self.relative(*actor), self.relative(*target), Some(*pai), meld_red_detail(&consumed[..])));
            }
            MjaiEvent::Daiminkan { actor, target, pai, consumed } => {
                self.history.push(self.event_token(7, self.relative(*actor), self.relative(*target), Some(*pai), meld_red_detail(&consumed[..])));
            }
            MjaiEvent::Ankan { actor, consumed } => {
                self.history.push(self.event_token(8, self.relative(*actor), ACTOR_SELF, Some(consumed[0]), meld_red_detail(&consumed[..])));
            }
            MjaiEvent::Kakan { actor, pai, consumed } => {
                self.history.push(self.event_token(9, self.relative(*actor), ACTOR_SELF, Some(*pai), meld_red_detail(&consumed[..])));
            }
            MjaiEvent::None | MjaiEvent::StartGame { .. } | MjaiEvent::StartKyoku { .. }
            | MjaiEvent::Hora { .. } | MjaiEvent::Ryukyoku { .. }
            | MjaiEvent::EndKyoku | MjaiEvent::EndGame => {}
        }
    }

    fn tokens(&self, snapshot: &DecisionSnapshot) -> Result<Vec<SemanticToken>, String> {
        if snapshot.player_id != self.absolute_seat || snapshot.oya >= NUM_PLAYERS as u8 {
            return Err("snapshot player_id/oya does not match table state".to_owned());
        }
        let mut result = self.history.clone();
        self.append_state_tokens(&mut result, snapshot)?;
        if result.len() + 1 > MAX_CONTEXT_TOKENS {
            return Err(format!("semantic-token context overflow: {} > {MAX_CONTEXT_TOKENS}", result.len() + 1));
        }
        Ok(result)
    }

    fn append_state_tokens(&self, out: &mut Vec<SemanticToken>, snapshot: &DecisionSnapshot) -> Result<(), String> {
        // Scores are semantic magnitudes, never categorical ids.
        for (seat, score) in snapshot.scores.iter().enumerate() {
            out.push(SemanticToken::categorical([
                SEGMENT_ACTOR_STATE, KIND_SCORE, 1, self.relative(seat as u8), 0, 0, 0, 0, 0, 0,
            ]).number(1, *score as f32));
        }
        for (field, value) in [
            (1, snapshot.round_wind as f32),
            (2, snapshot.kyoku_index as f32 + 1.0),
            (3, snapshot.honba as f32),
            (4, snapshot.riichi_sticks as f32),
            (5, self.live_wall as f32),
        ] {
            out.push(SemanticToken::categorical([
                SEGMENT_ACTOR_STATE, KIND_COUNTER, field, 0, 0, 0, 0, 0, 0, 0,
            ]).number(2, value));
        }
        out.push(SemanticToken::categorical([
            SEGMENT_ACTOR_STATE, KIND_COUNTER, 6, self.relative(snapshot.oya), 0, 0, 0, 0, 0, 0,
        ]));
        let self_wind = (self.absolute_seat + NUM_PLAYERS as u8 - snapshot.oya) % NUM_PLAYERS as u8 + 1;
        out.push(SemanticToken::categorical([
            SEGMENT_ACTOR_STATE, KIND_COUNTER, 7, ACTOR_SELF, 0, 0, 0, self_wind, 0, 0,
        ]));
        for indicator in &snapshot.dora_indicators {
            let tile = snapshot_tile(indicator)?;
            out.push(tile_count_token(3, 0, tile, 0, VIS_PUBLIC));
        }
        let drawn = snapshot.drawn_tile.as_deref().map(snapshot_tile).transpose()?;
        let flags = u8::from(snapshot.riichi_declared[self.absolute_seat as usize])
            | (u8::from(drawn.is_some()) << 1)
            | ((snapshot.decision_flags & 1) << 2);
        out.push(SemanticToken::categorical([
            SEGMENT_ACTOR_STATE, KIND_COUNTER, 8, ACTOR_SELF, 0, 0, 0, 0, flags, 0,
        ]));

        let mut counts = [0u8; 34];
        let mut red = [false; 34];
        for value in &snapshot.hand {
            let tile = snapshot_tile(value)?;
            let tile_type = tile.deaka().as_u8() as usize;
            counts[tile_type] = counts[tile_type].saturating_add(1);
            red[tile_type] |= is_red(tile);
        }
        for (tile_type, count) in counts.into_iter().enumerate() {
            if count > 0 {
                out.push(tile_type_count_token(1, ACTOR_SELF, tile_type as u8, red[tile_type], count, VIS_PUBLIC));
            }
        }
        if let Some(tile) = drawn {
            out.push(tile_count_token(5, ACTOR_SELF, tile, 1, VIS_PUBLIC));
        }
        Ok(())
    }

    fn event_token(&self, event: u8, actor: u8, source: u8, tile: Option<MjaiTile>, flag: u8) -> SemanticToken {
        let (suit, rank, red) = tile.map(tile_factors).unwrap_or((0, 0, 0));
        SemanticToken::categorical([
            SEGMENT_EVENT, KIND_EVENT, event, actor, suit, rank, red, source, flag, VIS_PUBLIC,
        ])
    }

    fn relative(&self, absolute: u8) -> u8 {
        match (absolute + NUM_PLAYERS as u8 - self.absolute_seat) % NUM_PLAYERS as u8 {
            0 => ACTOR_SELF,
            1 => ACTOR_SHIMOCHA,
            2 => ACTOR_TOIMEN,
            3 => ACTOR_KAMICHA,
            _ => unreachable!("four-player relative seat"),
        }
    }
}

fn numeric_features(field: u8, value: f32) -> [f32; NUMERIC_WIDTH] {
    if field == 0 { return [0.0; NUMERIC_WIDTH]; }
    let periods = match field {
        1 => [100.0, 1_000.0, 10_000.0, 100_000.0],
        2 => [2.0, 8.0, 32.0, 128.0],
        _ => return [0.0; NUMERIC_WIDTH],
    };
    let mut result = [0.0; NUMERIC_WIDTH];
    for (index, period) in periods.into_iter().enumerate() {
        let angle = std::f32::consts::TAU * value / period;
        result[index * 2] = angle.sin();
        result[index * 2 + 1] = angle.cos();
    }
    result
}

fn tile_factors(tile: MjaiTile) -> (u8, u8, u8) {
    let tile_type = tile.deaka().as_u8();
    let suit = if tile_type < 27 { tile_type / 9 + 1 } else { 4 };
    let rank = if tile_type < 27 { tile_type % 9 + 1 } else { tile_type - 26 };
    (suit, rank, u8::from(is_red(tile)))
}

fn tile_count_token(field: u8, seat: u8, tile: MjaiTile, count: u8, visibility: u8) -> SemanticToken {
    let (suit, rank, red) = tile_factors(tile);
    SemanticToken::categorical([
        SEGMENT_ACTOR_STATE, KIND_TILE_COUNT, field, seat, suit, rank, red, count, 0, visibility,
    ])
}

fn tile_type_count_token(field: u8, seat: u8, tile_type: u8, red: bool, count: u8, visibility: u8) -> SemanticToken {
    let tile = MjaiTile(tile_type);
    let (suit, rank, _) = tile_factors(tile);
    SemanticToken::categorical([
        SEGMENT_ACTOR_STATE, KIND_TILE_COUNT, field, seat, suit, rank, u8::from(red), count, 0, visibility,
    ])
}

fn is_red(tile: MjaiTile) -> bool { matches!(tile.as_u8(), 34..=36) }

fn meld_red_detail(tiles: &[MjaiTile]) -> u8 { 1 + u8::from(tiles.iter().copied().any(is_red)) }

fn chi_detail(pai: MjaiTile, consumed: &[MjaiTile; 2]) -> u8 {
    let mut types = [pai.deaka().as_u8(), consumed[0].deaka().as_u8(), consumed[1].deaka().as_u8()];
    types.sort_unstable();
    let offset = pai.deaka().as_u8().saturating_sub(types[0]).min(2);
    let red = is_red(pai) || consumed.iter().copied().any(is_red);
    1 + offset * 2 + u8::from(red)
}
