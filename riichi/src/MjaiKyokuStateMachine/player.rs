struct PlayerKyokuStateMachine {
    absolute_seat: u8,
    hand_counts: [u8; 38],
    open_melds: Vec<OpenMeld>,
    tokens: Vec<Token>,
    dora_indicators: u8,
}

#[derive(Clone, Copy)]
enum OpenMeld {
    Chi,
    Pon { tiles: [MjaiTile; 3] },
    Daiminkan,
    Kakan,
    Ankan,
}

impl PlayerKyokuStateMachine {
    fn new(absolute_seat: u8) -> Self {
        Self {
            absolute_seat,
            hand_counts: [0; 38],
            open_melds: Vec::new(),
            tokens: Vec::new(),
            dora_indicators: 0,
        }
    }

    fn start_kyoku(
        &mut self,
        event: &MjaiEvent,
        mode: GameMode,
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

        self.hand_counts.fill(0);
        self.open_melds.clear();
        for tile in tehais[self.absolute_seat as usize] {
            self.hand_counts[tile.as_usize()] += 1;
        }
        self.tokens.clear();
        self.dora_indicators = 1;

        self.push(token(
            TYPE_EVENT_START_KYOKU,
            ACTOR_NONE,
            self.relative(*oya),
            protocol_tile(*bakaze),
            protocol_tile(*dora_marker),
            TILE_NONE,
            encode_value(u32::from(*kyoku)),
            FLAG_NONE,
            0,
        ));
        self.push(token(
            TYPE_STATE_GAME_MODE,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            mode.flag(),
            0,
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
            0,
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
            0,
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
            0,
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
            0,
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
            0,
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
            0,
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
                0,
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
            0,
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
            0,
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
                    0,
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
                        0,
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
            0,
        ));
        Ok(())
    }

    fn apply_event(
        &mut self,
        event: &MjaiEvent,
        step: i64,
        is_double_reach: bool,
    ) -> Result<(), String> {
        self.update_visible_hand(event)?;

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
                    step,
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
                    step,
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
                    step,
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
                    step,
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
                    step,
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
                    step,
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
                    step,
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
                    step,
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
                    step,
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
                    step,
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
                    encode_value(u32::from(self.dora_indicators)),
                    FLAG_NONE,
                    step,
                ));
                self.dora_indicators = self.dora_indicators.saturating_add(1);
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
                    if is_double_reach {
                        FLAG_DOUBLE_REACH_DECLARE
                    } else {
                        FLAG_REACH_DECLARE
                    },
                    step,
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
                    step,
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
                    step,
                ));
                self.push_score_deltas(deltas.as_ref(), step);
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
                            step,
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
                    step,
                ));
                self.push_score_deltas(deltas.as_ref(), step);
            }
        }

        Ok(())
    }

    fn update_visible_hand(&mut self, event: &MjaiEvent) -> Result<(), String> {
        match event {
            MjaiEvent::Tsumo { actor, pai } if *actor == self.absolute_seat => self.add_tile(*pai),
            MjaiEvent::Dahai { actor, pai, .. } if *actor == self.absolute_seat => {
                self.remove_tile(*pai)
            }
            MjaiEvent::Chi { actor, consumed, .. } if *actor == self.absolute_seat => {
                self.remove_tiles(consumed)?;
                self.open_melds.push(OpenMeld::Chi);
                Ok(())
            }
            MjaiEvent::Pon {
                actor,
                pai,
                consumed,
                ..
            } if *actor == self.absolute_seat => {
                self.remove_tiles(consumed)?;
                self.open_melds.push(OpenMeld::Pon {
                    tiles: [*pai, consumed[0], consumed[1]],
                });
                Ok(())
            }
            MjaiEvent::Daiminkan { actor, consumed, .. } if *actor == self.absolute_seat => {
                self.remove_tiles(consumed)?;
                self.open_melds.push(OpenMeld::Daiminkan);
                Ok(())
            }
            MjaiEvent::Kakan { actor, pai, .. } if *actor == self.absolute_seat => {
                self.remove_tile(*pai)?;
                let matching_pon = self.open_melds.iter().position(|meld| {
                    matches!(meld, OpenMeld::Pon { tiles } if tiles[0].deaka() == pai.deaka())
                });
                if let Some(index) = matching_pon {
                    self.open_melds[index] = OpenMeld::Kakan;
                }
                Ok(())
            }
            MjaiEvent::Ankan { actor, consumed } if *actor == self.absolute_seat => {
                self.remove_tiles(consumed)?;
                self.open_melds.push(OpenMeld::Ankan);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn add_tile(&mut self, tile: MjaiTile) -> Result<(), String> {
        let count = &mut self.hand_counts[tile.as_usize()];
        if *count >= 4 {
            return Err(format!("cannot add a fifth visible tile {tile:?} to self hand"));
        }
        *count += 1;
        Ok(())
    }

    fn remove_tile(&mut self, tile: MjaiTile) -> Result<(), String> {
        let count = &mut self.hand_counts[tile.as_usize()];
        if *count == 0 {
            return Err(format!("cannot remove absent tile {tile:?} from self hand"));
        }
        *count -= 1;
        Ok(())
    }

    fn remove_tiles<const N: usize>(&mut self, tiles: &[MjaiTile; N]) -> Result<(), String> {
        for tile in tiles {
            self.remove_tile(*tile)?;
        }
        Ok(())
    }

    fn holds_tiles(&self, tiles: &[MjaiTile]) -> bool {
        let mut required = [0u8; 38];
        for tile in tiles {
            required[tile.as_usize()] = required[tile.as_usize()].saturating_add(1);
        }
        required
            .iter()
            .enumerate()
            .all(|(index, count)| *count <= self.hand_counts[index])
    }

    fn tiles_of_kind(&self, tile34: MjaiTile) -> Vec<MjaiTile> {
        let mut tiles = Vec::new();
        for tile_id in 0..37 {
            let tile = MjaiTile(tile_id as u8);
            if tile.deaka() == tile34 {
                for _ in 0..self.hand_counts[tile_id] {
                    tiles.push(tile);
                }
            }
        }
        tiles
    }

    fn pon_tiles(&self, tile34: MjaiTile) -> Option<[MjaiTile; 3]> {
        self.open_melds.iter().find_map(|meld| match meld {
            OpenMeld::Pon { tiles } if tiles[0].deaka() == tile34 => Some(*tiles),
            _ => None,
        })
    }

    fn push_score_deltas(&mut self, deltas: Option<&[i32; NUM_PLAYERS]>, step: i64) {
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
                step,
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
