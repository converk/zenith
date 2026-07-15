/// Per-view V4 state.  `blocks` is append-only and therefore cacheable.  A
/// partially-filled turn block deliberately stays out of `blocks`: it is
/// represented in the temporary board payload until it reaches four records
/// (or is flushed before a meld), so cached K/V is never mutated in place.
struct PlayerKyokuStateMachine {
    absolute_seat: u8,
    blocks: Vec<EventBlock>,
    pending_turn: [MicroEvent; 4],
    pending_len: usize,
    public: [PublicPlayerState; NUM_PLAYERS], // indexed by absolute seat
    live_wall: u8,
    is_first_turn: bool,
    is_rinshan: bool,
    pending_reach: [bool; NUM_PLAYERS],
    last_discard_actor: Option<u8>,
    last_discard_tile: Option<u8>,
}

impl PlayerKyokuStateMachine {
    fn new(absolute_seat: u8) -> Self {
        Self {
            absolute_seat,
            blocks: Vec::new(),
            pending_turn: [MicroEvent::default(); 4],
            pending_len: 0,
            public: std::array::from_fn(|_| PublicPlayerState::default()),
            live_wall: 70,
            is_first_turn: true,
            is_rinshan: false,
            pending_reach: [false; NUM_PLAYERS],
            last_discard_actor: None,
            last_discard_tile: None,
        }
    }

    fn apply_player_event(&mut self, event: &MjaiEvent) -> Result<(), String> {
        match event {
            MjaiEvent::StartGame { id, .. } => {
                if let Some(id) = id {
                    if *id >= NUM_PLAYERS as u8 { return Err("start_game.id must be in 0..4".to_owned()); }
                    self.absolute_seat = *id;
                }
            }
            MjaiEvent::StartKyoku { .. } => self.start_kyoku(),
            MjaiEvent::EndKyoku | MjaiEvent::EndGame | MjaiEvent::None => {}
            _ => self.apply_event(event)?,
        }
        Ok(())
    }

    fn start_kyoku(&mut self) {
        self.blocks.clear();
        self.pending_turn = [MicroEvent::default(); 4];
        self.pending_len = 0;
        self.public = std::array::from_fn(|_| PublicPlayerState::default());
        self.live_wall = 70;
        self.is_first_turn = true;
        self.is_rinshan = false;
        self.pending_reach = [false; NUM_PLAYERS];
        self.last_discard_actor = None;
        self.last_discard_tile = None;
    }

    fn push_micro(&mut self, micro: MicroEvent) {
        self.pending_turn[self.pending_len] = micro;
        self.pending_len += 1;
        if self.pending_len == 4 {
            self.blocks.push(EventBlock::Turn(self.pending_turn));
            self.pending_turn = [MicroEvent::default(); 4];
            self.pending_len = 0;
        }
    }

    fn flush_turn(&mut self) {
        if self.pending_len > 0 {
            self.blocks.push(EventBlock::Turn(self.pending_turn));
            self.pending_turn = [MicroEvent::default(); 4];
            self.pending_len = 0;
        }
    }

    fn push_meld(&mut self, meld_kind: u8, actor: u8, target: Option<u8>, pai: u8, tiles: [u8; 4]) {
        self.flush_turn();
        self.blocks.push(EventBlock::Meld {
            meld_kind,
            actor: self.relative(actor),
            target: target.map(|seat| self.relative(seat)).unwrap_or(ACTOR_NONE),
            pai,
            tiles,
        });
        self.public[actor as usize].melds.push(MeldEntry {
            meld_kind,
            target: target.map(|seat| self.relative(seat)).unwrap_or(ACTOR_NONE),
            pai,
            tiles,
        });
        if let Some(target) = target {
            if let Some(last) = self.public[target as usize].river.last_mut() {
                last.flag |= FLAG_CALLED;
            }
        }
        self.is_first_turn = false;
        self.is_rinshan = matches!(meld_kind, MELD_DAIMINKAN | MELD_KAKAN | MELD_ANKAN);
        for player in &mut self.public { player.ippatsu = false; }
    }

    fn apply_event(&mut self, event: &MjaiEvent) -> Result<(), String> {
        match event {
            MjaiEvent::Tsumo { .. } => {
                self.live_wall = self.live_wall.saturating_sub(1);
            }
            MjaiEvent::Dahai { actor, pai, tsumogiri } => {
                let mut flag = if *tsumogiri { FLAG_TSUMOGIRI } else { 0 };
                if self.pending_reach[*actor as usize] {
                    flag |= FLAG_RIICHI_DISCARD;
                    self.pending_reach[*actor as usize] = false;
                }
                let tile = tile_code(*pai);
                self.push_micro(MicroEvent { kind: MICRO_DISCARD, actor: self.relative(*actor), tile, flag });
                self.public[*actor as usize].river.push(RiverEntry { tile, flag });
                self.public[*actor as usize].ippatsu = false;
                self.last_discard_actor = Some(*actor);
                self.last_discard_tile = Some(tile);
                self.is_rinshan = false;
            }
            MjaiEvent::Reach { actor } => {
                self.push_micro(MicroEvent { kind: MICRO_REACH, actor: self.relative(*actor), tile: TILE_NONE, flag: 0 });
                self.pending_reach[*actor as usize] = true;
                if self.is_first_turn {
                    self.public[*actor as usize].double_riichi = true;
                }
            }
            MjaiEvent::ReachAccepted { actor } => {
                self.push_micro(MicroEvent { kind: MICRO_REACH_ACCEPTED, actor: self.relative(*actor), tile: TILE_NONE, flag: 0 });
                self.public[*actor as usize].ippatsu = true;
            }
            MjaiEvent::Dora { dora_marker } => {
                self.push_micro(MicroEvent { kind: MICRO_DORA, actor: ACTOR_NONE, tile: tile_code(*dora_marker), flag: 0 });
            }
            MjaiEvent::Chi { actor, target, pai, consumed } => {
                self.push_meld(MELD_CHI, *actor, Some(*target), tile_code(*pai), [tile_code(consumed[0]), tile_code(consumed[1]), TILE_NONE, TILE_NONE]);
            }
            MjaiEvent::Pon { actor, target, pai, consumed } => {
                self.push_meld(MELD_PON, *actor, Some(*target), tile_code(*pai), [tile_code(consumed[0]), tile_code(consumed[1]), TILE_NONE, TILE_NONE]);
            }
            MjaiEvent::Daiminkan { actor, target, pai, consumed } => {
                self.push_meld(MELD_DAIMINKAN, *actor, Some(*target), tile_code(*pai), [tile_code(consumed[0]), tile_code(consumed[1]), tile_code(consumed[2]), TILE_NONE]);
            }
            MjaiEvent::Kakan { actor, pai, consumed } => {
                self.push_meld(MELD_KAKAN, *actor, None, tile_code(*pai), [tile_code(consumed[0]), tile_code(consumed[1]), tile_code(consumed[2]), TILE_NONE]);
            }
            MjaiEvent::Ankan { actor, consumed } => {
                self.push_meld(MELD_ANKAN, *actor, None, TILE_NONE, [tile_code(consumed[0]), tile_code(consumed[1]), tile_code(consumed[2]), tile_code(consumed[3])]);
            }
            // Terminal events are intentionally retained only in RiichiEnv's
            // original log / reward path; no same-kyoku model decision follows.
            MjaiEvent::Hora { .. } | MjaiEvent::Ryukyoku { .. }
            | MjaiEvent::None | MjaiEvent::StartGame { .. }
            | MjaiEvent::StartKyoku { .. } | MjaiEvent::EndKyoku | MjaiEvent::EndGame => {}
        }
        Ok(())
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

    fn board_tokens(&self, snapshot: &DecisionSnapshot) -> Result<[[u8; BOARD_FIELDS]; BOARD_TOKENS], String> {
        if snapshot.player_id != self.absolute_seat || snapshot.oya >= NUM_PLAYERS as u8 {
            return Err("snapshot player_id/oya does not match table state".to_owned());
        }
        let globals = self.global_fields(snapshot)?;
        let mut result = [[0u8; BOARD_FIELDS]; BOARD_TOKENS];
        for relative_index in 0..NUM_PLAYERS {
            let absolute = (self.absolute_seat + relative_index as u8) % NUM_PLAYERS as u8;
            let state_index = relative_index * 3;
            result[state_index][..16].copy_from_slice(&globals);
            result[state_index + 1][..16].copy_from_slice(&globals);
            result[state_index + 2][..16].copy_from_slice(&globals);
            // Do not mutate cached block history to append a partial four-way
            // turn pack.  Its raw records instead live in this temporary
            // suffix and are visible immediately at the current decision.
            for (micro_index, micro) in self.pending_turn.iter().enumerate() {
                let offset = 120 + micro_index * 4;
                let values = [micro.kind, micro.actor, micro.tile, micro.flag];
                result[state_index][offset..offset + 4].copy_from_slice(&values);
                result[state_index + 1][offset..offset + 4].copy_from_slice(&values);
                result[state_index + 2][offset..offset + 4].copy_from_slice(&values);
            }
            self.fill_player_state(&mut result[state_index], snapshot, absolute, relative_index == 0)?;
            self.fill_river(&mut result[state_index + 1], absolute);
            self.fill_melds(&mut result[state_index + 2], absolute);
        }
        Ok(result)
    }

    fn global_fields(&self, snapshot: &DecisionSnapshot) -> Result<[u8; 16], String> {
        let mut fields = [0u8; 16];
        fields[0] = tile_code(round_wind_tile(snapshot.round_wind)?);
        fields[1] = self.relative(snapshot.oya);
        fields[2] = tile_code(jikaze_for(snapshot.player_id, snapshot.oya));
        fields[3] = snapshot.kyoku_index.saturating_add(1);
        fields[4] = bucket(snapshot.honba as u32);
        fields[5] = bucket(snapshot.riichi_sticks);
        fields[6] = bucket(u32::from(self.live_wall) / 4);
        fields[7] = self.last_discard_actor.map(|seat| self.relative(seat)).unwrap_or(ACTOR_NONE);
        fields[8] = self.last_discard_tile.or_else(|| snapshot.last_discard.as_deref().map(snapshot_tile).transpose().ok().flatten().map(tile_code)).unwrap_or(TILE_NONE);
        fields[9] = u8::from(self.is_first_turn);
        fields[10] = u8::from(self.is_rinshan);
        for (index, tile) in snapshot.dora_indicators.iter().take(DORA_SLOTS).enumerate() {
            fields[11 + index.min(4)] = tile_code(snapshot_tile(tile)?);
        }
        Ok(fields)
    }

    fn fill_player_state(&self, out: &mut [u8; BOARD_FIELDS], snapshot: &DecisionSnapshot, absolute: u8, is_self: bool) -> Result<(), String> {
        out[16] = self.relative(absolute);
        out[17] = bucket((snapshot.scores[absolute as usize].max(0) as u32) / 5_000);
        out[18] = u8::from(snapshot.riichi_declared[absolute as usize]);
        out[19] = u8::from(self.public[absolute as usize].double_riichi);
        out[20] = u8::from(self.public[absolute as usize].ippatsu);
        out[21] = snapshot.last_tedashis[absolute as usize].as_deref().map(snapshot_tile).transpose()?.map(tile_code).unwrap_or(TILE_NONE);
        if is_self {
            let mut counts = [0u8; 37];
            for tile in &snapshot.hand {
                let code = tile_code(snapshot_tile(tile)?);
                if code == TILE_NONE || code == TILE_UNKNOWN { return Err("snapshot hand contains invalid tile".to_owned()); }
                counts[(code - 1) as usize] = counts[(code - 1) as usize].saturating_add(1);
            }
            for (index, count) in counts.into_iter().enumerate() { out[32 + index] = count.saturating_add(1); }
            out[70] = snapshot.drawn_tile.as_deref().map(snapshot_tile).transpose()?.map(tile_code).unwrap_or(TILE_NONE);
        }
        Ok(())
    }

    fn fill_river(&self, out: &mut [u8; BOARD_FIELDS], absolute: u8) {
        out[16] = self.relative(absolute);
        for (index, river) in self.public[absolute as usize].river.iter().take(RIVER_SLOTS).enumerate() {
            out[20 + index] = river.tile;
            out[52 + index] = river.flag;
        }
        // The partial normal-event pack has not entered cache yet.  Its state
        // is still fully visible through river entries and global last discard.
    }

    fn fill_melds(&self, out: &mut [u8; BOARD_FIELDS], absolute: u8) {
        out[16] = self.relative(absolute);
        for (index, meld) in self.public[absolute as usize].melds.iter().take(MELD_SLOTS).enumerate() {
            let base = 20 + index * 7;
            out[base] = meld.meld_kind;
            out[base + 1] = meld.target;
            out[base + 2] = meld.pai;
            out[base + 3..base + 7].copy_from_slice(&meld.tiles);
        }
    }
}

fn tile_code(tile: MjaiTile) -> u8 { tile.as_u8() + 1 }
fn bucket(value: u32) -> u8 { value.min(17) as u8 + 1 }
