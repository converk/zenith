use super::*;

fn tile(value: &str) -> MjaiTile {
    tile_from_str(value).unwrap()
}

fn start_kyoku() -> MjaiEvent {
    MjaiEvent::StartKyoku {
        bakaze: tile("E"),
        dora_marker: tile("2p"),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25_000; NUM_PLAYERS],
        tehais: [
            [
                tile("1m"),
                tile("1m"),
                tile("2m"),
                tile("3m"),
                tile("4m"),
                tile("5mr"),
                tile("6m"),
                tile("7p"),
                tile("8p"),
                tile("9p"),
                tile("E"),
                tile("S"),
                tile("C"),
            ],
            [
                tile("5m"),
                tile("5m"),
                tile("5m"),
                tile("1p"),
                tile("1p"),
                tile("1p"),
                tile("1p"),
                tile("2p"),
                tile("2p"),
                tile("2p"),
                tile("2p"),
                tile("3p"),
                tile("3p"),
            ],
            [
                tile("1s"),
                tile("2s"),
                tile("3s"),
                tile("4s"),
                tile("5s"),
                tile("6s"),
                tile("7s"),
                tile("8s"),
                tile("9s"),
                tile("E"),
                tile("S"),
                tile("W"),
                tile("N"),
            ],
            [
                tile("1m"),
                tile("2m"),
                tile("3m"),
                tile("4m"),
                tile("5m"),
                tile("6m"),
                tile("7m"),
                tile("8m"),
                tile("9m"),
                tile("P"),
                tile("F"),
                tile("C"),
                tile("E"),
            ],
        ],
    }
}

impl TableStateMachine {
    fn apply_event(&mut self, event: MjaiEvent) -> Result<(), String> {
        for player_index in 0..NUM_PLAYERS {
            self.apply_player_events(player_index as u8, vec![event.clone()])?;
        }
        Ok(())
    }
}

fn token_of_type(tokens: &[Token], token_type: i64) -> &Token {
    tokens
        .iter()
        .find(|token| token[0] == token_type)
        .unwrap_or_else(|| panic!("missing token type {token_type}"))
}

fn assert_has_token(tokens: &[Token], expected: Token) {
    assert!(
        tokens.iter().any(|token| *token == expected),
        "missing expected token {expected:?} in {tokens:?}"
    );
}

fn assert_exact_mask(mask: &[bool; NUM_ACTIONS], expected_action_ids: &[usize]) {
    let mut actual = mask
        .iter()
        .enumerate()
        .filter_map(|(action_id, is_open)| is_open.then_some(action_id))
        .collect::<Vec<_>>();
    let mut expected = expected_action_ids.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

fn requested_json(
    table: &TableStateMachine,
    player_index: u8,
    action_id: usize,
) -> serde_json::Value {
    let response = table
        .requested_action_to_env(player_index, action_id)
        .unwrap()
        .unwrap_or_else(|| panic!("missing action_id {action_id}"));
    serde_json::from_str(&response).unwrap()
}

fn appended_by_event(
    table: &mut TableStateMachine,
    event: MjaiEvent,
    player_index: usize,
) -> Vec<Token> {
    let before = table.players[player_index].tokens.len();
    table.apply_event(event).unwrap();
    table.players[player_index].tokens[before..].to_vec()
}

fn started_table() -> TableStateMachine {
    let mut table = TableStateMachine::new(true);
    table.apply_event(start_kyoku()).unwrap();
    table
}

fn decision_snapshot() -> DecisionSnapshot {
    DecisionSnapshot {
        player_id: 2,
        oya: 0,
        round_wind: 1,
        kyoku_index: 2,
        honba: 1,
        riichi_sticks: 2,
        scores: [31_000, 24_000, 25_000, 20_000],
        dora_indicators: vec!["2p".to_owned(), "7s".to_owned()],
        hand: vec![
            "1m".to_owned(),
            "1m".to_owned(),
            "5m".to_owned(),
            "5mr".to_owned(),
            "E".to_owned(),
        ],
        drawn_tile: Some("5mr".to_owned()),
        riichi_declared: [false, true, false, true],
        last_discard: Some("9p".to_owned()),
        last_tedashis: [
            Some("3m".to_owned()),
            None,
            Some("5pr".to_owned()),
            Some("C".to_owned()),
        ],
    }
}

#[test]
fn starts_from_kyoku_and_appends_events_per_player_view() {
    let mut table = TableStateMachine::new(true);
    table
        .apply_event(MjaiEvent::StartGame {
            id: None,
            names: Default::default(),
            seed: None,
        })
        .unwrap();
    table.apply_event(start_kyoku()).unwrap();

    let initial_len = table.players[0].tokens.len();
    assert_eq!(table.players[0].tokens[0][0], TYPE_EVENT_START_KYOKU);
    assert_eq!(table.players[0].tokens.last().unwrap()[0], TYPE_SEP);
    assert_eq!(table.players[0].tokens[0][2], ACTOR_SELF);

    table
        .apply_event(MjaiEvent::Tsumo {
            actor: 0,
            pai: tile("5m"),
        })
        .unwrap();
    assert_eq!(table.players[0].tokens.len(), initial_len + 1);
    assert_eq!(table.players[0].tokens.last().unwrap()[0], TYPE_EVENT_DRAW);
    assert_eq!(
        table.players[0].tokens.last().unwrap()[3],
        protocol_tile(tile("5m"))
    );
    assert_eq!(table.players[1].tokens.last().unwrap()[3], TILE_UNKNOWN);

    let len_before_dora = table.players[0].tokens.len();
    table
        .apply_event(MjaiEvent::Dora {
            dora_marker: tile("7s"),
        })
        .unwrap();
    assert_eq!(table.players[0].tokens.len(), len_before_dora + 1);
    assert_eq!(table.players[0].tokens.last().unwrap()[0], TYPE_EVENT_DORA);
    assert_eq!(table.players[0].tokens[initial_len - 2][0], TYPE_STATE_HAND);
}

#[test]
fn decision_snapshot_appends_temporary_tokens_without_mutating_history() {
    let mut table = TableStateMachine::new(true);
    table
        .apply_event(MjaiEvent::StartGame {
            id: Some(2),
            names: Default::default(),
            seed: None,
        })
        .unwrap();
    table.apply_event(start_kyoku()).unwrap();

    let before_len = table.players[2].tokens.len();
    let row = table.players[2]
        .tokens_with_snapshot(&decision_snapshot())
        .unwrap();
    assert_eq!(table.players[2].tokens.len(), before_len);
    assert!(row.len() > before_len);

    let snapshot = &row[before_len..];
    assert_eq!(snapshot.first().unwrap()[0], TYPE_STATE_SNAPSHOT_BEGIN);
    assert_eq!(snapshot.last().unwrap()[0], TYPE_STATE_SNAPSHOT_END);
    assert_eq!(
        snapshot[1],
        token(
            TYPE_STATE_SELF_ID,
            ACTOR_SELF,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(2),
            FLAG_NONE,
        )
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_OYA,
            ACTOR_TOIMEN,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_JIKAZE,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(tile("W")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_BAKAZE,
            ACTOR_NONE,
            ACTOR_NONE,
            protocol_tile(tile("S")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_RIICHI_STICKS,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(2),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_DORA,
            ACTOR_NONE,
            ACTOR_NONE,
            protocol_tile(tile("7s")),
            TILE_NONE,
            TILE_NONE,
            encode_value(1),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_HAND,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(tile("1m")),
            TILE_NONE,
            TILE_NONE,
            encode_value(2),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_HAND,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(tile("5mr")),
            TILE_NONE,
            TILE_NONE,
            encode_value(1),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_DRAWN_TILE,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(tile("5mr")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_RIICHI_DECLARED,
            ACTOR_KAMICHA,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(1),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_LAST_DISCARD,
            ACTOR_NONE,
            ACTOR_NONE,
            protocol_tile(tile("9p")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_LAST_TEDASHI,
            ACTOR_KAMICHA,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
    assert_has_token(
        snapshot,
        token(
            TYPE_STATE_LAST_TEDASHI,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(tile("5pr")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
}

#[test]
fn start_kyoku_initial_tokens_have_expected_fields() {
    let mut table = TableStateMachine::new(true);
    table
        .apply_event(MjaiEvent::StartGame {
            id: Some(2),
            names: Default::default(),
            seed: Some((11, 22)),
        })
        .unwrap();
    table.apply_event(start_kyoku()).unwrap();

    let tokens = &table.players[2].tokens;
    assert_eq!(
        tokens[0],
        token(
            TYPE_EVENT_START_KYOKU,
            ACTOR_NONE,
            ACTOR_TOIMEN,
            protocol_tile(tile("E")),
            protocol_tile(tile("2p")),
            TILE_NONE,
            encode_value(1),
            FLAG_NONE,
        )
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_BAKAZE,
            ACTOR_NONE,
            ACTOR_NONE,
            protocol_tile(tile("E")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_JIKAZE,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(tile("W")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_OYA,
            ACTOR_TOIMEN,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_KYOKU_INDEX,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(1),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_HONBA,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(0),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_KYOTAKU,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(0),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_SCORE,
            ACTOR_SELF,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(5),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_LEFT_TILES,
            ACTOR_NONE,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            encode_value(70 / 4),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_DORA,
            ACTOR_NONE,
            ACTOR_NONE,
            protocol_tile(tile("2p")),
            TILE_NONE,
            TILE_NONE,
            encode_value(0),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_HAND,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(tile("1s")),
            TILE_NONE,
            TILE_NONE,
            encode_value(1),
            FLAG_NONE,
        ),
    );
    assert_has_token(
        tokens,
        token(
            TYPE_STATE_HAND,
            ACTOR_TOIMEN,
            ACTOR_NONE,
            protocol_tile(tile("1m")),
            TILE_NONE,
            TILE_NONE,
            encode_value(2),
            FLAG_NONE,
        ),
    );
    assert_eq!(tokens.last().unwrap()[0], TYPE_SEP);
}

#[test]
fn start_game_records_riichi_lab_seat_id() {
    let mut table = TableStateMachine::new(true);
    table
        .apply_player_events(
            2,
            vec![MjaiEvent::StartGame {
                id: Some(2),
                names: Default::default(),
                seed: Some((11, 22)),
            }],
        )
        .unwrap();

    assert_eq!(table.players[2].absolute_seat, 2);
    assert!(table.players[2].tokens.is_empty());
}

#[test]
fn event_tokens_have_expected_fields_for_each_riichi_env_event() {
    let mut table = started_table();

    let draw_self = appended_by_event(
        &mut table,
        MjaiEvent::Tsumo {
            actor: 0,
            pai: tile("5m"),
        },
        0,
    );
    assert_eq!(
        draw_self,
        vec![token(
            TYPE_EVENT_DRAW,
            ACTOR_SELF,
            ACTOR_NONE,
            protocol_tile(tile("5m")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        )]
    );

    let draw_opponent = appended_by_event(
        &mut table,
        MjaiEvent::Tsumo {
            actor: 1,
            pai: tile("7p"),
        },
        0,
    );
    assert_eq!(
        draw_opponent,
        vec![token(
            TYPE_EVENT_DRAW,
            ACTOR_SHIMOCHA,
            ACTOR_NONE,
            TILE_UNKNOWN,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        )]
    );

    let discard = appended_by_event(
        &mut table,
        MjaiEvent::Dahai {
            actor: 1,
            pai: tile("7p"),
            tsumogiri: true,
        },
        0,
    );
    assert_eq!(
        discard,
        vec![token(
            TYPE_EVENT_DISCARD,
            ACTOR_SHIMOCHA,
            ACTOR_NONE,
            protocol_tile(tile("7p")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_TSUMOGIRI,
        )]
    );

    let chi = appended_by_event(
        &mut table,
        MjaiEvent::Chi {
            actor: 1,
            target: 0,
            pai: tile("3p"),
            consumed: [tile("1p"), tile("2p")],
        },
        0,
    );
    assert_eq!(
        chi,
        vec![token(
            TYPE_EVENT_CHI,
            ACTOR_SHIMOCHA,
            ACTOR_SELF,
            protocol_tile(tile("3p")),
            protocol_tile(tile("1p")),
            protocol_tile(tile("2p")),
            VALUE_NONE,
            FLAG_CHI_HIGH,
        )]
    );

    let pon = appended_by_event(
        &mut table,
        MjaiEvent::Pon {
            actor: 1,
            target: 0,
            pai: tile("5m"),
            consumed: [tile("5m"), tile("5m")],
        },
        0,
    );
    assert_eq!(
        pon,
        vec![token(
            TYPE_EVENT_PON,
            ACTOR_SHIMOCHA,
            ACTOR_SELF,
            protocol_tile(tile("5m")),
            protocol_tile(tile("5m")),
            protocol_tile(tile("5m")),
            VALUE_NONE,
            FLAG_NONE,
        )]
    );

    let mut table = started_table();
    let daiminkan = appended_by_event(
        &mut table,
        MjaiEvent::Daiminkan {
            actor: 1,
            target: 0,
            pai: tile("5m"),
            consumed: [tile("5m"), tile("5m"), tile("5m")],
        },
        0,
    );
    assert_eq!(
        daiminkan,
        vec![
            token(
                TYPE_EVENT_DAIMINKAN,
                ACTOR_SHIMOCHA,
                ACTOR_SELF,
                protocol_tile(tile("5m")),
                protocol_tile(tile("5m")),
                protocol_tile(tile("5m")),
                VALUE_NONE,
                FLAG_MELD_DAIMINKAN,
            ),
            token(
                TYPE_EVENT_MELD_CONT,
                ACTOR_SHIMOCHA,
                ACTOR_SELF,
                protocol_tile(tile("5m")),
                TILE_NONE,
                TILE_NONE,
                encode_value(0),
                FLAG_MELD_DAIMINKAN,
            ),
        ]
    );

    let mut table = started_table();
    let kakan = appended_by_event(
        &mut table,
        MjaiEvent::Kakan {
            actor: 1,
            pai: tile("5m"),
            consumed: [tile("5m"), tile("5m"), tile("5m")],
        },
        0,
    );
    assert_eq!(
        kakan,
        vec![
            token(
                TYPE_EVENT_KAKAN,
                ACTOR_SHIMOCHA,
                ACTOR_NONE,
                protocol_tile(tile("5m")),
                protocol_tile(tile("5m")),
                protocol_tile(tile("5m")),
                VALUE_NONE,
                FLAG_MELD_KAKAN,
            ),
            token(
                TYPE_EVENT_MELD_CONT,
                ACTOR_SHIMOCHA,
                ACTOR_NONE,
                protocol_tile(tile("5m")),
                TILE_NONE,
                TILE_NONE,
                encode_value(0),
                FLAG_MELD_KAKAN,
            ),
        ]
    );

    let mut table = started_table();
    let ankan = appended_by_event(
        &mut table,
        MjaiEvent::Ankan {
            actor: 1,
            consumed: [tile("1p"), tile("1p"), tile("1p"), tile("1p")],
        },
        0,
    );
    assert_eq!(
        ankan,
        vec![
            token(
                TYPE_EVENT_ANKAN,
                ACTOR_SHIMOCHA,
                ACTOR_NONE,
                protocol_tile(tile("1p")),
                protocol_tile(tile("1p")),
                protocol_tile(tile("1p")),
                VALUE_NONE,
                FLAG_MELD_ANKAN,
            ),
            token(
                TYPE_EVENT_MELD_CONT,
                ACTOR_SHIMOCHA,
                ACTOR_NONE,
                protocol_tile(tile("1p")),
                TILE_NONE,
                TILE_NONE,
                encode_value(0),
                FLAG_MELD_ANKAN,
            ),
        ]
    );

    let mut table = started_table();
    let first_dora = appended_by_event(
        &mut table,
        MjaiEvent::Dora {
            dora_marker: tile("7s"),
        },
        0,
    );
    assert_eq!(
        first_dora,
        vec![token(
            TYPE_EVENT_DORA,
            ACTOR_NONE,
            ACTOR_NONE,
            protocol_tile(tile("7s")),
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        )]
    );
    let second_dora = appended_by_event(
        &mut table,
        MjaiEvent::Dora {
            dora_marker: tile("8s"),
        },
        0,
    );
    assert_eq!(second_dora[0][6], VALUE_NONE);

    let mut table = started_table();
    let reach = appended_by_event(&mut table, MjaiEvent::Reach { actor: 0 }, 0);
    assert_eq!(
        reach,
        vec![token(
            TYPE_EVENT_REACH,
            ACTOR_SELF,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_REACH_DECLARE,
        )]
    );

    let reach_accepted = appended_by_event(&mut table, MjaiEvent::ReachAccepted { actor: 0 }, 0);
    assert_eq!(
        reach_accepted,
        vec![token(
            TYPE_EVENT_REACH_ACCEPTED,
            ACTOR_SELF,
            ACTOR_NONE,
            TILE_NONE,
            TILE_NONE,
            TILE_NONE,
            VALUE_NONE,
            FLAG_NONE,
        )]
    );

    let mut table = started_table();
    let hora = appended_by_event(
        &mut table,
        MjaiEvent::Hora {
            actor: 1,
            target: 0,
            deltas: Some([-8_000, 8_000, 0, 0]),
            ura_markers: Some(vec![tile("3m")]),
        },
        0,
    );
    assert_eq!(
        hora,
        vec![
            token(
                TYPE_EVENT_HORA,
                ACTOR_SHIMOCHA,
                ACTOR_SELF,
                TILE_NONE,
                TILE_NONE,
                TILE_NONE,
                VALUE_NONE,
                FLAG_RON,
            ),
            token(
                TYPE_EVENT_SCORE_DELTA,
                ACTOR_SELF,
                ACTOR_NONE,
                TILE_NONE,
                TILE_NONE,
                TILE_NONE,
                encode_value(8),
                FLAG_DELTA_NEGATIVE,
            ),
            token(
                TYPE_EVENT_SCORE_DELTA,
                ACTOR_SHIMOCHA,
                ACTOR_NONE,
                TILE_NONE,
                TILE_NONE,
                TILE_NONE,
                encode_value(8),
                FLAG_DELTA_POSITIVE,
            ),
            token(
                TYPE_EVENT_SCORE_DELTA,
                ACTOR_TOIMEN,
                ACTOR_NONE,
                TILE_NONE,
                TILE_NONE,
                TILE_NONE,
                encode_value(0),
                FLAG_DELTA_ZERO,
            ),
            token(
                TYPE_EVENT_SCORE_DELTA,
                ACTOR_KAMICHA,
                ACTOR_NONE,
                TILE_NONE,
                TILE_NONE,
                TILE_NONE,
                encode_value(0),
                FLAG_DELTA_ZERO,
            ),
        ]
    );

    let mut table = started_table();
    let self_tsumo_hora = appended_by_event(
        &mut table,
        MjaiEvent::Hora {
            actor: 0,
            target: 0,
            deltas: Some([12_000, -4_000, -4_000, -4_000]),
            ura_markers: Some(vec![tile("3m")]),
        },
        0,
    );
    assert_eq!(self_tsumo_hora[0][1], ACTOR_SELF);
    assert_eq!(self_tsumo_hora[0][2], ACTOR_NONE);
    assert_eq!(self_tsumo_hora[0][7], FLAG_TSUMO);
    assert_eq!(self_tsumo_hora.last().unwrap()[0], TYPE_EVENT_URA_DORA);
    assert_eq!(
        self_tsumo_hora.last().unwrap()[3],
        protocol_tile(tile("3m"))
    );

    let mut table = started_table();
    let ryukyoku = appended_by_event(
        &mut table,
        MjaiEvent::Ryukyoku {
            deltas: Some([1_500, 1_500, -1_500, -1_500]),
        },
        0,
    );
    assert_eq!(ryukyoku[0][0], TYPE_EVENT_RYUKYOKU);
    assert_eq!(ryukyoku[1][0], TYPE_EVENT_SCORE_DELTA);
    assert_eq!(ryukyoku[1][1], ACTOR_SELF);
    assert_eq!(ryukyoku[1][6], encode_value(1));
    assert_eq!(ryukyoku[1][7], FLAG_DELTA_POSITIVE);
    assert_eq!(ryukyoku[3][1], ACTOR_TOIMEN);
    assert_eq!(ryukyoku[3][7], FLAG_DELTA_NEGATIVE);
}

#[test]
fn player_events_and_legal_actions_generate_mask_for_that_player() {
    let mut manager = MjaiKyokuStateMachineManager::new(1, true).unwrap();
    manager
            .apply_player_events(
                2,
                vec![
                    r#"{"type":"start_game","id":2}"#.to_owned(),
                    r#"{"type":"start_kyoku","bakaze":"E","dora_marker":"2p","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]]}"#.to_owned(),
                    r#"{"type":"tsumo","actor":2,"pai":"5s"}"#.to_owned(),
                ],
            )
            .unwrap();
    manager
        .set_legal_actions(
            2,
            vec![r#"{"type":"dahai","pai":"5s","tsumogiri":true}"#.to_owned()],
        )
        .unwrap();

    let mask_player_zero = manager.tables[0].action_mask(0);
    let mask_player_two = manager.tables[0].action_mask(2);
    assert!(!mask_player_zero[46]);
    assert!(mask_player_two[46]);
}

#[test]
fn player_events_cover_all_riichi_env_4p_event_tokens() {
    let mut manager = MjaiKyokuStateMachineManager::new(1, true).unwrap();
    manager
            .apply_player_events(
                0,
                vec![
                    r#"{"type":"start_game","id":0}"#.to_owned(),
                    r#"{"type":"start_kyoku","bakaze":"E","dora_marker":"2p","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","1m","2m","3m","4m","5mr","6m","7p","8p","9p","E","S","C"],["5m","5m","5m","1p","1p","1p","1p","2p","2p","2p","2p","3p","3p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","P","F","C","E"]]}"#.to_owned(),
                    r#"{"type":"tsumo","actor":0,"pai":"5m"}"#.to_owned(),
                    r#"{"type":"dahai","actor":0,"pai":"5m","tsumogiri":true}"#.to_owned(),
                    r#"{"type":"chi","actor":1,"target":0,"pai":"3p","consumed":["1p","2p"]}"#.to_owned(),
                    r#"{"type":"pon","actor":2,"target":1,"pai":"E","consumed":["E","E"]}"#.to_owned(),
                    r#"{"type":"daiminkan","actor":1,"target":0,"pai":"5m","consumed":["5m","5m","5m"]}"#.to_owned(),
                    r#"{"type":"ankan","actor":2,"consumed":["1s","1s","1s","1s"]}"#.to_owned(),
                    r#"{"type":"kakan","actor":1,"pai":"5m","consumed":["5m","5m","5m"]}"#.to_owned(),
                    r#"{"type":"dora","dora_marker":"7s"}"#.to_owned(),
                    r#"{"type":"reach","actor":3}"#.to_owned(),
                    r#"{"type":"reach_accepted","actor":3}"#.to_owned(),
                    r#"{"type":"hora","actor":0,"target":0,"deltas":[12000,-4000,-4000,-4000],"ura_markers":["3m"]}"#.to_owned(),
                    r#"{"type":"ryukyoku","deltas":[1500,1500,-1500,-1500]}"#.to_owned(),
                    r#"{"type":"end_kyoku"}"#.to_owned(),
                    r#"{"type":"end_game"}"#.to_owned(),
                ],
            )
            .unwrap();

    let tokens = &manager.tables[0].players[0].tokens;
    for token_type in [
        TYPE_EVENT_START_KYOKU,
        TYPE_EVENT_DRAW,
        TYPE_EVENT_DISCARD,
        TYPE_EVENT_CHI,
        TYPE_EVENT_PON,
        TYPE_EVENT_DAIMINKAN,
        TYPE_EVENT_ANKAN,
        TYPE_EVENT_KAKAN,
        TYPE_EVENT_DORA,
        TYPE_EVENT_REACH,
        TYPE_EVENT_REACH_ACCEPTED,
        TYPE_EVENT_HORA,
        TYPE_EVENT_RYUKYOKU,
        TYPE_EVENT_SCORE_DELTA,
        TYPE_EVENT_URA_DORA,
        TYPE_EVENT_MELD_CONT,
    ] {
        assert!(
            tokens.iter().any(|token| token[0] == token_type),
            "missing token type {token_type}"
        );
    }

    assert_eq!(manager.tables[0].players[0].absolute_seat, 0);
    assert_eq!(tokens[0][0], TYPE_EVENT_START_KYOKU);
    assert_eq!(tokens.last().unwrap()[0], TYPE_EVENT_SCORE_DELTA);
    assert!(!tokens.iter().any(|token| token[0] == TYPE_PAD));
}

#[test]
fn end_boundaries_do_not_append_model_tokens() {
    let mut manager = MjaiKyokuStateMachineManager::new(1, true).unwrap();
    manager
            .apply_player_events(
                0,
                vec![
                    r#"{"type":"start_game","id":0}"#.to_owned(),
                    r#"{"type":"start_kyoku","bakaze":"E","dora_marker":"2p","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","1m","2m","3m","4m","5mr","6m","7p","8p","9p","E","S","C"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]]}"#.to_owned(),
                ],
            )
            .unwrap();
    let before = manager.tables[0].players[0].tokens.len();
    manager
        .apply_player_events(
            0,
            vec![
                r#"{"type":"end_kyoku"}"#.to_owned(),
                r#"{"type":"end_game"}"#.to_owned(),
            ],
        )
        .unwrap();
    assert_eq!(manager.tables[0].players[0].tokens.len(), before);
}

#[test]
fn rare_terminal_events_have_expected_token_fields() {
    let mut table = TableStateMachine::new(true);
    table.apply_event(start_kyoku()).unwrap();
    table.apply_event(MjaiEvent::Reach { actor: 0 }).unwrap();
    table
        .apply_event(MjaiEvent::ReachAccepted { actor: 0 })
        .unwrap();
    table
        .apply_event(MjaiEvent::Hora {
            actor: 0,
            target: 0,
            deltas: Some([12_000, -4_000, -4_000, -4_000]),
            ura_markers: Some(vec![tile("3m")]),
        })
        .unwrap();
    table
        .apply_event(MjaiEvent::Ryukyoku {
            deltas: Some([1_500, 1_500, -1_500, -1_500]),
        })
        .unwrap();

    let tokens = &table.players[0].tokens;
    let reach = token_of_type(tokens, TYPE_EVENT_REACH);
    assert_eq!(reach[1], ACTOR_SELF);
    assert_eq!(reach[7], FLAG_REACH_DECLARE);

    let reach_accepted = token_of_type(tokens, TYPE_EVENT_REACH_ACCEPTED);
    assert_eq!(reach_accepted[1], ACTOR_SELF);

    let hora = token_of_type(tokens, TYPE_EVENT_HORA);
    assert_eq!(hora[1], ACTOR_SELF);
    assert_eq!(hora[2], ACTOR_NONE);
    assert_eq!(hora[7], FLAG_TSUMO);

    assert!(tokens.iter().any(|token| {
        token[0] == TYPE_EVENT_SCORE_DELTA
            && token[1] == ACTOR_SELF
            && token[7] == FLAG_DELTA_POSITIVE
    }));
    assert!(tokens.iter().any(|token| {
        token[0] == TYPE_EVENT_SCORE_DELTA
            && token[1] == ACTOR_SHIMOCHA
            && token[7] == FLAG_DELTA_NEGATIVE
    }));
    assert!(tokens
        .iter()
        .any(|token| { token[0] == TYPE_EVENT_URA_DORA && token[3] == protocol_tile(tile("3m")) }));
    assert!(tokens.iter().any(|token| token[0] == TYPE_EVENT_RYUKYOKU));
}

#[test]
fn set_legal_actions_covers_all_4p_action_types() {
    let mut table = TableStateMachine::new(true);
    table
        .set_legal_actions(
            0,
            vec![
                r#"{"type":"none"}"#.to_owned(),
                r#"{"type":"dahai","pai":"5m","tsumogiri":false}"#.to_owned(),
                r#"{"type":"dahai","pai":"5m","tsumogiri":true}"#.to_owned(),
                r#"{"type":"reach"}"#.to_owned(),
                r#"{"type":"chi","pai":"3p","consumed":["1p","2p"]}"#.to_owned(),
                r#"{"type":"pon","pai":"5m","consumed":["5m","5mr"]}"#.to_owned(),
                r#"{"type":"daiminkan","pai":"E","consumed":["E","E","E"]}"#.to_owned(),
                r#"{"type":"ankan","consumed":["1s","1s","1s","1s"]}"#.to_owned(),
                r#"{"type":"kakan","pai":"5m","consumed":["5m","5m","5m"]}"#.to_owned(),
                r#"{"type":"hora"}"#.to_owned(),
                r#"{"type":"ryukyoku"}"#.to_owned(),
            ],
        )
        .unwrap();

    let mask = table.action_mask(0);
    assert_exact_mask(&mask, &[0, 9, 10, 75, 95, 165, 170, 189, 209, 239, 240]);
}

#[test]
fn legal_action_mask_and_model_action_json_are_semantically_exact() {
    let mut table = TableStateMachine::new(true);
    let legal_actions = vec![
        (0, serde_json::json!({"type":"none","request_id":11})),
        (
            9,
            serde_json::json!({"type":"dahai","actor":0,"pai":"5m","tsumogiri":false,"request_id":11}),
        ),
        (
            10,
            serde_json::json!({"type":"dahai","actor":0,"pai":"5m","tsumogiri":true,"request_id":11}),
        ),
        (
            75,
            serde_json::json!({"type":"reach","actor":0,"request_id":11}),
        ),
        (
            95,
            serde_json::json!({"type":"chi","actor":0,"target":3,"pai":"3p","consumed":["1p","2p"],"request_id":11}),
        ),
        (
            165,
            serde_json::json!({"type":"pon","actor":0,"target":2,"pai":"5m","consumed":["5m","5mr"],"request_id":11}),
        ),
        (
            170,
            serde_json::json!({"type":"daiminkan","actor":0,"target":2,"pai":"E","consumed":["E","E","E"],"request_id":11}),
        ),
        (
            189,
            serde_json::json!({"type":"ankan","actor":0,"consumed":["1s","1s","1s","1s"],"request_id":11}),
        ),
        (
            209,
            serde_json::json!({"type":"kakan","actor":0,"pai":"5m","consumed":["5m","5m","5m"],"request_id":11}),
        ),
        (
            239,
            serde_json::json!({"type":"hora","actor":0,"target":2,"pai":"5p","request_id":11}),
        ),
        (
            240,
            serde_json::json!({"type":"ryukyoku","actor":0,"request_id":11}),
        ),
    ];
    table
        .set_legal_actions(
            0,
            legal_actions
                .iter()
                .map(|(_, action)| action.to_string())
                .collect(),
        )
        .unwrap();

    let expected_ids = legal_actions
        .iter()
        .map(|(action_id, _)| *action_id)
        .collect::<Vec<_>>();
    assert_exact_mask(&table.action_mask(0), &expected_ids);

    for (action_id, expected) in legal_actions {
        assert_eq!(requested_json(&table, 0, action_id), expected);
    }
}

#[test]
fn set_legal_actions_rejects_unmapped_or_3p_only_actions() {
    let mut table = TableStateMachine::new(true);
    assert!(table
        .set_legal_actions(0, vec![r#"{"type":"kita","pai":"N"}"#.to_owned()])
        .unwrap_err()
        .contains("unsupported"));
    assert!(table
        .set_legal_actions(
            0,
            vec![r#"{"type":"chi","pai":"5m","consumed":["E","S"]}"#.to_owned()],
        )
        .unwrap_err()
        .contains("cannot be mapped"));
}

#[test]
fn chi_and_pon_template_tables_cover_every_fixed_action_id() {
    let mut table = TableStateMachine::new(true);
    let mut actions = Vec::new();
    for index in 0..57 {
        let consumed = chi_consumed_pair(index).unwrap();
        actions.push(
            serde_json::json!({
                "type": "chi",
                "pai": "5m",
                "consumed": [tile_name(consumed[0]).unwrap(), tile_name(consumed[1]).unwrap()],
            })
            .to_string(),
        );
    }
    for index in 0..37 {
        let consumed = pon_consumed_pair(index).unwrap();
        let pai = consumed[0].deaka();
        actions.push(
            serde_json::json!({
                "type": "pon",
                "pai": tile_name(pai).unwrap(),
                "consumed": [tile_name(consumed[0]).unwrap(), tile_name(consumed[1]).unwrap()],
            })
            .to_string(),
        );
    }
    table.set_legal_actions(0, actions).unwrap();

    let mask = table.action_mask(0);
    for action_id in 76..=132 {
        assert!(mask[action_id], "missing chi action_id {action_id}");
    }
    for action_id in 133..=169 {
        assert!(mask[action_id], "missing pon action_id {action_id}");
    }
}

#[test]
fn model_action_to_mjai_reuses_current_riichi_env_legal_action_json() {
    let mut manager = MjaiKyokuStateMachineManager::new(1, true).unwrap();
    manager
        .set_legal_actions(
            0,
            vec![r#"{"type":"hora","actor":0,"target":2,"pai":"5p"}"#.to_owned()],
        )
        .unwrap();

    let response = manager.tables[0]
        .requested_action_to_env(0, 239)
        .unwrap()
        .unwrap();
    let response: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(
        response,
        serde_json::json!({"type":"hora","actor":0,"target":2,"pai":"5p"})
    );

    assert!(manager.tables[0].requested_action_to_env(0, 75).is_err());
}

#[test]
fn start_kyoku_can_reveal_or_hide_opponent_initial_hands() {
    let mut revealed = TableStateMachine::new(true);
    revealed.apply_event(start_kyoku()).unwrap();

    let player_zero_tokens = &revealed.players[0].tokens;
    assert!(player_zero_tokens.iter().any(|token| {
        token[0] == TYPE_STATE_HAND
            && token[1] == ACTOR_SHIMOCHA
            && token[3] == protocol_tile(tile("5m"))
            && token[6] == encode_value(3)
    }));

    let mut hidden = TableStateMachine::new(false);
    hidden.apply_event(start_kyoku()).unwrap();

    let hidden_opponent_hands = hidden.players[0]
        .tokens
        .iter()
        .filter(|token| token[0] == TYPE_STATE_HAND && token[1] != ACTOR_SELF)
        .collect::<Vec<_>>();
    assert_eq!(hidden_opponent_hands.len(), 3);
    assert!(hidden_opponent_hands
        .iter()
        .all(|token| { token[3] == TILE_UNKNOWN && token[6] == encode_value(13) }));
}

#[test]
fn daiminkan_uses_a_continuation_token() {
    let mut table = TableStateMachine::new(true);
    table.apply_event(start_kyoku()).unwrap();
    let length_before = table.players[0].tokens.len();
    table
        .apply_event(MjaiEvent::Daiminkan {
            actor: 1,
            target: 0,
            pai: tile("5mr"),
            consumed: [tile("5m"), tile("5m"), tile("5m")],
        })
        .unwrap();
    let appended = &table.players[0].tokens[length_before..];
    assert_eq!(appended.len(), 2);
    assert_eq!(appended[0][0], TYPE_EVENT_DAIMINKAN);
    assert_eq!(appended[1][0], TYPE_EVENT_MELD_CONT);
    assert_eq!(appended[1][3], protocol_tile(tile("5m")));
}
