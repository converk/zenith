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
                    tile("1m"), tile("1m"), tile("2m"), tile("3m"), tile("4m"), tile("5mr"),
                    tile("6m"), tile("7p"), tile("8p"), tile("9p"), tile("E"), tile("S"), tile("C"),
                ],
                [
                    tile("5m"), tile("5m"), tile("5m"), tile("1p"), tile("1p"), tile("1p"),
                    tile("1p"), tile("2p"), tile("2p"), tile("2p"), tile("2p"), tile("3p"), tile("3p"),
                ],
                [
                    tile("1s"), tile("2s"), tile("3s"), tile("4s"), tile("5s"), tile("6s"),
                    tile("7s"), tile("8s"), tile("9s"), tile("E"), tile("S"), tile("W"), tile("N"),
                ],
                [
                    tile("1m"), tile("2m"), tile("3m"), tile("4m"), tile("5m"), tile("6m"),
                    tile("7m"), tile("8m"), tile("9m"), tile("P"), tile("F"), tile("C"), tile("E"),
                ],
            ],
        }
    }

    fn decoded_action(
        table: &TableStateMachine,
        player_index: u8,
        action_id: usize,
    ) -> serde_json::Value {
        serde_json::from_str(&table.action_to_mjai(player_index, action_id).unwrap()).unwrap()
    }

    #[test]
    fn starts_from_kyoku_and_appends_events_per_player_view() {
        let mut table = TableStateMachine::new(GameMode::FourPlayerEast, true);
        table.apply_event(MjaiEvent::StartGame {
            names: Default::default(),
            seed: None,
        }).unwrap();
        table.apply_event(start_kyoku()).unwrap();

        let initial_len = table.players[0].tokens.len();
        assert_eq!(table.players[0].tokens[0][0], TYPE_EVENT_START_KYOKU);
        assert_eq!(table.players[0].tokens.last().unwrap()[0], TYPE_SEP);
        assert_eq!(table.players[0].tokens[0][2], ACTOR_SELF);

        table.apply_event(MjaiEvent::Tsumo {
            actor: 0,
            pai: tile("5m"),
        }).unwrap();
        assert_eq!(table.players[0].tokens.len(), initial_len + 1);
        assert_eq!(table.players[0].tokens.last().unwrap()[0], TYPE_EVENT_DRAW);
        assert_eq!(table.players[0].tokens.last().unwrap()[3], protocol_tile(tile("5m")));
        assert_eq!(table.players[1].tokens.last().unwrap()[3], TILE_UNKNOWN);

        let len_before_dora = table.players[0].tokens.len();
        table.apply_event(MjaiEvent::Dora { dora_marker: tile("7s") }).unwrap();
        assert_eq!(table.players[0].tokens.len(), len_before_dora + 1);
        assert_eq!(table.players[0].tokens.last().unwrap()[0], TYPE_EVENT_DORA);
        assert_eq!(table.players[0].tokens[initial_len - 2][0], TYPE_STATE_HAND);
    }

    #[test]
    fn start_kyoku_can_reveal_or_hide_opponent_initial_hands() {
        let mut revealed = TableStateMachine::new(GameMode::FourPlayerEast, true);
        revealed.apply_event(start_kyoku()).unwrap();

        let player_zero_tokens = &revealed.players[0].tokens;
        assert!(player_zero_tokens.iter().any(|token| {
            token[0] == TYPE_STATE_HAND
                && token[1] == ACTOR_SHIMOCHA
                && token[3] == protocol_tile(tile("5m"))
                && token[6] == encode_value(3)
        }));

        let mut hidden = TableStateMachine::new(GameMode::FourPlayerEast, false);
        hidden.apply_event(start_kyoku()).unwrap();

        let hidden_opponent_hands = hidden.players[0]
            .tokens
            .iter()
            .filter(|token| token[0] == TYPE_STATE_HAND && token[1] != ACTOR_SELF)
            .collect::<Vec<_>>();
        assert_eq!(hidden_opponent_hands.len(), 3);
        assert!(hidden_opponent_hands.iter().all(|token| {
            token[3] == TILE_UNKNOWN && token[6] == encode_value(13)
        }));
    }

    #[test]
    fn daiminkan_uses_a_continuation_token() {
        let mut table = TableStateMachine::new(GameMode::FourPlayerEast, true);
        table.apply_event(start_kyoku()).unwrap();
        let length_before = table.players[0].tokens.len();
        table.apply_event(MjaiEvent::Daiminkan {
            actor: 1,
            target: 0,
            pai: tile("5mr"),
            consumed: [tile("5m"), tile("5m"), tile("5m")],
        }).unwrap();
        let appended = &table.players[0].tokens[length_before..];
        assert_eq!(appended.len(), 2);
        assert_eq!(appended[0][0], TYPE_EVENT_DAIMINKAN);
        assert_eq!(appended[1][0], TYPE_EVENT_MELD_CONT);
        assert_eq!(appended[1][3], protocol_tile(tile("5m")));
    }

    #[test]
    fn decodes_self_turn_actions_to_mjai_json() {
        let mut table = TableStateMachine::new(GameMode::FourPlayerEast, true);
        table.apply_event(start_kyoku()).unwrap();
        table
            .apply_event(MjaiEvent::Tsumo {
                actor: 0,
                pai: tile("5m"),
            })
            .unwrap();

        assert_eq!(
            decoded_action(&table, 0, 10),
            serde_json::json!({"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": true})
        );
        assert_eq!(
            decoded_action(&table, 0, 75),
            serde_json::json!({"type": "reach", "actor": 0})
        );
        assert_eq!(
            decoded_action(&table, 0, 239),
            serde_json::json!({"type": "hora", "actor": 0, "target": 0})
        );
    }

    #[test]
    fn decodes_call_and_kan_actions_from_current_state() {
        let mut chi_table = TableStateMachine::new(GameMode::FourPlayerEast, true);
        chi_table.apply_event(start_kyoku()).unwrap();
        chi_table
            .apply_event(MjaiEvent::Tsumo {
                actor: 0,
                pai: tile("3p"),
            })
            .unwrap();
        chi_table
            .apply_event(MjaiEvent::Dahai {
                actor: 0,
                pai: tile("3p"),
                tsumogiri: true,
            })
            .unwrap();
        assert_eq!(
            decoded_action(&chi_table, 1, 95),
            serde_json::json!({
                "type": "chi", "actor": 1, "target": 0, "pai": "3p", "consumed": ["1p", "2p"]
            })
        );

        let mut kan_table = TableStateMachine::new(GameMode::FourPlayerEast, true);
        kan_table.apply_event(start_kyoku()).unwrap();
        kan_table
            .apply_event(MjaiEvent::Dahai {
                actor: 0,
                pai: tile("5mr"),
                tsumogiri: false,
            })
            .unwrap();
        assert_eq!(
            decoded_action(&kan_table, 1, 170),
            serde_json::json!({
                "type": "daiminkan", "actor": 1, "target": 0, "pai": "5mr",
                "consumed": ["5m", "5m", "5m"]
            })
        );
        assert_eq!(
            decoded_action(&kan_table, 1, 164),
            serde_json::json!({
                "type": "pon", "actor": 1, "target": 0, "pai": "5mr", "consumed": ["5m", "5m"]
            })
        );

        kan_table
            .apply_event(MjaiEvent::Pon {
                actor: 1,
                target: 0,
                pai: tile("5mr"),
                consumed: [tile("5m"), tile("5m")],
            })
            .unwrap();
        kan_table
            .apply_event(MjaiEvent::Dahai {
                actor: 1,
                pai: tile("3p"),
                tsumogiri: false,
            })
            .unwrap();
        kan_table
            .apply_event(MjaiEvent::Tsumo {
                actor: 1,
                pai: tile("4p"),
            })
            .unwrap();
        assert_eq!(
            decoded_action(&kan_table, 1, 209),
            serde_json::json!({
                "type": "kakan", "actor": 1, "pai": "5m", "consumed": ["5mr", "5m", "5m"]
            })
        );

        let mut ankan_table = TableStateMachine::new(GameMode::FourPlayerEast, true);
        ankan_table.apply_event(start_kyoku()).unwrap();
        ankan_table
            .apply_event(MjaiEvent::Tsumo {
                actor: 1,
                pai: tile("4p"),
            })
            .unwrap();
        assert_eq!(
            decoded_action(&ankan_table, 1, 180),
            serde_json::json!({
                "type": "ankan", "actor": 1, "consumed": ["1p", "1p", "1p", "1p"]
            })
        );
    }

    #[test]
    fn riichi_env_request_generates_mask_and_request_id_response() {
        let mut table = TableStateMachine::new(GameMode::FourPlayerEast, true);
        table.apply_event(start_kyoku()).unwrap();
        table
            .apply_event(MjaiEvent::Tsumo {
                actor: 0,
                pai: tile("5m"),
            })
            .unwrap();

        table
            .apply_request(
                0,
                RiichiEnvRequestAction {
                    request_id: 42,
                    possible_actions: vec![
                        serde_json::json!({"type": "dahai", "pai": "1m"}),
                        serde_json::json!({"type": "reach"}),
                        serde_json::json!({"type": "hora"}),
                    ],
                },
            )
            .unwrap();

        let mask = table.action_mask(0);
        assert!(mask[1]);
        assert!(mask[75]);
        assert!(mask[239]);
        assert!(!mask[0]);

        let response = table.requested_action_to_env(0, 75).unwrap().unwrap();
        let response: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response, serde_json::json!({"type": "reach", "actor": 0, "request_id": 42}));
    }

    #[test]
    fn riichi_env_request_can_fill_call_target_from_event_context() {
        let mut table = TableStateMachine::new(GameMode::FourPlayerEast, true);
        table.apply_event(start_kyoku()).unwrap();
        table
            .apply_event(MjaiEvent::Tsumo {
                actor: 0,
                pai: tile("3p"),
            })
            .unwrap();
        table
            .apply_event(MjaiEvent::Dahai {
                actor: 0,
                pai: tile("3p"),
                tsumogiri: true,
            })
            .unwrap();

        table
            .apply_request(
                1,
                RiichiEnvRequestAction {
                    request_id: 7,
                    possible_actions: vec![
                        serde_json::json!({"type": "none"}),
                        serde_json::json!({"type": "chi", "pai": "3p", "consumed": ["1p", "2p"]}),
                    ],
                },
            )
            .unwrap();

        let mask = table.action_mask(1);
        assert!(mask[0]);
        assert!(mask[95]);

        let response = table.requested_action_to_env(1, 95).unwrap().unwrap();
        let response: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            response,
            serde_json::json!({
                "type": "chi",
                "actor": 1,
                "target": 0,
                "pai": "3p",
                "consumed": ["1p", "2p"],
                "request_id": 7
            })
        );
    }
