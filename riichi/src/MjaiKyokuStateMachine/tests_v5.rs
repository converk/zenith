use super::*;

fn snapshot() -> DecisionSnapshot {
    DecisionSnapshot {
        player_id: 0, oya: 0, round_wind: 0, kyoku_index: 0, honba: 0,
        riichi_sticks: 0, scores: [25_000; NUM_PLAYERS], dora_indicators: vec!["2p".to_owned()],
        hand: vec!["1m".to_owned(); 13], drawn_tile: None, riichi_declared: [false; NUM_PLAYERS], decision_flags: 0,
    }
}

#[test]
fn history_uses_semantic_events_and_omits_draw_and_terminal_events() {
    let mut player = PlayerKyokuStateMachine::new(0);
    player.apply_player_event(&parse_event(r#"{"type":"start_kyoku","bakaze":"E","dora_marker":"2p","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m"],["2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m"],["3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m"],["4m","4m","4m","4m","4m","4m","4m","4m","4m","4m","4m","4m","4m"]]}"#).unwrap()).unwrap();
    player.apply_event(&MjaiEvent::Tsumo { actor: 0, pai: MjaiTile(0) });
    player.apply_event(&MjaiEvent::Reach { actor: 1 });
    player.apply_event(&MjaiEvent::Dahai { actor: 1, pai: MjaiTile(34), tsumogiri: true });
    player.apply_event(&MjaiEvent::Hora { actor: 1, target: 0, deltas: None, ura_markers: None });
    let history = &player.history;
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].factors[2], 2);
    assert_eq!(history[1].factors[2], 11);
    assert_eq!(history[2].factors[2], 4);
    assert_eq!(history[2].factors[3], ACTOR_SHIMOCHA);
    assert_eq!(history[2].factors[6], 1); // red-five factor
    assert_eq!(history[2].factors[8], 2); // tsumogiri detail
    assert_eq!(player.live_wall, 69);
}

#[test]
fn state_has_own_hand_draw_and_masks_but_no_river_or_meld_tokens() {
    let mut player = PlayerKyokuStateMachine::new(0);
    player.start_kyoku(MjaiTile(27), 1);
    let mut current = snapshot();
    current.hand = vec!["5mr".to_owned(), "5m".to_owned(), "1m".to_owned()];
    current.drawn_tile = Some("5mr".to_owned());
    let tokens = player.tokens(&current).unwrap();
    assert!(tokens.iter().any(|row| row.factors[1] == KIND_TILE_COUNT && row.factors[2] == 1 && row.factors[6] == 1 && row.factors[7] == 2));
    assert!(tokens.iter().any(|row| row.factors[1] == KIND_TILE_COUNT && row.factors[2] == 5 && row.factors[6] == 1));
    assert_eq!(tokens.iter().filter(|row| row.factors[1] == KIND_MASKED).count(), 3);
    assert!(!tokens.iter().any(|row| matches!(row.factors[1], 5 | 6))); // no state river/meld token kinds
}

#[test]
fn new_kyoku_discards_old_history() {
    let mut player = PlayerKyokuStateMachine::new(0);
    player.start_kyoku(MjaiTile(27), 1);
    player.apply_event(&MjaiEvent::Dahai { actor: 0, pai: MjaiTile(0), tsumogiri: false });
    player.start_kyoku(MjaiTile(28), 2);
    assert_eq!(player.history.len(), 1);
    assert_eq!(player.history[0].factors[2], 2);
    assert_eq!(player.history[0].factors[8], 6); // south / hand 2
}
