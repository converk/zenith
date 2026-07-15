use super::*;

#[test]
fn pon_is_one_meld_block_with_relative_seats() {
    let mut player = PlayerKyokuStateMachine::new(0);
    player.start_kyoku();
    player
        .apply_event(&MjaiEvent::Pon {
            actor: 3,
            target: 2,
            pai: MjaiTile(11), // 3p
            consumed: [MjaiTile(11), MjaiTile(11)],
        })
        .unwrap();
    assert_eq!(player.blocks.len(), 1);
    match &player.blocks[0] {
        EventBlock::Meld {
            meld_kind,
            actor,
            target,
            pai,
            tiles,
        } => {
            assert_eq!(
                (*meld_kind, *actor, *target, *pai),
                (MELD_PON, ACTOR_KAMICHA, ACTOR_TOIMEN, 12)
            );
            assert_eq!(*tiles, [12, 12, 0, 0]);
        }
        EventBlock::Turn(_) => panic!("pon must be a meld block"),
    }
}

#[test]
fn draw_updates_wall_without_a_history_block() {
    let mut player = PlayerKyokuStateMachine::new(0);
    player.start_kyoku();
    player
        .apply_event(&MjaiEvent::Tsumo {
            actor: 1,
            pai: MjaiTile(0),
        })
        .unwrap();
    assert_eq!(player.live_wall, 69);
    assert!(player.blocks.is_empty());
}

#[test]
fn every_four_player_mjai_event_variant_parses_and_applies() {
    let events = [
        r#"{"type":"start_game","id":0,"names":["a","b","c","d"]}"#,
        r#"{"type":"start_kyoku","bakaze":"E","dora_marker":"2p","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m","1m"],["2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m","2m"],["3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m","3m"],["4m","4m","4m","4m","4m","4m","4m","4m","4m","4m","4m","4m","4m"]]}"#,
        r#"{"type":"tsumo","actor":0,"pai":"5mr"}"#,
        r#"{"type":"reach","actor":0}"#,
        r#"{"type":"dahai","actor":0,"pai":"5mr","tsumogiri":true}"#,
        r#"{"type":"reach_accepted","actor":0}"#,
        r#"{"type":"chi","actor":1,"target":0,"pai":"3m","consumed":["4m","5mr"]}"#,
        r#"{"type":"pon","actor":2,"target":1,"pai":"P","consumed":["P","P"]}"#,
        r#"{"type":"daiminkan","actor":3,"target":2,"pai":"9s","consumed":["9s","9s","9s"]}"#,
        r#"{"type":"kakan","actor":3,"pai":"5pr","consumed":["5p","5p","5pr"]}"#,
        r#"{"type":"ankan","actor":2,"consumed":["5s","5s","5s","5sr"]}"#,
        r#"{"type":"dora","dora_marker":"C"}"#,
        r#"{"type":"hora","actor":1,"target":0,"deltas":[-8000,8000,0,0],"ura_markers":["1m"]}"#,
        r#"{"type":"ryukyoku","deltas":[0,0,0,0]}"#,
        r#"{"type":"end_kyoku"}"#,
        r#"{"type":"end_game"}"#,
    ];
    let mut player = PlayerKyokuStateMachine::new(0);
    for raw in events {
        let event = parse_event(raw).unwrap_or_else(|error| panic!("{raw}: {error}"));
        player.apply_player_event(&event).unwrap();
    }
    assert_eq!(player.live_wall, 69);
    assert!(player
        .blocks
        .iter()
        .any(|block| matches!(block, EventBlock::Turn(_))));
    assert_eq!(
        player
            .blocks
            .iter()
            .filter(|block| matches!(block, EventBlock::Meld { .. }))
            .count(),
        5
    );
}

#[test]
fn board_preserves_red_called_river_and_meld_source() {
    let mut player = PlayerKyokuStateMachine::new(0);
    player.start_kyoku();
    player
        .apply_event(&MjaiEvent::Dahai {
            actor: 1,
            pai: MjaiTile(34),
            tsumogiri: true,
        })
        .unwrap();
    player
        .apply_event(&MjaiEvent::Chi {
            actor: 2,
            target: 1,
            pai: MjaiTile(2),
            consumed: [MjaiTile(3), MjaiTile(34)],
        })
        .unwrap();
    let snapshot = DecisionSnapshot {
        player_id: 0,
        oya: 0,
        round_wind: 0,
        kyoku_index: 0,
        honba: 0,
        riichi_sticks: 0,
        scores: [25_000; NUM_PLAYERS],
        dora_indicators: vec!["2p".to_owned()],
        hand: vec!["1m".to_owned(); 13],
        drawn_tile: None,
        riichi_declared: [false; NUM_PLAYERS],
        last_discard: Some("5mr".to_owned()),
        last_tedashis: [None, None, None, None],
    };
    let board = player.board_tokens(&snapshot).unwrap();
    // Shimocha river: red five was tsumogiri and then called.
    assert_eq!(board[4][20], 35);
    assert_eq!(board[4][52], FLAG_TSUMOGIRI | FLAG_CALLED);
    // Toimen meld: chi source is shimocha and red consumed tile is retained.
    assert_eq!(
        &board[8][20..27],
        &[MELD_CHI, ACTOR_SHIMOCHA, 3, 4, 35, 0, 0]
    );
}

#[test]
fn pending_reach_discard_marks_the_turn_micro_event() {
    let mut player = PlayerKyokuStateMachine::new(0);
    player.start_kyoku();
    player.apply_event(&MjaiEvent::Reach { actor: 1 }).unwrap();
    player
        .apply_event(&MjaiEvent::Dahai {
            actor: 1,
            pai: MjaiTile(0),
            tsumogiri: false,
        })
        .unwrap();
    let snapshot = DecisionSnapshot {
        player_id: 0,
        oya: 0,
        round_wind: 0,
        kyoku_index: 0,
        honba: 0,
        riichi_sticks: 0,
        scores: [25_000; NUM_PLAYERS],
        dora_indicators: vec!["2p".to_owned()],
        hand: vec!["1m".to_owned(); 13],
        drawn_tile: None,
        riichi_declared: [false; NUM_PLAYERS],
        last_discard: Some("1m".to_owned()),
        last_tedashis: [None, None, None, None],
    };
    let board = player.board_tokens(&snapshot).unwrap();
    assert_eq!(
        &board[0][120..128],
        &[
            MICRO_REACH,
            ACTOR_SHIMOCHA,
            0,
            0,
            MICRO_DISCARD,
            ACTOR_SHIMOCHA,
            1,
            FLAG_RIICHI_DISCARD
        ]
    );
}
