fn parse_event(event_json: &str) -> PyResult<MjaiEvent> {
    let event = serde_json::from_str::<MjaiEvent>(event_json)
        .map_err(|error| PyValueError::new_err(format!("invalid MJAI event: {error}")))?;
    validate_event(&event).map_err(PyValueError::new_err)?;
    Ok(event)
}

fn action_id_from_mjai_value(action: &Value) -> Result<usize, String> {
    let action_type = action
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "possible action must contain a string type field".to_owned())?;
    match action_type {
        "none" => Ok(0),
        "dahai" => {
            let pai = action_tile_field(action, "pai")?;
            let tsumogiri = action
                .get("tsumogiri")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if pai.as_usize() >= 37 {
                return Err("dahai cannot use unknown tile".to_owned());
            }
            Ok(1 + pai.as_usize() * 2 + usize::from(tsumogiri))
        }
        "reach" => Ok(75),
        "chi" => {
            let pai = action_tile_field(action, "pai")?;
            let consumed = action_tiles_field::<2>(action, "consumed")?;
            if !is_valid_chi(pai, &consumed)? {
                return Err("chi pai and consumed tiles must form one suited sequence".to_owned());
            }
            for index in 0..57 {
                if same_tiles_unordered(&chi_consumed_pair(index)?, &consumed) {
                    return Ok(76 + index);
                }
            }
            Err("chi consumed tiles cannot be mapped to the fixed action space".to_owned())
        }
        "pon" => {
            let pai = action_tile_field(action, "pai")?;
            let consumed = action_tiles_field::<2>(action, "consumed")?;
            if !all_same_tile34(pai, &consumed)? {
                return Err("pon pai and consumed tiles must have the same tile type".to_owned());
            }
            for index in 0..37 {
                if same_tiles_unordered(&pon_consumed_pair(index)?, &consumed) {
                    return Ok(133 + index);
                }
            }
            Err("pon consumed tiles cannot be mapped to the fixed action space".to_owned())
        }
        "daiminkan" => {
            let pai = action_tile_field(action, "pai")?;
            let consumed = action_tiles_field::<3>(action, "consumed")?;
            if !all_same_tile34(pai, &consumed)? {
                return Err("daiminkan pai and consumed tiles must have the same tile type".to_owned());
            }
            Ok(170)
        }
        "ankan" => {
            let consumed = action_tiles_field::<4>(action, "consumed")?;
            let tile34 = tile34_from_tile(consumed[0])?;
            if consumed.iter().skip(1).any(|tile| tile34_from_tile(*tile) != Ok(tile34)) {
                return Err("ankan consumed tiles must have the same tile type".to_owned());
            }
            if let Some(pai) = optional_action_tile_field(action, "pai")? {
                if tile34_from_tile(pai)? != tile34 {
                    return Err("ankan pai must match consumed tiles".to_owned());
                }
            }
            Ok(171 + tile34)
        }
        "kakan" => {
            let consumed = action_tiles_field::<3>(action, "consumed")?;
            let tile34 = tile34_from_tile(consumed[0])?;
            if consumed.iter().skip(1).any(|tile| tile34_from_tile(*tile) != Ok(tile34)) {
                return Err("kakan consumed tiles must have the same tile type".to_owned());
            }
            if let Some(pai) = optional_action_tile_field(action, "pai")? {
                if tile34_from_tile(pai)? != tile34 {
                    return Err("kakan pai must match consumed tiles".to_owned());
                }
            }
            Ok(205 + tile34)
        }
        "hora" => Ok(239),
        "ryukyoku" => Ok(240),
        other => Err(format!("unsupported RiichiEnv possible action type {other:?}")),
    }
}

fn action_tile_field(action: &Value, field: &str) -> Result<MjaiTile, String> {
    let value = action
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("possible action requires string field {field:?}"))?;
    tile_from_str(value).ok_or_else(|| format!("invalid MJAI tile {value:?}"))
}

fn optional_action_tile_field(action: &Value, field: &str) -> Result<Option<MjaiTile>, String> {
    match action.get(field) {
        None => Ok(None),
        Some(value) => value
            .as_str()
            .and_then(tile_from_str)
            .map(Some)
            .ok_or_else(|| format!("invalid MJAI tile in optional field {field:?}")),
    }
}

fn action_tiles_field<const N: usize>(action: &Value, field: &str) -> Result<[MjaiTile; N], String> {
    let values = action
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("possible action requires array field {field:?}"))?;
    if values.len() != N {
        return Err(format!("possible action field {field:?} must contain {N} tiles"));
    }
    let mut tiles = [MjaiTile(0); N];
    for (index, value) in values.iter().enumerate() {
        let tile = value
            .as_str()
            .and_then(tile_from_str)
            .ok_or_else(|| format!("invalid MJAI tile in {field:?}"))?;
        tiles[index] = tile;
    }
    Ok(tiles)
}

fn same_tiles_unordered<const N: usize>(left: &[MjaiTile; N], right: &[MjaiTile; N]) -> bool {
    let mut left_ids = left.map(MjaiTile::as_u8);
    let mut right_ids = right.map(MjaiTile::as_u8);
    left_ids.sort_unstable();
    right_ids.sort_unstable();
    left_ids == right_ids
}

fn tile34_from_tile(tile: MjaiTile) -> Result<usize, String> {
    let tile_id = tile.deaka().as_usize();
    if tile_id < 34 {
        Ok(tile_id)
    } else {
        Err("tile cannot be mapped to TILE34".to_owned())
    }
}

fn all_same_tile34<const N: usize>(pai: MjaiTile, consumed: &[MjaiTile; N]) -> Result<bool, String> {
    let tile34 = tile34_from_tile(pai)?;
    consumed.iter().try_fold(true, |same, tile| Ok(same && tile34_from_tile(*tile)? == tile34))
}

fn is_valid_chi(pai: MjaiTile, consumed: &[MjaiTile; 2]) -> Result<bool, String> {
    let mut tile34 = [tile34_from_tile(pai)?, tile34_from_tile(consumed[0])?, tile34_from_tile(consumed[1])?];
    if tile34.iter().any(|tile| *tile >= 27) || tile34.iter().any(|tile| *tile / 9 != tile34[0] / 9) {
        return Ok(false);
    }
    tile34.sort_unstable();
    Ok(tile34[0] + 1 == tile34[1] && tile34[1] + 1 == tile34[2])
}

fn chi_consumed_pair(index: usize) -> Result<[MjaiTile; 2], String> {
    if index >= 57 {
        return Err("CHI index must be in 0..57".to_owned());
    }
    let suit = (index / 19) as u8;
    let local_index = index % 19;
    let normal = |rank: u8| MjaiTile(suit * 9 + rank - 1);
    let red_five = || MjaiTile(34 + suit);
    Ok(match local_index {
        0 => [normal(1), normal(2)],
        1 => [normal(2), normal(3)],
        2 => [normal(3), normal(4)],
        3 => [normal(4), normal(5)],
        4 => [normal(4), red_five()],
        5 => [normal(5), normal(6)],
        6 => [red_five(), normal(6)],
        7 => [normal(6), normal(7)],
        8 => [normal(7), normal(8)],
        9 => [normal(8), normal(9)],
        10 => [normal(1), normal(3)],
        11 => [normal(2), normal(4)],
        12 => [normal(3), normal(5)],
        13 => [normal(3), red_five()],
        14 => [normal(4), normal(6)],
        15 => [normal(5), normal(7)],
        16 => [red_five(), normal(7)],
        17 => [normal(6), normal(8)],
        18 => [normal(7), normal(9)],
        _ => unreachable!("CHI index was checked"),
    })
}

fn pon_consumed_pair(index: usize) -> Result<[MjaiTile; 2], String> {
    if index >= 37 {
        return Err("PON index must be in 0..37".to_owned());
    }
    if index < 31 {
        let mut tile_id = index as u8;
        for five in [4u8, 13, 22] {
            if tile_id >= five {
                tile_id = tile_id.saturating_add(1);
            }
        }
        let tile = MjaiTile(tile_id);
        return Ok([tile, tile]);
    }
    Ok(match index {
        31 => [MjaiTile(4), MjaiTile(4)],
        32 => [MjaiTile(4), MjaiTile(34)],
        33 => [MjaiTile(13), MjaiTile(13)],
        34 => [MjaiTile(13), MjaiTile(35)],
        35 => [MjaiTile(22), MjaiTile(22)],
        36 => [MjaiTile(22), MjaiTile(36)],
        _ => unreachable!("PON index was checked"),
    })
}

fn validate_event(event: &MjaiEvent) -> Result<(), String> {
    let valid_actor = |actor: u8| actor < NUM_PLAYERS as u8;
    let valid_pair = |actor: u8, target: u8| valid_actor(actor) && valid_actor(target);
    match event {
        MjaiEvent::StartKyoku { kyoku, oya, .. } => {
            if !(1..=4).contains(kyoku) || !valid_actor(*oya) {
                return Err("start_kyoku requires kyoku in 1..=4 and oya in 0..4".to_owned());
            }
        }
        MjaiEvent::Tsumo { actor, .. }
        | MjaiEvent::Dahai { actor, .. }
        | MjaiEvent::Kakan { actor, .. }
        | MjaiEvent::Ankan { actor, .. }
        | MjaiEvent::Reach { actor }
        | MjaiEvent::ReachAccepted { actor } => {
            if !valid_actor(*actor) {
                return Err("MJAI actor must be in 0..4".to_owned());
            }
        }
        MjaiEvent::Chi { actor, target, .. }
        | MjaiEvent::Pon { actor, target, .. }
        | MjaiEvent::Daiminkan { actor, target, .. }
        | MjaiEvent::Hora { actor, target, .. } => {
            if !valid_pair(*actor, *target) {
                return Err("MJAI actor and target must be in 0..4".to_owned());
            }
        }
        MjaiEvent::None
        | MjaiEvent::StartGame { .. }
        | MjaiEvent::Dora { .. }
        | MjaiEvent::Ryukyoku { .. }
        | MjaiEvent::EndKyoku
        | MjaiEvent::EndGame => {}
    }
    Ok(())
}

fn deserialize_tile<'de, D>(deserializer: D) -> Result<u8, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    tile_from_str(&value)
        .map(MjaiTile::as_u8)
        .ok_or_else(|| serde::de::Error::custom(format!("invalid MJAI tile {value:?}")))
}

const MJAI_TILE_NAMES: [&str; 38] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
    "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s",
    "9s", "E", "S", "W", "N", "P", "F", "C", "5mr", "5pr", "5sr", "?",
];

fn tile_from_str(value: &str) -> Option<MjaiTile> {
    MJAI_TILE_NAMES
        .iter()
        .position(|&tile| tile == value)
        .map(|index| MjaiTile(index as u8))
}

fn snapshot_tile(value: &str) -> Result<MjaiTile, String> {
    tile_from_str(value).ok_or_else(|| format!("invalid snapshot tile {value:?}"))
}

const fn default_scores() -> [i32; NUM_PLAYERS] {
    [25_000; NUM_PLAYERS]
}
