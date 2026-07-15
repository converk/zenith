const NUM_PLAYERS: usize = 4;
const ENVS_PER_THREAD: usize = 8;
const NUM_ACTIONS: usize = 241;

/// V4 contains twelve temporary board tokens: three groups for each relative
/// player (player state, river, melds).  The fields are deliberately kept as
/// small categorical integers; Python receives them pre-decoded as u8.
const BOARD_TOKENS: usize = NUM_PLAYERS * 3;
const BOARD_FIELDS: usize = 160;
const RIVER_SLOTS: usize = 32;
const MELD_SLOTS: usize = 4;
const DORA_SLOTS: usize = 5;

const BLOCK_PAD: u8 = 0;
const BLOCK_TURN: u8 = 1;
const BLOCK_MELD: u8 = 2;

const MICRO_DISCARD: u8 = 1;
const MICRO_REACH: u8 = 2;
const MICRO_REACH_ACCEPTED: u8 = 3;
const MICRO_DORA: u8 = 4;

const MELD_CHI: u8 = 1;
const MELD_PON: u8 = 2;
const MELD_DAIMINKAN: u8 = 3;
const MELD_KAKAN: u8 = 4;
const MELD_ANKAN: u8 = 5;

const ACTOR_NONE: u8 = 0;
const ACTOR_SELF: u8 = 1;
const ACTOR_SHIMOCHA: u8 = 2;
const ACTOR_TOIMEN: u8 = 3;
const ACTOR_KAMICHA: u8 = 4;

const TILE_NONE: u8 = 0;
const TILE_UNKNOWN: u8 = 38;

const FLAG_TSUMOGIRI: u8 = 1;
const FLAG_RIICHI_DISCARD: u8 = 2;
const FLAG_CALLED: u8 = 4;

#[derive(Clone, Copy, Default)]
struct MicroEvent {
    kind: u8,
    actor: u8,
    tile: u8,
    flag: u8,
}

#[derive(Clone)]
enum EventBlock {
    Turn([MicroEvent; 4]),
    Meld {
        meld_kind: u8,
        actor: u8,
        target: u8,
        pai: u8,
        tiles: [u8; 4],
    },
}

#[derive(Clone, Copy, Default)]
struct RiverEntry {
    tile: u8,
    flag: u8,
}

#[derive(Clone, Copy, Default)]
struct MeldEntry {
    meld_kind: u8,
    target: u8,
    pai: u8,
    tiles: [u8; 4],
}

#[derive(Clone, Default)]
struct PublicPlayerState {
    river: Vec<RiverEntry>,
    melds: Vec<MeldEntry>,
    double_riichi: bool,
    ippatsu: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MjaiEvent {
    None,
    StartGame {
        id: Option<u8>,
        #[allow(dead_code)]
        #[serde(default)]
        names: [String; NUM_PLAYERS],
        #[allow(dead_code)]
        seed: Option<(u64, u64)>,
    },
    StartKyoku {
        bakaze: MjaiTile,
        dora_marker: MjaiTile,
        kyoku: u8,
        honba: u8,
        kyotaku: u8,
        oya: u8,
        #[serde(default = "default_scores")]
        scores: [i32; NUM_PLAYERS],
        tehais: [[MjaiTile; 13]; NUM_PLAYERS],
    },
    Tsumo { actor: u8, pai: MjaiTile },
    Dahai { actor: u8, pai: MjaiTile, tsumogiri: bool },
    Chi { actor: u8, target: u8, pai: MjaiTile, consumed: [MjaiTile; 2] },
    Pon { actor: u8, target: u8, pai: MjaiTile, consumed: [MjaiTile; 2] },
    Daiminkan { actor: u8, target: u8, pai: MjaiTile, consumed: [MjaiTile; 3] },
    Kakan { actor: u8, pai: MjaiTile, consumed: [MjaiTile; 3] },
    Ankan { actor: u8, consumed: [MjaiTile; 4] },
    Dora { dora_marker: MjaiTile },
    Reach { actor: u8 },
    ReachAccepted { actor: u8 },
    Hora { actor: u8, target: u8, deltas: Option<[i32; NUM_PLAYERS]>, ura_markers: Option<Vec<MjaiTile>> },
    Ryukyoku { deltas: Option<[i32; NUM_PLAYERS]> },
    EndKyoku,
    EndGame,
}

#[derive(Clone, Debug, Deserialize)]
struct DecisionSnapshot {
    player_id: u8,
    oya: u8,
    round_wind: u8,
    kyoku_index: u8,
    honba: u8,
    riichi_sticks: u32,
    scores: [i32; NUM_PLAYERS],
    dora_indicators: Vec<String>,
    hand: Vec<String>,
    drawn_tile: Option<String>,
    riichi_declared: [bool; NUM_PLAYERS],
    last_discard: Option<String>,
    last_tedashis: [Option<String>; NUM_PLAYERS],
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
struct MjaiTile(#[serde(deserialize_with = "deserialize_tile")] u8);

impl MjaiTile {
    const fn as_u8(self) -> u8 { self.0 }
    const fn as_usize(self) -> usize { self.0 as usize }
    const fn deaka(self) -> Self {
        match self.0 {
            34 => Self(4),
            35 => Self(13),
            36 => Self(22),
            _ => self,
        }
    }
}
