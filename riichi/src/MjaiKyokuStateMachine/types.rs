const NUM_PLAYERS: usize = 4;
const ENVS_PER_THREAD: usize = 8;
const TOKEN_DIM: usize = 8;
const NUM_ACTIONS: usize = 241;

const TYPE_PAD: i64 = 0;
const TYPE_SEP: i64 = 1;
const TYPE_STATE_BAKAZE: i64 = 3;
const TYPE_STATE_JIKAZE: i64 = 4;
const TYPE_STATE_OYA: i64 = 5;
const TYPE_STATE_KYOKU_INDEX: i64 = 6;
const TYPE_STATE_HONBA: i64 = 7;
const TYPE_STATE_KYOTAKU: i64 = 8;
const TYPE_STATE_SCORE: i64 = 9;
const TYPE_STATE_LEFT_TILES: i64 = 11;
const TYPE_STATE_DORA: i64 = 12;
const TYPE_STATE_HAND: i64 = 13;
const TYPE_STATE_SNAPSHOT_BEGIN: i64 = 14;
const TYPE_STATE_SNAPSHOT_END: i64 = 15;
const TYPE_STATE_SELF_ID: i64 = 16;
const TYPE_STATE_RIICHI_STICKS: i64 = 17;
const TYPE_STATE_DRAWN_TILE: i64 = 18;
const TYPE_STATE_RIICHI_DECLARED: i64 = 19;
const TYPE_STATE_LAST_DISCARD: i64 = 20;
const TYPE_STATE_LAST_TEDASHI: i64 = 21;
const TYPE_EVENT_START_KYOKU: i64 = 26;
const TYPE_EVENT_DRAW: i64 = 27;
const TYPE_EVENT_DISCARD: i64 = 28;
const TYPE_EVENT_CHI: i64 = 29;
const TYPE_EVENT_PON: i64 = 30;
const TYPE_EVENT_DAIMINKAN: i64 = 31;
const TYPE_EVENT_KAKAN: i64 = 32;
const TYPE_EVENT_ANKAN: i64 = 33;
const TYPE_EVENT_MELD_CONT: i64 = 34;
const TYPE_EVENT_DORA: i64 = 35;
const TYPE_EVENT_REACH: i64 = 36;
const TYPE_EVENT_REACH_ACCEPTED: i64 = 37;
const TYPE_EVENT_HORA: i64 = 38;
const TYPE_EVENT_RYUKYOKU: i64 = 39;
const TYPE_EVENT_SCORE_DELTA: i64 = 40;
const TYPE_EVENT_URA_DORA: i64 = 41;

const ACTOR_NONE: i64 = 0;
const ACTOR_SELF: i64 = 1;
const ACTOR_SHIMOCHA: i64 = 2;
const ACTOR_TOIMEN: i64 = 3;
const ACTOR_KAMICHA: i64 = 4;

const TILE_NONE: i64 = 0;
const TILE_UNKNOWN: i64 = 38;
const VALUE_NONE: i64 = 0;

const FLAG_NONE: i64 = 0;
const FLAG_TSUMOGIRI: i64 = 3;
const FLAG_TEDASHI: i64 = 4;
const FLAG_REACH_DECLARE: i64 = 9;
const FLAG_MELD_DAIMINKAN: i64 = 13;
const FLAG_MELD_KAKAN: i64 = 14;
const FLAG_MELD_ANKAN: i64 = 15;
const FLAG_CHI_LOW: i64 = 16;
const FLAG_CHI_MID: i64 = 17;
const FLAG_CHI_HIGH: i64 = 18;
const FLAG_DELTA_POSITIVE: i64 = 19;
const FLAG_DELTA_NEGATIVE: i64 = 20;
const FLAG_DELTA_ZERO: i64 = 21;
const FLAG_RON: i64 = 22;
const FLAG_TSUMO: i64 = 23;

type Token = [i64; TOKEN_DIM];

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
struct MjaiTile(#[serde(deserialize_with = "deserialize_tile")] u8);

impl MjaiTile {
    const fn as_u8(self) -> u8 {
        self.0
    }

    const fn as_usize(self) -> usize {
        self.0 as usize
    }

    const fn deaka(self) -> Self {
        match self.0 {
            34 => Self(4),
            35 => Self(13),
            36 => Self(22),
            _ => self,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MjaiEvent {
    None,
    StartGame {
        id: Option<u8>,
        // RiichiEnv replay/test events may carry these fields. The state machine only
        // needs `id`, but accepting them keeps event deserialization compatible.
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
    Tsumo {
        actor: u8,
        pai: MjaiTile,
    },
    Dahai {
        actor: u8,
        pai: MjaiTile,
        tsumogiri: bool,
    },
    Chi {
        actor: u8,
        target: u8,
        pai: MjaiTile,
        consumed: [MjaiTile; 2],
    },
    Pon {
        actor: u8,
        target: u8,
        pai: MjaiTile,
        consumed: [MjaiTile; 2],
    },
    Daiminkan {
        actor: u8,
        target: u8,
        pai: MjaiTile,
        consumed: [MjaiTile; 3],
    },
    Kakan {
        actor: u8,
        pai: MjaiTile,
        consumed: [MjaiTile; 3],
    },
    Ankan {
        actor: u8,
        consumed: [MjaiTile; 4],
    },
    Dora {
        dora_marker: MjaiTile,
    },
    Reach {
        actor: u8,
    },
    ReachAccepted {
        actor: u8,
    },
    Hora {
        actor: u8,
        target: u8,
        deltas: Option<[i32; NUM_PLAYERS]>,
        ura_markers: Option<Vec<MjaiTile>>,
    },
    Ryukyoku {
        deltas: Option<[i32; NUM_PLAYERS]>,
    },
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
