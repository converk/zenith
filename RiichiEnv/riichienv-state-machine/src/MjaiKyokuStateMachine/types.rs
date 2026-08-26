const NUM_PLAYERS: usize = 4;
const ENVS_PER_THREAD: usize = 8;
// 固定 241 维动作空间与 34 类牌:领域不变常量,单一命名定义,禁止散落魔法数字。
pub(crate) const NUM_ACTIONS: usize = 241;
pub(crate) const TILE_KINDS: usize = 34;

#[allow(dead_code)]
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

#[allow(dead_code)]
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
