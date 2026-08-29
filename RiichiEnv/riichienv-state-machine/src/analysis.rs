use crate::mjai_kyoku_state_machine::TILE_KINDS;
use crate::shanten;

pub fn suji_safe(tile: usize, river: u64) -> bool {
    if tile >= 27 {
        return false;
    }
    let rank = tile % 9;
    let lower = (rank >= 3).then_some(tile - 3);
    let upper = (rank <= 5).then_some(tile + 3);
    // Middle tiles have two independent ryanmen routes.  Both suji anchors
    // must be present; e.g. a discarded 1m alone does not make 4m safe from
    // a 56m wait.  Edge tiles have only one applicable anchor.
    lower
        .into_iter()
        .chain(upper)
        .all(|anchor| river & (1_u64 << anchor) != 0)
}

pub fn wall_class(tile: usize, remaining: &[u8]) -> u8 {
    if tile >= 27 {
        return 0;
    }
    let rank = tile % 9;
    let base = tile - rank;
    // A kabe is made by related tiles needed alongside the candidate in a
    // sequence, not by copies of the candidate itself.  Preserve the compact
    // contract: 0=no wall, 1=three-visible wall, 2=four-visible wall.
    let strongest = (0..9)
        .filter(|&other| other != rank && other.abs_diff(rank) <= 2)
        .map(|other| 4_u8.saturating_sub(remaining[base + other]))
        .max()
        .unwrap_or(0);
    if strongest >= 4 {
        2
    } else if strongest >= 3 {
        1
    } else {
        0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OffenseRow {
    pub shanten: i8,
    pub effective_kinds: u8,
    pub effective_remaining: u16,
    pub wait_kinds: u8,
    pub wait_mask: u64,
    pub furiten: u8,
    pub can_riichi: u8,
}

/// 计算单个 13 张动作后形状的 进攻事实。
///
/// Python 批接口与融合编码器共用这一实现,避免两条热路径的语义漂移。
#[allow(clippy::too_many_arguments)]
pub(crate) fn offense_row(
    counts: &[u8; TILE_KINDS],
    open_melds: u8,
    remaining: &[u8; TILE_KINDS],
    own_river: u64,
    missed_doujun: bool,
    missed_riichi: bool,
    riichi_declared: bool,
    score: i32,
) -> Result<OffenseRow, String> {
    let total: u16 =
        counts.iter().map(|&value| u16::from(value)).sum::<u16>() + 3 * u16::from(open_melds);
    if total != 13 {
        return Err(format!(
            "shape represents {total} tiles (counts + 3×melds); expected 13"
        ));
    }
    if counts.iter().any(|&value| value > 4) {
        return Err("shape contains a count above 4".to_string());
    }
    if own_river >> TILE_KINDS != 0 {
        return Err(format!("own river mask exceeds {TILE_KINDS} bits"));
    }

    let shanten_value = shanten::calculate(counts, open_melds).overall;
    let after_draws = shanten::calculate_after_draws(counts, open_melds);
    let mut improving = 0_u64;
    if shanten_value > 0 {
        for tile in 0..TILE_KINDS {
            if counts[tile] >= 4 {
                continue;
            }
            if after_draws[tile] < shanten_value {
                improving |= 1_u64 << tile;
            }
        }
    }
    let mut wait_mask = 0_u64;
    if shanten_value == 0 {
        for tile in 0..TILE_KINDS {
            if counts[tile] >= 4 {
                continue;
            }
            if after_draws[tile] < 0 {
                wait_mask |= 1_u64 << tile;
            }
        }
    }
    let furiten = if shanten_value != 0 || wait_mask == 0 {
        0
    } else if (0..TILE_KINDS)
        .any(|tile| wait_mask & (1_u64 << tile) != 0 && own_river & (1_u64 << tile) != 0)
    {
        2
    } else if missed_doujun || missed_riichi {
        3
    } else {
        1
    };
    let can_riichi = if riichi_declared {
        0
    } else if open_melds > 0 || score < 1000 || wait_mask == 0 {
        2
    } else {
        1
    };
    let effective_remaining = (0..TILE_KINDS)
        .filter(|&tile| improving & (1_u64 << tile) != 0)
        .map(|tile| u16::from(remaining[tile]))
        .sum::<u16>();

    Ok(OffenseRow {
        shanten: shanten_value,
        effective_kinds: improving.count_ones() as u8,
        effective_remaining,
        wait_kinds: u8::try_from(wait_mask.count_ones()).unwrap_or(u8::MAX),
        wait_mask,
        furiten,
        can_riichi,
    })
}

/// 单行防守事实(D0–D9),与 Python 侧 `encoding_protocol.py` 的编码下标一致。
pub(crate) fn defense_row(
    tile: i16,
    rivers: &[u64; 3],
    hand: &[u8; TILE_KINDS],
    remaining: &[u8; TILE_KINDS],
) -> ([u8; 3], [u8; 3], [u8; 3], u8) {
    let mut genbutsu = [2_u8; 3];
    let mut suji = [2_u8; 3];
    let mut stock = [0_u8; 3];
    for opponent in 0..3 {
        let river = rivers[opponent];
        let mut count = 0_u8;
        for kind in 0..TILE_KINDS {
            if hand[kind] > 0 && river & (1_u64 << kind) != 0 {
                count += 1;
            }
        }
        stock[opponent] = count.min(4);
    }
    if tile < 0 {
        return (genbutsu, suji, stock, 5);
    }
    let tile = tile as usize;
    let visible = (4_usize.saturating_sub(usize::from(remaining[tile]))).min(4) as u8;
    for opponent in 0..3 {
        let river = rivers[opponent];
        genbutsu[opponent] = u8::from(river & (1_u64 << tile) != 0);
        suji[opponent] = u8::from(suji_safe(tile, river));
    }
    (genbutsu, suji, stock, visible)
}

#[cfg(test)]
mod defense_tests {
    use super::*;

    #[test]
    fn test_genbutsu_suji_and_stock() {
        // 对手 0 河含 1m/4m/7m:4m 是现物,且两侧筋锚(1m、7m)齐全故成筋。
        let rivers = [1_u64 << 0 | 1_u64 << 3 | 1_u64 << 6, 0, 1_u64 << 8];
        let mut hand = [0_u8; TILE_KINDS];
        hand[2] = 1; // 3m
        hand[3] = 1; // 4m
        let remaining = [4_u8; TILE_KINDS];
        let (genbutsu, suji, stock, visible) = defense_row(3, &rivers, &hand, &remaining);
        assert_eq!(genbutsu, [1, 0, 0]);
        assert_eq!(suji, [1, 0, 0]);
        // 现物库存只数「手中牌种 ∩ 对手河」:手中 3m/4m,仅 4m 在对手 0 河内。
        assert_eq!(stock, [1, 0, 0]);
        assert_eq!(visible, 0);
    }

    #[test]
    fn test_non_discard_na_and_visible() {
        let rivers = [1_u64 << 3, 0, 0];
        let mut hand = [0_u8; TILE_KINDS];
        hand[3] = 2;
        let mut remaining = [4_u8; TILE_KINDS];
        remaining[5] = 1;
        let (genbutsu, suji, stock, visible) = defense_row(-1, &rivers, &hand, &remaining);
        assert_eq!(genbutsu, [2, 2, 2]);
        assert_eq!(suji, [2, 2, 2]);
        // 非打牌动作仍计算安全牌库存:手中 4m 是对手 0 的现物。
        assert_eq!(stock, [1, 0, 0]);
        assert_eq!(visible, 5);
        let (_g, _s, _k, visible_discard) = defense_row(5, &rivers, &hand, &remaining);
        assert_eq!(visible_discard, 3);
    }
}
