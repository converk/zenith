use crate::shanten_table;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shanten {
    pub overall: i8,
    pub standard: i8,
    pub seven_pairs: i8,
    pub thirteen_orphans: i8,
}

const ORPHANS: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

pub fn calculate(counts: &[u8; 34], open_melds: u8) -> Shanten {
    let standard = standard(counts, open_melds);
    let seven_pairs = if open_melds == 0 {
        seven_pairs(counts)
    } else {
        crate::SHANTEN_UNAVAILABLE
    };
    let thirteen_orphans = if open_melds == 0 {
        thirteen_orphans(counts)
    } else {
        crate::SHANTEN_UNAVAILABLE
    };
    let overall = standard.min(seven_pairs).min(thirteen_orphans);
    Shanten {
        overall,
        standard,
        seven_pairs,
        thirteen_orphans,
    }
}

pub fn seven_pairs(counts: &[u8; 34]) -> i8 {
    let pairs = counts.iter().filter(|&&c| c >= 2).count() as i8;
    let distinct = counts.iter().filter(|&&c| c > 0).count() as i8;
    6 - pairs + (7 - distinct).max(0)
}

pub fn thirteen_orphans(counts: &[u8; 34]) -> i8 {
    let unique = ORPHANS.iter().filter(|&&i| counts[i] > 0).count() as i8;
    let pair = ORPHANS.iter().any(|&i| counts[i] >= 2) as i8;
    13 - unique - pair
}

pub fn standard(counts: &[u8; 34], open_melds: u8) -> i8 {
    shanten_table::standard(counts, open_melds)
}

#[cfg(test)]
fn calculate_recursive(counts: &[u8; 34], open_melds: u8) -> Shanten {
    let standard = standard_recursive(counts, open_melds);
    let seven_pairs = if open_melds == 0 {
        seven_pairs(counts)
    } else {
        crate::SHANTEN_UNAVAILABLE
    };
    let thirteen_orphans = if open_melds == 0 {
        thirteen_orphans(counts)
    } else {
        crate::SHANTEN_UNAVAILABLE
    };
    Shanten {
        overall: standard.min(seven_pairs).min(thirteen_orphans),
        standard,
        seven_pairs,
        thirteen_orphans,
    }
}

#[cfg(test)]
fn standard_recursive(counts: &[u8; 34], open_melds: u8) -> i8 {
    let mut work = *counts;
    let mut best = 8;
    search(&mut work, 0, open_melds as i8, 0, 0, &mut best);
    best
}

#[cfg(test)]
fn search(c: &mut [u8; 34], start: usize, melds: i8, pairs: i8, taatsu: i8, best: &mut i8) {
    let mut i = start;
    while i < 34 && c[i] == 0 {
        i += 1;
    }
    if i == 34 {
        let capped_taatsu = taatsu.min(4 - melds);
        *best = (*best).min(8 - 2 * melds - capped_taatsu - pairs.min(1));
        return;
    }
    if melds < 4 && c[i] >= 3 {
        c[i] -= 3;
        search(c, i, melds + 1, pairs, taatsu, best);
        c[i] += 3;
    }
    if melds < 4 && i < 27 && i % 9 <= 6 && c[i + 1] > 0 && c[i + 2] > 0 {
        c[i] -= 1;
        c[i + 1] -= 1;
        c[i + 2] -= 1;
        search(c, i, melds + 1, pairs, taatsu, best);
        c[i] += 1;
        c[i + 1] += 1;
        c[i + 2] += 1;
    }
    if pairs == 0 && c[i] >= 2 {
        c[i] -= 2;
        search(c, i, melds, 1, taatsu, best);
        c[i] += 2;
    }
    if taatsu < 4 && c[i] >= 2 {
        c[i] -= 2;
        search(c, i, melds, pairs, taatsu + 1, best);
        c[i] += 2;
    }
    if taatsu < 4 && i < 27 && i % 9 <= 7 && c[i + 1] > 0 {
        c[i] -= 1;
        c[i + 1] -= 1;
        search(c, i, melds, pairs, taatsu + 1, best);
        c[i] += 1;
        c[i + 1] += 1;
    }
    if taatsu < 4 && i < 27 && i % 9 <= 6 && c[i + 2] > 0 {
        c[i] -= 1;
        c[i + 2] -= 1;
        search(c, i, melds, pairs, taatsu + 1, best);
        c[i] += 1;
        c[i + 2] += 1;
    }
    c[i] -= 1;
    search(c, i, melds, pairs, taatsu, best);
    c[i] += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_matches_recursive_over_legal_closed_and_open_hands() {
        let mut state = 0x9e37_79b9_u32;
        for open_melds in 0..=4_u8 {
            let concealed_tiles = 13_usize.saturating_sub(3 * open_melds as usize);
            for _ in 0..2_000 {
                let mut counts = [0; 34];
                let mut placed = 0;
                while placed < concealed_tiles {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    let tile = state as usize % 34;
                    if counts[tile] < 4 {
                        counts[tile] += 1;
                        placed += 1;
                    }
                }
                assert_eq!(
                    calculate(&counts, open_melds).overall,
                    calculate_recursive(&counts, open_melds).overall,
                    "open_melds={open_melds}, counts={counts:?}"
                );
            }
        }
    }

    #[test]
    fn families() {
        let mut complete = [0; 34];
        for i in [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28, 28] {
            complete[i] += 1;
        }
        assert_eq!(calculate(&complete, 0).overall, -1);
        let mut pairs = [0; 34];
        for count in pairs.iter_mut().take(7) {
            *count = 2;
        }
        assert_eq!(seven_pairs(&pairs), -1);
        let mut kokushi = [0; 34];
        for i in ORPHANS {
            kokushi[i] = 1;
        }
        kokushi[0] = 2;
        assert_eq!(thirteen_orphans(&kokushi), -1);
    }
}
