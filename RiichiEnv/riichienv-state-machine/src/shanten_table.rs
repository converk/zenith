use std::{
    env,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
};

const MAGIC: &[u8; 8] = b"ZNSHANT1";
const VERSION: u32 = 1;
const SUIT_STATES: usize = 1_953_125; // 5^9
const HONOR_STATES: usize = 78_125; // 5^7
const HEADER_LEN: usize = 32;
const POW5: [usize; 9] = [1, 5, 25, 125, 625, 3_125, 15_625, 78_125, 390_625];
const PAIR_ZERO_MASK: u64 = 0x155_5555_5555_5555;

static TABLE: OnceLock<ShantenTable> = OnceLock::new();

struct ShantenTable {
    suits: Vec<u64>,
    honors: Vec<u64>,
}

impl ShantenTable {
    fn generate() -> Self {
        Self {
            suits: generate_group_table(9, true),
            honors: generate_group_table(7, false),
        }
    }

    fn standard(&self, counts: &[u8; 34], open_melds: u8) -> i8 {
        let groups = [
            self.suits[encode(&counts[0..9])],
            self.suits[encode(&counts[9..18])],
            self.suits[encode(&counts[18..27])],
            self.honors[encode(&counts[27..34])],
        ];
        best_from_combined(combine_groups(&groups), open_melds)
    }

    /// 同一 13 张形状的 34 种摸牌共享三个未变化牌组的合并结果。
    fn standard_after_draws(&self, counts: &[u8; 34], open_melds: u8) -> [i8; 34] {
        let codes = [
            encode(&counts[0..9]),
            encode(&counts[9..18]),
            encode(&counts[18..27]),
            encode(&counts[27..34]),
        ];
        let groups = [
            self.suits[codes[0]],
            self.suits[codes[1]],
            self.suits[codes[2]],
            self.honors[codes[3]],
        ];
        let mut unchanged = [0_u64; 4];
        for changed in 0..4 {
            let others = groups
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| (index != changed).then_some(value))
                .collect::<Vec<_>>();
            unchanged[changed] = combine_groups(&others);
        }

        let mut out = [crate::SHANTEN_UNAVAILABLE; 34];
        for tile in 0..34 {
            if counts[tile] >= 4 {
                continue;
            }
            let group = if tile < 27 { tile / 9 } else { 3 };
            let local = if tile < 27 { tile % 9 } else { tile - 27 };
            let changed_code = codes[group] + POW5[local];
            let changed_value = if group < 3 {
                self.suits[changed_code]
            } else {
                self.honors[changed_code]
            };
            out[tile] = best_from_combined(combine(unchanged[group], changed_value), open_melds);
        }
        out
    }

    fn payload_checksum(&self) -> u64 {
        let mut hash = FNV_OFFSET;
        for value in self.suits.iter().chain(&self.honors) {
            for byte in value.to_le_bytes() {
                hash = fnv_byte(hash, byte);
            }
        }
        hash
    }

    fn write(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("shanten-v1.bin");
        let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
        let result = (|| {
            let mut file = File::create(&temporary)?;
            file.write_all(MAGIC)?;
            file.write_all(&VERSION.to_le_bytes())?;
            file.write_all(&(SUIT_STATES as u32).to_le_bytes())?;
            file.write_all(&(HONOR_STATES as u32).to_le_bytes())?;
            file.write_all(&8_u32.to_le_bytes())?;
            file.write_all(&self.payload_checksum().to_le_bytes())?;
            for value in self.suits.iter().chain(&self.honors) {
                file.write_all(&value.to_le_bytes())?;
            }
            file.sync_all()?;
            fs::rename(&temporary, path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    fn read(path: &Path) -> io::Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        let expected_len = HEADER_LEN + (SUIT_STATES + HONOR_STATES) * 8;
        if bytes.len() != expected_len
            || &bytes[0..8] != MAGIC
            || read_u32(&bytes[8..12]) != VERSION
            || read_u32(&bytes[12..16]) as usize != SUIT_STATES
            || read_u32(&bytes[16..20]) as usize != HONOR_STATES
            || read_u32(&bytes[20..24]) != 8
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported shanten cache header",
            ));
        }
        let expected_checksum = read_u64(&bytes[24..32]);
        let mut checksum = FNV_OFFSET;
        for &byte in &bytes[HEADER_LEN..] {
            checksum = fnv_byte(checksum, byte);
        }
        if checksum != expected_checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "shanten cache checksum mismatch",
            ));
        }

        let mut values = bytes[HEADER_LEN..]
            .chunks_exact(8)
            .map(read_u64)
            .collect::<Vec<_>>();
        let honors = values.split_off(SUIT_STATES);
        Ok(Self {
            suits: values,
            honors,
        })
    }
}

fn configured_cache_path() -> PathBuf {
    if let Some(path) = env::var_os("ZENITH_SHANTEN_CACHE") {
        return PathBuf::from(path);
    }
    if let Some(root) = env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(root).join("zenith-riichi/shanten-v1.bin");
    }
    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home).join(".cache/zenith-riichi/shanten-v1.bin");
    }
    env::temp_dir().join("zenith-riichi/shanten-v1.bin")
}

pub(super) fn standard(counts: &[u8; 34], open_melds: u8) -> i8 {
    TABLE
        .get_or_init(|| {
            let path = configured_cache_path();
            ShantenTable::read(&path).unwrap_or_else(|_| {
                let table = ShantenTable::generate();
                let _ = table.write(&path);
                table
            })
        })
        .standard(counts, open_melds)
}

pub(super) fn standard_after_draws(counts: &[u8; 34], open_melds: u8) -> [i8; 34] {
    TABLE
        .get_or_init(|| {
            let path = configured_cache_path();
            ShantenTable::read(&path).unwrap_or_else(|_| {
                let table = ShantenTable::generate();
                let _ = table.write(&path);
                table
            })
        })
        .standard_after_draws(counts, open_melds)
}

fn generate_group_table(width: usize, sequences: bool) -> Vec<u64> {
    let state_count = 5_usize.pow(width as u32);
    let mut table = vec![0_u64; state_count];
    table[0] = 1;
    let valid_meld_mask = mask_where(|melds, _, _| melds < 4);
    let valid_taatsu_mask = mask_where(|_, taatsu, _| taatsu < 4);

    for code in 1..state_count {
        let mut digits = [0_u8; 9];
        let mut value = code;
        let mut total = 0_u8;
        let mut first = width;
        for (i, digit) in digits.iter_mut().enumerate().take(width) {
            *digit = (value % 5) as u8;
            value /= 5;
            total += *digit;
            if *digit > 0 && first == width {
                first = i;
            }
        }
        if total > 14 {
            continue;
        }

        let unit = POW5[first];
        let mut outcomes = table[code - unit];
        if digits[first] >= 3 {
            outcomes |= (table[code - 3 * unit] & valid_meld_mask) << 10;
        }
        if digits[first] >= 2 {
            outcomes |= (table[code - 2 * unit] & PAIR_ZERO_MASK) << 1;
            outcomes |= (table[code - 2 * unit] & valid_taatsu_mask) << 2;
        }
        if sequences && first + 2 < width && digits[first + 1] > 0 && digits[first + 2] > 0 {
            let lower = code - unit - POW5[first + 1] - POW5[first + 2];
            outcomes |= (table[lower] & valid_meld_mask) << 10;
        }
        if sequences && first + 1 < width && digits[first + 1] > 0 {
            let lower = code - unit - POW5[first + 1];
            outcomes |= (table[lower] & valid_taatsu_mask) << 2;
        }
        if sequences && first + 2 < width && digits[first + 2] > 0 {
            let lower = code - unit - POW5[first + 2];
            outcomes |= (table[lower] & valid_taatsu_mask) << 2;
        }
        table[code] = outcomes & ((1_u64 << 50) - 1);
    }
    table
}

fn combine(left: u64, right: u64) -> u64 {
    let mut result = 0_u64;
    for lhs in BitIndices(left) {
        let (lm, lt, lp) = unpack(lhs);
        for rhs in BitIndices(right) {
            let (rm, rt, rp) = unpack(rhs);
            let melds = lm + rm;
            let taatsu = lt + rt;
            let pairs = lp + rp;
            if melds <= 4 && taatsu <= 4 && pairs <= 1 {
                result |= 1_u64 << pack(melds, taatsu, pairs);
            }
        }
    }
    result
}

fn combine_groups(groups: &[u64]) -> u64 {
    let mut values = groups.iter().copied();
    let Some(mut combined) = values.next() else {
        return 1;
    };
    for group in values {
        combined = combine(combined, group);
    }
    combined
}

fn best_from_combined(combined: u64, open_melds: u8) -> i8 {
    let mut best = 8_i8;
    for index in BitIndices(combined) {
        let (melds, taatsu, pair) = unpack(index);
        let melds = melds + open_melds.min(4);
        if melds <= 4 {
            let useful_taatsu = taatsu.min(4 - melds);
            best = best.min(8 - 2 * melds as i8 - useful_taatsu as i8 - pair as i8);
        }
    }
    best
}

fn encode(counts: &[u8]) -> usize {
    counts
        .iter()
        .enumerate()
        .map(|(i, &count)| count as usize * POW5[i])
        .sum()
}

const fn pack(melds: u8, taatsu: u8, pair: u8) -> usize {
    ((melds as usize * 5 + taatsu as usize) * 2) + pair as usize
}

const fn unpack(index: usize) -> (u8, u8, u8) {
    (
        (index / 10) as u8,
        ((index % 10) / 2) as u8,
        (index % 2) as u8,
    )
}

fn mask_where(predicate: fn(u8, u8, u8) -> bool) -> u64 {
    let mut mask = 0;
    let mut index = 0;
    while index < 50 {
        let (melds, taatsu, pair) = unpack(index);
        if predicate(melds, taatsu, pair) {
            mask |= 1_u64 << index;
        }
        index += 1;
    }
    mask
}

struct BitIndices(u64);

impl Iterator for BitIndices {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        let index = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(index)
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const fn fnv_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four-byte cache field"))
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("eight-byte cache field"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_cache_round_trips_and_rejects_corruption() {
        let root = env::temp_dir().join(format!("zenith-shanten-test-{}", std::process::id()));
        let path = root.join("table.bin");
        let table = ShantenTable::generate();
        table.write(&path).unwrap();
        let loaded = ShantenTable::read(&path).unwrap();
        assert_eq!(loaded.payload_checksum(), table.payload_checksum());

        let mut bytes = fs::read(&path).unwrap();
        bytes[HEADER_LEN + 7] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(ShantenTable::read(&path).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
