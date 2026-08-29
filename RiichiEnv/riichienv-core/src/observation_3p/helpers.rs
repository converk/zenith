/// Compact tile dimension for 3-player mahjong (sanma).
/// 27 valid tile types: 1m, 9m, 1-9p, 1-9s, 4 winds, 3 dragons (no 2m-8m).
pub(crate) const TILE_DIM_3P: usize = 27;

/// Map tile34 index (0-33) to compact 27-tile index.
/// Returns None for tile34 in 1..=7 (2m-8m, excluded in sanma).
#[inline]
pub(crate) fn tile34_to_compact(tile34: usize) -> Option<usize> {
    match tile34 {
        0 => Some(0),
        1..=7 => None,
        8..=33 => Some(tile34 - 7),
        _ => None,
    }
}

/// Helper: write a scalar value broadcast across 27 compact tile positions into a flat buffer.
/// buf layout: channel-major, i.e. buf[(ch_offset + ch) * TILE_DIM_3P + tile] = val
#[inline]
pub(crate) fn broadcast_scalar(buf: &mut [f32], ch_offset: usize, ch: usize, val: f32) {
    let start = (ch_offset + ch) * TILE_DIM_3P;
    for j in 0..TILE_DIM_3P {
        buf[start + j] = val;
    }
}

/// Helper: set a single value in the flat buffer.
#[inline]
pub(crate) fn set_val(buf: &mut [f32], ch_offset: usize, ch: usize, tile: usize, val: f32) {
    buf[(ch_offset + ch) * TILE_DIM_3P + tile] = val;
}

/// Helper: add a value in the flat buffer.
#[inline]
pub(crate) fn add_val(buf: &mut [f32], ch_offset: usize, ch: usize, tile: usize, val: f32) {
    buf[(ch_offset + ch) * TILE_DIM_3P + tile] += val;
}

/// Sanma dora indicator -> dora tile mapping (tile IDs, i.e. tile/4 in 0..34).
/// 万子只有 1m/9m:1m↔9m 直接回绕;其余花色/字牌复用 `types::standard_next_dora_tile`
/// 这一 4P 单源实现,再映射回 136 空间(tile_id = type*4)。
pub(crate) fn get_next_tile_sanma(tile: u32) -> u8 {
    let tile34 = (tile / 4) as u8;
    match tile34 {
        0 => 8 * 4,          // 1m -> 9m
        8 => 0,              // 9m -> 1m (tile34=0, times 4 = 0)
        1..=7 => tile as u8, // 2m-8m 不存在于三麻牌山;原样回退保持历史行为
        _ => crate::types::standard_next_dora_tile(tile34) * 4,
    }
}
