//! 4 人麻将游戏模式的数值约定(单源)。
//!
//! 规则配置由 `GameState.rule`(GameRule)单独持有,本模块只提供与
//! 游戏模式绑定的固定数值,避免出现第二份规则/子模式状态。

/// 4P 游戏人数。
pub fn num_players() -> u8 {
    4
}

/// 4P 配点(25000 返し)。
pub fn starting_score() -> i32 {
    25000
}

/// 流局听牌罚符总池。
pub fn tenpai_pool() -> i32 {
    3000
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::standard_next_dora_tile;

    #[test]
    fn test_game_mode_config_four_player() {
        assert_eq!(num_players(), 4);
        assert_eq!(starting_score(), 25000);
        assert_eq!(tenpai_pool(), 3000);
    }

    #[test]
    fn test_four_player_dora_wrapping() {
        assert_eq!(standard_next_dora_tile(0), 1); // 1m -> 2m
        assert_eq!(standard_next_dora_tile(8), 0); // 9m -> 1m
        assert_eq!(standard_next_dora_tile(27), 28); // E -> S
        assert_eq!(standard_next_dora_tile(30), 27); // N -> E
        assert_eq!(standard_next_dora_tile(31), 32); // Haku -> Hatsu
        assert_eq!(standard_next_dora_tile(33), 31); // Chun -> Haku
    }
}
