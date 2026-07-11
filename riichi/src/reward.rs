use crate::game::Cache;

pub(crate) fn distance_delta_reward(
    cache: &Cache,
    previous_distance: u8,
    current_tiles: &[u8],
) -> f32 {
    distance_delta_reward_with_completion(cache, previous_distance, current_tiles).0
}

pub(crate) fn distance_delta_reward_with_completion(
    cache: &Cache,
    previous_distance: u8,
    current_tiles: &[u8],
) -> (f32, bool) {
    let current_distance = cache.completion_distance(current_tiles, 1, 4);
    let is_complete = current_distance == 0;
    let reward = previous_distance as f32 - current_distance as f32;
    (reward, is_complete)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewards_distance_improvement_and_completion() {
        let cache = Cache::new();

        let mut complete_hand = [0u8; 34];
        for tile in [
            0, 1, 2, // 1m 2m 3m
            3, 4, 5, // 4m 5m 6m
            6, 7, 8, // 7m 8m 9m
            18, 19, 20, // 1s 2s 3s
            24, 24, // 7s 7s
        ] {
            complete_hand[tile] += 1;
        }

        assert_eq!(distance_delta_reward(&cache, 1, &complete_hand), 1f32);
    }
}
