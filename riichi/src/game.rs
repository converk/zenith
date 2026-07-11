use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use rand::Rng;
use rand_pcg::{rand_core::SeedableRng, Pcg32};

use crate::reward::{distance_delta_reward, distance_delta_reward_with_completion};

#[derive(Clone, Debug)]
struct Form {
    pairs: [u8; 9],
    tris: [u8; 9],
    seqs: [u8; 7],
    num_pairs: u8,
    num_tris: u8,
    num_seqs: u8,
    num_tiles: u8,
}

impl Form {
    fn new() -> Form {
        Form {
            pairs: [0u8; 9],
            tris: [0u8; 9],
            seqs: [0u8; 7],
            num_pairs: 0,
            num_tris: 0,
            num_seqs: 0,
            num_tiles: 0,
        }
    }
}

pub(crate) struct Cache {
    form_map: HashMap<[u8; 9], Vec<Form>>,
    edit_distance: HashMap<[u8; 9], [u8; 10]>,
}

impl Cache {
    fn search_forms(&mut self, tiles: [u8; 9], form: Form) {
        let num_tiles_with_pair = form.num_tiles + 2;
        let num_tiles_with_group = form.num_tiles + 3;

        if form.num_pairs <= 1 && num_tiles_with_group <= 14 {
            for i in 0..7 {
                let j = i + 1;
                let k = j + 1;
                if tiles[i] <= 3 && tiles[j] <= 3 && tiles[k] <= 3 {
                    let mut new_tiles = tiles.clone();
                    let mut new_form = form.clone();

                    new_tiles[i] += 1;
                    new_tiles[j] += 1;
                    new_tiles[k] += 1;
                    new_form.seqs[i] += 1;
                    new_form.num_seqs += 1;
                    new_form.num_tiles = num_tiles_with_group;

                    self.search_forms(new_tiles, new_form);
                }
            }
        }

        if form.num_pairs <= 1 && num_tiles_with_group <= 14 && form.num_seqs == 0 {
            for i in 0..9 {
                if tiles[i] <= 1 {
                    let mut new_tiles = tiles.clone();
                    let mut new_form = form.clone();

                    new_tiles[i] += 3;
                    new_form.tris[i] += 1;
                    new_form.num_tris += 1;
                    new_form.num_tiles = num_tiles_with_group;

                    self.search_forms(new_tiles, new_form);
                }
            }
        }

        if form.num_seqs == 0 && form.num_tris == 0 && num_tiles_with_pair <= 14 {
            for i in 0..9 {
                if tiles[i] <= 2 && form.pairs[i] == 0 {
                    let mut new_tiles = tiles.clone();
                    let mut new_form = form.clone();

                    new_tiles[i] += 2;
                    new_form.pairs[i] += 1;
                    new_form.num_pairs += 1;
                    new_form.num_tiles = num_tiles_with_pair;

                    self.search_forms(new_tiles, new_form);
                }
            }
        }

        self.form_map.entry(tiles).or_default().push(form);
    }

    fn search_edit_distance(&mut self) {
        let mut queue = VecDeque::<([u8; 9], u8, u8)>::new();

        for (tiles, forms) in &self.form_map {
            let mut distances = [14u8; 10];

            let mut t = 10;
            for form in forms {
                if form.num_pairs <= 1 {
                    t = form.num_pairs * 5 + form.num_tris + form.num_seqs;
                    break;
                }
            }
            if t < 10 {
                queue.push_back((tiles.clone(), t, 0));
                distances[t as usize] = 0;
                self.edit_distance.insert(tiles.clone(), distances);
            }
        }

        while let Some((tiles, t, edit_distance)) = queue.pop_front() {
            let mut update_edit_distance = |tiles: [u8; 9], edit_distance: u8| {
                let distance = &mut self
                    .edit_distance
                    .entry(tiles.clone())
                    .or_insert([14u8; 10])[t as usize];
                if *distance == 14 {
                    queue.push_back((tiles, t, edit_distance));
                    *distance = edit_distance;
                }
            };

            let num_tiles = tiles.iter().sum::<u8>();

            for i in 0..9 {
                if tiles[i] < 4 && num_tiles < 14 {
                    let mut new_tiles = tiles.clone();
                    new_tiles[i] += 1;

                    update_edit_distance(new_tiles, edit_distance);
                }
            }

            for i in 0..9 {
                if tiles[i] > 0 {
                    let mut new_tiles = tiles.clone();
                    new_tiles[i] -= 1;

                    update_edit_distance(new_tiles, edit_distance + 1);
                }
            }
        }
    }

    pub fn new() -> Cache {
        let mut cache = Cache {
            form_map: HashMap::new(),
            edit_distance: HashMap::new(),
        };

        cache.search_forms([0u8; 9], Form::new());
        cache.search_edit_distance();

        cache
    }

    pub(crate) fn completion_distance(&self, tiles: &[u8], num_pairs: u8, num_groups: u8) -> u8 {
        let mut distance = [14u8; 10];
        let honours = &tiles[27..34];
        let mut honour_forms = [0u8; 5];

        for h in 0..7 {
            honour_forms[honours[h] as usize] += 1;
        }

        distance[0] = 0;

        let mut num_replicas = 4;
        let mut num_replica_groups = 0;

        while num_replica_groups < 5 {
            if honour_forms[num_replicas] > 0 {
                if num_replica_groups < 4 {
                    distance[num_replica_groups + 1] = distance[num_replica_groups]
                        + if num_replicas < 3 {
                            3 - num_replicas
                        } else {
                            0
                        } as u8;
                }
                distance[5 + num_replica_groups] = distance[num_replica_groups]
                    + if num_replicas < 2 {
                        2 - num_replicas
                    } else {
                        0
                    } as u8;
                honour_forms[num_replicas] -= 1;
                num_replica_groups += 1;
            } else {
                num_replicas -= 1;
            }
        }

        for s in 0..3 {
            let suit_tiles = &tiles[s * 9..(s + 1) * 9];
            let suit_distance = self.edit_distance.get(suit_tiles).unwrap();

            for t1 in (0..10).rev() {
                for t0 in 0..=t1 {
                    let compose_distance = distance[t1 - t0] + suit_distance[t0];
                    if compose_distance < distance[t1] {
                        distance[t1] = compose_distance;
                    }
                }
            }
        }

        distance[(num_pairs * 5 + num_groups) as usize]
    }
}

pub(crate) struct State {
    cache: Arc<Cache>,
    rng: Pcg32,

    wall: [u8; 136],
    player_tiles: [u8; 136],
    next_draw: u8,
    next_player: u8,
}

impl State {
    pub fn new(cache: Arc<Cache>, seed: u64) -> State {
        let mut wall = [0u8; 136];
        for i in 0..34 {
            for j in 0..4 {
                wall[4 * i + j] = i as u8;
            }
        }

        State {
            cache,
            rng: Pcg32::seed_from_u64(seed),
            wall,
            player_tiles: [0u8; 136],
            next_draw: 0,
            next_player: 0,
        }
    }

    fn next_move(&mut self) {
        self.next_player += 1;
        if self.next_player == 4 {
            self.next_player = 0;
        }
    }

    fn draw(&mut self) {
        let tile = self.wall[self.next_draw as usize];
        self.player_tiles[self.next_player as usize * 34 + tile as usize] += 1;
        self.next_draw += 1;
        self.next_move();
    }

    fn discard(&mut self, tile: u8) {
        self.player_tiles[self.next_player as usize * 34 + tile as usize] -= 1;
        self.next_move();
    }

    pub fn reset(&mut self) -> [u8; 136] {
        // shuffle the wall
        for i in (1..136).rev() {
            let j = self.rng.random_range(0..=i);
            self.wall.swap(i, j);
        }

        // initialize player tiles
        for i in 0..136 {
            self.player_tiles[i] = 0;
        }

        for p in 0..4 {
            for i in (p * 13)..((p + 1) * 13) {
                self.player_tiles[p * 34 + self.wall[i] as usize] += 1;
            }
        }

        self.next_draw = 4 * 13;
        self.next_player = 0;

        for _p in 0..4 {
            self.draw();
        }

        self.player_tiles
    }

    #[allow(dead_code)]
    fn enumerate_points(forms: &[&Vec<Form>], honours: &[u8]) -> f32 {
        let mut num_honour_pairs = 0;
        let mut num_honour_tris = 0;

        for i in 0..7 {
            if honours[i] == 2 {
                num_honour_pairs += 1;
            } else if honours[i] == 3 {
                num_honour_tris += 1;
            }
        }

        for form_0 in forms[0] {
            for form_1 in forms[1] {
                for form_2 in forms[2] {
                    let num_pairs =
                        form_0.num_pairs + form_1.num_pairs + form_2.num_pairs + num_honour_pairs;
                    let num_groups = form_0.num_tris
                        + form_1.num_tris
                        + form_2.num_tris
                        + form_0.num_seqs
                        + form_1.num_seqs
                        + form_2.num_seqs
                        + num_honour_tris;

                    if (num_pairs == 1 && num_groups == 4) || (num_pairs == 7) {
                        return 1f32;
                    }
                }
            }
        }

        0f32
    }

    pub fn step(&mut self, discard: &[u8]) -> ([u8; 136], [f32; 4], bool) {
        let (player_tiles, reward, done, _winners) = self.step_with_winners(discard);
        (player_tiles, reward, done)
    }

    pub fn step_with_winners(&mut self, discard: &[u8]) -> ([u8; 136], [f32; 4], bool, [bool; 4]) {
        let mut previous_distance = [0u8; 4];
        for p in 0..4 {
            let player_tiles = &self.player_tiles[p * 34..(p + 1) * 34];
            previous_distance[p] = self.cache.completion_distance(player_tiles, 1, 4);
        }

        for p in 0..4 {
            self.discard(discard[p]);
        }

        if self.next_draw + 4 > 122 {
            let mut reward = [0f32; 4];
            for p in 0..4 {
                let player_tiles = &self.player_tiles[p * 34..(p + 1) * 34];
                reward[p] = distance_delta_reward(&self.cache, previous_distance[p], player_tiles);
            }
            (self.reset(), reward, true, [false; 4])
        } else {
            let mut reward = [0f32; 4];
            for _p in 0..4 {
                self.draw();
            }

            let mut has_complete_hand = false;
            let mut winners = [false; 4];
            for p in 0..4 {
                let player_tiles = &self.player_tiles[p * 34..(p + 1) * 34];
                let (player_reward, is_complete) = distance_delta_reward_with_completion(
                    &self.cache,
                    previous_distance[p],
                    player_tiles,
                );
                reward[p] = player_reward;
                has_complete_hand |= is_complete;
                winners[p] = is_complete;
            }

            // Previous average discard reward:
            // reward[p] = average_distance_after_discard(before_discard)
            //     - completion_distance(after_discard_13_tiles);
            //
            // Previous distance reward:
            // reward[p] = -completion_distance(after_discard_13_tiles)
            //
            // Original sparse reward:
            // let mut forms = Vec::with_capacity(3);
            // for s in 0..3 {
            //     let suit_tiles = &player_tiles[s * 9..(s + 1) * 9];
            //     match self.cache.form_map.get(suit_tiles) {
            //         Some(v) => forms.push(v),
            //         None => continue 'player,
            //     }
            // }
            // reward[p] = State::enumerate_points(&forms, &player_tiles[27..34]);
            if has_complete_hand {
                (self.reset(), reward, true, winners)
            } else {
                (self.player_tiles, reward, false, winners)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state() {
        let cache = Arc::new(Cache::new());
        let mut state = State::new(cache.clone(), 1);

        let observation = state.reset();
        for chunk in observation.chunks_exact(34) {
            assert_eq!(chunk.iter().sum::<u8>(), 14);
        }

        assert_eq!(
            cache.completion_distance(
                &[
                    1, 4, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ],
                1,
                4
            ),
            0
        );
        assert_eq!(
            cache.completion_distance(
                &[
                    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 1, 0, 3, 0, 0, 0,
                ],
                1,
                4
            ),
            3
        );
        assert_eq!(
            cache.completion_distance(
                &[
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 0, 0,
                    0, 1, 0, 0, 0, 2, 0, 1
                ],
                1,
                4
            ),
            4
        );
        assert_eq!(
            cache.completion_distance(
                &[
                    0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
                    0, 1, 1, 0, 1, 1, 0, 1
                ],
                1,
                4
            ),
            7
        );
        assert_eq!(
            cache.completion_distance(
                &[
                    0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0,
                    3, 0, 0, 1, 0, 0, 0, 0
                ],
                1,
                4
            ),
            4
        );
    }
}
