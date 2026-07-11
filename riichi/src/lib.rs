#[path = "MjaiKyokuStateMachine/mod.rs"]
mod mjai_kyoku_state_machine;
mod game;
mod reward;

use std::{sync::Arc, thread};

use numpy::{IntoPyArray, PyArray3, PyArrayMethods, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};

use crate::mjai_kyoku_state_machine::MjaiKyokuStateMachineManager;
use crate::game::{Cache, State};

const ENVS_PER_THREAD: usize = 8;

#[pyclass]
struct VecEnv {
    states: Vec<State>,
}

#[pymethods]
impl VecEnv {
    #[new]
    fn new(num_envs: usize, seed: u64) -> VecEnv {
        let cache = Arc::new(Cache::new());
        let mut states = Vec::with_capacity(num_envs);
        for i in 0..num_envs {
            states.push(State::new(cache.clone(), seed + i as u64))
        }
        VecEnv { states }
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let num_envs = self.states.len();
        let mut player_tiles_by_env = vec![[0u8; 136]; num_envs];

        py.detach(|| {
            thread::scope(|scope| {
                for (state_chunk, output_chunk) in self
                    .states
                    .chunks_mut(ENVS_PER_THREAD)
                    .zip(player_tiles_by_env.chunks_mut(ENVS_PER_THREAD))
                {
                    scope.spawn(move || {
                        for (state, output) in state_chunk.iter_mut().zip(output_chunk.iter_mut()) {
                            *output = state.reset();
                        }
                    });
                }
            });
        });

        let mut player_tiles_vec = Vec::with_capacity(num_envs * 136);
        for player_tiles in &player_tiles_by_env {
            player_tiles_vec.extend_from_slice(player_tiles);
        }
        player_tiles_vec
            .into_pyarray(py)
            .reshape((num_envs, 4, 34))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        discard_vec: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (player_tiles_vec, reward_vec, done_vec, _winner_vec, num_envs) =
            self.step_batch(py, discard_vec)?;

        Ok((
            player_tiles_vec
                .into_pyarray(py)
                .reshape((num_envs, 4, 34))?,
            reward_vec.into_pyarray(py).reshape((num_envs, 4))?,
            done_vec.into_pyarray(py).reshape((num_envs,))?,
        )
            .into_pyobject(py)?)
    }

    fn step_with_winners<'py>(
        &mut self,
        py: Python<'py>,
        discard_vec: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (player_tiles_vec, reward_vec, done_vec, winner_vec, num_envs) =
            self.step_batch(py, discard_vec)?;

        Ok((
            player_tiles_vec
                .into_pyarray(py)
                .reshape((num_envs, 4, 34))?,
            reward_vec.into_pyarray(py).reshape((num_envs, 4))?,
            done_vec.into_pyarray(py).reshape((num_envs,))?,
            winner_vec.into_pyarray(py).reshape((num_envs, 4))?,
        )
            .into_pyobject(py)?)
    }

    fn step_batch(
        &mut self,
        py: Python<'_>,
        discard_vec: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<(Vec<u8>, Vec<f32>, Vec<bool>, Vec<bool>, usize)> {
        let num_envs = self.states.len();
        let discard_array = discard_vec.to_owned_array();
        if discard_array.shape() != [num_envs, 4] {
            return Err(PyValueError::new_err("discard_vec must have shape [num_envs, 4]"));
        }
        let discard_by_env = discard_array
            .outer_iter()
            .map(|row| [row[0], row[1], row[2], row[3]])
            .collect::<Vec<[u8; 4]>>();
        let mut player_tiles_by_env = vec![[0u8; 136]; num_envs];
        let mut reward_by_env = vec![[0f32; 4]; num_envs];
        let mut done_vec = vec![false; num_envs];
        let mut winner_by_env = vec![[false; 4]; num_envs];

        py.detach(|| {
            thread::scope(|scope| {
                for (
                    (((state_chunk, discard_chunk), player_tiles_chunk), reward_chunk),
                    (done_chunk, winner_chunk),
                ) in
                    self.states
                        .chunks_mut(ENVS_PER_THREAD)
                        .zip(discard_by_env.chunks(ENVS_PER_THREAD))
                        .zip(player_tiles_by_env.chunks_mut(ENVS_PER_THREAD))
                        .zip(reward_by_env.chunks_mut(ENVS_PER_THREAD))
                        .zip(
                            done_vec
                                .chunks_mut(ENVS_PER_THREAD)
                                .zip(winner_by_env.chunks_mut(ENVS_PER_THREAD)),
                        )
                {
                    scope.spawn(move || {
                        for index in 0..state_chunk.len() {
                            let (player_tiles, reward, done, winners) =
                                state_chunk[index].step_with_winners(&discard_chunk[index]);
                            player_tiles_chunk[index] = player_tiles;
                            reward_chunk[index] = reward;
                            done_chunk[index] = done;
                            winner_chunk[index] = winners;
                        }
                    });
                }
            });
        });

        let mut player_tiles_vec = Vec::with_capacity(num_envs * 136);
        for player_tiles in &player_tiles_by_env {
            player_tiles_vec.extend_from_slice(player_tiles);
        }
        let mut reward_vec = Vec::with_capacity(num_envs * 4);
        for reward in &reward_by_env {
            reward_vec.extend_from_slice(reward);
        }
        let mut winner_vec = Vec::with_capacity(num_envs * 4);
        for winners in &winner_by_env {
            winner_vec.extend_from_slice(winners);
        }

        Ok((player_tiles_vec, reward_vec, done_vec, winner_vec, num_envs))
    }
}

#[pymodule]
fn riichi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_class::<MjaiKyokuStateMachineManager>()?;
    Ok(())
}
