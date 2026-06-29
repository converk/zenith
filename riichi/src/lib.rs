mod game;

use std::sync::Arc;

use numpy::{ndarray::Axis, IntoPyArray, PyArray3, PyArrayMethods, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyTuple};

use crate::game::{Cache, State};

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
        let mut player_tiles_vec = Vec::new();
        for state in &mut self.states {
            let player_tiles = state.reset();
            player_tiles_vec.extend_from_slice(&player_tiles);
        }
        player_tiles_vec
            .into_pyarray(py)
            .reshape((self.states.len(), 4, 34))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        discard_vec: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let discard_vec = discard_vec.to_owned_array();
        let mut player_tiles_vec = Vec::new();
        let mut reward_vec = Vec::new();
        let mut done_vec = Vec::new();

        for (index, state) in self.states.iter_mut().enumerate() {
            let discard = discard_vec.index_axis(Axis(0), index).to_vec();
            let (player_tiles, reward, done) = state.step(&discard);

            player_tiles_vec.extend_from_slice(&player_tiles);
            reward_vec.extend_from_slice(&reward);
            done_vec.push(done);
        }

        Ok((
            player_tiles_vec
                .into_pyarray(py)
                .reshape((self.states.len(), 4, 34))?,
            reward_vec
                .into_pyarray(py)
                .reshape((self.states.len(), 4))?,
            done_vec.into_pyarray(py).reshape((self.states.len(),))?,
        )
            .into_pyobject(py)?)
    }
}

#[pymodule]
fn riichi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    Ok(())
}
