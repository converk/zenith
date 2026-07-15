#[pyclass]
pub struct MjaiKyokuStateMachineManager {
    tables: Vec<TableStateMachine>,
}

#[pymethods]
impl MjaiKyokuStateMachineManager {
    #[new]
    #[pyo3(signature = (num_envs, reveal_opponent_initial_hands=true))]
    fn new(num_envs: usize, reveal_opponent_initial_hands: bool) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be greater than 0"));
        }
        Ok(Self {
            tables: (0..num_envs)
                .map(|_| TableStateMachine::new(reveal_opponent_initial_hands))
                .collect(),
        })
    }

    #[getter]
    fn num_envs(&self) -> usize { self.tables.len() }

    #[getter]
    fn num_players(&self) -> usize { NUM_PLAYERS }

    #[getter]
    fn token_dim(&self) -> usize { TOKEN_DIM }

    /// Apply one event delta per player for every selected table.
    ///
    /// `events_by_env_player` has shape `[B][4][events]`.  A player may have
    /// an empty list: RiichiEnv event cursors are intentionally per-player.
    /// The returned `(end_kyoku, end_game)` arrays are aligned with
    /// `env_indices`, and are true when any player view saw that boundary.
    fn apply_events_batch<'py>(
        &mut self,
        py: Python<'py>,
        env_indices: Vec<usize>,
        events_by_env_player: Vec<Vec<Vec<String>>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if env_indices.len() != events_by_env_player.len() {
            return Err(PyValueError::new_err("env_indices and events_by_env_player must have the same length"));
        }
        let table_count = self.tables.len();
        let mut all_events: Vec<Option<[Vec<MjaiEvent>; NUM_PLAYERS]>> =
            (0..table_count).map(|_| None).collect();
        let mut end_kyoku = vec![false; env_indices.len()];
        let mut end_game = vec![false; env_indices.len()];

        for (row, (env_index, by_player)) in env_indices.iter().copied().zip(events_by_env_player).enumerate() {
            if env_index >= table_count {
                return Err(PyValueError::new_err("env_index is out of range"));
            }
            if by_player.len() != NUM_PLAYERS {
                return Err(PyValueError::new_err(format!("each event row must contain {NUM_PLAYERS} player lists")));
            }
            if all_events[env_index].is_some() {
                return Err(PyValueError::new_err("env_indices must not contain duplicates"));
            }
            let mut parsed: [Vec<MjaiEvent>; NUM_PLAYERS] = std::array::from_fn(|_| Vec::new());
            for (player_index, jsons) in by_player.into_iter().enumerate() {
                let mut events = Vec::with_capacity(jsons.len());
                for json in jsons {
                    let event = parse_event(&json)?;
                    end_kyoku[row] |= matches!(event, MjaiEvent::EndKyoku);
                    end_game[row] |= matches!(event, MjaiEvent::EndGame);
                    events.push(event);
                }
                parsed[player_index] = events;
            }
            all_events[env_index] = Some(parsed);
        }

        py.detach(|| {
            thread::scope(|scope| -> Result<(), String> {
                let mut handles = Vec::new();
                for (tables, event_rows) in self.tables.chunks_mut(ENVS_PER_THREAD).zip(all_events.chunks_mut(ENVS_PER_THREAD)) {
                    handles.push(scope.spawn(move || -> Result<(), String> {
                        for (table, row) in tables.iter_mut().zip(event_rows.iter_mut()) {
                            if let Some(by_player) = row.take() {
                                for (player_index, events) in by_player.into_iter().enumerate() {
                                    if !events.is_empty() {
                                        table.apply_player_events(player_index as u8, events)?;
                                    }
                                }
                            }
                        }
                        Ok(())
                    }));
                }
                for handle in handles {
                    handle.join().map_err(|_| "state-machine worker thread panicked".to_owned())??;
                }
                Ok(())
            })
        }).map_err(PyValueError::new_err)?;

        Ok((end_kyoku.into_pyarray(py), end_game.into_pyarray(py)).into_pyobject(py)?)
    }

    /// Atomically records legal actions and materializes only active model
    /// rows. Returns `(input_ids, attention_mask, sequence_lengths, mask,
    /// history_lengths, history_generations)`.
    fn prepare_decisions<'py>(
        &mut self,
        py: Python<'py>,
        batch_indices: Vec<usize>,
        legal_actions: Vec<Vec<String>>,
        snapshot_jsons: Vec<String>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if batch_indices.len() != legal_actions.len() || batch_indices.len() != snapshot_jsons.len() {
            return Err(PyValueError::new_err("batch_indices, legal_actions and snapshot_jsons must have the same length"));
        }
        let batch_size = batch_indices.len();
        let mut seen = vec![false; self.tables.len() * NUM_PLAYERS];
        let mut snapshots = Vec::with_capacity(batch_size);
        for (row, ((batch_index, actions), snapshot_json)) in batch_indices.iter().copied().zip(legal_actions).zip(snapshot_jsons).enumerate() {
            if batch_index >= seen.len() {
                return Err(PyValueError::new_err("batch_index is out of range"));
            }
            if seen[batch_index] {
                return Err(PyValueError::new_err("batch_indices must not contain duplicates"));
            }
            seen[batch_index] = true;
            let env_index = batch_index / NUM_PLAYERS;
            let player_index = (batch_index % NUM_PLAYERS) as u8;
            let snapshot = serde_json::from_str::<DecisionSnapshot>(&snapshot_json)
                .map_err(|error| PyValueError::new_err(format!("invalid snapshot JSON: {error}")))?;
            let table = &mut self.tables[env_index];
            table.set_legal_action_jsons(player_index, actions).map_err(PyValueError::new_err)?;
            let player = &table.players[player_index as usize];
            let snapshot_tokens = player.snapshot_tokens(&snapshot).map_err(PyValueError::new_err)?;
            snapshots.push((row, batch_index, snapshot_tokens));
        }

        let max_length = snapshots.iter().map(|(_, batch_index, snapshot)| {
            let table = &self.tables[*batch_index / NUM_PLAYERS];
            table.players[*batch_index % NUM_PLAYERS].tokens.len() + snapshot.len()
        }).max().unwrap_or(0);
        let mut input_ids = vec![TYPE_PAD; batch_size * max_length * TOKEN_DIM];
        let mut attention_mask = vec![false; batch_size * max_length];
        let mut sequence_lengths = vec![0i64; batch_size];
        let mut history_lengths = vec![0i64; batch_size];
        let mut history_generations = vec![0i64; batch_size];
        let mut masks = vec![false; batch_size * NUM_ACTIONS];
        for (row, batch_index, snapshot) in snapshots {
            let table = &self.tables[batch_index / NUM_PLAYERS];
            let player = &table.players[batch_index % NUM_PLAYERS];
            let length = player.tokens.len() + snapshot.len();
            sequence_lengths[row] = length as i64;
            history_lengths[row] = player.tokens.len() as i64;
            history_generations[row] = table.history_generations[batch_index % NUM_PLAYERS] as i64;
            let input_offset = row * max_length * TOKEN_DIM;
            for (token_index, token) in player.tokens.iter().chain(snapshot.iter()).enumerate() {
                let offset = input_offset + token_index * TOKEN_DIM;
                input_ids[offset..offset + TOKEN_DIM].copy_from_slice(token);
                attention_mask[row * max_length + token_index] = true;
            }
            masks[row * NUM_ACTIONS..(row + 1) * NUM_ACTIONS]
                .copy_from_slice(&table.action_mask((batch_index % NUM_PLAYERS) as u8));
        }
        Ok((
            input_ids.into_pyarray(py).reshape((batch_size, max_length, TOKEN_DIM))?,
            attention_mask.into_pyarray(py).reshape((batch_size, max_length))?,
            sequence_lengths.into_pyarray(py),
            masks.into_pyarray(py).reshape((batch_size, NUM_ACTIONS))?,
            history_lengths.into_pyarray(py),
            history_generations.into_pyarray(py),
        ).into_pyobject(py)?)
    }

    /// Decode selected fixed action ids into the exact corresponding RiichiEnv
    /// MJAI JSON templates.
    fn decode_actions(&self, batch_indices: Vec<usize>, action_ids: Vec<usize>) -> PyResult<Vec<String>> {
        if batch_indices.len() != action_ids.len() {
            return Err(PyValueError::new_err("batch_indices and action_ids must have the same length"));
        }
        batch_indices.into_iter().zip(action_ids).map(|(batch_index, action_id)| {
            let env_index = batch_index / NUM_PLAYERS;
            let player_index = (batch_index % NUM_PLAYERS) as u8;
            let table = self.tables.get(env_index).ok_or_else(|| PyValueError::new_err("batch_index is out of range"))?;
            table.requested_action_to_env(player_index, action_id)
                .map_err(PyValueError::new_err)?
                .ok_or_else(|| PyValueError::new_err("no legal-action request for batch_index"))
        }).collect()
    }
}

// Legacy one-row helpers remain available to the Rust-only protocol tests so
// those tests can construct compact fixtures. They are intentionally not
// Python methods and therefore are not part of the training extension API.
#[cfg(test)]
impl MjaiKyokuStateMachineManager {
    fn apply_player_events(&mut self, batch_index: usize, event_jsons: Vec<String>) -> PyResult<()> {
        let env_index = batch_index / NUM_PLAYERS;
        let player_index = (batch_index % NUM_PLAYERS) as u8;
        let table = self.tables.get_mut(env_index).ok_or_else(|| PyValueError::new_err("batch_index is out of range"))?;
        let events = event_jsons.into_iter().map(|json| parse_event(&json)).collect::<PyResult<Vec<_>>>()?;
        table.apply_player_events(player_index, events).map_err(PyValueError::new_err)
    }

    fn set_legal_actions(&mut self, batch_index: usize, action_jsons: Vec<String>) -> PyResult<()> {
        let env_index = batch_index / NUM_PLAYERS;
        let player_index = (batch_index % NUM_PLAYERS) as u8;
        self.tables.get_mut(env_index)
            .ok_or_else(|| PyValueError::new_err("batch_index is out of range"))?
            .set_legal_action_jsons(player_index, action_jsons)
            .map_err(PyValueError::new_err)
    }
}
