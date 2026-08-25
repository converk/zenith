#[pyclass]
pub struct MjaiKyokuStateMachineManager {
    tables: Vec<TableStateMachine>,
}

#[pymethods]
impl MjaiKyokuStateMachineManager {
    #[new]
    fn new(num_envs: usize) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be greater than 0"));
        }
        Ok(Self {
            tables: (0..num_envs)
                .map(|_| TableStateMachine::new())
                .collect(),
        })
    }

    #[getter]
    fn num_envs(&self) -> usize { self.tables.len() }

    #[getter]
    fn num_players(&self) -> usize { NUM_PLAYERS }

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

    /// Atomically records legal actions and materializes semantic tokens.
    /// The append-only public event prefix and temporary current-state suffix
    /// are concatenated here; the learned query token belongs to the model.
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
            let tokens = table.players[player_index as usize].tokens(&snapshot).map_err(PyValueError::new_err)?;
            snapshots.push((row, batch_index, tokens));
        }

        let max_length = snapshots.iter().map(|(_, _, tokens)| tokens.len()).max().unwrap_or(0);
        let mut factors = vec![0u8; batch_size * max_length * TOKEN_WIDTH];
        let mut numeric = vec![0f32; batch_size * max_length * NUMERIC_WIDTH];
        let mut token_lengths = vec![0i64; batch_size];
        let mut history_generations = vec![0i64; batch_size];
        let mut masks = vec![false; batch_size * NUM_ACTIONS];
        for (row, batch_index, tokens) in snapshots {
            let table = &self.tables[batch_index / NUM_PLAYERS];
            token_lengths[row] = tokens.len() as i64;
            history_generations[row] = table.history_generations[batch_index % NUM_PLAYERS] as i64;
            for (token_index, token) in tokens.into_iter().enumerate() {
                let factor_offset = (row * max_length + token_index) * TOKEN_WIDTH;
                factors[factor_offset..factor_offset + TOKEN_WIDTH].copy_from_slice(&token.factors);
                let numeric_offset = (row * max_length + token_index) * NUMERIC_WIDTH;
                numeric[numeric_offset..numeric_offset + NUMERIC_WIDTH].copy_from_slice(&token.numeric);
            }
            masks[row * NUM_ACTIONS..(row + 1) * NUM_ACTIONS]
                .copy_from_slice(&table.action_mask((batch_index % NUM_PLAYERS) as u8));
        }
        Ok((
            factors.into_pyarray(py).reshape((batch_size, max_length, TOKEN_WIDTH))?,
            numeric.into_pyarray(py).reshape((batch_size, max_length, NUMERIC_WIDTH))?,
            token_lengths.into_pyarray(py),
            masks.into_pyarray(py).reshape((batch_size, NUM_ACTIONS))?,
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

    /// 返回固定 action id 到调用方合法动作列表下标的直接映射。
    ///
    /// prepare 可据此复用原始 Action 对象,无需把 MJAI JSON 解码后再解析匹配。
    fn action_ids_with_source_indices(
        &self,
        batch_indices: Vec<usize>,
    ) -> PyResult<Vec<Vec<(usize, usize)>>> {
        batch_indices
            .into_iter()
            .map(|batch_index| {
                let env_index = batch_index / NUM_PLAYERS;
                let player_index = (batch_index % NUM_PLAYERS) as u8;
                let table = self
                    .tables
                    .get(env_index)
                    .ok_or_else(|| PyValueError::new_err("batch_index is out of range"))?;
                table
                    .action_source_indices(player_index)
                    .map_err(PyValueError::new_err)
            })
            .collect()
    }
}
