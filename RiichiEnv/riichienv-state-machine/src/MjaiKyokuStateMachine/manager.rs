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

        (end_kyoku.into_pyarray(py), end_game.into_pyarray(py)).into_pyobject(py)
    }

    /// Atomically records legal actions and returns the fixed 241-action legal mask。
    ///
    /// 旧 per-player semantic-token 历史/快照材料已删除(V18 输入由
    /// `prepare_current_state_batch` + `encode_query_batch` 独立装配);
    /// 本方法只保留「固定动作空间」登记与合法掩码职责。
    fn prepare_decisions<'py>(
        &mut self,
        py: Python<'py>,
        batch_indices: Vec<usize>,
        legal_actions: Vec<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<bool>>> {
        if batch_indices.len() != legal_actions.len() {
            return Err(PyValueError::new_err(
                "batch_indices and legal_actions must have the same length",
            ));
        }
        let batch_size = batch_indices.len();
        let mut seen = vec![false; self.tables.len() * NUM_PLAYERS];
        let mut masks = vec![false; batch_size * NUM_ACTIONS];
        for (row, (batch_index, actions)) in batch_indices.iter().copied().zip(legal_actions).enumerate() {
            if batch_index >= seen.len() {
                return Err(PyValueError::new_err("batch_index is out of range"));
            }
            if seen[batch_index] {
                return Err(PyValueError::new_err(
                    "batch_indices must not contain duplicates",
                ));
            }
            seen[batch_index] = true;
            let player_index = (batch_index % NUM_PLAYERS) as u8;
            let table = &mut self.tables[batch_index / NUM_PLAYERS];
            table
                .set_legal_action_jsons(player_index, actions)
                .map_err(PyValueError::new_err)?;
            masks[row * NUM_ACTIONS..(row + 1) * NUM_ACTIONS]
                .copy_from_slice(&table.action_mask(player_index));
        }
        masks
            .into_pyarray(py)
            .reshape((batch_size, NUM_ACTIONS))
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
