#[pyclass]
pub struct MjaiKyokuStateMachineManager {
    tables: Vec<TableStateMachine>,
}

#[pymethods]
impl MjaiKyokuStateMachineManager {
    #[new]
    #[pyo3(signature = (num_envs, game_mode=FLAG_GAME_4P_EAST as u8, reveal_opponent_initial_hands=true))]
    fn new(
        num_envs: usize,
        game_mode: u8,
        reveal_opponent_initial_hands: bool,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be greater than 0"));
        }
        let mode = GameMode::from_flag(game_mode).map_err(PyValueError::new_err)?;
        Ok(Self {
            tables: (0..num_envs)
                .map(|_| TableStateMachine::new(mode, reveal_opponent_initial_hands))
                .collect(),
        })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.tables.len()
    }

    #[getter]
    fn num_players(&self) -> usize {
        NUM_PLAYERS
    }

    #[getter]
    fn token_dim(&self) -> usize {
        TOKEN_DIM
    }

    fn reset(&mut self, py: Python<'_>) {
        py.detach(|| {
            thread::scope(|scope| {
                for table_chunk in self.tables.chunks_mut(ENVS_PER_THREAD) {
                    scope.spawn(move || {
                        for table in table_chunk {
                            table.reset();
                        }
                    });
                }
            });
        });
    }

    fn reset_env(&mut self, env_index: usize) -> PyResult<()> {
        let table = self
            .tables
            .get_mut(env_index)
            .ok_or_else(|| PyValueError::new_err("env_index is out of range"))?;
        table.reset();
        Ok(())
    }

    /// Applies one MJAI JSON event to one table and broadcasts it to four views.
    fn apply_event(&mut self, env_index: usize, event_json: &str) -> PyResult<()> {
        let event = parse_event(event_json)?;
        let table = self
            .tables
            .get_mut(env_index)
            .ok_or_else(|| PyValueError::new_err("env_index is out of range"))?;
        table.apply_event(event).map_err(PyValueError::new_err)
    }

    /// Applies one optional MJAI JSON event per table. `None` means no event for that table.
    fn apply_events(&mut self, py: Python<'_>, event_jsons: Vec<Option<String>>) -> PyResult<()> {
        if event_jsons.len() != self.tables.len() {
            return Err(PyValueError::new_err(
                "event_jsons must contain one entry per environment",
            ));
        }
        let mut events = Vec::with_capacity(event_jsons.len());
        for event_json in event_jsons {
            events.push(event_json.as_deref().map(parse_event).transpose()?);
        }
        let mut errors = vec![None; self.tables.len()];
        py.detach(|| {
            thread::scope(|scope| {
                for ((table_chunk, event_chunk), error_chunk) in self
                    .tables
                    .chunks_mut(ENVS_PER_THREAD)
                    .zip(events.chunks(ENVS_PER_THREAD))
                    .zip(errors.chunks_mut(ENVS_PER_THREAD))
                {
                    scope.spawn(move || {
                        for ((table, event), error) in table_chunk
                            .iter_mut()
                            .zip(event_chunk)
                            .zip(error_chunk.iter_mut())
                        {
                            if let Some(event) = event {
                                if let Err(message) = table.apply_event(event.clone()) {
                                    *error = Some(message);
                                }
                            }
                        }
                    });
                }
            });
        });
        if let Some((env_index, message)) = errors
            .into_iter()
            .enumerate()
            .find_map(|(index, error)| error.map(|message| (index, message)))
        {
            return Err(PyValueError::new_err(format!("env {env_index}: {message}")));
        }
        Ok(())
    }

    /// Handles a RiichiEnv/RiichiLab-style message. `action_ack` is ignored, `request_action`
    /// records possible actions for one player, and all other messages are treated as MJAI events.
    fn apply_env_message(
        &mut self,
        env_index: usize,
        player_index: u8,
        message_json: &str,
    ) -> PyResult<()> {
        let table = self
            .tables
            .get_mut(env_index)
            .ok_or_else(|| PyValueError::new_err("env_index is out of range"))?;
        match message_type(message_json)?.as_str() {
            "action_ack" => Ok(()),
            "request_action" => {
                let request = parse_request_action(message_json)?;
                table
                    .apply_request(player_index, request)
                    .map_err(PyValueError::new_err)
            }
            _ => {
                let event = parse_event(message_json)?;
                table.apply_event(event).map_err(PyValueError::new_err)
            }
        }
    }

    /// Records one RiichiEnv/RiichiLab `request_action` message for a player.
    fn apply_request(
        &mut self,
        env_index: usize,
        player_index: u8,
        request_json: &str,
    ) -> PyResult<()> {
        let request = parse_request_action(request_json)?;
        let table = self
            .tables
            .get_mut(env_index)
            .ok_or_else(|| PyValueError::new_err("env_index is out of range"))?;
        table
            .apply_request(player_index, request)
            .map_err(PyValueError::new_err)
    }

    /// Records optional `request_action` messages in env-major player order.
    fn apply_requests(&mut self, py: Python<'_>, requests: Vec<Option<String>>) -> PyResult<()> {
        let expected = self.tables.len() * NUM_PLAYERS;
        if requests.len() != expected {
            return Err(PyValueError::new_err(format!(
                "requests must contain {expected} entries in env-major player order"
            )));
        }
        let mut parsed_requests = Vec::with_capacity(expected);
        for request_json in requests {
            parsed_requests.push(request_json.as_deref().map(parse_request_action).transpose()?);
        }

        let mut errors = vec![None; self.tables.len()];
        py.detach(|| {
            thread::scope(|scope| {
                for (chunk_index, (table_chunk, error_chunk)) in self
                    .tables
                    .chunks_mut(ENVS_PER_THREAD)
                    .zip(errors.chunks_mut(ENVS_PER_THREAD))
                    .enumerate()
                {
                    let request_start = chunk_index * ENVS_PER_THREAD * NUM_PLAYERS;
                    let request_end = request_start + table_chunk.len() * NUM_PLAYERS;
                    let request_chunk = &parsed_requests[request_start..request_end];
                    scope.spawn(move || {
                        for (env_offset, table) in table_chunk.iter_mut().enumerate() {
                            for player_index in 0..NUM_PLAYERS {
                                let request_index = env_offset * NUM_PLAYERS + player_index;
                                if let Some(request) = &request_chunk[request_index] {
                                    if let Err(message) =
                                        table.apply_request(player_index as u8, request.clone())
                                    {
                                        error_chunk[env_offset] = Some(message);
                                        break;
                                    }
                                }
                            }
                        }
                    });
                }
            });
        });
        if let Some((env_index, message)) = errors
            .into_iter()
            .enumerate()
            .find_map(|(index, error)| error.map(|message| (index, message)))
        {
            return Err(PyValueError::new_err(format!("env {env_index}: {message}")));
        }
        Ok(())
    }

    /// Converts currently recorded RiichiEnv/RiichiLab possible actions to a 241-dim mask.
    fn action_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let batch_size = self.tables.len() * NUM_PLAYERS;
        let mut mask = vec![false; batch_size * NUM_ACTIONS];
        py.detach(|| {
            thread::scope(|scope| {
                for (table_chunk, mask_chunk) in self
                    .tables
                    .chunks(ENVS_PER_THREAD)
                    .zip(mask.chunks_mut(ENVS_PER_THREAD * NUM_PLAYERS * NUM_ACTIONS))
                {
                    scope.spawn(move || {
                        for (env_offset, table) in table_chunk.iter().enumerate() {
                            for player_index in 0..NUM_PLAYERS {
                                let player_mask = table.action_mask(player_index as u8);
                                let offset =
                                    (env_offset * NUM_PLAYERS + player_index) * NUM_ACTIONS;
                                mask_chunk[offset..offset + NUM_ACTIONS]
                                    .copy_from_slice(&player_mask);
                            }
                        }
                    });
                }
            });
        });
        mask.into_pyarray(py).reshape((batch_size, NUM_ACTIONS))
    }

    /// Decodes one player's fixed KyokuActionSpace V2 action id into MJAI JSON.
    fn action_to_mjai(
        &self,
        env_index: usize,
        player_index: u8,
        action_id: usize,
    ) -> PyResult<String> {
        let table = self
            .tables
            .get(env_index)
            .ok_or_else(|| PyValueError::new_err("env_index is out of range"))?;
        table
            .action_to_mjai(player_index, action_id)
            .map_err(PyValueError::new_err)
    }

    /// Decodes one action id for every `(env, player)` batch row.
    ///
    /// The return order is `[env0_player0, ..., env0_player3, env1_player0, ...]`.
    /// Players without an active decision window return `None`; invalid actions for an active
    /// player raise an error instead of silently emitting a malformed MJAI response.
    fn actions_to_mjai(
        &self,
        py: Python<'_>,
        action_ids: Vec<usize>,
    ) -> PyResult<Vec<Option<String>>> {
        let expected = self.tables.len() * NUM_PLAYERS;
        if action_ids.len() != expected {
            return Err(PyValueError::new_err(format!(
                "action_ids must contain {expected} entries in env-major player order"
            )));
        }

        let mut responses = vec![None; expected];
        let mut errors = vec![None; expected];
        py.detach(|| {
            thread::scope(|scope| {
                for (chunk_index, ((action_chunk, response_chunk), error_chunk)) in action_ids
                    .chunks(PLAYERS_PER_THREAD)
                    .zip(responses.chunks_mut(PLAYERS_PER_THREAD))
                    .zip(errors.chunks_mut(PLAYERS_PER_THREAD))
                    .enumerate()
                {
                    let batch_start = chunk_index * PLAYERS_PER_THREAD;
                    scope.spawn(move || {
                        for (local_index, action_id) in action_chunk.iter().enumerate() {
                            let batch_index = batch_start + local_index;
                            let env_index = batch_index / NUM_PLAYERS;
                            let player_index = (batch_index % NUM_PLAYERS) as u8;
                            let table = &self.tables[env_index];
                            if table.player_can_act(player_index) {
                                match table.action_to_mjai(player_index, *action_id) {
                                    Ok(response) => response_chunk[local_index] = Some(response),
                                    Err(message) => error_chunk[local_index] = Some(message),
                                }
                            }
                        }
                    });
                }
            });
        });
        if let Some((batch_index, message)) = errors
            .into_iter()
            .enumerate()
            .find_map(|(index, error)| error.map(|message| (index, message)))
        {
            return Err(PyValueError::new_err(format!("batch {batch_index}: {message}")));
        }
        Ok(responses)
    }

    /// Converts model action ids to RiichiEnv/RiichiLab responses. Only players with a recorded
    /// request return a JSON action, and the response echoes that request_id.
    fn model_to_env(
        &self,
        py: Python<'_>,
        action_ids: Vec<usize>,
    ) -> PyResult<Vec<Option<String>>> {
        let expected = self.tables.len() * NUM_PLAYERS;
        if action_ids.len() != expected {
            return Err(PyValueError::new_err(format!(
                "action_ids must contain {expected} entries in env-major player order"
            )));
        }

        let mut responses = vec![None; expected];
        let mut errors = vec![None; expected];
        py.detach(|| {
            thread::scope(|scope| {
                for (chunk_index, ((action_chunk, response_chunk), error_chunk)) in action_ids
                    .chunks(PLAYERS_PER_THREAD)
                    .zip(responses.chunks_mut(PLAYERS_PER_THREAD))
                    .zip(errors.chunks_mut(PLAYERS_PER_THREAD))
                    .enumerate()
                {
                    let batch_start = chunk_index * PLAYERS_PER_THREAD;
                    scope.spawn(move || {
                        for (local_index, action_id) in action_chunk.iter().enumerate() {
                            let batch_index = batch_start + local_index;
                            let env_index = batch_index / NUM_PLAYERS;
                            let player_index = (batch_index % NUM_PLAYERS) as u8;
                            match self.tables[env_index]
                                .requested_action_to_env(player_index, *action_id)
                            {
                                Ok(response) => response_chunk[local_index] = response,
                                Err(message) => error_chunk[local_index] = Some(message),
                            }
                        }
                    });
                }
            });
        });
        if let Some((batch_index, message)) = errors
            .into_iter()
            .enumerate()
            .find_map(|(index, error)| error.map(|message| (index, message)))
        {
            return Err(PyValueError::new_err(format!("batch {batch_index}: {message}")));
        }
        Ok(responses)
    }

    /// Returns `(input_ids, attention_mask, sequence_lengths)` for all envs and players.
    fn model_inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let batch_size = self.tables.len() * NUM_PLAYERS;
        let max_length = self
            .tables
            .iter()
            .flat_map(|table| table.players.iter())
            .map(|player| player.tokens.len())
            .max()
            .unwrap_or(0);
        let mut input_ids = vec![TYPE_PAD; batch_size * max_length * TOKEN_DIM];
        let mut attention_mask = vec![false; batch_size * max_length];
        let mut sequence_lengths = Vec::with_capacity(batch_size);

        sequence_lengths.resize(batch_size, 0);
        if max_length > 0 {
            py.detach(|| {
                thread::scope(|scope| {
                    for (((table_chunk, length_chunk), input_chunk), attention_chunk) in self
                        .tables
                        .chunks(ENVS_PER_THREAD)
                        .zip(sequence_lengths.chunks_mut(ENVS_PER_THREAD * NUM_PLAYERS))
                        .zip(
                            input_ids.chunks_mut(
                                ENVS_PER_THREAD * NUM_PLAYERS * max_length * TOKEN_DIM,
                            ),
                        )
                        .zip(
                            attention_mask
                                .chunks_mut(ENVS_PER_THREAD * NUM_PLAYERS * max_length),
                        )
                    {
                        scope.spawn(move || {
                            for (env_offset, table) in table_chunk.iter().enumerate() {
                                for (player_index, player) in table.players.iter().enumerate() {
                                    let local_batch_index =
                                        env_offset * NUM_PLAYERS + player_index;
                                    length_chunk[local_batch_index] = player.tokens.len() as i64;
                                    for (token_index, token) in player.tokens.iter().enumerate() {
                                        let offset = (local_batch_index * max_length + token_index)
                                            * TOKEN_DIM;
                                        input_chunk[offset..offset + TOKEN_DIM]
                                            .copy_from_slice(token);
                                        attention_chunk
                                            [local_batch_index * max_length + token_index] = true;
                                    }
                                }
                            }
                        });
                    }
                });
            });
        }

        Ok((
            input_ids
                .into_pyarray(py)
                .reshape((batch_size, max_length, TOKEN_DIM))?,
            attention_mask.into_pyarray(py).reshape((batch_size, max_length))?,
            sequence_lengths.into_pyarray(py),
        )
            .into_pyobject(py)?)
    }

    /// Returns one player's append-only token sequence for tests and debugging.
    fn player_tokens<'py>(
        &self,
        py: Python<'py>,
        env_index: usize,
        player_index: usize,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let table = self
            .tables
            .get(env_index)
            .ok_or_else(|| PyValueError::new_err("env_index is out of range"))?;
        let player = table
            .players
            .get(player_index)
            .ok_or_else(|| PyValueError::new_err("player_index must be in 0..4"))?;
        let mut tokens = Vec::with_capacity(player.tokens.len() * TOKEN_DIM);
        for token in &player.tokens {
            tokens.extend_from_slice(token);
        }
        tokens.into_pyarray(py).reshape((player.tokens.len(), TOKEN_DIM))
    }
}
