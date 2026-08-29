use pyo3::IntoPyObject;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::thread;

use riichienv_core::action::{Action, Phase};
use riichienv_core::replay::MjaiEvent;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use riichienv_core::state::legal_actions::GameStateLegalActions;
use riichienv_core::types::{Meld, WinResult};

#[pyclass(module = "riichienv._riichienv", from_py_object)]
#[derive(Debug, Clone)]
pub struct RiichiEnv {
    pub state: GameState,
}

/// Shared logic for apply_event / observe_event: optionally reset logs on
/// start_game, apply the parsed event, and push it into mjai_log.
macro_rules! apply_and_log {
    ($state:expr, $ev:expr, $json_val:expr, $is_start_game:expr) => {
        if $is_start_game {
            // Reset logs and related counters/caches so the viewer and
            // observation helpers (e.g. new_events) stay in sync.
            $state.reset();
            // reset() pushes a generic start_game; clear it so we log
            // the caller's original event (which may carry extra fields).
            $state.mjai_log.clear();
            for pl in $state.mjai_log_per_player.iter_mut() {
                pl.clear();
            }
        }
        $state.apply_mjai_event($ev);
        $state._push_mjai_event($json_val);
    };
}

/// Native vector environment for high-throughput self-play.
///
/// It intentionally supports only the training protocol's 4p-red-half mode.
/// Python `Action` objects are extracted while holding the GIL; all native
/// state transitions then run in parallel without it.
#[pyclass(module = "riichienv._riichienv")]
pub struct BatchedRiichiEnv {
    envs: Vec<RiichiEnv>,
    step_threads: usize,
}

impl BatchedRiichiEnv {
    fn reset_native(env: &mut RiichiEnv) {
        let s = &mut env.state;
        s.reset();
        s._initialize_round(0, 0, 0, 0, None, Some(vec![25_000; 4]));
    }

    fn observations_for_env<'py>(env: &mut RiichiEnv, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let state = &mut env.state;
        let mut map = HashMap::new();
        for player in 0..4 {
            map.insert(player, state.get_observation(player));
        }
        map.into_pyobject(py).map(|value| value.unbind().into())
    }

    fn observations_all<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let list = pyo3::types::PyList::empty(py);
        for env in &mut self.envs {
            list.append(Self::observations_for_env(env, py)?)?;
        }
        Ok(list.unbind().into())
    }
}

#[pymethods]
impl BatchedRiichiEnv {
    #[new]
    #[pyo3(signature = (num_envs, seed=0, step_threads=4, game_mode="4p-red-half"))]
    fn new(num_envs: usize, seed: u64, step_threads: usize, game_mode: &str) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("num_envs must be greater than 0"));
        }
        if game_mode != "4p-red-half" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "BatchedRiichiEnv only supports game_mode='4p-red-half'",
            ));
        }
        Ok(Self {
            envs: (0..num_envs)
                .map(|index| RiichiEnv {
                    state: GameState::new(2, false, Some(seed + index as u64), 0, GameRule::default()),
                })
                .collect(),
            step_threads: step_threads.max(1).min(num_envs),
        })
    }

    #[getter]
    fn num_envs(&self) -> usize { self.envs.len() }

    #[getter]
    fn step_threads(&self) -> usize { self.step_threads }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        py.detach(|| {
            thread::scope(|scope| {
                let chunk = self.envs.len().div_ceil(self.step_threads);
                for envs in self.envs.chunks_mut(chunk) {
                    scope.spawn(move || for env in envs { Self::reset_native(env); });
                }
            });
        });
        self.observations_all(py)
    }

    /// Advance every table once. Empty action maps keep an inactive table in
    /// place, which makes mixed response windows safe in a synchronous batch.
    fn step_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions_by_env: Vec<HashMap<u8, Action>>,
    ) -> PyResult<Py<PyAny>> {
        if actions_by_env.len() != self.envs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "actions_by_env length must equal num_envs",
            ));
        }
        let mut actions: Vec<HashMap<u8, Action>> = actions_by_env;
        py.detach(|| {
            thread::scope(|scope| {
                let chunk = self.envs.len().div_ceil(self.step_threads);
                for (envs, action_rows) in self.envs.chunks_mut(chunk).zip(actions.chunks_mut(chunk)) {
                    scope.spawn(move || {
                        for (env, row) in envs.iter_mut().zip(action_rows.iter()) {
                            if !row.is_empty() {
                                env.state.step(row);
                            }
                        }
                    });
                }
            });
        });
        self.observations_all(py)
    }

    /// Reset just-completed games and return observations for every table.
    fn reset_indices<'py>(&mut self, py: Python<'py>, env_indices: Vec<usize>) -> PyResult<Py<PyAny>> {
        let mut selected = vec![false; self.envs.len()];
        for index in env_indices {
            let entry = selected.get_mut(index).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("env_index is out of range"))?;
            if *entry {
                return Err(pyo3::exceptions::PyValueError::new_err("env_indices must not contain duplicates"));
            }
            *entry = true;
        }
        py.detach(|| {
            thread::scope(|scope| {
                let chunk = self.envs.len().div_ceil(self.step_threads);
                for (envs, selected) in self.envs.chunks_mut(chunk).zip(selected.chunks(chunk)) {
                    scope.spawn(move || {
                        for (env, reset) in envs.iter_mut().zip(selected.iter()) {
                            if *reset { Self::reset_native(env); }
                        }
                    });
                }
            });
        });
        self.observations_all(py)
    }

    fn done(&self) -> Vec<bool> {
        self.envs.iter().map(|env| env.state.is_done).collect()
    }

    fn scores(&self) -> Vec<Vec<i32>> {
        self.envs
            .iter()
            .map(|env| env.state.players.iter().map(|p| p.score).collect())
            .collect()
    }

    /// Return every table's remaining wall tiles, one row per env.
    ///
    /// The native wall stores tiles in reverse draw order and pops the next
    /// normal draw from the back, so the public getter returns the vector in
    /// draw order: the live wall first (``tiles[0]`` = next draw, 69 tiles)
    /// followed by the 14 dead-wall tiles.  ``tiles[:5]`` are therefore the
    /// next five live-wall tiles in order.
    fn walls(&self) -> Vec<Vec<u32>> {
        self.envs
            .iter()
            .map(|env| {
                env.state
                    .wall
                    .tiles
                    .iter()
                    .rev()
                    .map(|&tile| tile as u32)
                    .collect()
            })
            .collect()
    }

    fn ranks(&self) -> Vec<Vec<usize>> { self.envs.iter().map(RiichiEnv::ranks).collect() }
}

#[pymethods]
impl RiichiEnv {
    #[new]
    #[pyo3(signature = (game_mode=None, skip_mjai_logging=false, seed=None, round_wind=None, rule=None))]
    pub fn new(
        game_mode: Option<Bound<'_, PyAny>>,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: Option<u8>,
        rule: Option<GameRule>,
    ) -> PyResult<Self> {
        // 本仓库仅支持四麻:3p-* 模式与 >=3 的数值模式一律拒绝。
        let gt = if let Some(val) = game_mode {
            if let Ok(s) = val.extract::<String>() {
                match s.as_str() {
                    "4p-red-single" => 0,
                    "4p-red-east" => 1,
                    "4p-red-half" => 2,
                    other if other.starts_with("3p") => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "3-player (sanma) game modes are not supported",
                        ));
                    }
                    _ => 0,
                }
            } else {
                let gt = val.extract::<u8>().unwrap_or_default();
                if gt >= 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "3-player (sanma) game modes are not supported",
                    ));
                }
                gt
            }
        } else {
            0
        };

        Ok(RiichiEnv {
            state: GameState::new(
                gt,
                skip_mjai_logging,
                seed,
                round_wind.unwrap_or(0),
                rule.unwrap_or_default(),
            ),
        })
    }

    // --- Backward-compatible state getter (4P only) ---

    #[getter]
    pub fn get_state(&self) -> PyResult<GameState> {
        Ok(self.state.clone())
    }

    // --- Delegation Getters/Setters ---

    #[getter]
    pub fn get_wall(&self) -> Vec<u32> {
        self.state.wall.tiles.iter().map(|&x| x as u32).collect()
    }
    #[setter]
    pub fn set_wall(&mut self, v: Vec<u32>) {
        self.state.wall.tiles = v.iter().map(|&x| x as u8).collect();
    }

    #[getter]
    pub fn get_hands(&self) -> Vec<Vec<u32>> {
        self.state
            .players
            .iter()
            .map(|p| p.hand.iter().map(|&x| x as u32).collect())
            .collect()
    }
    #[setter]
    pub fn set_hands(&mut self, v: Vec<Vec<u32>>) {
        if v.len() == 4 {
            for (i, h) in v.into_iter().enumerate() {
                self.state.players[i].hand = h.iter().map(|&x| x as u8).collect();
            }
        }
    }

    #[getter]
    pub fn get_melds(&self) -> Vec<Vec<Meld>> {
        self.state.players.iter().map(|p| p.melds.clone()).collect()
    }
    #[setter]
    pub fn set_melds(&mut self, v: Vec<Vec<Meld>>) {
        if v.len() == 4 {
            for (i, m) in v.into_iter().enumerate() {
                self.state.players[i].melds = m;
            }
        }
    }

    #[getter]
    pub fn get_discards(&self) -> Vec<Vec<u32>> {
        self.state
            .players
            .iter()
            .map(|p| p.discards.iter().map(|&x| x as u32).collect())
            .collect()
    }
    #[setter]
    pub fn set_discards(&mut self, v: Vec<Vec<u32>>) {
        if v.len() == 4 {
            for (i, d) in v.into_iter().enumerate() {
                self.state.players[i].discards = d.iter().map(|&x| x as u8).collect();
            }
        }
    }

    #[getter]
    pub fn get_discard_from_hand(&self) -> Vec<Vec<bool>> {
        self.state
            .players
            .iter()
            .map(|p| p.discard_from_hand.clone())
            .collect()
    }
    #[setter]
    pub fn set_discard_from_hand(&mut self, v: Vec<Vec<bool>>) {
        if v.len() == 4 {
            for (i, d) in v.into_iter().enumerate() {
                self.state.players[i].discard_from_hand = d;
            }
        }
    }

    #[getter]
    pub fn get_discard_is_riichi(&self) -> Vec<Vec<bool>> {
        self.state
            .players
            .iter()
            .map(|p| p.discard_is_riichi.clone())
            .collect()
    }
    #[setter]
    pub fn set_discard_is_riichi(&mut self, v: Vec<Vec<bool>>) {
        if v.len() == 4 {
            for (i, d) in v.into_iter().enumerate() {
                self.state.players[i].discard_is_riichi = d;
            }
        }
    }

    #[getter]
    pub fn get_dora_indicators(&self) -> Vec<u32> {
        self.state
            .wall
            .dora_indicators
            .iter()
            .map(|&x| x as u32)
            .collect()
    }
    #[setter]
    pub fn set_dora_indicators(&mut self, v: Vec<u32>) {
        self.state.wall.dora_indicators = v.iter().map(|&x| x as u8).collect();
    }

    #[getter]
    pub fn get_rinshan_draw_count(&self) -> u8 {
        self.state.wall.rinshan_draw_count
    }
    #[setter]
    pub fn set_rinshan_draw_count(&mut self, v: u8) {
        self.state.wall.rinshan_draw_count = v;
    }

    #[getter]
    pub fn get_pending_kan_dora_count(&self) -> u8 {
        self.state.wall.pending_kan_dora_count
    }
    #[setter]
    pub fn set_pending_kan_dora_count(&mut self, v: u8) {
        self.state.wall.pending_kan_dora_count = v;
    }

    #[getter]
    pub fn get_is_rinshan_flag(&self) -> bool {
        self.state.is_rinshan_flag
    }
    #[setter]
    pub fn set_is_rinshan_flag(&mut self, v: bool) {
        self.state.is_rinshan_flag = v;
    }

    #[getter]
    pub fn get_riichi_declaration_index(&self) -> Vec<Option<usize>> {
        self.state
            .players
            .iter()
            .map(|p| p.riichi_declaration_index)
            .collect()
    }
    #[setter]
    pub fn set_riichi_declaration_index(&mut self, v: Vec<Option<usize>>) {
        if v.len() == 4 {
            for (i, d) in v.into_iter().enumerate() {
                self.state.players[i].riichi_declaration_index = d;
            }
        }
    }

    #[getter]
    pub fn get_current_player(&self) -> u8 {
        self.state.current_player
    }
    #[setter]
    pub fn set_current_player(&mut self, v: u8) {
        self.state.current_player = v;
    }

    #[getter]
    pub fn get_game_mode(&self) -> u8 {
        self.state.game_mode
    }

    #[getter]
    pub fn get_num_players(&self) -> u8 {
        riichienv_core::state::game_mode::num_players()
    }

    #[getter]
    pub fn get_action_space_size(&self) -> usize {
        riichienv_core::action::ACTION_SPACE_4P
    }

    #[getter]
    pub fn get_turn_count(&self) -> u32 {
        self.state.turn_count
    }
    #[setter]
    pub fn set_turn_count(&mut self, v: u32) {
        self.state.turn_count = v;
    }

    #[getter]
    pub fn get_kyoku_idx(&self) -> u8 {
        self.state.kyoku_idx
    }

    #[pyo3(name = "done")]
    pub fn done_method(&self) -> bool {
        self.state.is_done
    }

    /// Return a deep copy of this environment (full game state clone).
    #[pyo3(name = "clone")]
    pub fn py_clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }

    pub fn __copy__(&self) -> Self {
        self.py_clone()
    }

    pub fn __deepcopy__(&self, _memo: &pyo3::Bound<'_, pyo3::types::PyAny>) -> Self {
        self.py_clone()
    }

    #[getter]
    pub fn get_is_done(&self) -> bool {
        self.state.is_done
    }
    #[setter]
    pub fn set_is_done(&mut self, v: bool) {
        self.state.is_done = v;
    }

    #[getter]
    pub fn get_needs_tsumo(&self) -> bool {
        self.state.needs_tsumo
    }
    #[setter]
    pub fn set_needs_tsumo(&mut self, v: bool) {
        self.state.needs_tsumo = v;
    }

    #[getter]
    pub fn get_needs_initialize_next_round(&self) -> bool {
        self.state.needs_initialize_next_round
    }
    #[setter]
    pub fn set_needs_initialize_next_round(&mut self, v: bool) {
        self.state.needs_initialize_next_round = v;
    }

    #[pyo3(name = "scores")]
    pub fn scores_method(&self) -> Vec<i32> {
        self.state.players.iter().map(|p| p.score).collect()
    }
    #[pyo3(name = "set_scores")]
    pub fn set_scores_method(&mut self, v: Vec<i32>) {
        if v.len() == 4 {
            for (i, &sc) in v.iter().enumerate() {
                self.state.players[i].score = sc;
            }
        }
    }
    #[getter]
    pub fn get_scores(&self) -> Vec<i32> {
        self.state.players.iter().map(|p| p.score).collect()
    }
    #[setter]
    pub fn set_scores(&mut self, v: Vec<i32>) {
        if v.len() == 4 {
            for (i, &sc) in v.iter().enumerate() {
                self.state.players[i].score = sc;
            }
        }
    }

    #[getter]
    pub fn get_riichi_sticks(&self) -> u32 {
        self.state.riichi_sticks
    }
    #[setter]
    pub fn set_riichi_sticks(&mut self, v: u32) {
        self.state.riichi_sticks = v;
    }

    #[getter]
    pub fn get_riichi_declared(&self) -> Vec<bool> {
        self.state
            .players
            .iter()
            .map(|p| p.riichi_declared)
            .collect()
    }
    #[setter]
    pub fn set_riichi_declared(&mut self, v: Vec<bool>) {
        if v.len() == 4 {
            for (i, &val) in v.iter().enumerate() {
                self.state.players[i].riichi_declared = val;
            }
        }
    }

    #[getter]
    pub fn get_riichi_stage(&self) -> Vec<bool> {
        self.state.players.iter().map(|p| p.riichi_stage).collect()
    }
    #[setter]
    pub fn set_riichi_stage(&mut self, v: Vec<bool>) {
        if v.len() == 4 {
            for (i, &val) in v.iter().enumerate() {
                self.state.players[i].riichi_stage = val;
            }
        }
    }

    #[getter]
    pub fn get_phase(&self) -> Phase {
        self.state.phase
    }
    #[setter]
    pub fn set_phase(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        let phase = if let Ok(p) = v.extract::<Phase>() {
            p
        } else if let Ok(i) = v.extract::<i32>() {
            match i {
                0 => Phase::WaitAct,
                1 => Phase::WaitResponse,
                _ => Phase::WaitAct,
            }
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected Phase or int",
            ));
        };
        self.state.phase = phase;
        Ok(())
    }

    #[getter]
    pub fn get_active_players(&self) -> Vec<u32> {
        self.state
            .active_players
            .iter()
            .map(|&x| x as u32)
            .collect()
    }
    #[setter]
    pub fn set_active_players(&mut self, v: Vec<u32>) {
        self.state.active_players = v.iter().map(|&x| x as u8).collect();
    }

    #[getter]
    pub fn get_oya(&self) -> u8 {
        self.state.oya
    }
    #[setter]
    pub fn set_oya(&mut self, v: u8) {
        self.state.oya = v;
    }

    #[getter]
    pub fn get_honba(&self) -> u8 {
        self.state.honba
    }
    #[setter]
    pub fn set_honba(&mut self, v: u8) {
        self.state.honba = v;
    }

    #[getter]
    pub fn is_first_turn(&self) -> bool {
        self.state.is_first_turn
    }
    #[setter]
    pub fn set_is_first_turn(&mut self, v: bool) {
        self.state.is_first_turn = v;
    }

    #[getter]
    pub fn get_drawn_tile(&self) -> Option<u8> {
        self.state.drawn_tile
    }
    #[setter]
    pub fn set_drawn_tile(&mut self, v: Option<u8>) {
        self.state.drawn_tile = v;
    }

    #[getter]
    pub fn current_claims(&self) -> HashMap<u8, Vec<Action>> {
        self.state.current_claims.clone()
    }
    #[setter]
    pub fn set_current_claims(&mut self, v: HashMap<u8, Vec<Action>>) {
        self.state.current_claims = v;
    }

    #[getter]
    pub fn get_last_discard(&self) -> Option<(u32, u32)> {
        self.state.last_discard.map(|(a, b)| (a as u32, b as u32))
    }
    #[setter]
    pub fn set_last_discard(&mut self, v: Option<(u32, u32)>) {
        let ld = v.map(|(pid, tile)| (pid as u8, tile as u8));
        self.state.last_discard = ld;
    }

    #[getter]
    pub fn get_pao(&self) -> Vec<HashMap<u8, u8>> {
        self.state.players.iter().map(|p| p.pao.clone()).collect()
    }
    #[setter]
    pub fn set_pao(&mut self, v: Vec<HashMap<u8, u8>>) {
        if v.len() == 4 {
            for (i, p) in v.into_iter().enumerate() {
                self.state.players[i].pao = p;
            }
        }
    }

    #[getter]
    pub fn get_missed_agari_doujun(&self) -> Vec<bool> {
        self.state
            .players
            .iter()
            .map(|p| p.missed_agari_doujun)
            .collect()
    }
    #[setter]
    pub fn set_missed_agari_doujun(&mut self, v: Vec<bool>) {
        if v.len() == 4 {
            for (i, &val) in v.iter().enumerate() {
                self.state.players[i].missed_agari_doujun = val;
            }
        }
    }

    #[getter]
    pub fn get_win_results(&self) -> HashMap<u8, WinResult> {
        self.state.win_results.clone()
    }

    #[getter]
    pub fn get_score_deltas(&self) -> Vec<i32> {
        self.state.players.iter().map(|p| p.score_delta).collect()
    }

    #[getter]
    pub fn get_round_wind(&self) -> u8 {
        self.state.round_wind
    }
    #[setter]
    pub fn set_round_wind(&mut self, v: u8) {
        self.state.round_wind = v;
    }

    pub fn _reveal_kan_dora(&mut self) {
        self.state._reveal_kan_dora();
    }

    pub fn _get_ura_markers(&self) -> Vec<String> {
        self.state._get_ura_markers()
    }

    #[getter(_custom_round_wind)]
    pub fn get_custom_round_wind(&self) -> u8 {
        self.state.round_wind
    }

    // --- Methods ---

    #[pyo3(signature = (oya=None, honba=None, riichi_sticks=None, scores=None, round_wind=None))]
    pub fn set_state(
        &mut self,
        oya: Option<u8>,
        honba: Option<u8>,
        riichi_sticks: Option<u32>,
        scores: Option<Vec<i32>>,
        round_wind: Option<u8>,
    ) {
        let np = riichienv_core::state::game_mode::num_players() as usize;
        let s = &mut self.state;
        if let Some(o) = oya {
            s.oya = o;
            s.kyoku_idx = o;
        }
        if let Some(h) = honba {
            s.honba = h;
        }
        if let Some(r) = riichi_sticks {
            s.riichi_sticks = r;
        }
        if let Some(ref sc) = scores
            && sc.len() == np
        {
            for (i, &val) in sc.iter().enumerate() {
                s.players[i].score = val;
            }
        }
        if let Some(rw) = round_wind {
            s.round_wind = rw;
        }
    }

    pub fn ranks(&self) -> Vec<usize> {
        let np = 4usize;
        let scores: Vec<i32> = self.state.players.iter().map(|p| p.score).collect();
        let mut indices: Vec<usize> = (0..np).collect();
        indices.sort_by(|&a, &b| {
            if scores[a] != scores[b] {
                scores[b].cmp(&scores[a])
            } else {
                a.cmp(&b)
            }
        });
        let mut result = vec![0; np];
        for (rank, &pid) in indices.iter().enumerate() {
            result[pid] = rank + 1;
        }
        result
    }

    pub fn points(&self, rule_name: &str) -> PyResult<Vec<f64>> {
        let np = 4usize;
        let (soten_weight, soten_base, jun_weight) = match rule_name {
            "basic" => (1.0, 25000.0, vec![50.0, 10.0, -10.0, -50.0]),
            "ouza-tyoujyo" => (0.0, 25000.0, vec![100.0, 40.0, -40.0, -100.0]),
            "ouza-normal" => (0.0, 25000.0, vec![50.0, 20.0, -20.0, -50.0]),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown preset rule: {}",
                    rule_name
                )));
            }
        };

        let scores: Vec<i32> = self.state.players.iter().map(|p| p.score).collect();
        let ranks = self.ranks();
        let mut points = vec![0.0; np];
        for i in 0..np {
            let score = scores[i] as f64;
            let rank = ranks[i];
            let uma = jun_weight[rank - 1];
            points[i] = (score - soten_base) / 1000.0 * soten_weight + uma;
        }
        points.into_iter().map(Ok).collect()
    }

    #[getter]
    pub fn mjai_log(&self, py: Python) -> PyResult<Py<PyAny>> {
        let json = py.import("json")?;
        let loads = json.getattr("loads")?;
        let list = pyo3::types::PyList::empty(py);
        for s in &self.state.mjai_log {
            list.append(loads.call1((s,))?)?;
        }
        Ok(list.unbind().into())
    }

    #[pyo3(signature = (players=None))]
    pub fn get_observations<'py>(
        &mut self,
        py: Python<'py>,
        players: Option<Vec<u8>>,
    ) -> PyResult<Py<PyAny>> {
        let np = riichienv_core::state::game_mode::num_players();
        let targets = players.unwrap_or_else(|| (0..np).collect());
        let s = &mut self.state;
        let mut map = HashMap::new();
        for p in targets {
            map.insert(p, s.get_observation(p));
        }
        map.into_pyobject(py).map(|o| o.unbind().into())
    }

    pub fn get_observation<'py>(&mut self, py: Python<'py>, player_id: u8) -> PyResult<Py<PyAny>> {
        self.state
            .get_observation(player_id)
            .into_pyobject(py)
            .map(|o| o.unbind().into())
    }

    fn get_obs_py<'py>(
        &mut self,
        py: Python<'py>,
        players: Option<Vec<u8>>,
    ) -> PyResult<Py<PyAny>> {
        self.get_observations(py, players)
    }

    /// Start a new game (hanchan, tonpuusen, or single-round).
    ///
    /// This resets the entire game to its initial state: scores return to their
    /// starting values, oya/round_wind/honba/kyotaku default to 0 when omitted,
    /// and logs from the previous game are cleared. A new game is then started,
    /// emitting a fresh `start_game` event into the log. All parameters are
    /// optional — when omitted they default to the initial game state
    /// (NOT the previous round's values).
    ///
    /// To re-initialize a single round within an ongoing game (e.g. for
    /// replay validation), pass explicit values for every parameter.
    #[pyo3(signature = (oya=None, wall=None, round_wind=None, scores=None, honba=None, kyotaku=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        oya: Option<u8>,
        wall: Option<Vec<u8>>,
        round_wind: Option<u8>,
        scores: Option<Vec<i32>>,
        honba: Option<u8>,
        kyotaku: Option<u32>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let np = 4usize;

        // Validate scores length when explicitly provided.
        if let Some(ref sc) = scores
            && sc.len() != np
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "scores length {} does not match number of players {}",
                sc.len(),
                np,
            )));
        }

        // For a new game, default to the 4P starting scores (25000).
        let default_scores = vec![riichienv_core::state::game_mode::starting_score(); 4];

        let s = &mut self.state;
        if let Some(sd) = seed {
            s.seed = Some(sd);
        }
        s.reset();
        s._initialize_round(
            oya.unwrap_or(0),
            round_wind.unwrap_or(0),
            honba.unwrap_or(0),
            kyotaku.unwrap_or(0),
            wall,
            Some(scores.unwrap_or(default_scores)),
        );

        let active = s.active_players.clone();
        self.get_obs_py(py, Some(active))
    }

    pub fn _get_legal_actions(&mut self, pid: u8) -> Vec<Action> {
        self.state._get_legal_actions_internal(pid)
    }

    #[pyo3(signature = (actions))]
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: HashMap<u8, Action>,
    ) -> PyResult<Py<PyAny>> {
        self.state.step(&actions);
        if self.state.last_error.is_some() {
            let dict = pyo3::types::PyDict::new(py);
            return Ok(dict.unbind().into());
        }
        let active = self.state.active_players.clone();
        self.get_obs_py(py, Some(active))
    }

    /// Apply an MJAI event to advance the game state.
    ///
    /// Use this for replay parsing and training data generation where
    /// observations are obtained separately via `get_observation()`.
    /// For online inference, prefer `observe_event()` which combines
    /// event application with observation retrieval.
    pub fn apply_event(&mut self, py: Python, event: Py<PyAny>) -> PyResult<()> {
        let (ev, json_val) = Self::parse_mjai_event(py, event)?;
        let is_start_game = matches!(ev, MjaiEvent::StartGame { .. });
        apply_and_log!(self.state, ev, json_val, is_start_game);
        Ok(())
    }

    /// Apply an MJAI event and return the observation for `player_id`
    /// if that player has legal actions available. Returns `None` otherwise.
    ///
    /// This is the recommended API for online inference: feed events
    /// one at a time and act whenever a non-None observation is returned.
    #[pyo3(signature = (event, player_id))]
    pub fn observe_event<'py>(
        &mut self,
        py: Python<'py>,
        event: Py<PyAny>,
        player_id: u8,
    ) -> PyResult<Option<Py<PyAny>>> {
        let (ev, json_val) = Self::parse_mjai_event(py, event)?;

        // Events that never require a player decision.
        // Skipping the observation check for these avoids returning
        // stale legal_actions from uncleared state (e.g. after reset).
        let skip_check = matches!(
            ev,
            MjaiEvent::StartGame { .. }
                | MjaiEvent::StartKyoku { .. }
                | MjaiEvent::ReachAccepted { .. }
                | MjaiEvent::Dora { .. }
                | MjaiEvent::Hora { .. }
                | MjaiEvent::Ryukyoku { .. }
                | MjaiEvent::EndKyoku
                | MjaiEvent::EndGame
                | MjaiEvent::Other
        );

        let is_start_game = matches!(ev, MjaiEvent::StartGame { .. });
        apply_and_log!(self.state, ev, json_val, is_start_game);

        if skip_check {
            return Ok(None);
        }

        // Check if this player has legal actions
        let obs = self.state.get_observation(player_id);
        let has_actions = if obs.legal_actions_method().is_empty() {
            None
        } else {
            Some(obs.into_pyobject(py).map(|o| o.unbind().into())?)
        };
        Ok(has_actions)
    }
}

impl RiichiEnv {
    /// Parse a Python event dict into (MjaiEvent, serde_json::Value).
    /// Parses the JSON string once into a Value, then converts to MjaiEvent
    /// via `from_value` to avoid double-parsing.
    fn parse_mjai_event(py: Python, event: Py<PyAny>) -> PyResult<(MjaiEvent, serde_json::Value)> {
        let json = py.import("json")?;
        let json_str: String = json.call_method1("dumps", (event,))?.extract()?;
        let json_val: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JSON Parse Error: {}", e))
        })?;
        let ev: MjaiEvent = serde_json::from_value(json_val.clone()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JSON Parse Error: {}", e))
        })?;
        Ok((ev, json_val))
    }
}
