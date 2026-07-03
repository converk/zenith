from __future__ import annotations

from typing import Any

import numpy as np


AGENTS = tuple(f"player_{index}" for index in range(4))
NUM_TILE_TYPES = 34


class RiichiVectorEnv:
    """Training wrapper that uses the Rust VecEnv batch interface directly."""

    def __init__(
        self,
        num_envs: int,
        seed: int = 1,
        raw_env: Any | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.num_envs = num_envs
        self.seed_value = seed
        self.raw_env = raw_env if raw_env is not None else self._make_raw_env(num_envs, seed)

    @staticmethod
    def _make_raw_env(num_envs: int, seed: int) -> Any:
        import riichi

        return riichi.VecEnv(num_envs, seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.seed_value = seed
            if self.raw_env.__class__.__module__ == "riichi":
                self.raw_env = self._make_raw_env(self.num_envs, seed)

        observations = self._extract_observation(self.raw_env.reset())
        return observations, self._make_action_masks(observations)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        action_array = np.asarray(actions, dtype=np.uint8)
        if action_array.shape != (self.num_envs, len(AGENTS)):
            raise ValueError("actions must have shape [num_envs, 4]")

        raw_observation, raw_rewards, raw_dones = self.raw_env.step(action_array)
        observations = self._extract_observation(raw_observation)
        rewards = np.asarray(raw_rewards, dtype=np.float32).reshape(self.num_envs, len(AGENTS))
        dones = np.asarray(raw_dones, dtype=np.bool_).reshape(self.num_envs)
        return observations, self._make_action_masks(observations), rewards, dones

    def close(self) -> None:
        return None

    def _extract_observation(self, raw_observation: Any) -> np.ndarray:
        observation = np.asarray(raw_observation, dtype=np.uint8)
        expected_shape = (self.num_envs, len(AGENTS), NUM_TILE_TYPES)
        if observation.shape != expected_shape:
            raise ValueError("raw riichi observation must have shape [num_envs, 4, 34]")
        return observation

    @staticmethod
    def _make_action_masks(observations: np.ndarray) -> np.ndarray:
        return observations > 0