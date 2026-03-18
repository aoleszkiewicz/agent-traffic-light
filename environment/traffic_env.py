"""PettingZoo ParallelEnv for a 2-agent traffic light intersection."""

from __future__ import annotations

import functools
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from config import EnvConfig, DEFAULT_CONFIG
from environment.traffic_logic import (
    process_departures,
    resolve_phase_change,
    sample_arrivals,
)


class TrafficLightEnv(ParallelEnv):
    metadata = {"name": "traffic_light_v0", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = env_config or DEFAULT_CONFIG.env

        self.possible_agents = ["light_A", "light_B"]
        self.render_mode = render_mode

        self._queue_a: int = 0
        self._queue_b: int = 0
        self._current_green: str = "A"
        self._time_in_phase: int = 0
        self._yellow_remaining: int = 0
        self._step_count: int = 0
        self._rng = np.random.default_rng()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.Space:
        # [own_queue, other_queue, own_phase, time_in_phase] — all normalized to [0, 1]
        return spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        return spaces.Discrete(2)  # 0=keep, 1=request change

    def _get_obs(self) -> dict[str, np.ndarray]:
        # phase encoding: 0=red, 1=green, 2=yellow
        if self._yellow_remaining > 0:
            phase_a = 2.0
            phase_b = 2.0
        else:
            phase_a = 1.0 if self._current_green == "A" else 0.0
            phase_b = 1.0 if self._current_green == "B" else 0.0

        # Normalize observations to [0, 1] range for stable training
        mq = self.cfg.max_queue
        ms = self.cfg.max_steps
        return {
            "light_A": np.array(
                [self._queue_a / mq, self._queue_b / mq, phase_a / 2.0, self._time_in_phase / ms],
                dtype=np.float32,
            ),
            "light_B": np.array(
                [self._queue_b / mq, self._queue_a / mq, phase_b / 2.0, self._time_in_phase / ms],
                dtype=np.float32,
            ),
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self._queue_a = 0
        self._queue_b = 0
        self._current_green = "A"
        self._time_in_phase = 0
        self._yellow_remaining = 0
        self._step_count = 0

        return self._get_obs(), {a: {} for a in self.agents}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        self._step_count += 1

        # --- arrivals ---
        arr_a, arr_b = sample_arrivals(self.cfg.lambda_a, self.cfg.lambda_b, self._rng)
        self._queue_a = min(self._queue_a + arr_a, self.cfg.max_queue)
        self._queue_b = min(self._queue_b + arr_b, self.cfg.max_queue)

        # --- yellow transition ---
        if self._yellow_remaining > 0:
            self._yellow_remaining -= 1
            self._time_in_phase += 1
        else:
            # --- phase change logic ---
            action_a = actions.get("light_A", 0)
            action_b = actions.get("light_B", 0)

            next_green, changed = resolve_phase_change(
                action_a,
                action_b,
                self._current_green,
                self._time_in_phase,
                self.cfg.min_green_steps,
            )

            if changed:
                self._yellow_remaining = self.cfg.yellow_steps
                self._current_green = next_green
                self._time_in_phase = 0
            else:
                self._time_in_phase += 1

            # --- departures (no departures during yellow) ---
            if self._yellow_remaining == 0:
                self._queue_a = process_departures(
                    self._queue_a,
                    self._current_green == "A",
                    self.cfg.green_drain_rate,
                )
                self._queue_b = process_departures(
                    self._queue_b,
                    self._current_green == "B",
                    self.cfg.green_drain_rate,
                )

        # --- rewards (normalized to approx [-1, 0]) ---
        scale = self.cfg.max_queue * (1.0 + self.cfg.alpha)
        reward_a = -(self._queue_a + self.cfg.alpha * self._queue_b) / scale
        reward_b = -(self._queue_b + self.cfg.alpha * self._queue_a) / scale

        rewards = {"light_A": float(reward_a), "light_B": float(reward_b)}

        # --- termination ---
        truncated = self._step_count >= self.cfg.max_steps
        terminations = {a: False for a in self.agents}
        truncations = {a: truncated for a in self.agents}

        if truncated:
            self.agents = []

        obs = self._get_obs()
        infos = {
            a: {
                "queue_a": self._queue_a,
                "queue_b": self._queue_b,
                "green": self._current_green,
                "step": self._step_count,
            }
            for a in self.possible_agents
        }

        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "human":
            from visualization.renderer import render_intersection

            render_intersection(
                queue_a=self._queue_a,
                queue_b=self._queue_b,
                green=self._current_green,
                yellow=self._yellow_remaining > 0,
                step=self._step_count,
            )
        elif self.render_mode == "rgb_array":
            from visualization.renderer import render_intersection_rgb

            return render_intersection_rgb(
                queue_a=self._queue_a,
                queue_b=self._queue_b,
                green=self._current_green,
                yellow=self._yellow_remaining > 0,
                step=self._step_count,
            )
        return None
