"""Tests for traffic environment logic and PettingZoo compliance."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EnvConfig
from environment.traffic_logic import (
    process_departures,
    resolve_phase_change,
    sample_arrivals,
)
from environment.traffic_env import TrafficLightEnv


# --- traffic_logic tests ---


class TestSampleArrivals:
    def test_returns_non_negative(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            a, b = sample_arrivals(0.5, 0.3, rng)
            assert a >= 0
            assert b >= 0

    def test_mean_approximates_lambda(self):
        rng = np.random.default_rng(42)
        arrivals = [sample_arrivals(2.0, 3.0, rng) for _ in range(10000)]
        mean_a = np.mean([a for a, _ in arrivals])
        mean_b = np.mean([b for _, b in arrivals])
        assert abs(mean_a - 2.0) < 0.1
        assert abs(mean_b - 3.0) < 0.1


class TestProcessDepartures:
    def test_green_drains(self):
        assert process_departures(10, True, 3) == 7

    def test_red_no_drain(self):
        assert process_departures(10, False, 3) == 10

    def test_no_negative_queue(self):
        assert process_departures(2, True, 5) == 0

    def test_empty_queue(self):
        assert process_departures(0, True, 3) == 0


class TestResolvePhaseChange:
    def test_min_green_prevents_change(self):
        phase, changed = resolve_phase_change(1, 0, "A", 2, 4)
        assert phase == "A"
        assert not changed

    def test_change_from_a_to_b(self):
        phase, changed = resolve_phase_change(1, 0, "A", 5, 4)
        assert phase == "B"
        assert changed

    def test_change_from_b_to_a(self):
        phase, changed = resolve_phase_change(0, 1, "B", 5, 4)
        assert phase == "A"
        assert changed

    def test_no_change_when_action_keep(self):
        phase, changed = resolve_phase_change(0, 0, "A", 10, 4)
        assert phase == "A"
        assert not changed

    def test_only_current_green_agent_can_switch(self):
        # light_B requests change but A is green — no change
        phase, changed = resolve_phase_change(0, 1, "A", 10, 4)
        assert phase == "A"
        assert not changed


# --- TrafficLightEnv tests ---


class TestTrafficLightEnv:
    def setup_method(self):
        self.cfg = EnvConfig(max_steps=50)
        self.env = TrafficLightEnv(env_config=self.cfg)

    def test_reset(self):
        obs, infos = self.env.reset(seed=42)
        assert set(obs.keys()) == {"light_A", "light_B"}
        assert obs["light_A"].shape == (4,)
        assert obs["light_B"].shape == (4,)

    def test_step_returns_correct_structure(self):
        self.env.reset(seed=42)
        actions = {"light_A": 0, "light_B": 0}
        obs, rewards, terms, truncs, infos = self.env.step(actions)

        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert "light_A" in rewards
        assert "light_B" in rewards

    def test_episode_truncates(self):
        self.env.reset(seed=42)
        for _ in range(self.cfg.max_steps):
            if not self.env.agents:
                break
            actions = {a: 0 for a in self.env.agents}
            self.env.step(actions)
        assert len(self.env.agents) == 0

    def test_observation_space_compliance(self):
        self.env.reset(seed=42)
        obs, _, _, _, _ = self.env.step({"light_A": 0, "light_B": 0})
        for agent in ["light_A", "light_B"]:
            assert self.env.observation_space(agent).contains(obs[agent])

    def test_rewards_are_negative(self):
        self.env.reset(seed=42)
        # Run some steps to build up queues
        for _ in range(20):
            if not self.env.agents:
                break
            _, rewards, _, _, _ = self.env.step({"light_A": 0, "light_B": 0})
        # Rewards should be non-positive (negative queue penalty)
        assert rewards["light_A"] <= 0
        assert rewards["light_B"] <= 0


class TestPettingZooCompliance:
    def test_parallel_api(self):
        """Run PettingZoo's official parallel API test."""
        from pettingzoo.test import parallel_api_test

        env = TrafficLightEnv(env_config=EnvConfig(max_steps=100))
        parallel_api_test(env, num_cycles=50)
