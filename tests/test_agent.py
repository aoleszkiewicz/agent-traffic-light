"""Tests for the agent training and baseline modules."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EnvConfig
from agent.baseline import FixedCyclePolicy, evaluate_baseline


def _green_obs():
    """Fake observation where agent is green (phase=0.5 normalized)."""
    return np.array([0.1, 0.2, 0.5, 0.02], dtype=np.float32)


def _red_obs():
    """Fake observation where agent is red (phase=0.0 normalized)."""
    return np.array([0.1, 0.2, 0.0, 0.02], dtype=np.float32)


class TestFixedCyclePolicy:
    def test_returns_valid_actions(self):
        policy = FixedCyclePolicy(cycle_length=10)
        for _ in range(100):
            action = policy.predict(_green_obs())
            assert action in (0, 1)

    def test_changes_at_cycle_boundary_when_green(self):
        policy = FixedCyclePolicy(cycle_length=5)
        actions = [policy.predict(_green_obs()) for _ in range(15)]
        assert actions[4] == 1   # step 5
        assert actions[9] == 1   # step 10
        assert actions[14] == 1  # step 15

    def test_no_change_when_red(self):
        policy = FixedCyclePolicy(cycle_length=5)
        actions = [policy.predict(_red_obs()) for _ in range(15)]
        # Red agent should never request a change
        assert all(a == 0 for a in actions)

    def test_reset(self):
        policy = FixedCyclePolicy(cycle_length=5)
        for _ in range(3):
            policy.predict(_green_obs())
        policy.reset()
        actions = [policy.predict(_green_obs()) for _ in range(5)]
        assert actions[4] == 1


class TestEvaluateBaseline:
    def test_returns_metrics(self):
        cfg = EnvConfig(max_steps=50)
        results = evaluate_baseline(n_episodes=3, env_config=cfg, seed=42)
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "mean_queue" in results
        assert results["n_episodes"] == 3
