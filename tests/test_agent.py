"""Tests for the agent training and baseline modules."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EnvConfig
from agent.baseline import FixedCyclePolicy, evaluate_baseline


class TestFixedCyclePolicy:
    def test_returns_valid_actions(self):
        policy = FixedCyclePolicy(cycle_length=10)
        for _ in range(100):
            action = policy.predict(None)
            assert action in (0, 1)

    def test_changes_at_cycle_boundary(self):
        policy = FixedCyclePolicy(cycle_length=5)
        actions = [policy.predict(None) for _ in range(15)]
        # Should request change at steps 5, 10, 15
        assert actions[4] == 1  # step 5
        assert actions[9] == 1  # step 10
        assert actions[14] == 1  # step 15

    def test_reset(self):
        policy = FixedCyclePolicy(cycle_length=5)
        for _ in range(3):
            policy.predict(None)
        policy.reset()
        # After reset, next change should be at step 5 again
        actions = [policy.predict(None) for _ in range(5)]
        assert actions[4] == 1


class TestEvaluateBaseline:
    def test_returns_metrics(self):
        cfg = EnvConfig(max_steps=50)
        results = evaluate_baseline(n_episodes=3, env_config=cfg, seed=42)
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "mean_queue" in results
        assert results["n_episodes"] == 3
