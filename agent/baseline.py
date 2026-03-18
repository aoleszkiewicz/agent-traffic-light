"""Fixed-cycle baseline policy for comparison with RL agent."""

from __future__ import annotations

import numpy as np

from config import EnvConfig, DEFAULT_CONFIG
from environment.traffic_env import TrafficLightEnv


class FixedCyclePolicy:
    """Alternates green phase every `cycle_length` steps.

    Only the currently-green agent requests a change. The red agent always
    outputs 0 (keep) to avoid triggering the cancel-out rule.
    """

    def __init__(self, cycle_length: int = 15):
        self.cycle_length = cycle_length
        self._step = 0

    def predict(self, obs: np.ndarray) -> int:
        """Return action: 1 (request change) at cycle boundary if green, else 0."""
        # obs[2] is own_phase (normalized: 0.5=green, 0=red, 1=yellow)
        own_phase = obs[2] if obs is not None else 0
        is_green = abs(own_phase - 0.5) < 0.01  # 0.5 = green (1.0/2.0 normalized)

        self._step += 1
        if is_green and self._step % self.cycle_length == 0:
            return 1
        return 0

    def reset(self):
        self._step = 0


def evaluate_baseline(
    cycle_length: int = 15,
    n_episodes: int = 20,
    env_config: EnvConfig | None = None,
    seed: int = 42,
) -> dict:
    """Run the fixed-cycle baseline and return aggregate metrics."""
    cfg = env_config or DEFAULT_CONFIG.env
    env = TrafficLightEnv(env_config=cfg)

    policy_a = FixedCyclePolicy(cycle_length)
    policy_b = FixedCyclePolicy(cycle_length)

    episode_rewards = []
    episode_queues = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        policy_a.reset()
        policy_b.reset()

        total_reward = 0.0
        total_queue = 0.0
        steps = 0

        while env.agents:
            actions = {
                "light_A": policy_a.predict(obs["light_A"]),
                "light_B": policy_b.predict(obs["light_B"]),
            }
            obs, rewards, _, _, infos = env.step(actions)
            total_reward += rewards.get("light_A", 0) + rewards.get("light_B", 0)
            info = infos.get("light_A", {})
            total_queue += info.get("queue_a", 0) + info.get("queue_b", 0)
            steps += 1

        episode_rewards.append(total_reward)
        episode_queues.append(total_queue / max(steps, 1))

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_queue": float(np.mean(episode_queues)),
        "std_queue": float(np.std(episode_queues)),
        "n_episodes": n_episodes,
        "cycle_length": cycle_length,
    }
