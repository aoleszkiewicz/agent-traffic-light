"""SuperSuit wrappers for SB3 compatibility."""

from __future__ import annotations

import supersuit as ss

from config import EnvConfig
from environment.traffic_env import TrafficLightEnv


def make_sb3_env(
    env_config: EnvConfig | None = None,
    n_envs: int = 1,
    seed: int = 42,
):
    """Create an SB3-compatible vectorized environment.

    Uses SuperSuit to convert the PettingZoo ParallelEnv into a VecEnv
    with a shared policy across both agents.
    """
    env = TrafficLightEnv(env_config=env_config)
    env.reset(seed=seed)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=1, base_class="stable_baselines3")

    # SuperSuit's ConcatVecEnv lacks a seed() method that SB3 expects.
    # SB3's wrapper delegates seed() to env.venv, so patch the inner env.
    inner = env.venv if hasattr(env, "venv") else env
    if not hasattr(inner, "seed"):
        inner.seed = lambda seed=None: [seed] * env.num_envs

    return env
