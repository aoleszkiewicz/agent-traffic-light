"""PPO training with SB3, including Redis metrics callback."""

from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from config import AppConfig, DEFAULT_CONFIG
from environment.wrappers import make_sb3_env


class RedisMetricsCallback(BaseCallback):
    """Push training metrics to Redis after each rollout."""

    def __init__(self, redis_manager, verbose: int = 0):
        super().__init__(verbose)
        self.redis_manager = redis_manager

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0 and len(self.model.ep_info_buffer) > 0:
            recent = list(self.model.ep_info_buffer)[-10:]
            avg_reward = sum(ep["r"] for ep in recent) / len(recent)
            avg_length = sum(ep["l"] for ep in recent) / len(recent)

            self.redis_manager.push_metrics({
                "timestep": self.num_timesteps,
                "avg_reward": round(avg_reward, 2),
                "avg_length": round(avg_length, 2),
            })

            if self.redis_manager.is_connected():
                self.redis_manager.set_env_state({
                    "timestep": self.num_timesteps,
                    "avg_reward": round(avg_reward, 2),
                })
        return True


def train_ppo(
    config: AppConfig | None = None,
    redis_manager=None,
) -> PPO:
    """Train a PPO agent on the traffic light environment."""
    config = config or DEFAULT_CONFIG
    tc = config.train

    model_dir = Path(tc.model_dir)
    model_dir.mkdir(exist_ok=True)

    env = make_sb3_env(env_config=config.env, seed=tc.seed)
    eval_env = make_sb3_env(env_config=config.env, seed=tc.seed + 1000)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=tc.learning_rate,
        n_steps=tc.n_steps,
        batch_size=tc.batch_size,
        n_epochs=tc.n_epochs,
        gamma=tc.gamma,
        gae_lambda=tc.gae_lambda,
        clip_range=tc.clip_range,
        ent_coef=tc.ent_coef,
        vf_coef=tc.vf_coef,
        seed=tc.seed,
        verbose=1,
    )

    callbacks = []

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(model_dir / "logs"),
        eval_freq=tc.eval_freq,
        n_eval_episodes=tc.n_eval_episodes,
        deterministic=True,
    )
    callbacks.append(eval_callback)

    if redis_manager is not None:
        callbacks.append(RedisMetricsCallback(redis_manager))

    if redis_manager is not None:
        redis_manager.set_status("running")

    model.learn(total_timesteps=tc.total_timesteps, callback=callbacks)

    final_path = str(model_dir / "ppo_traffic_final")
    model.save(final_path)

    if redis_manager is not None:
        redis_manager.set_status("finished")

    return model


def load_model(path: str) -> PPO:
    return PPO.load(path)
