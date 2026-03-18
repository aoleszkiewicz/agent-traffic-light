"""Optuna hyperparameter optimization for the PPO agent."""

from __future__ import annotations

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from config import AppConfig, DEFAULT_CONFIG
from environment.wrappers import make_sb3_env


def objective(trial: optuna.Trial, base_config: AppConfig | None = None) -> float:
    cfg = base_config or DEFAULT_CONFIG

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 3, 15)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)

    env_cfg = cfg.env.model_copy(update={"alpha": alpha})

    env = make_sb3_env(env_config=env_cfg, seed=cfg.train.seed)
    eval_env = make_sb3_env(env_config=env_cfg, seed=cfg.train.seed + 1000)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        seed=cfg.train.seed,
        verbose=0,
    )

    model.learn(total_timesteps=100_000)

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    return mean_reward


def run_hyperopt(
    n_trials: int = 50,
    storage: str = "sqlite:///optuna_study.db",
    base_config: AppConfig | None = None,
) -> optuna.Study:
    study = optuna.create_study(
        study_name="traffic_ppo",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, base_config),
        n_trials=n_trials,
    )
    return study
