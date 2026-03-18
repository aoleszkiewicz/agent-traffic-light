"""FastAPI REST API for the traffic light RL system."""

from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import AppConfig, DEFAULT_CONFIG, EnvConfig, TrainConfig
from state.redis_manager import RedisStateManager

app = FastAPI(title="Traffic Light RL API", version="1.0.0")

_config = DEFAULT_CONFIG.model_copy(deep=True)
_redis = RedisStateManager(_config.redis)
_train_thread: threading.Thread | None = None


class ConfigUpdate(BaseModel):
    env: EnvConfig | None = None
    train: TrainConfig | None = None


@app.get("/state")
def get_state():
    """Current intersection state from Redis."""
    if not _redis.is_connected():
        raise HTTPException(503, "Redis not available")
    return _redis.get_env_state()


@app.get("/metrics")
def get_metrics(start: int = 0, end: int = -1):
    """Training metrics history."""
    if not _redis.is_connected():
        raise HTTPException(503, "Redis not available")
    return _redis.get_metrics(start, end)


@app.post("/train/start")
def start_training():
    """Launch PPO training in a background thread."""
    global _train_thread

    if _train_thread is not None and _train_thread.is_alive():
        raise HTTPException(409, "Training already in progress")

    from agent.ppo_agent import train_ppo

    _redis.clear_metrics()
    _redis.set_status("starting")

    def _run():
        train_ppo(config=_config, redis_manager=_redis)

    _train_thread = threading.Thread(target=_run, daemon=True)
    _train_thread.start()

    return {"status": "started"}


@app.get("/train/status")
def train_status():
    """Current training status."""
    if not _redis.is_connected():
        return {"status": "unknown", "detail": "Redis not available"}
    return {"status": _redis.get_status()}


@app.get("/config")
def get_config():
    return _config.model_dump()


@app.put("/config")
def update_config(update: ConfigUpdate):
    global _config
    updates = {}
    if update.env is not None:
        updates["env"] = update.env
    if update.train is not None:
        updates["train"] = update.train
    _config = _config.model_copy(update=updates)
    if _redis.is_connected():
        _redis.set_config(_config.model_dump())
    return _config.model_dump()


@app.post("/evaluate")
def run_evaluation(episodes: int = 20, compare_baseline: bool = True):
    """Run evaluation of the trained model and optionally compare with baseline."""
    from pathlib import Path

    from stable_baselines3.common.evaluation import evaluate_policy

    from agent.baseline import evaluate_baseline
    from environment.wrappers import make_sb3_env

    model_path = Path(_config.train.model_dir) / "ppo_traffic_final.zip"
    if not model_path.exists():
        raise HTTPException(404, "No trained model found. Run training first.")

    from agent.ppo_agent import load_model

    model = load_model(str(model_path.with_suffix("")))
    eval_env = make_sb3_env(env_config=_config.env, seed=99)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=episodes
    )

    result = {
        "ppo": {
            "mean_reward": round(float(mean_reward), 2),
            "std_reward": round(float(std_reward), 2),
        }
    }

    if compare_baseline:
        baseline_results = evaluate_baseline(
            n_episodes=episodes, env_config=_config.env
        )
        result["baseline"] = baseline_results

    return result


@app.get("/hyperopt/results")
def hyperopt_results():
    """Return Optuna study results if available."""
    try:
        import optuna

        study = optuna.load_study(
            study_name="traffic_ppo",
            storage="sqlite:///optuna_study.db",
        )
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }
    except Exception as e:
        raise HTTPException(404, f"No hyperopt results found: {e}")
