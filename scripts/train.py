#!/usr/bin/env python3
"""CLI script to train the PPO traffic light agent."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig, DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train PPO traffic light agent")
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Total training timesteps"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--hyperopt", action="store_true", help="Run Optuna hyperparameter optimization"
    )
    parser.add_argument(
        "--hyperopt-trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--no-redis", action="store_true", help="Disable Redis metrics logging"
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG.model_copy(deep=True)

    if args.timesteps is not None:
        config = config.model_copy(
            update={"train": config.train.model_copy(update={"total_timesteps": args.timesteps})}
        )
    if args.seed is not None:
        config = config.model_copy(
            update={"train": config.train.model_copy(update={"seed": args.seed})}
        )

    if args.hyperopt:
        from agent.hyperopt import run_hyperopt

        print(f"Running Optuna hyperparameter optimization ({args.hyperopt_trials} trials)...")
        study = run_hyperopt(n_trials=args.hyperopt_trials, base_config=config)
        print(f"Best params: {study.best_params}")
        print(f"Best value: {study.best_value}")
        return

    redis_manager = None
    if not args.no_redis:
        try:
            from state.redis_manager import RedisStateManager

            redis_manager = RedisStateManager(config.redis)
            if not redis_manager.is_connected():
                print("Warning: Redis not available, training without metrics logging.")
                redis_manager = None
        except Exception:
            print("Warning: Redis not available, training without metrics logging.")

    from agent.ppo_agent import train_ppo

    print(f"Starting training for {config.train.total_timesteps} timesteps...")
    model = train_ppo(config=config, redis_manager=redis_manager)
    print(f"Training complete. Model saved to {config.train.model_dir}/")


if __name__ == "__main__":
    main()
