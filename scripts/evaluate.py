#!/usr/bin/env python3
"""CLI script to evaluate the trained agent and compare with baseline."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Evaluate traffic light agent")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_traffic_final",
        help="Path to trained model",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument(
        "--compare-baseline", action="store_true", help="Compare with fixed-cycle baseline"
    )
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = DEFAULT_CONFIG

    if args.render:
        _evaluate_with_render(args, config)
    else:
        _evaluate_headless(args, config)


def _evaluate_headless(args, config):
    from stable_baselines3.common.evaluation import evaluate_policy

    from agent.ppo_agent import load_model
    from environment.wrappers import make_sb3_env

    model = load_model(args.model)
    eval_env = make_sb3_env(env_config=config.env, seed=args.seed)

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.episodes
    )
    print(f"PPO Agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    if args.compare_baseline:
        from agent.baseline import evaluate_baseline

        baseline = evaluate_baseline(
            n_episodes=args.episodes, env_config=config.env, seed=args.seed
        )
        print(
            f"Baseline:  mean_reward={baseline['mean_reward']:.2f} "
            f"+/- {baseline['std_reward']:.2f}"
        )
        improvement = mean_reward - baseline["mean_reward"]
        print(f"PPO improvement over baseline: {improvement:+.2f}")


def _evaluate_with_render(args, config):
    from agent.ppo_agent import load_model
    from environment.traffic_env import TrafficLightEnv

    model = load_model(args.model)
    env = TrafficLightEnv(env_config=config.env, render_mode="human")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0

        while env.agents:
            # Predict independently for each agent (shared policy, different obs)
            action_a, _ = model.predict(obs["light_A"], deterministic=True)
            action_b, _ = model.predict(obs["light_B"], deterministic=True)
            actions = {"light_A": int(action_a), "light_B": int(action_b)}
            obs, rewards, _, _, _ = env.step(actions)
            total_reward += rewards.get("light_A", 0) + rewards.get("light_B", 0)
            env.render()

        print(f"Episode {ep + 1}: total_reward={total_reward:.2f}")


if __name__ == "__main__":
    main()
