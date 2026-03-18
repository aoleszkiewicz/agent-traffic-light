#!/usr/bin/env python3
"""Generate all visualization plots for the project report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_CONFIG
from environment.traffic_env import TrafficLightEnv
from agent.ppo_agent import load_model
from agent.baseline import FixedCyclePolicy

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

config = DEFAULT_CONFIG
model = load_model("models/ppo_traffic_final")


# --- 1. Queue dynamics: PPO vs Baseline side by side ---
def run_episode(policy_fn, seed=42):
    env = TrafficLightEnv(env_config=config.env)
    obs, _ = env.reset(seed=seed)
    queues_a, queues_b, rewards_log, greens = [], [], [], []
    while env.agents:
        actions = policy_fn(obs)
        obs, rewards, _, _, infos = env.step(actions)
        info = infos["light_A"]
        queues_a.append(info["queue_a"])
        queues_b.append(info["queue_b"])
        rewards_log.append(rewards["light_A"] + rewards["light_B"])
        greens.append(info["green"])
    return queues_a, queues_b, rewards_log, greens


def ppo_policy(obs):
    action_a, _ = model.predict(obs["light_A"], deterministic=True)
    action_b, _ = model.predict(obs["light_B"], deterministic=True)
    return {"light_A": int(action_a), "light_B": int(action_b)}


baseline_a = FixedCyclePolicy(15)
baseline_b = FixedCyclePolicy(15)


def baseline_policy(obs):
    return {
        "light_A": baseline_a.predict(obs["light_A"]),
        "light_B": baseline_b.predict(obs["light_B"]),
    }


print("Running PPO episode...")
ppo_qa, ppo_qb, ppo_r, ppo_g = run_episode(ppo_policy, seed=42)

print("Running Baseline episode...")
baseline_a.reset()
baseline_b.reset()
bl_qa, bl_qb, bl_r, bl_g = run_episode(baseline_policy, seed=42)

# Plot 1: Queue dynamics comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].plot(ppo_qa, label="Queue A", color="tab:blue", linewidth=1)
axes[0, 0].plot(ppo_qb, label="Queue B", color="tab:orange", linewidth=1)
axes[0, 0].set_ylabel("Queue Length")
axes[0, 0].set_title("PPO Agent — Queue Dynamics")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(bl_qa, label="Queue A", color="tab:blue", linewidth=1)
axes[0, 1].plot(bl_qb, label="Queue B", color="tab:orange", linewidth=1)
axes[0, 1].set_ylabel("Queue Length")
axes[0, 1].set_title("Fixed-Cycle Baseline — Queue Dynamics")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Phase timeline
ppo_phase = [1 if g == "A" else 0 for g in ppo_g]
bl_phase = [1 if g == "A" else 0 for g in bl_g]

axes[1, 0].fill_between(range(len(ppo_phase)), ppo_phase, alpha=0.4, color="tab:green", label="A green")
axes[1, 0].fill_between(range(len(ppo_phase)), [1 - p for p in ppo_phase], alpha=0.4, color="tab:red", label="B green")
axes[1, 0].set_ylabel("Phase")
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_title("PPO — Traffic Light Phases")
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_yticklabels(["B green", "A green"])
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].fill_between(range(len(bl_phase)), bl_phase, alpha=0.4, color="tab:green", label="A green")
axes[1, 1].fill_between(range(len(bl_phase)), [1 - p for p in bl_phase], alpha=0.4, color="tab:red", label="B green")
axes[1, 1].set_ylabel("Phase")
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_title("Baseline — Traffic Light Phases")
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_yticklabels(["B green", "A green"])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "queue_dynamics_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIGURES_DIR / 'queue_dynamics_comparison.png'}")


# --- 2. Reward comparison bar chart ---
print("Running evaluation episodes...")
ppo_episode_rewards = []
bl_episode_rewards = []

for ep in range(20):
    _, _, ppo_r_ep, _ = run_episode(ppo_policy, seed=100 + ep)
    ppo_episode_rewards.append(sum(ppo_r_ep))

    baseline_a.reset()
    baseline_b.reset()
    _, _, bl_r_ep, _ = run_episode(baseline_policy, seed=100 + ep)
    bl_episode_rewards.append(sum(bl_r_ep))

fig, ax = plt.subplots(figsize=(8, 5))
labels = ["PPO Agent", "Fixed-Cycle Baseline"]
means = [np.mean(ppo_episode_rewards), np.mean(bl_episode_rewards)]
stds = [np.std(ppo_episode_rewards), np.std(bl_episode_rewards)]

bars = ax.bar(labels, means, yerr=stds, capsize=10, color=["tab:blue", "tab:orange"], alpha=0.85)
ax.set_ylabel("Mean Episode Reward (sum of both agents)")
ax.set_title("RL Agent vs Fixed-Cycle Baseline")
ax.grid(True, alpha=0.3, axis="y")

for bar, mean in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 1,
        f"{mean:.1f}",
        ha="center", va="top", fontweight="bold", color="white", fontsize=14,
    )

plt.tight_layout()
fig.savefig(FIGURES_DIR / "reward_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIGURES_DIR / 'reward_comparison.png'}")


# --- 3. Cumulative reward over episode ---
fig, ax = plt.subplots(figsize=(10, 5))
_, _, ppo_r, _ = run_episode(ppo_policy, seed=42)
baseline_a.reset()
baseline_b.reset()
_, _, bl_r, _ = run_episode(baseline_policy, seed=42)

ax.plot(np.cumsum(ppo_r), label="PPO Agent", color="tab:blue", linewidth=2)
ax.plot(np.cumsum(bl_r), label="Fixed-Cycle Baseline", color="tab:orange", linewidth=2)
ax.set_xlabel("Step")
ax.set_ylabel("Cumulative Reward")
ax.set_title("Cumulative Reward Over Episode")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "cumulative_reward.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIGURES_DIR / 'cumulative_reward.png'}")


# --- 4. Intersection snapshot ---
from visualization.renderer import render_intersection_rgb

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Snapshot at different points
env = TrafficLightEnv(env_config=config.env)
obs, _ = env.reset(seed=42)
snapshots = []
snapshot_steps = [1, 100, 400]
step = 0
while env.agents:
    actions = ppo_policy(obs)
    obs, _, _, _, _ = env.step(actions)
    step += 1
    if step in snapshot_steps:
        img = render_intersection_rgb(
            queue_a=env._queue_a, queue_b=env._queue_b,
            green=env._current_green, yellow=env._yellow_remaining > 0,
            step=step,
        )
        snapshots.append(img)

for ax, img, s in zip(axes, snapshots, snapshot_steps):
    ax.imshow(img)
    ax.set_title(f"Step {s}")
    ax.axis("off")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "intersection_snapshots.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIGURES_DIR / 'intersection_snapshots.png'}")

print("\nAll plots generated in figures/")
