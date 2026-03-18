"""Training curves and comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    metrics: list[dict],
    save_path: str | None = None,
    show: bool = True,
):
    """Plot reward and queue length over training timesteps."""
    if not metrics:
        print("No metrics to plot.")
        return

    timesteps = [m["timestep"] for m in metrics]
    rewards = [m["avg_reward"] for m in metrics]
    lengths = [m.get("avg_length", 0) for m in metrics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(timesteps, rewards, color="tab:blue", linewidth=1.5)
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Training Curves")
    ax1.grid(True, alpha=0.3)

    ax2.plot(timesteps, lengths, color="tab:orange", linewidth=1.5)
    ax2.set_ylabel("Average Episode Length")
    ax2.set_xlabel("Timestep")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison(
    ppo_rewards: list[float],
    baseline_rewards: list[float],
    save_path: str | None = None,
    show: bool = True,
):
    """Bar chart comparing PPO vs fixed-cycle baseline."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["PPO Agent", "Fixed-Cycle Baseline"]
    means = [np.mean(ppo_rewards), np.mean(baseline_rewards)]
    stds = [np.std(ppo_rewards), np.std(baseline_rewards)]

    bars = ax.bar(labels, means, yerr=stds, capsize=10, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("RL Agent vs Fixed-Cycle Baseline")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_optuna_results(
    study,
    save_path: str | None = None,
    show: bool = True,
):
    """Plot Optuna optimization history and parameter importance."""
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )

        fig1 = plot_optimization_history(study)
        if save_path:
            base = Path(save_path)
            fig1.figure.savefig(
                str(base.with_stem(base.stem + "_history")),
                dpi=150, bbox_inches="tight",
            )

        fig2 = plot_param_importances(study)
        if save_path:
            fig2.figure.savefig(
                str(base.with_stem(base.stem + "_importance")),
                dpi=150, bbox_inches="tight",
            )

        if show:
            plt.show()
        plt.close("all")
    except ImportError:
        print("Optuna matplotlib visualization not available.")
