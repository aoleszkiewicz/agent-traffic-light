"""Matplotlib-based intersection renderer."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def _draw_intersection(ax, queue_a, queue_b, green, yellow, step):
    """Draw a top-down intersection view on the given axes."""
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d2d2d")
    ax.set_title(f"Step {step}", color="white", fontsize=14)

    # Roads
    ax.add_patch(patches.Rectangle((-2, -10), 4, 20, fc="#555555"))  # vertical
    ax.add_patch(patches.Rectangle((-10, -2), 20, 4, fc="#555555"))  # horizontal

    # Intersection
    ax.add_patch(patches.Rectangle((-2, -2), 4, 4, fc="#444444"))

    # Lane markings
    for y in range(-9, 10, 2):
        if -2 <= y <= 2:
            continue
        ax.plot([0, 0], [y, y + 0.8], color="white", linewidth=1, alpha=0.5)
    for x in range(-9, 10, 2):
        if -2 <= x <= 2:
            continue
        ax.plot([x, x + 0.8], [0, 0], color="white", linewidth=1, alpha=0.5)

    # Traffic lights
    if yellow:
        color_a, color_b = "yellow", "yellow"
    else:
        color_a = "lime" if green == "A" else "red"
        color_b = "lime" if green == "B" else "red"

    # Light A (vertical road, south side)
    ax.add_patch(patches.Circle((-3, -3), 0.6, fc=color_a, ec="white", lw=2))
    ax.text(-3, -4.2, "A", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    # Light B (horizontal road, west side)
    ax.add_patch(patches.Circle((3, 3), 0.6, fc=color_b, ec="white", lw=2))
    ax.text(4.2, 3, "B", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    # Queue bars
    bar_max = 50
    # Road A queue (vertical, below intersection)
    bar_h_a = min(queue_a / bar_max, 1.0) * 6
    ax.add_patch(patches.Rectangle((-1.5, -9), 1, bar_h_a, fc="cyan", alpha=0.7))
    ax.text(-1, -9.7, f"Q_A={queue_a}", ha="center", va="center", color="white", fontsize=9)

    # Road B queue (horizontal, left of intersection)
    bar_w_b = min(queue_b / bar_max, 1.0) * 6
    ax.add_patch(patches.Rectangle((-9, 0.5), bar_w_b, 1, fc="orange", alpha=0.7))
    ax.text(-9, -0.5, f"Q_B={queue_b}", ha="left", va="center", color="white", fontsize=9)

    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("white")


_fig = None
_ax = None


def render_intersection(queue_a, queue_b, green, yellow, step):
    """Render intersection in an interactive matplotlib window."""
    global _fig, _ax
    if _fig is None:
        matplotlib.use("TkAgg")
        _fig, _ax = plt.subplots(1, 1, figsize=(8, 8))
    _draw_intersection(_ax, queue_a, queue_b, green, yellow, step)
    _fig.canvas.draw()
    plt.pause(0.05)


def render_intersection_rgb(queue_a, queue_b, green, yellow, step) -> np.ndarray:
    """Render intersection and return as RGB numpy array."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=80)
    _draw_intersection(ax, queue_a, queue_b, green, yellow, step)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img
