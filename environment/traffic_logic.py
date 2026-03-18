"""Pure simulation functions for the traffic intersection (no PettingZoo dependency)."""

import numpy as np


def sample_arrivals(
    lambda_a: float, lambda_b: float, rng: np.random.Generator
) -> tuple[int, int]:
    """Sample Poisson-distributed vehicle arrivals for both roads."""
    return int(rng.poisson(lambda_a)), int(rng.poisson(lambda_b))


def process_departures(queue: int, is_green: bool, rate: int) -> int:
    """Return updated queue after vehicles depart on green."""
    if not is_green:
        return queue
    return max(0, queue - rate)


def resolve_phase_change(
    action_a: int,
    action_b: int,
    current_green: str,
    time_in_phase: int,
    min_green: int,
) -> tuple[str, bool]:
    """Determine next green phase given agent actions and constraints.

    Returns:
        (next_green, changed): the phase that will be green, and whether a
        transition (yellow) is needed.

    Mutual exclusion: only one road can be green at a time.
    Min-green: the current phase must last at least `min_green` steps.
    """
    if time_in_phase < min_green:
        return current_green, False

    # action=1 means "request change"
    if current_green == "A" and action_a == 1:
        return "B", True
    if current_green == "B" and action_b == 1:
        return "A", True

    return current_green, False
