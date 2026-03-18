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
    # Either the green agent voluntarily yields, or the red agent demands green.
    # If both request change simultaneously, requests cancel out (no switch).
    wants_change_a = action_a == 1
    wants_change_b = action_b == 1

    if wants_change_a and wants_change_b:
        # Conflicting requests cancel out
        return current_green, False

    if current_green == "A" and (wants_change_a or wants_change_b):
        return "B", True
    if current_green == "B" and (wants_change_a or wants_change_b):
        return "A", True

    return current_green, False
