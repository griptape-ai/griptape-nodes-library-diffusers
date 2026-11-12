"""Helpers that map schedule steps to img2img strengths and protect scheduler state while doing so."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from contextlib import contextmanager


def strength_for_step_index(step_index: int, num_steps: int) -> float:
    """Return the img2img strength that selects a given schedule index.

    The strength sits one float step below the exact ratio, which keeps ``int(N - N * strength)``
    (diffusers' ``get_timesteps``) and ``round(N * strength)`` from landing one step off. Pipelines
    that truncate ``int(N * strength)`` instead need :func:`truncation_safe_strength`.
    """
    if num_steps < 1:
        raise ValueError(
            f"Attempted to compute the img2img strength for a schedule step. Failed with num_steps={num_steps} "
            "because it must be a positive integer."
        )
    if step_index < 0 or step_index >= num_steps:
        raise ValueError(
            f"Attempted to compute the img2img strength for a schedule step. Failed with step_index={step_index} "
            f"and num_steps={num_steps} because step_index must be in [0, {num_steps})."
        )

    strength = (num_steps - step_index) / num_steps
    if step_index > 0:
        strength = math.nextafter(strength, 0.0)
    return strength


def truncation_safe_strength(strength: float, num_steps: int) -> float:
    """Return ``strength`` nudged so ``int(num_steps * strength)`` keeps every whole step it was meant to.

    Pipelines such as SDXL truncate ``int(N * strength)``, which drops a step when the product falls
    below a whole number. A strength that is not within float error of a whole step is returned as is.
    """
    steps_kept = round(num_steps * strength)
    if not math.isclose(num_steps * strength, steps_kept, rel_tol=0.0, abs_tol=1e-6):
        return strength

    adjusted = steps_kept / num_steps
    while int(num_steps * adjusted) < steps_kept:
        adjusted = math.nextafter(adjusted, math.inf)
    return adjusted


@contextmanager
def preserved_scheduler_state(scheduler: object) -> Iterator[None]:
    """Restore the scheduler's attributes on exit, undoing any ``set_timesteps`` / ``set_begin_index`` made inside."""
    saved_state = copy.copy(scheduler.__dict__)
    try:
        yield
    finally:
        scheduler.__dict__.clear()
        scheduler.__dict__.update(saved_state)
