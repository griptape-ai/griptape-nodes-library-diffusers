from __future__ import annotations

from types import SimpleNamespace

import pytest

from modular_diffusion_nodes_library.misc.schedule_utils import (
    preserved_scheduler_state,
    strength_for_step_index,
    truncation_safe_strength,
)

MAX_STEPS = 300


@pytest.mark.parametrize("num_steps", [1, 4, 20, 22, 25, 28, 30, 49, 50, MAX_STEPS])
def test_strength_for_step_index_selects_the_step_for_float_and_rounding_formulas(num_steps: int) -> None:
    for step_index in range(1, num_steps):
        strength = strength_for_step_index(step_index, num_steps)

        # Flux, Flux 2, Qwen, SD3, Z-Image, LTX, Wan and Hunyuan: int(max(N - min(N * strength, N), 0)).
        assert int(max(num_steps - min(num_steps * strength, num_steps), 0)) == step_index
        # Flux 3: N - round(N * strength).
        assert num_steps - round(num_steps * strength) == step_index


@pytest.mark.parametrize("num_steps", [1, 4, 20, 22, 25, 28, 30, 49, 50, MAX_STEPS])
def test_truncation_safe_strength_selects_the_step_for_truncating_formula(num_steps: int) -> None:
    for step_index in range(1, num_steps):
        strength = truncation_safe_strength(strength_for_step_index(step_index, num_steps), num_steps)

        # SDXL: N - int(N * strength).
        assert num_steps - int(num_steps * strength) == step_index


def test_truncation_safe_strength_leaves_non_step_aligned_strengths_alone() -> None:
    assert truncation_safe_strength(0.57, 20) == 0.57


def test_strength_for_step_index_is_exactly_one_at_step_zero() -> None:
    assert strength_for_step_index(0, 20) == 1.0


def test_strength_for_step_index_rejects_out_of_range_steps() -> None:
    with pytest.raises(ValueError, match="step_index=20"):
        strength_for_step_index(20, 20)
    with pytest.raises(ValueError, match="num_steps=0"):
        strength_for_step_index(0, 0)


def test_preserved_scheduler_state_restores_attributes_set_inside() -> None:
    scheduler = SimpleNamespace(timesteps=[3, 2, 1], _begin_index=None)

    with preserved_scheduler_state(scheduler):
        scheduler.timesteps = [9]
        scheduler._begin_index = 4
        scheduler.extra = True

    assert scheduler.timesteps == [3, 2, 1]
    assert scheduler._begin_index is None
    assert not hasattr(scheduler, "extra")


def test_preserved_scheduler_state_restores_after_an_error() -> None:
    scheduler = SimpleNamespace(_begin_index=None)

    with pytest.raises(RuntimeError), preserved_scheduler_state(scheduler):
        scheduler._begin_index = 4
        raise RuntimeError

    assert scheduler._begin_index is None
