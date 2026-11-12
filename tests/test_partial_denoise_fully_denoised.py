from __future__ import annotations

from typing import Any

import pytest

from modular_diffusion_nodes_library.misc.partial_denoise import PartialDenoiseSchedulerProxy

TORCH_SKIP_REASON = "Reads real torch; run `make test/exec`."


class _FakeScheduler:
    """Flow-matching style schedule: n timesteps and n + 1 sigmas ending at 0."""

    def __init__(self) -> None:
        self.timesteps: Any = None
        self.sigmas: Any = None
        self.num_inference_steps: Any = None
        self._step_index = 5

    def set_timesteps(self, num_inference_steps: int, device: object = None) -> None:
        import torch

        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        self.sigmas = sigmas
        self.timesteps = sigmas[:-1] * 1000
        self.num_inference_steps = num_inference_steps


def test_return_fully_denoised_retargets_last_step_without_adding_one() -> None:
    torch = pytest.importorskip("torch", reason=TORCH_SKIP_REASON)

    scheduler = _FakeScheduler()
    proxy = PartialDenoiseSchedulerProxy(scheduler, 0.0, 0.5, return_fully_denoised=True)
    proxy.set_timesteps(num_inference_steps=10)

    assert scheduler.num_inference_steps == 5
    assert len(scheduler.timesteps) == 5
    assert len(scheduler.sigmas) == 6
    assert scheduler.sigmas[-1].item() == 0.0
    # The model still sees the real timesteps; only the final target moved.
    assert torch.allclose(scheduler.timesteps, torch.linspace(1.0, 0.0, 11)[:5] * 1000)
    assert scheduler.sigmas[-2].item() == pytest.approx(torch.linspace(1.0, 0.0, 11)[4].item())


def test_without_return_fully_denoised_keeps_schedule_target() -> None:
    torch = pytest.importorskip("torch", reason=TORCH_SKIP_REASON)

    scheduler = _FakeScheduler()
    proxy = PartialDenoiseSchedulerProxy(scheduler, 0.0, 0.5, return_fully_denoised=False)
    proxy.set_timesteps(num_inference_steps=10)

    assert scheduler.sigmas[-1].item() == pytest.approx(torch.linspace(1.0, 0.0, 11)[5].item())


def test_return_fully_denoised_does_not_mutate_original_sigmas() -> None:
    pytest.importorskip("torch", reason=TORCH_SKIP_REASON)

    scheduler = _FakeScheduler()
    proxy = PartialDenoiseSchedulerProxy(scheduler, 0.0, 0.5, return_fully_denoised=True)
    proxy.set_timesteps(num_inference_steps=10)
    proxy.set_timesteps(num_inference_steps=10)

    # A second call rebuilds the schedule from scratch, so the retarget is not compounded.
    assert len(scheduler.sigmas) == 6
    assert scheduler.sigmas[-2].item() > 0.0
