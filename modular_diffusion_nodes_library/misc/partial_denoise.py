"""Utilities for running a partial denoise window of a diffusion pipeline.

Classes
-------
PartialDenoiseSchedulerProxy
    Wraps a scheduler and slices its timesteps to a fractional range when
    ``set_timesteps`` is called, so the pipeline only denoises over
    ``[denoise_begin, denoise_end)`` of the full schedule.

PartialDenoisePipelineRunner
    Context-manager that temporarily swaps ``pipe.scheduler`` with a
    ``PartialDenoiseSchedulerProxy`` for a single pipeline call, then
    restores the original scheduler.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch  # type: ignore[import]

logger = logging.getLogger("modular_diffusers_nodes_library")


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp *value* to the range [lo, hi]."""
    return max(lo, min(value, hi))


# ---------------------------------------------------------------------------
# Scheduler proxy for partial denoising
# ---------------------------------------------------------------------------

_STEP_INDEX_ATTRS = ("_step_index", "_begin_index")


class PartialDenoiseSchedulerProxy:
    """Forwards all attribute access to *scheduler* except ``set_timesteps``.

    Parameters
    ----------
    scheduler:
        The real scheduler instance owned by the pipeline.
    denoise_begin:
        Fraction of total steps to skip at the start (0.0 = beginning).
    denoise_end:
        Fraction of total steps at which to stop (1.0 = end, exclusive).
    return_fully_denoised:
        When ``True``, the terminal sigma is always appended to the sliced
        sigma sequence so the denoiser reaches a fully clean latent, even
        when ``denoise_end < 1.0``.  Defaults to ``False``.
    """

    def __init__(self, scheduler: Any, denoise_begin: float, denoise_end: float, return_fully_denoised: bool) -> None:
        if not (0.0 <= denoise_begin < denoise_end <= 1.0):
            raise ValueError(
                f"denoise_begin and denoise_end must satisfy 0 ≤ begin < end ≤ 1, "
                f"got begin={denoise_begin}, end={denoise_end}"
            )
        # Store on __dict__ directly to avoid going through __setattr__
        object.__setattr__(self, "_scheduler", scheduler)
        object.__setattr__(self, "_denoise_begin", denoise_begin)
        object.__setattr__(self, "_denoise_end", denoise_end)
        object.__setattr__(self, "_return_fully_denoised", return_fully_denoised)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the real scheduler."""
        return getattr(object.__getattribute__(self, "_scheduler"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute setting to the real scheduler, except for internal attributes."""
        if name in ("_scheduler", "_denoise_begin", "_denoise_end", "_return_fully_denoised"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_scheduler"), name, value)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        **kwargs,
    ):
        """
        Intercept the scheduler's ``set_timesteps`` call and slice the timesteps to the specified range.
        """
        scheduler = object.__getattribute__(self, "_scheduler")
        denoise_begin = object.__getattribute__(self, "_denoise_begin")
        denoise_end = object.__getattribute__(self, "_denoise_end")
        return_fully_denoised = object.__getattribute__(self, "_return_fully_denoised")

        # Let the real scheduler build its full schedule first.
        if timesteps is not None:
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        elif sigmas is not None:
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        else:
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)

        timesteps = scheduler.timesteps
        n = len(timesteps)

        if n == 0:
            return

        # Compute the begin and end index.
        begin = _clamp(math.floor(denoise_begin * n), 0, n - 1)
        end = _clamp(math.ceil(denoise_end * n), begin + 1, n)

        timesteps_slice = timesteps[begin:end]

        sigmas_slice = None
        if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None:
            sigmas = scheduler.sigmas
            # Slice from begin to end+1 (to include the sigma for the next step boundary)
            sigmas_slice = sigmas[begin : min(end + 1, len(sigmas))]

        # Optionally append the final timestep/sigma to ensure a fully denoised output, even when denoise_end < 1.0.
        if return_fully_denoised and timesteps_slice[-1] != timesteps[-1]:
            timesteps_slice = torch.cat([timesteps_slice, timesteps[-1:]])
            if sigmas_slice is not None:
                sigmas_slice = torch.cat([sigmas_slice, sigmas[-1:]])

        m = len(timesteps_slice)
        logger.debug(
            "PartialDenoiseSchedulerProxy: slicing %d steps to [%d:%d] (%d steps, begin=%.3f, end=%.3f), timesteps=%s",
            n,
            begin,
            end,
            m,
            denoise_begin,
            denoise_end,
            timesteps_slice,
        )

        scheduler.timesteps = timesteps_slice
        if sigmas_slice is not None:
            scheduler.sigmas = sigmas_slice
        scheduler.num_inference_steps = m

        for attr in _STEP_INDEX_ATTRS:
            if hasattr(scheduler, attr):
                setattr(scheduler, attr, 0)


class PartialDenoisePipelineRunner:
    """Temporarily swaps *pipe.scheduler* with a ``PartialDenoiseSchedulerProxy``.

    Parameters
    ----------
    pipe:
        A diffusers ``DiffusionPipeline`` instance.
    """

    def __init__(self, pipe: Any, *, proxy_class: type = PartialDenoiseSchedulerProxy) -> None:
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "_proxy_class", proxy_class)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        denoise_begin: float,
        denoise_end: float,
        return_fully_denoised: bool,
        **pipe_kwargs: Any,
    ) -> Any:
        """Run *pipe* with the scheduler restricted to ``[denoise_begin, denoise_end)``.

        Parameters
        ----------
        denoise_begin, denoise_end:
            Fractional positions in the full schedule (0 … 1).
        **pipe_kwargs:
            Forwarded verbatim to the pipeline's ``__call__``.
            Specify ``output_type="latent"`` to receive raw latents back.
        """
        pipe = object.__getattribute__(self, "_pipe")

        conflicting_kwargs = {"denoising_start", "denoising_end"} & pipe_kwargs.keys()
        if conflicting_kwargs:
            raise ValueError(
                f"pipe_kwargs must not contain {sorted(conflicting_kwargs)} argument(s) — "
                f"use denoise_begin/denoise_end on PartialDenoisePipelineRunner instead."
            )

        self._check_latent_support(pipe)

        # Always output latents.
        pipe_kwargs.setdefault("output_type", "latent")

        proxy_class = object.__getattribute__(self, "_proxy_class")
        proxy = proxy_class(pipe.scheduler, denoise_begin, denoise_end, return_fully_denoised)
        with self._swap_scheduler(pipe, proxy):
            return pipe(**pipe_kwargs)

    @staticmethod
    def _check_latent_support(
        pipe: Any,
    ) -> None:
        """Validate latent input capability of the pipeline."""
        call_sig = inspect.signature(pipe.__call__)
        accepts_latents = "latents" in call_sig.parameters

        if not accepts_latents:
            logger.warning(
                "Pipeline %s does not accept a 'latents' argument.",
                type(pipe).__name__,
            )
            raise TypeError(f"Pipeline {type(pipe).__name__} does not accept a 'latents' argument.")

    @staticmethod
    @contextmanager
    def _swap_scheduler(pipe: Any, proxy: PartialDenoiseSchedulerProxy) -> Generator[None, None, None]:
        """Context manager to temporarily swap the pipeline's scheduler with the proxy."""
        original = pipe.scheduler
        pipe.scheduler = proxy
        try:
            yield
        finally:
            pipe.scheduler = original
