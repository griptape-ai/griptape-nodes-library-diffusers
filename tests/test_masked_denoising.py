from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pytest

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import _DRIVER_REGISTRY
from modular_diffusion_nodes_library.masked_denoise.driver_integration import make_masked_denoising_callback
from modular_diffusion_nodes_library.misc.masked_denoising import (
    blend_differential_diffusion_latents,
    validate_masked_denoising_options,
    validate_schedule_length,
)

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

TORCH_SKIP_REASON = "Reads real torch; run `make test/exec`."
DIFFUSERS_SKIP_REASON = "Reads real diffusers classes; run `make test/exec` for the execution environment."


def test_masked_denoising_specs_only_advertise_pipelines_with_step_callbacks() -> None:
    diffusers = pytest.importorskip("diffusers", reason=DIFFUSERS_SKIP_REASON)

    for name, spec in _DRIVER_REGISTRY.items():
        if not spec.supports_masked_denoising:
            continue
        if name.endswith("ModularPipeline"):
            # Modular pipelines apply the mask through a block in the denoise loop, not a callback.
            continue

        pipeline_cls = getattr(diffusers, name)
        parameters = inspect.signature(pipeline_cls.__call__).parameters
        assert "callback_on_step_end" in parameters, (
            f"{name} advertises supports_masked_denoising but its __call__ has no 'callback_on_step_end'."
        )


class _FakeScheduler:
    """Scheduler stand-in whose schedule length is ``num_inference_steps + extra_steps``."""

    def __init__(self, *, extra_steps: int = 0, use_dynamic_shifting: bool = False) -> None:
        self.extra_steps = extra_steps
        self.config = {"use_dynamic_shifting": use_dynamic_shifting}
        self.timesteps: list[int] = []

    def set_timesteps(self, num_inference_steps: int, device: str) -> None:  # noqa: ARG002
        self.timesteps = list(range(num_inference_steps + self.extra_steps))


def test_validate_schedule_length_accepts_one_timestep_per_inference_step() -> None:
    validate_schedule_length(scheduler=_FakeScheduler(), num_inference_steps=4)


def test_validate_schedule_length_rejects_extra_timesteps_from_order_one_scheduler() -> None:
    with pytest.raises(ValueError, match="producing 5 timesteps for num_inference_steps=4"):
        validate_schedule_length(scheduler=_FakeScheduler(extra_steps=1), num_inference_steps=4)


def test_validate_schedule_length_skips_dynamic_shift_schedulers() -> None:
    validate_schedule_length(scheduler=_FakeScheduler(extra_steps=1, use_dynamic_shifting=True), num_inference_steps=4)


def test_masked_denoise_block_declares_the_state_it_reads() -> None:
    torch = pytest.importorskip("torch", reason=TORCH_SKIP_REASON)
    pytest.importorskip("diffusers", reason=DIFFUSERS_SKIP_REASON)
    from modular_diffusion_nodes_library.masked_denoise.driver_integration import (
        PreparedMaskedDenoising,
        make_masked_denoise_block,
    )

    prepared = PreparedMaskedDenoising(
        latents=torch.zeros(1, 1, 1),
        mask=torch.ones(1, 1, 1),
        noised_sources={0: torch.zeros(1, 1, 1)},
        generator_state=None,  # type: ignore[arg-type]
    )

    block = make_masked_denoise_block(model_name="flux3", masked_denoising=prepared)()

    assert {param.name for param in block.inputs} == {"latents", "num_inference_steps"}


def test_differential_releases_mask_values_as_schedule_advances() -> None:
    torch = pytest.importorskip("torch", reason=TORCH_SKIP_REASON)
    latents = torch.ones(1, 1, 1, 4)
    source = torch.zeros(1, 1, 1, 4)
    mask = torch.tensor([[[[0.0, 0.3, 0.8, 1.0]]]])

    def blend(step: int) -> list[float]:
        blended = blend_differential_diffusion_latents(
            latents=latents,
            source=source,
            mask=mask,
            absolute_step=step,
            num_inference_steps=4,
        )
        return blended.flatten().tolist()

    # Release threshold is 1 - (step + 1)/4: only mask >= threshold keeps the denoised value.
    assert blend(0) == [0.0, 0.0, 1.0, 1.0]
    assert blend(2) == [0.0, 1.0, 1.0, 1.0]
    assert blend(3) == [1.0, 1.0, 1.0, 1.0]


def test_mismatched_shapes_are_rejected() -> None:
    torch = pytest.importorskip("torch", reason=TORCH_SKIP_REASON)

    with pytest.raises(ValueError, match="shapes must match"):
        blend_differential_diffusion_latents(
            latents=torch.ones(1, 1, 1, 4),
            source=torch.zeros(1, 1, 1, 4),
            mask=torch.ones(1, 1, 1, 2),
            absolute_step=0,
            num_inference_steps=4,
        )


def test_non_unit_strength_is_rejected_in_favour_of_start_step() -> None:
    with pytest.raises(ValueError, match="Use start_step"):
        validate_masked_denoising_options(scheduler=object(), kwargs={"strength": 0.5})


def test_unit_strength_is_accepted() -> None:
    validate_masked_denoising_options(scheduler=object(), kwargs={"strength": 1.0})


def test_callback_uses_absolute_step_and_forwards_blended_latents() -> None:
    torch = pytest.importorskip("torch", reason=TORCH_SKIP_REASON)
    seen: list[Any] = []

    def user_callback(pipe: object, step_index: int, timestep: object, kwargs: dict) -> dict:  # noqa: ARG001
        seen.append(kwargs["latents"])
        return {}

    callback = make_masked_denoising_callback(
        mask=torch.ones(1, 1, 1, 1),
        noised_sources={2: torch.full((1, 1, 1, 1), 7.0)},
        start_step=2,
        num_inference_steps=5,
        callback=user_callback,
    )

    # Step index 0 of this call is absolute step 2; threshold is 0.4, so a mask of 1 keeps the generated latents.
    # The callback never reads the pipeline, and building a real one needs model weights.
    pipe: DiffusionPipeline = None  # type: ignore[assignment]
    timestep = torch.tensor(0)
    result = callback(pipe, 0, timestep, {"latents": torch.full((1, 1, 1, 1), 3.0)})

    assert result["latents"].item() == 3.0
    assert seen[0].item() == 3.0

    with pytest.raises(ValueError, match="no source was prepared"):
        callback(pipe, 1, timestep, {"latents": torch.zeros(1, 1, 1, 1)})


def test_callback_needs_no_source_at_the_final_step() -> None:
    torch = pytest.importorskip("torch", reason=TORCH_SKIP_REASON)
    callback = make_masked_denoising_callback(
        mask=torch.zeros(1, 1, 1, 1),
        noised_sources={},
        start_step=3,
        num_inference_steps=4,
        callback=None,
    )
    pipe: DiffusionPipeline = None  # type: ignore[assignment]

    result = callback(pipe, 0, torch.tensor(0), {"latents": torch.full((1, 1, 1, 1), 3.0)})

    assert result["latents"].item() == 3.0


def test_flux3_loop_adds_masked_denoise_block_after_after_denoiser() -> None:
    pytest.importorskip("torch", reason=TORCH_SKIP_REASON)
    pytest.importorskip("diffusers", reason=DIFFUSERS_SKIP_REASON)
    from modular_diffusion_nodes_library.latent_pipeline_drivers.flux3 import _make_callback_denoise_step
    from modular_diffusion_nodes_library.masked_denoise.driver_integration import PreparedMaskedDenoising

    prepared = PreparedMaskedDenoising(
        latents=None,  # type: ignore[arg-type]
        mask=None,  # type: ignore[arg-type]
        noised_sources={},
        generator_state=None,  # type: ignore[arg-type]
    )

    step = _make_callback_denoise_step(
        distilled=False,
        callback=None,
        start_step=0,
        end_step=-1,
        height=64,
        width=64,
        masked_denoising=prepared,
    )

    assert list(step.block_names) == ["denoiser", "after_denoiser", "masked_denoise"]
