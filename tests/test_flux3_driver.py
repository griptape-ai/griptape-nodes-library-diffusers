from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

LAYOUT_PROMPT = (
    "Minimalist graphic illustration featuring a black silhouette of a person <silhouette_1> centered against a "
    "solid, vibrant chartreuse background <background_1>.\n"
    '[{"id": "background_1", "bbox": [0, 0, 1000, 1000], "desc": "A flat field of neon yellow-green."}, '
    '{"id": "silhouette_1", "bbox": [150, 150, 850, 850], "desc": "A black silhouette in motion."}]'
)


def test_flux3_layout_canvas_matches_diffusers() -> None:
    prompt_upsampler = pytest.importorskip("diffusers.modular_pipelines.flux3.prompt_upsampler")

    from modular_diffusion_nodes_library.utils.flux3_layout import _LAYOUT_CANVAS

    assert _LAYOUT_CANVAS == prompt_upsampler.CANVAS


@pytest.fixture
def torch_module() -> Any:
    torch_module = pytest.importorskip("torch", reason="Reads real torch tensors; run `make test/exec`.")
    if not hasattr(torch_module, "randn"):
        pytest.skip("Requires an installed PyTorch execution environment; run `make test/exec`.")
    return torch_module


@pytest.fixture
def flux3_module(torch_module: Any) -> Any:
    import modular_diffusion_nodes_library.latent_pipeline_drivers.flux3 as flux3

    return flux3


@pytest.fixture
def flux3_layout_driver(torch_module: Any) -> Any:
    pytest.importorskip("diffusers.modular_pipelines.flux3", reason="Needs diffusers flux3; run `make test/exec`.")

    from modular_diffusion_nodes_library.latent_pipeline_drivers.flux3 import Flux3LatentPipelineDriver

    return Flux3LatentPipelineDriver


@pytest.mark.parametrize("shape", [(1, 32, 64, 96), (2, 32, 64, 96)])
def test_patchification_round_trips_public_latents(
    shape: tuple[int, ...], torch_module: Any, flux3_module: Any
) -> None:
    public_latents = torch_module.randn(shape)

    patchified = flux3_module._patchify_latents(public_latents)

    assert patchified.shape == (shape[0], 128, shape[2] // 2, shape[3] // 2)
    assert torch_module.equal(flux3_module._unpatchify_latents(patchified), public_latents)


def test_prepare_output_latent_converts_flux3_patchified_shape_to_public_shape(
    torch_module: Any, flux3_module: Any
) -> None:
    driver = flux3_module.Flux3LatentPipelineDriver.__new__(flux3_module.Flux3LatentPipelineDriver)
    patchified = torch_module.zeros(1, 128, 64, 64)

    public_latents = driver.prepare_output_latent(patchified, (1, 3, 1024, 1024))

    assert public_latents.shape == (1, 32, 128, 128)


def test_add_noise_patchifies_inputs_and_returns_public_shape(
    monkeypatch: pytest.MonkeyPatch, torch_module: Any, flux3_module: Any
) -> None:
    from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
    from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState

    driver = flux3_module.Flux3LatentPipelineDriver.__new__(flux3_module.Flux3LatentPipelineDriver)
    source_shape = (1, 3, 512, 512)
    public_latents = torch_module.zeros(1, 32, 64, 64)
    public_noise = torch_module.ones_like(public_latents)
    latent = LatentArtifact.from_torch(public_latents, source_shape=source_shape)
    noise_artifact = LatentArtifact.from_torch(public_noise, source_shape=source_shape)
    scheduler_inputs: dict[str, Any] = {}

    class FakeScheduler:
        def set_begin_index(self, index):
            scheduler_inputs["begin_index"] = index

        def scale_noise(self, sample, timestep, noise):
            scheduler_inputs["sample"] = tuple(sample.shape)
            scheduler_inputs["noise"] = tuple(noise.shape)
            return sample + noise

    driver._pipe = cast(Any, SimpleNamespace(scheduler=FakeScheduler()))
    monkeypatch.setattr(driver, "_get_device_and_type", lambda: (torch_module.device("cpu"), torch_module.float32))
    monkeypatch.setattr(driver, "_call_block", lambda block, **kwargs: {"timesteps": torch_module.tensor([999, 0])})
    monkeypatch.setattr(driver, "create_noise_latent", lambda shape, state: noise_artifact)

    result = driver.add_noise_to_latent(latent, GeneratorState.from_seed(7), num_inference_steps=2, strength=1.0)

    assert scheduler_inputs == {
        "begin_index": 0,
        "sample": (1, 128, 32, 32),
        "noise": (1, 128, 32, 32),
    }
    assert tuple(result.to_torch().shape) == tuple(public_latents.shape)


def test_prepare_input_latent_converts_public_shape_to_flux3_patchified_shape(
    torch_module: Any, flux3_module: Any
) -> None:
    driver = flux3_module.Flux3LatentPipelineDriver.__new__(flux3_module.Flux3LatentPipelineDriver)
    public_latents = torch_module.zeros(1, 32, 128, 128)

    patchified = driver._prepare_input_latent(public_latents, (1, 3, 1024, 1024))

    assert patchified.shape == (1, 128, 64, 64)


def test_layout_prompt_is_split_from_json_and_boxes_scaled_to_unit_range(flux3_layout_driver: Any) -> None:
    layout = flux3_layout_driver._parse_layout_prompt(LAYOUT_PROMPT)

    assert "[{" not in layout.prompt
    assert layout.prompt.endswith("<background_1>.")
    assert layout.boxes is not None
    assert [box["id"] for box in layout.boxes] == ["background_1", "silhouette_1"]
    assert layout.boxes[1]["bbox"] == pytest.approx([0.15, 0.15, 0.85, 0.85])


def test_plain_prompt_has_no_layout(flux3_layout_driver: Any) -> None:
    layout = flux3_layout_driver._parse_layout_prompt("A red fox in snow.")

    assert layout.prompt == "A red fox in snow."
    assert layout.boxes is None


def test_out_of_range_box_raises_with_context(flux3_layout_driver: Any) -> None:
    bad_prompt = 'A scene.\n[{"id": "subject", "bbox": [0, 0, 1200, 500], "desc": "x"}]'

    with pytest.raises(ValueError, match="subject"):
        flux3_layout_driver._parse_layout_prompt(bad_prompt)


def test_boxes_are_ignored_when_upsampler_is_not_loaded(flux3_layout_driver: Any) -> None:
    layout = flux3_layout_driver._resolve_layout(LAYOUT_PROMPT, None, upsampler_enabled=False)

    assert layout.boxes is None
    assert layout.prompt == LAYOUT_PROMPT


def test_prompt_upsampler_is_enabled_only_when_both_components_are_loaded(flux3_module: Any) -> None:
    driver = flux3_module.Flux3LatentPipelineDriver.__new__(flux3_module.Flux3LatentPipelineDriver)
    pipe = SimpleNamespace(components={"prompt_upsampler": object()})
    driver._pipe = cast(Any, pipe)

    assert not driver._has_prompt_upsampler()

    pipe.components["prompt_upsampler_processor"] = object()

    assert driver._has_prompt_upsampler()
