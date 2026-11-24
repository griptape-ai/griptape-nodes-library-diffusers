"""Post-load optimization of a pipeline whose loader owns placement and weights."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("strategy", ["Manual", "Automatic"])
@pytest.mark.parametrize("offload", ["None", "Model", "Sequential"])
def test_loader_owned_placement_preserves_memory_options(
    monkeypatch: pytest.MonkeyPatch, strategy: str, offload: str
) -> None:
    torch = pytest.importorskip("torch")
    from modular_diffusion_nodes_library.utils import pipeline_utils

    events: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Loader-owned placement or weights were modified")

    pipe: Any = SimpleNamespace(
        _manages_runtime_optimisation=True,
        vae=SimpleNamespace(
            use_slicing=False,
            use_tiling=False,
            enable_slicing=lambda: events.append("slicing"),
            enable_tiling=lambda: events.append("tiling"),
        ),
        transformer=SimpleNamespace(enable_layerwise_casting=forbidden),
        enable_model_cpu_offload=forbidden,
        enable_sequential_cpu_offload=forbidden,
        to=forbidden,
    )
    monkeypatch.setattr(pipeline_utils, "get_best_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(pipeline_utils, "_log_memory_info", lambda *args: None)
    monkeypatch.setattr(pipeline_utils, "_quantize_diffusion_pipeline", forbidden)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda, "sdp_kernel", lambda **kwargs: events.append("backend"))

    pipeline_utils.optimize_diffusion_pipeline(
        pipe,
        memory_optimization_strategy=strategy,
        vae_slicing=True,
        vae_tiling=True,
        quantization_mode="int4",
        transformer_layerwise_casting=True,
        cpu_offload_strategy=offload,
    )

    assert events == ["slicing", "tiling", "backend"]


@pytest.fixture
def flux3_vae_override(loader_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> Any:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from modular_diffusion_nodes_library.utils import pipeline_utils

    vae = diffusers.AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8),
        norm_num_groups=4,
        layers_per_block=1,
        latent_channels=32,
        sample_size=64,
        mid_block_add_attention=False,
    )
    pipe = loader_environment.make_loader(workflows=("active",), device="cpu").from_pretrained(
        "org/base", overrides={"transformer": object(), "vae": vae}
    )
    assert pipe.vae is vae
    monkeypatch.setattr(pipeline_utils, "get_best_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda, "sdp_kernel", lambda **kwargs: None)
    return pipe


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_vae_override_slicing_splits_actual_decoder_batches(
    flux3_vae_override: Any, monkeypatch: pytest.MonkeyPatch, enabled: bool, batch_size: int
) -> None:
    torch = pytest.importorskip("torch")
    from modular_diffusion_nodes_library.utils.pipeline_utils import optimize_diffusion_pipeline

    pipe = flux3_vae_override
    vae = pipe.vae
    batches: list[int] = []
    scale = 2 ** (len(vae.config.block_out_channels) - 1)

    def decode(latents: Any) -> Any:
        batches.append(latents.shape[0])
        return latents.new_zeros((latents.shape[0], 3, latents.shape[-2] * scale, latents.shape[-1] * scale))

    monkeypatch.setattr(vae.decoder, "forward", decode)
    optimize_diffusion_pipeline(pipe, vae_slicing=enabled)

    output = vae.decode(torch.zeros((batch_size, 32, 8, 8)), return_dict=False)[0]

    expected_batches = [batch_size]
    if enabled and batch_size > 1:
        expected_batches = [1] * batch_size
    assert vae.use_slicing is enabled
    assert batches == expected_batches
    assert output.shape == (batch_size, 3, 8 * scale, 8 * scale)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
@pytest.mark.parametrize("offset", [-1, 0, 1], ids=["below", "at", "above"])
@pytest.mark.parametrize("axis", [-2, -1], ids=["height", "width"])
def test_vae_override_tiling_splits_actual_decoder_inputs(
    flux3_vae_override: Any, monkeypatch: pytest.MonkeyPatch, enabled: bool, offset: int, axis: int
) -> None:
    torch = pytest.importorskip("torch")
    from modular_diffusion_nodes_library.utils.pipeline_utils import optimize_diffusion_pipeline

    pipe = flux3_vae_override
    vae = pipe.vae
    decoded_shapes: list[tuple[int, ...]] = []
    scale = 2 ** (len(vae.config.block_out_channels) - 1)

    def decode(latents: Any) -> Any:
        decoded_shapes.append(tuple(latents.shape))
        return latents.new_zeros((latents.shape[0], 3, latents.shape[-2] * scale, latents.shape[-1] * scale))

    monkeypatch.setattr(vae.decoder, "forward", decode)
    optimize_diffusion_pipeline(pipe, vae_tiling=enabled)
    threshold = vae.tile_latent_min_size
    shape = [1, 32, threshold, threshold]
    shape[axis] += offset

    output = vae.decode(torch.zeros(shape), return_dict=False)[0]

    assert vae.use_tiling is enabled
    assert output.shape == (1, 3, shape[-2] * scale, shape[-1] * scale)
    if enabled and offset > 0:
        assert len(decoded_shapes) > 1
        assert all(height <= threshold and width <= threshold for _, _, height, width in decoded_shapes)
    else:
        assert decoded_shapes == [tuple(shape)]
