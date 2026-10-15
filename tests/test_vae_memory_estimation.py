"""Tests for config-only VAE encode/decode liveness tracing."""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch
from accelerate import init_empty_weights
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import (
    AutoencoderKLHunyuanVideo15,
    HunyuanVideo15AttnBlock,
)
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.memory_estimation import pipeline_memory_estimator as pme
from modular_diffusion_nodes_library.memory_estimation.vae_trace import (
    VaeMetaTraceError,
    trace_vae_activation_memory,
    vae_estimate_warning,
)
from modular_diffusion_nodes_library.utils.pipeline_utils import (
    _automatic_optimize_diffusion_pipeline,
    _manual_optimize_diffusion_pipeline,
    apply_vae_memory_options,
)

BF16 = 2


_VAE_CASES: list[tuple[str, type[Any], dict[str, Any], tuple[int, ...], tuple[int, ...]]] = [
    (
        "AutoencoderKL",
        AutoencoderKL,
        {
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["DownEncoderBlock2D"] * 4,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "layers_per_block": 2,
        },
        (1, 3, 32, 32),
        (1, 4, 4, 4),
    ),
    ("AutoencoderKLFlux2", AutoencoderKLFlux2, {}, (1, 3, 32, 32), (1, 32, 4, 4)),
    (
        "AutoencoderKLHunyuanVideo15",
        AutoencoderKLHunyuanVideo15,
        {},
        (1, 3, 17, 32, 32),
        (1, 32, 5, 2, 2),
    ),
    ("AutoencoderKLLTXVideo", AutoencoderKLLTXVideo, {}, (1, 3, 17, 32, 32), (1, 128, 3, 1, 1)),
    (
        "AutoencoderKLLTX2Video",
        AutoencoderKLLTX2Video,
        {},
        (1, 3, 17, 128, 128),
        (1, 128, 3, 4, 4),
    ),
    (
        "AutoencoderKLMiniMaxH3",
        AutoencoderKLMiniMaxH3,
        {"decoder_num_attention_heads": 1, "decoder_attention_head_dim": 64},
        (1, 3, 34, 64, 64),
        (1, 24, 7, 4, 4),
    ),
    ("AutoencoderKLQwenImage", AutoencoderKLQwenImage, {}, (1, 3, 17, 32, 32), (1, 16, 5, 4, 4)),
    ("AutoencoderKLWan", AutoencoderKLWan, {}, (1, 3, 17, 32, 32), (1, 16, 5, 4, 4)),
]


def _meta_vae(vae_class: type[Any], config: dict[str, Any] | None = None) -> Any:
    with init_empty_weights():
        vae = vae_class.from_config(config or {})
    torch.nn.Module.to(vae, device="meta", dtype=torch.bfloat16)
    vae.eval()
    return vae


def _latent(source_shape: tuple[int, ...], latent_shape: tuple[int, ...]) -> Any:
    return NS(source_shape=source_shape, shape=latent_shape)


def _case(name: str) -> tuple[type[Any], dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    for family, vae_class, config, source_shape, latent_shape in _VAE_CASES:
        if family == name:
            return vae_class, config, source_shape, latent_shape
    raise AssertionError(f"Unknown VAE test family: {name}")


@pytest.mark.parametrize("family", [case[0] for case in _VAE_CASES])
def test_public_encode_decode_paths_trace_on_meta(family: str) -> None:
    vae_class, config, source_shape, latent_shape = _case(family)
    vae = _meta_vae(vae_class, config)
    trace = trace_vae_activation_memory(vae, _latent(source_shape, latent_shape), BF16)

    assert trace.encode_bytes > 0
    assert trace.decode_bytes > 0
    assert trace.estimated_bytes == max(trace.encode_bytes, trace.decode_bytes)
    assert all(parameter.device.type == "meta" for parameter in vae.parameters())


def test_encode_trace_samples_posterior() -> None:
    """Include temporary tensors created by posterior sampling in the peak."""

    class Posterior:
        def __init__(self, mean: torch.Tensor) -> None:
            """Hold the posterior mean for this test."""
            self.mean = mean

        def sample(self) -> torch.Tensor:
            """Return a fresh tensor to represent sampling allocations."""
            return self.mean + torch.ones_like(self.mean)

    class SamplingVae(torch.nn.Module):
        config = {"in_channels": 3}

        def __init__(self) -> None:
            """Create a meta VAE for sampling-liveness coverage."""
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty((1,), device="meta"))

        @classmethod
        def from_config(cls, _config: dict[str, Any]) -> SamplingVae:
            """Build the synthetic VAE from test config."""
            return cls()

        def encode(self, input_tensor: torch.Tensor, return_dict: bool = True) -> Any:
            """Return a posterior whose sample allocates a fresh latent."""
            del return_dict
            mean = input_tensor[:, :1]
            return NS(latent_dist=Posterior(mean))

        def decode(self, input_tensor: torch.Tensor, return_dict: bool = True) -> torch.Tensor:
            """Pass through the latent for the decode trace."""
            return input_tensor

    trace = trace_vae_activation_memory(SamplingVae(), _latent((1, 3, 8, 8), (1, 1, 8, 8)), BF16)

    assert trace.encode_bytes > 3 * 8 * 8 * BF16


def test_trace_excludes_registered_buffer_storage() -> None:
    """Do not count persistent buffers as temporary activations."""

    class Posterior:
        def __init__(self, mean: torch.Tensor) -> None:
            """Hold the encoded tensor for this test."""
            self.mean = mean

        def sample(self) -> torch.Tensor:
            """Return the encoded tensor without adding another allocation."""
            return self.mean

    class BufferVae(torch.nn.Module):
        config = {"in_channels": 3}

        def __init__(self) -> None:
            """Create a meta VAE with a persistent buffer."""
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty((1,), device="meta"))
            self.register_buffer("resident", torch.empty((1, 1, 8, 8), device="meta"))

        @classmethod
        def from_config(cls, _config: dict[str, Any]) -> BufferVae:
            """Build the synthetic VAE from test config."""
            return cls()

        def encode(self, input_tensor: torch.Tensor, return_dict: bool = True) -> Any:
            """Read the buffer and return its encoded activation."""
            del return_dict
            encoded = input_tensor[:, :1] + self.resident
            return NS(latent_dist=Posterior(encoded))

        def decode(self, input_tensor: torch.Tensor, return_dict: bool = True) -> torch.Tensor:
            """Pass through the latent for the decode trace."""
            return input_tensor

    trace = trace_vae_activation_memory(BufferVae(), _latent((1, 3, 8, 8), (1, 1, 8, 8)), BF16)

    assert trace.encode_bytes == (3 + 1) * 8 * 8 * BF16


def test_trace_cache_reuses_equivalent_vaes_and_separates_options() -> None:
    """Reuse structurally identical VAEs but distinguish trace-affecting options."""
    vae_class, config, source_shape, latent_shape = _case("AutoencoderKL")
    first_vae = _meta_vae(vae_class, config)
    second_vae = _meta_vae(vae_class, config)
    latent = _latent(source_shape, latent_shape)

    first = trace_vae_activation_memory(first_vae, latent, BF16)
    second = trace_vae_activation_memory(second_vae, latent, BF16)
    second_vae.use_slicing = True
    with_option = trace_vae_activation_memory(second_vae, latent, BF16)

    assert first is second
    assert with_option is not second


def test_trace_cache_separates_source_shapes() -> None:
    vae_class, config, _source_shape, _latent_shape = _case("AutoencoderKL")
    vae = _meta_vae(vae_class, config)
    first = trace_vae_activation_memory(vae, _latent((1, 3, 32, 32), (1, 4, 4, 4)), BF16)
    second = trace_vae_activation_memory(vae, _latent((1, 3, 64, 64), (1, 4, 8, 8)), BF16)

    assert first is not second


def test_loaded_vae_is_cloned_to_meta_for_tracing() -> None:
    loaded_vae = AutoencoderKL.from_config(
        {
            "block_out_channels": [32, 32],
            "down_block_types": ["DownEncoderBlock2D"] * 2,
            "up_block_types": ["UpDecoderBlock2D"] * 2,
            "layers_per_block": 1,
        }
    )
    vae = loaded_vae[0] if isinstance(loaded_vae, tuple) else loaded_vae
    source_shape = (1, 3, 32, 32)
    latent_shape = (1, 4, 16, 16)

    trace = trace_vae_activation_memory(vae, _latent(source_shape, latent_shape), BF16)

    assert next(vae.parameters()).device.type == "cpu"
    assert trace.estimated_bytes > 0


def test_init_empty_vae_with_cpu_buffer_is_rebuilt_fully_on_meta() -> None:
    _, config, source_shape, latent_shape = _case("AutoencoderKLMiniMaxH3")
    vae_class, _, _, _ = _case("AutoencoderKLMiniMaxH3")
    with init_empty_weights():
        vae = vae_class.from_config(config)
    assert next(vae.parameters()).device.type == "meta"
    assert any(buffer.device.type != "meta" for buffer in vae.buffers())

    trace = trace_vae_activation_memory(vae, _latent(source_shape, latent_shape), 4)

    assert trace.estimated_bytes > 0


def test_hunyuan_trace_restores_mask_implementation() -> None:
    vae_class, config, source_shape, latent_shape = _case("AutoencoderKLHunyuanVideo15")
    vae = _meta_vae(vae_class, config)
    original = HunyuanVideo15AttnBlock.prepare_causal_attention_mask

    trace = trace_vae_activation_memory(vae, _latent(source_shape, latent_shape), BF16)

    assert trace.estimated_bytes > 0
    assert HunyuanVideo15AttnBlock.prepare_causal_attention_mask is original


def test_unsupported_meta_operation_raises_trace_error() -> None:
    class UnsupportedVae(torch.nn.Module):
        config = {"in_channels": 3}

        def __init__(self) -> None:
            """Create a meta VAE whose encode path deliberately fails."""
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty((1,), device="meta"))

        @classmethod
        def from_config(cls, _config: dict[str, Any]) -> UnsupportedVae:
            """Build the synthetic VAE from test config."""
            return cls()

        def encode(self, _input_tensor: torch.Tensor, return_dict: bool = True) -> Any:
            """Raise the test's unsupported meta-operation error."""
            raise RuntimeError("synthetic unsupported meta op")

        def decode(self, input_tensor: torch.Tensor, return_dict: bool = True) -> Any:
            """Pass through latents for the decode trace."""
            return input_tensor

    vae = UnsupportedVae()
    with pytest.raises(VaeMetaTraceError, match="encode.*synthetic unsupported meta op"):
        trace_vae_activation_memory(vae, _latent((1, 3, 8, 8), (1, 1, 8, 8)), BF16)


def test_only_minimax_gets_its_estimate_warning() -> None:
    minimax_class, config, *_ = _case("AutoencoderKLMiniMaxH3")
    minimax = _meta_vae(minimax_class, config)
    kl_class, kl_config, *_ = _case("AutoencoderKL")
    kl = _meta_vae(kl_class, kl_config)

    assert (
        vae_estimate_warning(minimax)
        == "VAE activation memory estimate omits backend-dependent attention workspaces; treat it as approximate."
    )
    assert vae_estimate_warning(kl) is None


def test_vae_memory_options_apply_supported_flags() -> None:
    vae = _meta_vae(AutoencoderKLLTXVideo)
    apply_vae_memory_options(vae, slicing=True, tiling=True)
    assert vae.use_slicing
    assert vae.use_tiling
    assert vae.use_framewise_decoding
    assert vae.use_framewise_encoding


def test_vae_memory_options_skip_unsupported_component() -> None:
    component = NS()
    apply_vae_memory_options(component, slicing=True, tiling=True)
    assert vars(component) == {}


def test_runtime_optimization_paths_use_shared_vae_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("modular_diffusion_nodes_library.utils.pipeline_utils._log_memory_info", lambda *_args: None)
    monkeypatch.setattr(
        "modular_diffusion_nodes_library.utils.pipeline_utils.should_enable_attention_slicing", lambda _device: False
    )
    automatic_vae = _meta_vae(AutoencoderKLLTXVideo)
    automatic_pipe: Any = NS(vae=automatic_vae)
    _automatic_optimize_diffusion_pipeline(automatic_pipe, torch.device("cuda"), requires_device_map=True)
    assert automatic_vae.use_slicing
    assert automatic_vae.use_tiling
    assert automatic_vae.use_framewise_decoding
    assert automatic_vae.use_framewise_encoding

    manual_vae = _meta_vae(AutoencoderKLLTXVideo)
    manual_pipe: Any = NS(vae=manual_vae)
    _manual_optimize_diffusion_pipeline(
        manual_pipe,
        torch.device("cpu"),
        attention_slicing=False,
        vae_slicing=True,
        vae_tiling=True,
        transformer_layerwise_casting=False,
        cpu_offload_strategy="None",
        quantization_mode="None",
        requires_device_map=True,
    )
    assert manual_vae.use_slicing
    assert manual_vae.use_tiling
    assert manual_vae.use_framewise_decoding
    assert manual_vae.use_framewise_encoding


def test_prebuild_vae_options_follow_saved_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vae_class, config, source_shape, latent_shape = _case("AutoencoderKL")
    vae = _meta_vae(vae_class, config)
    observed_state: list[tuple[bool, bool, bool, bool]] = []
    monkeypatch.setattr(pme, "_resolve_pipeline_cls", lambda _artifact: object)
    monkeypatch.setattr(pme, "_weight_bearing_slots", lambda *_args, **_kwargs: ["vae"])
    monkeypatch.setattr(pme, "_stored_bytes_for_base_component", lambda *_args: (0, BF16, vae))

    def record_trace(component: Any, _latent: Any, _element_size: int) -> int:
        observed_state.append(
            (
                component.use_slicing,
                component.use_tiling,
                getattr(component, "use_framewise_decoding", False),
                getattr(component, "use_framewise_encoding", False),
            )
        )
        return 1

    monkeypatch.setattr(pme, "estimate_vae_activation_bytes", record_trace)
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        build_data={"base_repo_id": "org/model"},
        optimization_kwargs={
            "memory_optimization_strategy": "Manual",
            "vae_slicing": True,
            "vae_tiling": True,
        },
    )

    pme.estimate_pipeline_memory_from_build_data(artifact, _latent(source_shape, latent_shape))

    assert observed_state == [(True, True, False, False)]


def test_unsupported_trace_becomes_weights_only_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    vae_class, config, source_shape, latent_shape = _case("AutoencoderKL")
    vae = _meta_vae(vae_class, config)
    monkeypatch.setattr(pme, "_resolve_pipeline_cls", lambda _artifact: object)
    monkeypatch.setattr(pme, "_weight_bearing_slots", lambda *_args, **_kwargs: ["vae"])
    monkeypatch.setattr(pme, "_stored_bytes_for_base_component", lambda *_args: (1024, BF16, vae))
    monkeypatch.setattr(
        pme,
        "estimate_vae_activation_bytes",
        lambda *_args: (_ for _ in ()).throw(VaeMetaTraceError("unsupported meta op")),
    )
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        build_data={"base_repo_id": "org/model"},
    )

    estimate = pme.estimate_pipeline_memory_from_build_data(artifact, _latent(source_shape, latent_shape))

    vae_estimate = estimate.components[0]
    assert vae_estimate.activation_bytes == 0
    assert vae_estimate.is_estimated
    assert vae_estimate.warning is not None
    assert "unsupported meta op" in vae_estimate.warning
