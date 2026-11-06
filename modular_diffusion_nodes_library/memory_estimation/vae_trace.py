"""Estimate VAE activation memory by running encode/decode on meta tensors.

Meta tensors carry shape and dtype but no data, so a VAE built from its config
alone runs the real diffusers `encode`/`decode` code (tiling, slicing, caching)
without weights or GPU memory. `LivenessTracker` sees every tensor op and
records the most storage bytes alive at once; that peak is the estimate.

Call flow:
    trace_vae_activation_memory(vae, latent, element_size)
      -> _trace_cached                  lru cache keyed on config, shapes, dtype size, options
         -> _construct_meta_vae         weight-free copy (also cached)
         -> _trace_pass("encode"), _trace_pass("decode")
              grid <= 5 tiles per axis -> _trace_exact -> _trace_phases   (one full trace)
              larger grid              -> _trace_phases on 4/5-tile samples -> _extrapolate

`_trace_phases` patches the VAE via `tiled_vae_trace.reuse_tiled_forwards` so
repeated encoder/decoder forwards are replayed instead of re-run.
"""

from __future__ import annotations

import itertools
import json
import math
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch  # type: ignore[reportMissingImports]
from accelerate import init_empty_weights  # type: ignore[reportMissingImports]
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import HunyuanVideo15AttnBlock
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.modular_pipelines.flux2.decoders import Flux2DecodeStep
from diffusers.modular_pipelines.qwenimage.modular_pipeline import QwenImagePachifier

from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import build_component_on_meta_device
from modular_diffusion_nodes_library.memory_estimation.tiled_vae_trace import reuse_tiled_forwards, spatial_tile_strides
from modular_diffusion_nodes_library.memory_estimation.vae_trace_common import (
    LivenessTracker,
    VaeMetaTraceError,
    patched_attribute,
)

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact


_TRACE_CACHE_MAXSIZE = 128
_VAE_ESTIMATE_WARNINGS = {
    "AutoencoderKLMiniMaxH3": (
        "MiniMax H3 traces representative encoder/decoder tiles and wrapper assembly; "
        "backend-dependent attention workspaces may be omitted."
    )
}
_META_VAE_CACHE_MAXSIZE = 8
# Tiles per axis traced when extrapolating a larger grid; 3-tile grids still show edge effects (Qwen).
_EXTRAPOLATION_TILES = (4, 5)
_HUNYUAN_MASK_PATCH_LOCK = threading.RLock()


@dataclass(frozen=True)
class VaeActivationTrace:
    """Peak live tensor-storage bytes for independent encode and decode passes."""

    encode_bytes: int
    decode_bytes: int

    @property
    def estimated_bytes(self) -> int:
        """Return the larger peak from encode and decode."""
        return max(self.encode_bytes, self.decode_bytes)


def _config_dict(vae: torch.nn.Module) -> dict[str, Any]:
    """Return the VAE config as a plain dictionary."""
    config = getattr(vae, "config", None)
    if config is None:
        msg = f"Cannot meta-trace VAE '{type(vae).__name__}': it exposes no config."
        raise VaeMetaTraceError(msg)
    if hasattr(config, "to_dict"):
        return config.to_dict()
    try:
        return dict(config)
    except (TypeError, ValueError) as error:
        msg = f"Cannot meta-trace VAE '{type(vae).__name__}': its config cannot be converted to a dict."
        raise VaeMetaTraceError(msg) from error


def _option_key(vae: torch.nn.Module) -> tuple[Any, ...]:
    """Return the VAE's `use_*`/`tile_*` runtime settings (e.g. tiling on/off) as a hashable key."""
    values = []
    for name, value in vars(vae).items():
        if name.startswith(("use_", "tile_")) and isinstance(value, (str, int, float, type(None))):
            values.append((name, value))
    return tuple(sorted(values))


def _meta_dtype(element_size: int) -> torch.dtype:
    """Return bfloat16 for 16-bit or smaller element sizes, float32 otherwise."""
    # Only the byte width matters on meta; quantized widths round up to 16-bit activations.
    if element_size <= 2:
        return torch.bfloat16
    return torch.float32


def _trace_input_shapes(
    vae: torch.nn.Module,
    source_shape: tuple[int, ...],
    latent_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Apply pipeline encode/decode shape transforms to public artifact shapes."""
    source = torch.empty(source_shape, device="meta")
    vae_latents = torch.empty(latent_shape, device="meta")

    if isinstance(vae, AutoencoderKLQwenImage):
        if source.ndim == 4:
            source = source.unsqueeze(2)
        if vae_latents.ndim == 4:
            pachifier = QwenImagePachifier()
            packed_latents = pachifier.pack_latents(vae_latents)
            vae_latents = pachifier.unpack_latents(
                packed_latents,
                source_shape[-2],
                source_shape[-1],
                vae_scale_factor=vae.spatial_compression_ratio,
            )
    elif isinstance(vae, AutoencoderKLFlux2) and vae_latents.shape[1] == vae.config["latent_channels"] * 4:
        vae_latents = Flux2DecodeStep._unpatchify_latents(vae_latents)

    return tuple(source.shape), tuple(vae_latents.shape)


def _keep_fp32_modules(vae: torch.nn.Module) -> None:
    """Restore float32 on modules diffusers pins via `_keep_in_fp32_modules` at load time."""
    keep_names = set(getattr(type(vae), "_keep_in_fp32_modules", None) or ())
    if not keep_names:
        return
    for name, module in vae.named_modules():
        if keep_names.intersection(name.split(".")):
            torch.nn.Module.to(module, dtype=torch.float32)


@lru_cache(maxsize=_META_VAE_CACHE_MAXSIZE)
def _construct_meta_vae(
    vae_class: type[torch.nn.Module], config_json: str, element_size: int, options: tuple[Any, ...]
) -> torch.nn.Module:
    """Build a weight-free meta VAE and apply cached runtime options."""
    dtype = _meta_dtype(element_size)
    try:
        with init_empty_weights():
            meta_vae = build_component_on_meta_device(vae_class, json.loads(config_json), dtype)
        torch.nn.Module.to(meta_vae, device="meta", dtype=dtype)
        _keep_fp32_modules(meta_vae)
    except Exception as error:
        msg = f"Cannot construct config-only meta VAE '{vae_class.__name__}': {error}"
        raise VaeMetaTraceError(msg) from error

    for name, value in options:
        setattr(meta_vae, name, value)
    return meta_vae


def _meta_causal_mask(n_frame, n_hw, dtype, device, batch_size=None):
    """Create the causal mask shape without filling its values."""
    seq_len = n_frame * n_hw
    mask = torch.empty((seq_len, seq_len), dtype=dtype, device=device)
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


@contextmanager
def _fast_hunyuan_meta_mask(vae: torch.nn.Module) -> Iterator[None]:
    """Skip Hunyuan mask-value construction while preserving its meta shape."""
    if type(vae).__name__ != "AutoencoderKLHunyuanVideo15":
        yield
        return

    with _HUNYUAN_MASK_PATCH_LOCK, ExitStack() as patches:
        for module in vae.modules():
            if isinstance(module, HunyuanVideo15AttnBlock):
                patches.enter_context(patched_attribute(module, "prepare_causal_attention_mask", _meta_causal_mask))
        yield


def _reduced_extent(extent: int, stride: int, tiles: int) -> int:
    """Shrink an axis to `tiles` tiles while keeping the same trailing edge-tile remainder."""
    return extent - (math.ceil(extent / stride) - tiles) * stride


@dataclass(frozen=True)
class _PassPeaks:
    """Peak live bytes between forwards, inside forwards, and after the final forward."""

    loop_bytes: int
    forward_bytes: int
    tail_bytes: int

    @property
    def total_bytes(self) -> int:
        return max(self.loop_bytes, self.forward_bytes, self.tail_bytes)


def _extrapolate(samples: dict[tuple[int, ...], int], steps: list[int]) -> int:
    """Extrapolate a peak from the 4- and 5-tile samples to the real tile counts.

    Models the peak as linear in tiles per axis (1 axis) or bilinear
    `a + b*h + c*w + d*h*w` (2 axes); the 4/5 samples fix the coefficients.
    `steps` = real tiles minus 4 per axis. Never returns less than a sample.
    """
    low, high = _EXTRAPOLATION_TILES
    if len(steps) == 1:
        a = samples[(low,)]
        b = samples[(high,)] - a
        estimate = a + b * steps[0]
    else:
        # Fit f(h, w) = a + b*h + c*w + d*h*w through f(0,0), f(1,0), f(0,1), f(1,1).
        a = samples[(low, low)]
        b = samples[(high, low)] - a
        c = samples[(low, high)] - a
        d = samples[(high, high)] - samples[(high, low)] - samples[(low, high)] + a
        estimate = a + steps[0] * b + steps[1] * c + steps[0] * steps[1] * d
    return max(estimate, *samples.values())


def _trace_pass(
    vae: torch.nn.Module,
    operation: str,
    input_shape: tuple[int, ...],
    element_size: int,
    *,
    reuse_tiles: bool = True,
) -> int:
    """Return one pass's peak bytes, extrapolating when the tile grid is large.

    Diffusers wrappers loop over tiles in Python, so a huge grid (e.g. 64800x115200
    px) would mean hundreds of thousands of traced ops. Instead, any axis with more
    than 5 tiles is shrunk to 4 and 5 tiles (keeping the edge-tile size), those 2 or
    4 grids are traced, and each phase peak is extrapolated. 3-tile samples are too
    small for Qwen.
    """
    strides = None
    if reuse_tiles and len(input_shape) in (4, 5):
        strides = spatial_tile_strides(vae, operation)
    if strides is None:
        return _trace_exact(vae, operation, input_shape, element_size, reuse_tiles=reuse_tiles)

    extents = input_shape[-2:]
    tile_counts = [math.ceil(extent / stride) for extent, stride in zip(extents, strides, strict=True)]
    large_axes = [index for index, count in enumerate(tile_counts) if count > _EXTRAPOLATION_TILES[-1]]
    if not large_axes:
        return _trace_exact(vae, operation, input_shape, element_size, reuse_tiles=reuse_tiles)

    # Sample only 4- and 5-tile grids, preserving the original edge tile in each reduced shape.
    samples: dict[tuple[int, ...], _PassPeaks] = {}
    # Shared traces avoid repeating core forwards across these representative grids.
    shared_traces: dict[str, dict[Any, Any]] = {}
    for sample in itertools.product(_EXTRAPOLATION_TILES, repeat=len(large_axes)):
        reduced = list(extents)
        for axis, tiles in zip(large_axes, sample, strict=True):
            reduced[axis] = _reduced_extent(extents[axis], strides[axis], tiles)
        samples[sample] = _trace_phases(
            vae, operation, (*input_shape[:-2], *reduced), element_size, shared_traces=shared_traces
        )

    # Extrapolate each peak phase independently; the peak phase may change on larger grids.
    steps = [tile_counts[axis] - _EXTRAPOLATION_TILES[0] for axis in large_axes]
    return max(
        _extrapolate({key: peaks.loop_bytes for key, peaks in samples.items()}, steps),
        _extrapolate({key: peaks.forward_bytes for key, peaks in samples.items()}, steps),
        _extrapolate({key: peaks.tail_bytes for key, peaks in samples.items()}, steps),
    )


def _trace_exact(
    vae: torch.nn.Module,
    operation: str,
    input_shape: tuple[int, ...],
    element_size: int,
    *,
    reuse_tiles: bool = True,
) -> int:
    """Trace one VAE pass over the full input and return its peak live activation bytes."""
    return _trace_phases(vae, operation, input_shape, element_size, reuse_tiles=reuse_tiles).total_bytes


def _trace_phases(
    vae: torch.nn.Module,
    operation: str,
    input_shape: tuple[int, ...],
    element_size: int,
    *,
    reuse_tiles: bool = True,
    shared_traces: dict[str, dict[Any, Any]] | None = None,
) -> _PassPeaks:
    """Trace one VAE pass on a meta input and return its peak live bytes per phase."""
    if len(input_shape) not in (4, 5) or any(dimension <= 0 for dimension in input_shape):
        msg = f"Cannot meta-trace VAE {operation}: invalid tensor shape {input_shape}."
        raise VaeMetaTraceError(msg)

    vae_operation = getattr(vae, operation, None)
    if not callable(vae_operation):
        msg = f"VAE '{type(vae).__name__}' has no callable {operation} method."
        raise VaeMetaTraceError(msg)

    dtype = _meta_dtype(element_size)
    input_tensor = torch.empty(input_shape, device="meta", dtype=dtype)
    tracker = LivenessTracker(vae)
    tracker.register(input_tensor)

    try:
        # Wrapper code and outputs are always traced; replay only skips repeated encoder/decoder forwards.
        with (
            torch.inference_mode(),
            reuse_tiled_forwards(vae, tracker, enabled=reuse_tiles, shared_traces=shared_traces) as phase_peaks,
            _fast_hunyuan_meta_mask(vae),
            tracker,
        ):
            operation_kwargs: dict[str, Any] = {"return_dict": True}
            if operation == "decode" and isinstance(vae, AutoencoderKLLTXVideo) and vae.config["timestep_conditioning"]:
                operation_kwargs["temb"] = torch.empty((input_shape[0],), device="meta", dtype=dtype)
            output = vae_operation(input_tensor, **operation_kwargs)
            # Encode returns a distribution; sampling it allocates the latent, which counts too.
            posterior = getattr(output, "latent_dist", None)
            if posterior is not None:
                posterior.sample()
    except Exception as error:
        msg = f"Meta trace failed for VAE '{type(vae).__name__}' {operation} shape {input_shape}: {error}"
        raise VaeMetaTraceError(msg) from error
    return _PassPeaks(
        loop_bytes=phase_peaks.loop_peak_bytes,
        forward_bytes=phase_peaks.forward_peak_bytes,
        tail_bytes=tracker.peak_bytes,
    )


@lru_cache(maxsize=_TRACE_CACHE_MAXSIZE)
def _trace_cached(
    vae_class: type[torch.nn.Module],
    config_json: str,
    source_shape: tuple[int, ...],
    latent_shape: tuple[int, ...],
    element_size: int,
    options: tuple[Any, ...],
) -> VaeActivationTrace:
    """Cache structural VAE traces; simultaneous first misses may trace twice."""
    meta_vae = _construct_meta_vae(vae_class, config_json, element_size, options)
    encode_bytes = _trace_pass(meta_vae, "encode", source_shape, element_size)
    decode_bytes = _trace_pass(meta_vae, "decode", latent_shape, element_size)
    return VaeActivationTrace(encode_bytes=encode_bytes, decode_bytes=decode_bytes)


def trace_vae_activation_memory(
    vae: torch.nn.Module,
    latent: LatentArtifact,
    element_size: int,
) -> VaeActivationTrace:
    """Trace public encode/decode paths on meta tensors and cache exact-shape results."""
    config_json = json.dumps(_config_dict(vae), sort_keys=True, default=repr)
    source_shape, latent_shape = _trace_input_shapes(vae, latent.source_shape, latent.shape)
    return _trace_cached(
        type(vae),
        config_json,
        source_shape,
        latent_shape,
        element_size,
        _option_key(vae),
    )


def vae_estimate_warning(vae: torch.nn.Module) -> str | None:
    """Return the class-specific warning for a VAE memory estimate, if any."""
    return _VAE_ESTIMATE_WARNINGS.get(type(vae).__name__)


def estimate_vae_activation_bytes(
    vae: torch.nn.Module,
    latent: LatentArtifact,
    element_size: int,
) -> int:
    """Return the larger traced peak from VAE encode or decode."""
    return trace_vae_activation_memory(vae, latent, element_size).estimated_bytes


__all__ = [
    "VaeActivationTrace",
    "VaeMetaTraceError",
    "estimate_vae_activation_bytes",
    "trace_vae_activation_memory",
    "vae_estimate_warning",
]
