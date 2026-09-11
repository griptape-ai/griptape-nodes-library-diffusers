"""Conv/tile-based VAE activation-memory formula.

Deliberately its own module, not folded into activation_formulas.py: VAE activation
memory is conv/resolution-based, not attention/token-based, and it scales with
pixel-space resolution -- it can swing by multiple GB depending on whether tiling is
enabled, which a fixed constant (as used for text encoders) cannot represent.

Tile-size attribute names differ across VAE classes (verified against diffusers
source):
- AutoencoderKL (image): `tile_sample_min_size` (square).
- AutoencoderKLLTXVideo: `tile_sample_min_width` / `tile_sample_min_height` /
  `tile_sample_min_num_frames`.
- AutoencoderKLWan: `tile_sample_min_width` / `tile_sample_min_height` (no temporal
  tiling).

Per-level channel width is summed across resolution levels rather than pairing the
single deepest channel count with the single largest (pixel-space) resolution -- those
never coexist in a real forward pass; the deepest channel count only exists at the
most-downsampled internal resolution. Mirrors the same per-resolution-level shape
already used for the UNet-SDPA family (activation_formulas.py's
estimate_unet_sdpa_activation_bytes / family_registry.get_sdxl_unet_levels).

Within a single level, diffusers' decoder stacks more than one ResnetBlock2D (e.g.
`layers_per_block + 1` for AutoencoderKL -- vae.py's `Decoder.__init__`), each holding
its own live activation tensor at that level's resolution -- a single per-level tensor
undercounts this. Rather than hardcoding a per-family resnet count (it varies: uniform
+1 for AutoencoderKL/Flux2/Qwen/WAN, a per-level tuple for LTX, absent entirely for
MiniMax H3's ViT-style decoder), `_resnets_per_up_block` reads the real count off the
loaded (or meta-built) `decoder.up_blocks[i].resnets` module list, since the actual
module graph is already available in both the post-load and meta-device estimation
paths. The decoder's `mid_block` resnets are counted separately by
`_mid_block_extra_bytes` since `block_out_channels` never lists the mid-block's
resolution. Both fall back to the prior single-tensor behavior when a VAE doesn't
expose this conventional structure.

The config field naming this list comes from also differs across VAE classes:
AutoencoderKL, AutoencoderKLLTXVideo, AutoencoderKLLTX2, AutoencoderKLHunyuanVideo15,
and AutoencoderKLMiniMaxH3 all expose `block_out_channels` directly (verified against
diffusers source). AutoencoderKLWan instead expresses it as `base_dim * dim_mult[i]`
(autoencoder_kl_wan.py:978-981) -- no `block_out_channels` field exists on it at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact


def _resolve_block_out_channels(vae: torch.nn.Module) -> list[int]:
    """Return this VAE's per-level channel widths, regardless of which config field
    name the class uses to express them.

    Decoder-specific fields win when present: this formula estimates decode, and several
    classes let the decoder be wider than the encoder -- `AutoencoderKLFlux2` takes
    `decoder_block_out_channels` and passes `decoder_block_out_channels or
    block_out_channels` to its `Decoder` (autoencoder_kl_flux2.py:94, :128), and
    `AutoencoderKLWan` has the `decoder_base_dim` analog (autoencoder_kl_wan.py:979).
    Reading only the encoder's field silently builds the whole ladder from the wrong
    widths for any checkpoint that sets them.
    """
    decoder_block_out_channels = getattr(vae.config, "decoder_block_out_channels", None)
    if decoder_block_out_channels is not None:
        return list(decoder_block_out_channels)

    block_out_channels = getattr(vae.config, "block_out_channels", None)
    if block_out_channels is not None:
        return list(block_out_channels)

    base_dim = getattr(vae.config, "decoder_base_dim", None) or getattr(vae.config, "base_dim", None)
    dim_mult = getattr(vae.config, "dim_mult", None)
    if base_dim is not None and dim_mult is not None:
        return [base_dim * multiplier for multiplier in dim_mult]

    msg = (
        f"Attempted to resolve VAE per-level channel widths. Failed because "
        f"'{type(vae).__name__}' exposes neither 'block_out_channels' nor "
        f"'base_dim'/'dim_mult' in its config."
    )
    raise AttributeError(msg)


def _resolve_level_downsample_factors(vae: torch.nn.Module, num_levels: int) -> tuple[list[int], list[int]]:
    """Return per-level cumulative (spatial, temporal) downsample factors, ordered
    shallow-to-deep to match `block_out_channels`.

    A flat `2 ** level_index` is only correct for image VAEs, where every level halves
    spatially and nothing is temporally compressed. Video VAEs publish which levels scale
    and along which axes, and they do not scale uniformly:
    - `AutoencoderKLLTXVideo.spatio_temporal_scaling` (autoencoder_kl_ltx.py:763) -- a
      per-level bool scaling BOTH axes together; the last level is False by default.
    - `AutoencoderKLWan.temperal_downsample` (autoencoder_kl_wan.py:984, diffusers'
      spelling) -- temporal only, e.g. `[False, True, True]`, while spatial still halves
      every level.

    Holding frames constant across levels (the previous behavior) overcounts a WAN
    decoder's deepest levels by up to 4x, since they run at T/4.
    """
    spatio_temporal_scaling = getattr(vae.config, "spatio_temporal_scaling", None)
    temperal_downsample = getattr(vae.config, "temperal_downsample", None)

    spatial_factors: list[int] = []
    temporal_factors: list[int] = []
    spatial_factor = 1
    temporal_factor = 1
    for level_index in range(num_levels):
        spatial_factors.append(spatial_factor)
        temporal_factors.append(temporal_factor)
        if spatio_temporal_scaling is not None:
            if level_index < len(spatio_temporal_scaling) and spatio_temporal_scaling[level_index]:
                spatial_factor *= 2
                temporal_factor *= 2
        else:
            spatial_factor *= 2
            if temperal_downsample is not None and level_index < len(temperal_downsample):
                if temperal_downsample[level_index]:
                    temporal_factor *= 2
    return spatial_factors, temporal_factors


def _resolve_tiled_extent(vae: torch.nn.Module, source_shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return (frames, height, width) actually processed at once, honoring tiling.

    Tiles overlap by `tile_overlap_factor` (0.25 in every diffusers VAE that defines it),
    so the tensor a tile actually allocates is larger than the nominal tile size.
    """
    *_, height, width = source_shape
    num_frames = source_shape[-3] if len(source_shape) >= 5 else 1  # noqa: PLR2004

    overlap_scale = 1.0 + getattr(vae, "tile_overlap_factor", 0.0)

    if getattr(vae, "tile_sample_min_size", None) is not None:
        tile = int(vae.tile_sample_min_size * overlap_scale)
        return num_frames, min(height, tile), min(width, tile)

    tile_height = getattr(vae, "tile_sample_min_height", None)
    tile_width = getattr(vae, "tile_sample_min_width", None)
    if tile_height is not None and tile_width is not None:
        tile_frames = getattr(vae, "tile_sample_min_num_frames", num_frames)
        return (
            min(num_frames, tile_frames),
            min(height, int(tile_height * overlap_scale)),
            min(width, int(tile_width * overlap_scale)),
        )

    return num_frames, height, width


def _resnets_per_up_block(vae: torch.nn.Module) -> list[int] | None:
    """Return the decoder's per-level resnet counts, ordered shallow-to-deep to match
    `block_out_channels`, or None if the VAE has no conventional `decoder.up_blocks`
    (e.g. MiniMax H3's ViT-style decoder).
    """
    decoder = getattr(vae, "decoder", None)
    up_blocks = getattr(decoder, "up_blocks", None)
    if not up_blocks:
        return None
    return list(reversed([len(block.resnets) for block in up_blocks]))


def _mid_block_extra_bytes(
    vae: torch.nn.Module,
    deepest_channels: int,
    deepest_height: int,
    deepest_width: int,
    batch: int,
    frames: int,
    element_size: int,
) -> int:
    """Extra activation bytes for the decoder's mid-block resnets, which run at the
    deepest (smallest) resolution before the first up-block and are never listed in
    `block_out_channels`. Returns 0 if the VAE has no conventional `decoder.mid_block`.
    """
    decoder = getattr(vae, "decoder", None)
    mid_block = getattr(decoder, "mid_block", None)
    mid_block_resnets = getattr(mid_block, "resnets", None)
    if not mid_block_resnets:
        return 0
    return len(mid_block_resnets) * batch * frames * deepest_height * deepest_width * deepest_channels * element_size


def estimate_vae_activation_bytes(
    vae: torch.nn.Module,
    latent: LatentArtifact,
    optimization_kwargs: dict[str, Any],
    element_size: int,
) -> int:
    """Estimate VAE encode/decode activation memory, honoring vae_tiling/vae_slicing."""
    vae_tiling = optimization_kwargs.get("vae_tiling", False)
    vae_slicing = optimization_kwargs.get("vae_slicing", False)
    source_shape = latent.source_shape

    *_, full_height, full_width = source_shape
    full_frames = source_shape[-3] if len(source_shape) >= 5 else 1  # noqa: PLR2004

    if vae_tiling and getattr(vae, "use_tiling", False):
        frames, height, width = _resolve_tiled_extent(vae, source_shape)
    else:
        frames, height, width = full_frames, full_height, full_width

    batch = 1 if vae_slicing else source_shape[0] if len(source_shape) >= 4 else 1  # noqa: PLR2004

    block_out_channels = _resolve_block_out_channels(vae)
    resnets_per_level = _resnets_per_up_block(vae)
    spatial_factors, temporal_factors = _resolve_level_downsample_factors(vae, len(block_out_channels))

    total_bytes = 0
    for level_index, channels in enumerate(block_out_channels):
        level_height = max(height // spatial_factors[level_index], 1)
        level_width = max(width // spatial_factors[level_index], 1)
        level_frames = max(frames // temporal_factors[level_index], 1)
        if resnets_per_level is not None and level_index < len(resnets_per_level):
            resnet_count = resnets_per_level[level_index]
        else:
            resnet_count = 1
        total_bytes += resnet_count * batch * level_frames * level_height * level_width * channels * element_size

        # Upsample intermediate. UpDecoderBlock2D runs its resnets and THEN upsamples at
        # the same channel count (unet_2d_blocks.py, UpDecoderBlock2D.forward), so the
        # transition from the next-deeper level materialises a `block_out_channels[
        # level_index + 1]` tensor at THIS level's resolution -- a tensor that appears at
        # no level of the ladder. It is the single largest omission in the pre-fix
        # formula: 0.81 GiB of a measured 2.38 GiB for Flux2 at 1024x1024.
        if level_index + 1 < len(block_out_channels):
            deeper_channels = block_out_channels[level_index + 1]
            total_bytes += batch * level_frames * level_height * level_width * deeper_channels * element_size

    # Decoder output head: `conv_norm_out` -> silu -> `conv_out` holds a
    # block_out_channels[0]-wide tensor at the shallowest (full) resolution, outside
    # every up-block and so outside the level loop above.
    total_bytes += batch * frames * height * width * block_out_channels[0] * element_size

    deepest_height = max(height // spatial_factors[-1], 1)
    deepest_width = max(width // spatial_factors[-1], 1)
    deepest_frames = max(frames // temporal_factors[-1], 1)
    total_bytes += _mid_block_extra_bytes(
        vae, block_out_channels[-1], deepest_height, deepest_width, batch, deepest_frames, element_size
    )

    # Tiled decode still materialises the full-resolution output tensor that the tiles are
    # blended into, whatever the tile size -- a floor the per-tile terms above never
    # cover, and the term that keeps a tiled estimate from collapsing toward zero.
    out_channels = getattr(vae.config, "out_channels", 3)
    total_bytes += batch * full_frames * full_height * full_width * out_channels * element_size

    return total_bytes
