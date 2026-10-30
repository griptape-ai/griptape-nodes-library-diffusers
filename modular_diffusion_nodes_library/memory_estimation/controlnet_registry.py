"""Maps a diffusers pipeline class name to the denoiser-config adapter for its ControlNet.
The ControlNet model class comes from the driver's `controlnet_model_class`, and the
activation-memory family from the driver's `memory_family`.

Each ControlNet class is its own `ConfigMixin` with its own `config.json`,
hosted on its own HF repo -- its shape does NOT match the base pipeline's
denoiser (e.g. FluxControlNetModel defaults to 19 dual + 38 single layers vs.
FluxTransformer2DModel's identical defaults, but real published ControlNet
checkpoints are commonly truncated, e.g. InstantX's Union checkpoint ships
num_layers=5/num_single_layers=10). Field names were verified against each
class's real `__init__` signature and confirmed to match the same
DenoiserConfigFields field names family_registry.py's adapters already use
for the base pipeline (num_layers, attention_head_dim, num_attention_heads,
patch_size / block_out_channels).

Only pipelines whose driver's `can_make_control_pipe_from_standard()` can
return True are registered here (verified by reading every driver's
implementation): Flux, StableDiffusion3, StableDiffusionXL, QwenImage,
ZImage. All other drivers (Flux2, Flux2Klein, FluxKontext, FluxFill, LTX,
LTX2, WAN family, HunyuanVideo15, MiniMaxH3, QwenImageEdit) hardcode `return
False`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from modular_diffusion_nodes_library.memory_estimation.family_registry import DenoiserConfigFields


def _adapt_flux_controlnet(config: Any) -> DenoiserConfigFields:
    return DenoiserConfigFields(
        hidden_dim=config.num_attention_heads * config.attention_head_dim,
        num_layers=config.num_layers + config.num_single_layers,
        patch_size_spatial=2 * config.patch_size,
        patch_size_temporal=1,
    )


def _adapt_sd3_controlnet(config: Any) -> DenoiserConfigFields:
    return DenoiserConfigFields(
        hidden_dim=config.num_attention_heads * config.attention_head_dim,
        num_layers=config.num_layers,
        patch_size_spatial=config.patch_size,
        patch_size_temporal=1,
    )


def _adapt_qwen_controlnet(config: Any) -> DenoiserConfigFields:
    return DenoiserConfigFields(
        hidden_dim=config.num_attention_heads * config.attention_head_dim,
        num_layers=config.num_layers,
        patch_size_spatial=config.patch_size,
        patch_size_temporal=1,
    )


def _adapt_zimage_controlnet(config: Any) -> DenoiserConfigFields:
    return DenoiserConfigFields(
        hidden_dim=config.dim,
        num_layers=len(config.control_layers_places) + config.n_refiner_layers,
        patch_size_spatial=config.all_patch_size[0],
        patch_size_temporal=1,
    )


_CONTROLNET_DENOISER_FIELD_ADAPTERS: dict[str, Callable[[Any], DenoiserConfigFields]] = {
    "FluxPipeline": _adapt_flux_controlnet,
    "StableDiffusion3Pipeline": _adapt_sd3_controlnet,
    "QwenImagePipeline": _adapt_qwen_controlnet,
    "ZImagePipeline": _adapt_zimage_controlnet,
}


def get_controlnet_denoiser_config_fields(pipeline_name: str, controlnet_config: Any) -> DenoiserConfigFields | None:
    """Return normalized IMAGE_DIT_SDPA-family fields for a ControlNet's own config.

    Not used for StableDiffusionXLPipeline (UNET_SDPA family) -- that path reads
    block_out_channels/transformer_layers_per_block directly via
    family_registry.get_sdxl_unet_levels()-equivalent logic instead.
    """
    adapter = _CONTROLNET_DENOISER_FIELD_ADAPTERS.get(pipeline_name)
    if adapter is None:
        return None
    return adapter(controlnet_config)
