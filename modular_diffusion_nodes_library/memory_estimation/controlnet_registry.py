"""Maps a diffusers pipeline class name to the ControlNet model class and
memory-estimation family it uses, mirroring family_registry.py's pattern.

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

from diffusers.models.controlnets.controlnet import ControlNetModel  # type: ignore[reportMissingImports]
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel  # type: ignore[reportMissingImports]
from diffusers.models.controlnets.controlnet_qwenimage import (  # type: ignore[reportMissingImports]
    QwenImageControlNetModel,
)
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel  # type: ignore[reportMissingImports]
from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.memory_estimation.family_registry import DenoiserConfigFields, MemoryFamily

# pipeline_name -> ControlNet model class this pipeline attaches when a
# ControlNet Pipeline node wraps it.
_CONTROLNET_CLASS_BY_PIPELINE_NAME: dict[str, type] = {
    "FluxPipeline": FluxControlNetModel,
    "StableDiffusion3Pipeline": SD3ControlNetModel,
    "StableDiffusionXLPipeline": ControlNetModel,
    "QwenImagePipeline": QwenImageControlNetModel,
    "ZImagePipeline": ZImageControlNetModel,
}

# pipeline_name -> memory family used to estimate that pipeline's ControlNet
# activation cost. StableDiffusionXL's ControlNetModel has no flat
# num_layers/hidden_dim (UNet-shaped, not DiT-shaped), so it reuses the
# UNET_SDPA level-based formula like the base SDXL UNet does -- this
# overestimates somewhat since the real ControlNetModel has no up-blocks,
# but matches this codebase's bias-toward-overestimate stance.
_CONTROLNET_ACTIVATION_FAMILY_BY_PIPELINE_NAME: dict[str, MemoryFamily] = {
    "FluxPipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "StableDiffusion3Pipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "StableDiffusionXLPipeline": MemoryFamily.UNET_SDPA,
    "QwenImagePipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "ZImagePipeline": MemoryFamily.IMAGE_DIT_SDPA,
}


def get_controlnet_class(pipeline_name: str) -> type | None:
    """Return the ControlNet model class *pipeline_name* attaches, or None if unregistered."""
    return _CONTROLNET_CLASS_BY_PIPELINE_NAME.get(pipeline_name)


def get_controlnet_activation_family(pipeline_name: str) -> MemoryFamily | None:
    """Return the memory family used for *pipeline_name*'s ControlNet activation cost."""
    return _CONTROLNET_ACTIVATION_FAMILY_BY_PIPELINE_NAME.get(pipeline_name)


def _adapt_flux_controlnet(config: Any) -> DenoiserConfigFields:
    return DenoiserConfigFields(
        hidden_dim=config.num_attention_heads * config.attention_head_dim,
        num_layers=config.num_layers + config.num_single_layers,
        patch_size_spatial=config.patch_size,
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
