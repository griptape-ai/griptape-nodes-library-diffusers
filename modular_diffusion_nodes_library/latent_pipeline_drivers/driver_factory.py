from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver

_DRIVER_PACKAGE = "modular_diffusion_nodes_library.latent_pipeline_drivers"

# Maps a pipeline class name to the driver that handles it, as "module:attribute" rather than the
# class itself. Every driver subclasses a diffusers pipeline block, so importing them here would
# pull diffusers and torch into any process that merely builds node classes -- which the
# orchestrator does for every library it registers, without ever running one.
_DRIVER_REGISTRY: dict[str, str] = {
    "FluxFillPipeline": "flux_fill:FluxFillLatentPipelineDriver",
    "HunyuanVideo15Pipeline": "hunyuan_video1_5:HunyuanVideo15TextToVideoLatentPipelineDriver",
    "HunyuanVideo15ImageToVideoPipeline": "hunyuan_video1_5_i2v:HunyuanVideo15ImageToVideoLatentPipelineDriver",
    "FluxKontextPipeline": "flux_kontext:FluxKontextLatentPipelineDriver",
    "FluxPipeline": "flux:FluxLatentPipelineDriver",
    "Flux2Pipeline": "flux2:Flux2LatentPipelineDriver",
    "Flux2KleinPipeline": "flux2_klein:Flux2KleinLatentPipelineDriver",
    "LTX2Pipeline": "ltx2:LTX2PipelineDriver",
    "MiniMaxH3ModularPipeline": "minimax_h3:MiniMaxH3LatentPipelineDriver",
    "QwenImagePipeline": "qwen:QwenLatentPipelineDriver",
    "QwenImageEditPipeline": "qwen_edit:QwenEditLatentPipelineDriver",
    "StableDiffusion3Pipeline": "stable_diffusion_3:StableDiffusion3LatentPipelineDriver",
    "StableDiffusionXLPipeline": "stable_diffusion_xl:StableDiffusionXLLatentPipelineDriver",
    "ZImagePipeline": "z_image:ZImageLatentPipelineDriver",
    "WanPipeline": "wan:WanTextToVideoLatentPipelineDriver",
    "LTXPipeline": "ltx:LTXLatentPipelineDriver",
    "WanImageToVideoPipeline": "wan_i2v:WanImageToVideoLatentPipelineDriver",
    "WanAnimatePipeline": "wan_animate:WanAnimateLatentPipelineDriver",
    "WanVACEPipeline": "wan_vace:WanVaceLatentPipelineDriver",
}


def get_driver_class(pipeline_class: str | None) -> type[LatentPipelineDriver] | None:
    """Return the driver class for *pipeline_class*, or ``None`` if unsupported.

    Imports the driver's module on first lookup. A registered driver that fails to import raises
    rather than reading as unsupported: the name is known, so the failure is a broken driver and
    not a pipeline this library does not handle.
    """
    if pipeline_class is None:
        return None
    target = _DRIVER_REGISTRY.get(pipeline_class)
    if target is None:
        return None
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(f"{_DRIVER_PACKAGE}.{module_name}")
    return getattr(module, attribute)


def create_driver(pipe: DiffusionPipeline, pipeline_class: str | None) -> LatentPipelineDriver:
    """Instantiate and return the appropriate driver for *pipe*.

    Raises
    ------
    ValueError
        If *pipeline_class* has no registered driver.
    """
    driver_cls = get_driver_class(pipeline_class)
    if driver_cls is None:
        supported = ", ".join(sorted(_DRIVER_REGISTRY))
        msg = (
            f"Attempted to create a latent pipeline driver. Failed with pipeline_class={pipeline_class!r} "
            f"because it is not supported. Supported pipeline classes: {supported}."
        )
        raise ValueError(msg)
    return driver_cls(pipe)
