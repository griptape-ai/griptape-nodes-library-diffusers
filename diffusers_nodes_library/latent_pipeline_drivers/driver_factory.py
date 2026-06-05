from __future__ import annotations

from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from diffusers_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.flux import FluxLatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.flux2 import Flux2LatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.flux2_klein import Flux2KleinLatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.flux_fill import FluxFillLatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.ltx import LTXLatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.qwen import QwenLatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.qwen_edit import QwenEditLatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.stable_diffusion_xl import (
    StableDiffusionXLLatentPipelineDriver,
)
from diffusers_nodes_library.latent_pipeline_drivers.wan import WanTextToVideoLatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.wan_i2v import WanImageToVideoLatentPipelineDriver
from diffusers_nodes_library.latent_pipeline_drivers.z_image import ZImageLatentPipelineDriver

# Maps pipeline class name prefix to the corresponding driver class.
_DRIVER_REGISTRY: dict[str, type[LatentPipelineDriver]] = {
    "FluxFillPipeline": FluxFillLatentPipelineDriver,
    "FluxPipeline": FluxLatentPipelineDriver,
    "Flux2Pipeline": Flux2LatentPipelineDriver,
    "Flux2KleinPipeline": Flux2KleinLatentPipelineDriver,
    "QwenImagePipeline": QwenLatentPipelineDriver,
    "QwenImageEditPipeline": QwenEditLatentPipelineDriver,
    "StableDiffusionXLPipeline": StableDiffusionXLLatentPipelineDriver,
    "ZImagePipeline": ZImageLatentPipelineDriver,
    "WanPipeline": WanTextToVideoLatentPipelineDriver,
    "LTXPipeline": LTXLatentPipelineDriver,
    "WanImageToVideoPipeline": WanImageToVideoLatentPipelineDriver,
}


def get_driver_class(pipeline_class: str | None) -> type[LatentPipelineDriver] | None:
    """Return the driver class for *pipeline_class*, or ``None`` if unsupported."""
    if pipeline_class is None:
        return None
    return _DRIVER_REGISTRY.get(pipeline_class)


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
