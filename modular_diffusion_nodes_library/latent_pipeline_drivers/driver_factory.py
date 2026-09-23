from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver

_DRIVER_PACKAGE = "modular_diffusion_nodes_library.latent_pipeline_drivers"


@dataclass(frozen=True)
class DriverSpec:
    """What can be known about a driver without importing it.

    `target` is the driver, as "module:attribute" rather than the class itself. Every driver subclasses
    a diffusers pipeline block, so holding the class here would pull diffusers and torch into any
    process that merely builds node classes -- which the orchestrator does for every library it
    registers, without ever running one.

    The remaining fields restate class attributes of that driver, for the same reason: the orchestrator
    decides which parameters to show from them, and importing a driver to read a boolean would put the
    whole execution stack on the orchestrator. `tests/test_driver_specs.py` fails if a restatement
    stops matching the driver it describes.
    """

    target: str
    #: Whether this pipeline produces video, which decides whether frame and fps parameters appear.
    produces_video: bool
    #: Default frame rate offered for a video pipeline.
    video_fps: int
    #: Whether the driver declares an inpaint pipeline class, which decides whether mask inputs appear.
    supports_inpainting: bool


_DRIVER_REGISTRY: dict[str, DriverSpec] = {
    "FluxFillPipeline": DriverSpec(
        "flux_fill:FluxFillLatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=True,
    ),
    "FluxKontextPipeline": DriverSpec(
        "flux_kontext:FluxKontextLatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=True,
    ),
    "FluxPipeline": DriverSpec(
        "flux:FluxLatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=True,
    ),
    "Flux2Pipeline": DriverSpec(
        "flux2:Flux2LatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=False,
    ),
    "Flux2KleinPipeline": DriverSpec(
        "flux2_klein:Flux2KleinLatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=True,
    ),
    "HunyuanVideo15Pipeline": DriverSpec(
        "hunyuan_video1_5:HunyuanVideo15TextToVideoLatentPipelineDriver",
        produces_video=True,
        video_fps=15,
        supports_inpainting=False,
    ),
    "HunyuanVideo15ImageToVideoPipeline": DriverSpec(
        "hunyuan_video1_5_i2v:HunyuanVideo15ImageToVideoLatentPipelineDriver",
        produces_video=True,
        video_fps=24,
        supports_inpainting=False,
    ),
    "LTXPipeline": DriverSpec(
        "ltx:LTXLatentPipelineDriver",
        produces_video=True,
        video_fps=25,
        supports_inpainting=False,
    ),
    "LTX2Pipeline": DriverSpec(
        "ltx2:LTX2PipelineDriver",
        produces_video=True,
        video_fps=24,
        supports_inpainting=False,
    ),
    "MiniMaxH3ModularPipeline": DriverSpec(
        "minimax_h3:MiniMaxH3LatentPipelineDriver",
        produces_video=True,
        video_fps=24,
        supports_inpainting=False,
    ),
    "QwenImagePipeline": DriverSpec(
        "qwen:QwenLatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=True,
    ),
    "QwenImageEditPipeline": DriverSpec(
        "qwen_edit:QwenEditLatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=True,
    ),
    "StableDiffusion3Pipeline": DriverSpec(
        "stable_diffusion_3:StableDiffusion3LatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=True,
    ),
    "StableDiffusionXLPipeline": DriverSpec(
        "stable_diffusion_xl:StableDiffusionXLLatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=True,
    ),
    "WanPipeline": DriverSpec(
        "wan:WanTextToVideoLatentPipelineDriver",
        produces_video=True,
        video_fps=16,
        supports_inpainting=False,
    ),
    "WanImageToVideoPipeline": DriverSpec(
        "wan_i2v:WanImageToVideoLatentPipelineDriver",
        produces_video=True,
        video_fps=16,
        supports_inpainting=False,
    ),
    "WanAnimatePipeline": DriverSpec(
        "wan_animate:WanAnimateLatentPipelineDriver",
        produces_video=True,
        video_fps=16,
        supports_inpainting=False,
    ),
    "WanVACEPipeline": DriverSpec(
        "wan_vace:WanVaceLatentPipelineDriver",
        produces_video=True,
        video_fps=16,
        supports_inpainting=False,
    ),
    "ZImagePipeline": DriverSpec(
        "z_image:ZImageLatentPipelineDriver",
        produces_video=False,
        video_fps=16,
        supports_inpainting=True,
    ),
}


def supported_pipeline_classes() -> list[str]:
    """Every pipeline class this library has a driver for, sorted."""
    return sorted(_DRIVER_REGISTRY)


def get_driver_spec(pipeline_class: str | None) -> DriverSpec | None:
    """The driver's declared facts, or `None` if the pipeline class is unsupported.

    Answers "is this supported" and "does it produce video" without importing a driver, which is what
    makes those questions askable on the orchestrator. Use `get_driver_class` only where the real class
    is needed, in the process that runs the node.
    """
    if pipeline_class is None:
        return None
    return _DRIVER_REGISTRY.get(pipeline_class)


def get_driver_class(pipeline_class: str | None) -> type[LatentPipelineDriver] | None:
    """Return the driver class for *pipeline_class*, or ``None`` if unsupported.

    Imports the driver's module, and with it diffusers and torch, so this belongs in the process that
    executes a node. A registered driver that fails to import raises rather than reading as
    unsupported: the name is known, so the failure is a broken driver and not a pipeline this library
    does not handle.
    """
    spec = get_driver_spec(pipeline_class)
    if spec is None:
        return None
    module_name, _, attribute = spec.target.partition(":")
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
        supported = ", ".join(supported_pipeline_classes())
        msg = (
            f"Attempted to create a latent pipeline driver. Failed with pipeline_class={pipeline_class!r} "
            f"because it is not supported. Supported pipeline classes: {supported}."
        )
        raise ValueError(msg)
    return driver_cls(pipe)
