import logging
from typing import Any, override

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.wan.modular_blocks_wan22_i2v import (
    Wan22Image2VideoBlocks,  # type: ignore[reportMissingImports]
)
from diffusers.modular_pipelines.wan.modular_blocks_wan_i2v import (
    WanImage2VideoAutoBlocks,  # type: ignore[reportMissingImports]
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState
from modular_diffusion_nodes_library.latent_pipeline_drivers.wan import WanTextToVideoLatentPipelineDriver
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import normalize_to_payloads
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    MediaGenConditioningKey,
    resolve_conditioning_image,
    resolve_frame_index,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class WanImageToVideoLatentPipelineDriver(WanTextToVideoLatentPipelineDriver):
    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        if getattr(self.pipe, "transformer_2", None) is not None:
            return Wan22Image2VideoBlocks().init_pipeline()
        return WanImage2VideoAutoBlocks().init_pipeline()

    @override
    def denoise_latent(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        latent: LatentArtifact,
        num_inference_steps: int,
        generator_state: GeneratorState,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        """Denoise a WAN i2v video latent."""

        update_kwargs = kwargs.copy()
        payloads = normalize_to_payloads(update_kwargs.pop(MediaGenConditioningKey.OUTPUT, None))

        height, width = latent.source_shape[-2], latent.source_shape[-1]

        if payloads is not None:
            num_frames = latent.source_shape[-3]
            for payload in payloads:
                if payload.mode is ConditioningMode.IMAGE:
                    for entry in payload.entries:
                        image = resolve_conditioning_image(entry.artifact)
                        frame_index = resolve_frame_index(entry.frame_index, num_frames)
                        output_image = self.preprocess_image(image, width, height)
                        if frame_index == 0:
                            update_kwargs["image"] = output_image
                        elif frame_index == -1 or frame_index == num_frames - 1:
                            update_kwargs["last_image"] = output_image
                        else:
                            msg = (
                                f"Attempted to build WAN i2v conditioning. "
                                f"Failed with frame_index={frame_index} because only 0 and -1/{num_frames - 1} "
                                f"are supported."
                            )
                            raise ValueError(msg)
                elif payload.mode is ConditioningMode.VIDEO:
                    logger.warning("Unsupported media_gen_conditioning mode 'video' for WAN i2v; ignoring.")
                else:
                    msg = f"Failed to build WAN video conditioning because mode '{payload.mode.value}' is unsupported."
                    raise ValueError(msg)

        if "image" not in update_kwargs:
            raise ValueError(
                f"{self.driver_namespace}: WAN i2v requires a first-frame image (frame_index=0) "
                "via media_gen_conditioning."
            )

        update_kwargs["height"] = height
        update_kwargs["width"] = width

        return super().denoise_latent(
            latent,
            num_inference_steps,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **update_kwargs,
        )

    def preprocess_image(self, image: Image, width: int, height: int) -> Image:
        """Resize a conditioning image to the video dimensions."""
        return image.resize((width, height))
