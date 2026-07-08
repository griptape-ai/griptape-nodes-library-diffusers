import logging
from typing import Any, ClassVar, override

from PIL.Image import Image, Resampling

from modular_diffusion_nodes_library.latent_pipeline_drivers.hunyuan_video1_5 import HunyuanVideo15TextToVideoLatentPipelineDriver
from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    DecodeResult,
    GeneratorState,
)
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import normalize_to_payloads
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    MediaGenConditioningKey,
    resolve_conditioning_image,
    resolve_frame_index,
)

logger = logging.getLogger("modular_diffusers_nodes_library")
class HunyuanVideo15ImageToVideoLatentPipelineDriver(HunyuanVideo15TextToVideoLatentPipelineDriver):
    # EXAMPLE_DOC_STRING in pipeline_hunyuan_video1_5_image2video.py specifies fps=24
    video_fps: ClassVar[int] = 24

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        width, height = self.get_resize_dimensions(source_shape[-1], source_shape[-2])
        resized_source_shape = (*source_shape[:-2], height, width)
        resized_output = super().create_noise_latent(resized_source_shape, generator_state)
        return self._make_latent_artifact(
            resized_output.to_torch(),
            source_shape=source_shape,
            upstream=resized_output,
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        frames = super().decode_latent(latent)
        source_shape = latent.source_shape
        output_frames = [frame.resize((source_shape[-1], source_shape[-2]), Resampling.LANCZOS) for frame in frames]  # type: ignore[reportGeneralTypeIssues]
        return output_frames

    @override
    def denoise_latent(
        self,
        latent: LatentArtifact | InpaintMaskArtifact,
        num_inference_steps: int,
        generator_state: GeneratorState,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        update_kwargs = kwargs.copy()
        payloads = normalize_to_payloads(update_kwargs.pop(MediaGenConditioningKey.OUTPUT, None))

        guidance_scale = update_kwargs.pop("guidance_scale", None)
        if guidance_scale is not None:
            self._pipe.guider.guidance_scale = guidance_scale

        if payloads is not None:
            num_frames = latent.source_shape[-3]
            for payload in payloads:
                if payload.mode is ConditioningMode.IMAGE:
                    for entry in payload.entries:
                        image = resolve_conditioning_image(entry.artifact)
                        image = self.preprocess_image(image, latent.source_shape[-1], latent.source_shape[-2])
                        frame_index = resolve_frame_index(entry.frame_index, num_frames)
                        if frame_index == 0:
                            update_kwargs["image"] = image
                        else:
                            msg = (
                                f"Attempted to build HunyuanVideo I2V conditioning. "
                                f"Failed with frame_index={frame_index} because only frame_index=0 is supported."
                            )
                            raise ValueError(msg)
                else:
                    msg = (
                        f"Failed to build HunyuanVideo I2V conditioning because "
                        f"mode '{payload.mode.value}' is unsupported."
                    )
                    raise ValueError(msg)

        if "image" not in update_kwargs:
            raise ValueError(
                f"{self.driver_namespace}: HunyuanVideo I2V requires a first-frame image (frame_index=0) "
                "via media_gen_conditioning."
            )

        update_kwargs["num_frames"] = latent.source_shape[-3]
        # height/width are NOT injected — HunyuanVideo I2V derives them from the image internally.

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


    def get_resize_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Calculate the resize dimensions for a given width and height."""
        target_size = self.pipe.transformer.config.target_size if getattr(self.pipe, "transformer", None) else 640
        height, width = self.pipe.video_processor.calculate_default_height_width(height=height, width=width, target_size=target_size)
        return width, height

    def preprocess_image(self, image: Image, width: int, height: int) -> Image:
        """Preprocess a PIL image for WAN i2v encoding."""
        # Automatically resize image based on model capabilities
        resized_width, resized_height = self.get_resize_dimensions(width, height)
        image = image.resize((resized_width, resized_height))
        return image