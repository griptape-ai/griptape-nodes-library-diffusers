import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.hunyuan_video1_5 import (  # type: ignore[reportMissingImports]
    HunyuanVideo15AutoBlocks,
)
from diffusers.modular_pipelines.hunyuan_video1_5.before_denoise import (  # type: ignore[reportMissingImports]
    HunyuanVideo15PrepareLatentsStep,
)
from diffusers.modular_pipelines.hunyuan_video1_5.decoders import (  # type: ignore[reportMissingImports]
    HunyuanVideo15VaeDecoderStep,
)
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    DecodeResult,
    GeneratorState,
    ImageMedia,
    VideoMedia,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class HunyuanVideo15TextToVideoLatentPipelineDriver(LatentPipelineDriver):
    produces_video: ClassVar[bool] = True
    # EXAMPLE_DOC_STRING in pipeline_hunyuan_video1_5.py specifies fps=15
    video_fps: ClassVar[int] = 15

    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return HunyuanVideo15AutoBlocks().init_pipeline()

    @override
    def _extract_latents_from_output(self, pipe_output: Any) -> torch.Tensor:
        """HunyuanVideo pipelines return video frames under `.frames` instead of `.images`."""
        return pipe_output.frames

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        generator = generator_state.to_generator()
        num_frames, height, width = source_shape[-3], source_shape[-2], source_shape[-1]
        prepare_latents = HunyuanVideo15PrepareLatentsStep()
        output_state = self._call_block(
            prepare_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_videos_per_prompt=1,
            generator=generator,
            batch_size=1,
        )
        latents = output_state.get("latents")
        return self._make_latent_artifact(
            latents,
            source_shape=source_shape,
            meta=GeneratorState.from_generator(generator).as_meta(),
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        """Decode a 5-D HunyuanVideo latent and return the video frames."""
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        vae_decoder_step = HunyuanVideo15VaeDecoderStep()
        output_state = self._call_block(vae_decoder_step, latents=latents, output_type="pil")
        video_frames = output_state.get("videos")[0]
        return video_frames

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        raise NotImplementedError(
            f"Pipeline '{self.pipe.__class__.__name__}' does not support media encoding. "
            "This is a text-to-video pipeline."
        )

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        raise NotImplementedError(
            f"Pipeline '{self.pipe.__class__.__name__}' does not support img2img noise injection."
        )

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
        update_kwargs.pop("media_gen_conditioning", None)

        guidance_scale = update_kwargs.pop("guidance_scale", None)
        if guidance_scale is not None:
            self._pipe.guider.guidance_scale = guidance_scale

        update_kwargs["num_frames"] = latent.source_shape[-3]

        # HunyuanVideo15Pipeline.__call__ has no callback_on_step_end parameter,
        # so execution status, live preview and mid-run cancellation are not available for this pipeline.
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
