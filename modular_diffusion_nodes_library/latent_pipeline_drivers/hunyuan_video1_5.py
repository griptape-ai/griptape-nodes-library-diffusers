import logging
from typing import Any, ClassVar, cast, override

import torch  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.hunyuan_video1_5.before_denoise import (  # type: ignore[reportMissingImports]
    HunyuanVideo15PrepareLatentsStep,
    HunyuanVideo15SetTimestepsStep,
)
from diffusers.modular_pipelines.hunyuan_video1_5.decoders import (  # type: ignore[reportMissingImports]
    HunyuanVideo15VaeDecoderStep,
)
from diffusers.modular_pipelines.hunyuan_video1_5.modular_blocks_hunyuan_video1_5 import (
    HunyuanVideo15AutoBlocks,
)
from diffusers.modular_pipelines.hunyuan_video1_5.modular_pipeline import (  # type: ignore[reportMissingImports]
    HunyuanVideo15ModularPipeline,
)
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
    InputParam,
    OutputParam,
)
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


class _HunyuanVideo15EncodeVideoStep(ModularPipelineBlocks):
    """Encode a list of PIL frames into a normalised HunyuanVideo 1.5 video latent [B, C, T_lat, H/vsf, W/vsf]."""

    model_name = "hunyuan_video1_5"

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam("frames", required=True), InputParam("generator")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "video_latents",
                type_hint=torch.Tensor,
                description="Normalised video latent in the HunyuanVideo 1.5 VAE's output space.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: HunyuanVideo15ModularPipeline, state: PipelineState
    ) -> tuple[HunyuanVideo15ModularPipeline, PipelineState]:
        block_state = cast(Any, self.get_block_state(state))

        device = components._execution_device
        dtype = components.vae.dtype

        frame_tensors = [components.video_processor.preprocess(frame) for frame in block_state.frames]
        video_tensor = torch.stack([t.squeeze(0) for t in frame_tensors], dim=0)  # [T, C, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)  # [1, C, T, H, W]

        latents = components.vae.encode(video_tensor).latent_dist.sample(generator=block_state.generator)
        block_state.video_latents = latents * components.vae.config.scaling_factor

        self.set_block_state(state, block_state)
        return components, state


class _HunyuanVideo15AddNoiseStep(ModularPipelineBlocks):
    """Add flow-matching noise to a HunyuanVideo 1.5 video latent at a strength-derived timestep."""

    model_name = "hunyuan_video1_5"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True),
            InputParam("noise", required=True),
            InputParam("num_inference_steps", required=True),
            InputParam("strength", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "noisy_latents",
                type_hint=torch.Tensor,
                description="Latents with noise added at the strength-derived timestep.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: HunyuanVideo15ModularPipeline, state: PipelineState
    ) -> tuple[HunyuanVideo15ModularPipeline, PipelineState]:
        block_state = cast(Any, self.get_block_state(state))

        _, state = HunyuanVideo15SetTimestepsStep()(components, state)  # type: ignore[reportOperatorIssue]
        timesteps = state.values.get("timesteps")
        if timesteps is None or len(timesteps) == 0:
            raise ValueError("HunyuanVideo15SetTimestepsStep did not return valid timesteps.")

        num_inference_steps = block_state.num_inference_steps
        strength = block_state.strength
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        latent_timestep = timesteps[t_start * components.scheduler.order :][:1]

        block_state.noisy_latents = components.scheduler.scale_noise(
            block_state.latents, latent_timestep, block_state.noise
        )

        self.set_block_state(state, block_state)
        return components, state


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
        latents = self._get_required(output_state, "latents", torch.Tensor)
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
        video_frames = self._get_required(output_state, "videos", list)[0]
        return video_frames

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        if isinstance(media, ImageMedia):
            raise NotImplementedError(
                f"Pipeline '{self.pipe.__class__.__name__}' does not support image encoding. Use a video input instead."
            )
        generator = generator_state.to_generator()
        output = self._call_block(_HunyuanVideo15EncodeVideoStep(), frames=media.frames, generator=generator)
        return self._make_latent_artifact(
            self._get_required(output, "video_latents", torch.Tensor),
            source_shape=media.source_shape,
        )

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        device, dtype = self._get_device_and_type()
        source_shape = latent.source_shape
        latents = latent.to_torch(device=device, dtype=dtype)
        noise_artifact = self.create_noise_latent(source_shape, generator_state)
        noise = noise_artifact.to_torch(device=device, dtype=dtype)
        noise_generator_state = GeneratorState.from_artifact(noise_artifact) or generator_state
        output = self._call_block(
            _HunyuanVideo15AddNoiseStep(),
            latents=latents,
            noise=noise,
            num_inference_steps=num_inference_steps,
            strength=strength,
        )
        result = self._get_required(output, "noisy_latents", torch.Tensor)
        return self._make_latent_artifact(
            result,
            source_shape=source_shape,
            upstream=latent,
            meta=noise_generator_state.as_meta(),
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
