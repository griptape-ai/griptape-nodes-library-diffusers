import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import LTXConditionPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.ltx.before_denoise import (  # type: ignore[reportMissingImports]
    LTXPrepareLatentsStep,
    LTXSetTimestepsStep,
)
from diffusers.modular_pipelines.ltx.decoders import LTXVaeDecoderStep  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.ltx.modular_blocks_ltx import (  # type: ignore[reportMissingImports]
    LTXAutoBlocks,
    LTXAutoVaeEncoderStep,
)
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ComponentSpec,
    InputParam,
    ModularPipeline,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # type: ignore[reportMissingImports]
from diffusers.video_processor import VideoProcessor  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import (
    DecodeResult,
    ImageMedia,
    LatentPipelineDriver,
    VideoMedia,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    resolve_conditioning_image,
    resolve_conditioning_video,
)
from modular_diffusion_nodes_library.utils.pipeline_utils import create_pipe_variant

logger = logging.getLogger("modular_diffusers_nodes_library")


# ------------------------------------------------------------------
# Custom composite blocks
# ------------------------------------------------------------------


class LTXSetTimestepsWithStrengthStep(ModularPipelineBlocks):
    """Slice scheduler timesteps based on img2img strength.

    Must run after ``LTXSetTimestepsStep`` which populates
    ``components.scheduler.timesteps``.
    """

    model_name = "ltx"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", required=True),
            InputParam("num_inference_steps", required=True),
            InputParam("strength", default=0.6),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)

        init_timestep = min(
            block_state.num_inference_steps * block_state.strength,
            block_state.num_inference_steps,
        )
        t_start = int(max(block_state.num_inference_steps - init_timestep, 0))
        timesteps = components.scheduler.timesteps[t_start * components.scheduler.order :]
        if hasattr(components.scheduler, "set_begin_index"):
            components.scheduler.set_begin_index(t_start * components.scheduler.order)

        block_state.timesteps = timesteps
        block_state.num_inference_steps = block_state.num_inference_steps - t_start

        self.set_block_state(state, block_state)
        return components, state


class LTXScaleNoiseStep(ModularPipelineBlocks):
    """Add noise to image latents via ``scheduler.scale_noise``.

    Expects noise in ``latents``, clean encoded latent in ``image_latents``,
    and the timestep schedule (already sliced for strength) in ``timesteps``.
    """

    model_name = "ltx"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True),
            InputParam("image_latents", required=True),
            InputParam("timesteps", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)

        latent_timestep = block_state.timesteps[:1].repeat(block_state.latents.shape[0])
        block_state.latents = components.scheduler.scale_noise(
            block_state.image_latents, latent_timestep, block_state.latents
        )

        self.set_block_state(state, block_state)
        return components, state


class LTXAddNoiseStep(SequentialPipelineBlocks):
    """Add noise to image latents for img2img workflows.

    Composes SetTimesteps -> SetTimestepsWithStrength -> ScaleNoise.
    The scheduler handles dynamic shift (mu) via LTXSetTimestepsStep,
    then strength slicing, then scale_noise.
    """

    model_name = "ltx"
    block_classes = [
        LTXSetTimestepsStep,
        LTXSetTimestepsWithStrengthStep,
        LTXScaleNoiseStep,
    ]
    block_names = ["set_timesteps", "set_timesteps_with_strength", "scale_noise"]


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------


class LTXLatentPipelineDriver(LatentPipelineDriver):
    """Hybrid driver for LTX Video pipelines.

    Uses the standard ``LTXPipeline`` for denoising and modular pipeline
    blocks for create_noise_latent, decode, and encode.

    Latents at the driver boundary are in **unpacked 5D** format
    ``[B, C, T, H, W]``.  Packing/unpacking is handled internally via
    ``prepare_input_latent`` / ``prepare_output_latent``.
    """

    produces_video: ClassVar[bool] = True
    video_fps: ClassVar[int] = 25

    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return LTXAutoBlocks().init_pipeline()

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @property
    def _vae_spatial_compression_ratio(self) -> int:
        return self.modular_pipe.vae_spatial_compression_ratio

    @property
    def _vae_temporal_compression_ratio(self) -> int:
        return self.modular_pipe.vae_temporal_compression_ratio

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Unpack 2D token latents back to 5D ``[B, C, T, H, W]``."""
        if latents_from_pipe.ndim == 5:
            return latents_from_pipe

        num_frames, height, width = latents_source_shape[-3], latents_source_shape[-2], latents_source_shape[-1]
        return self._unpack_latents(latents_from_pipe, height, width, num_frames)

    @override
    def _extract_latents_from_output(self, pipe_output: Any) -> torch.Tensor:
        """LTX pipeline with output_type='latent' returns packed latents under .frames."""
        return pipe_output.frames

    # ------------------------------------------------------------------
    # Latent creation
    # ------------------------------------------------------------------

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], seed: int) -> LatentArtifact:
        """Create a raw noise latent via modular pipeline block.

        Returns unpacked 5D latent ``[B, C, T, H, W]``.
        The block produces packed latents; we unpack before returning.
        """
        num_frames, height, width = source_shape[-3], source_shape[-2], source_shape[-1]
        output_state = self._call_block(
            LTXPrepareLatentsStep(),
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=1,
            num_videos_per_prompt=1,
            generator=torch.Generator().manual_seed(seed),
        )
        packed_latents = output_state.get("latents")
        latents = self._unpack_latents(packed_latents, height, width, num_frames)
        return self._make_latent_artifact(latents, source_shape=source_shape)

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        """Add noise to image latents via modular pipeline blocks.

        Uses LTXSetTimesteps -> SetTimestepsWithStrength -> ScaleNoise
        so the scheduler computes the correct timestep from strength
        and calls scale_noise internally.
        """
        device, dtype = self._get_device_and_type()
        source_shape = latent.source_shape
        latents = latent.to_torch(device=device, dtype=dtype)
        num_frames, height, width = source_shape[-3], source_shape[-2], source_shape[-1]

        noise = self.create_noise_latent(source_shape, seed).to_torch(device=device, dtype=dtype)
        output_state = self._call_block(
            LTXAddNoiseStep(),
            latents=noise,
            image_latents=latents,
            num_inference_steps=num_inference_steps,
            strength=strength,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=self.video_fps,
        )
        return self._make_latent_artifact(output_state.get("latents"), source_shape=source_shape, upstream=latent)

    # ------------------------------------------------------------------
    # VAE encode / decode (modular blocks)
    # ------------------------------------------------------------------

    @override
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        """Decode a 5D video latent and return frames as PIL images."""
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)

        packed_latents = self._pack_latents(latents)

        num_frames = latents.shape[2]
        pixel_height = latents.shape[3] * self._vae_spatial_compression_ratio
        pixel_width = latents.shape[4] * self._vae_spatial_compression_ratio
        pixel_num_frames = (num_frames - 1) * self._vae_temporal_compression_ratio + 1

        output_state = self._call_block(
            LTXVaeDecoderStep(),
            latents=packed_latents,
            height=pixel_height,
            width=pixel_width,
            num_frames=pixel_num_frames,
            batch_size=1,
            dtype=dtype,
            output_type="pil",
            decode_timestep=0.0,
        )
        videos = output_state.get("videos")
        return videos[0]

    @override
    def encode_media(self, media: ImageMedia | VideoMedia) -> LatentArtifact:
        """Encode an image or video into a 5D latent ``[B, C, T, H, W]``."""
        if isinstance(media, ImageMedia):
            output_state = self._call_block(LTXAutoVaeEncoderStep(), image=media.image)
            return self._make_latent_artifact(output_state.get("image_latents"), source_shape=media.source_shape)

        device, dtype = self._get_device_and_type()
        vae = self.modular_pipe.vae

        video_processor = VideoProcessor(vae_scale_factor=self._vae_spatial_compression_ratio)
        video_tensor = video_processor.preprocess_video(media.frames)  # [1, 3, T, H, W]
        video_tensor = video_tensor.to(device=device, dtype=dtype)

        with torch.no_grad():
            latents = vae.encode(video_tensor).latent_dist.sample()  # [1, C, T_latent, H_latent, W_latent]

        latents = latents.to(dtype=dtype)
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
        latents = (latents - latents_mean) * vae.config.scaling_factor / latents_std
        return self._make_latent_artifact(latents, source_shape=media.source_shape)

    # ------------------------------------------------------------------
    # Denoise (standard pipeline)
    # ------------------------------------------------------------------

    @override
    def denoise_latent(
        self,
        latent: LatentArtifact | InpaintMaskArtifact,
        num_inference_steps: int,
        seed: int = 0,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        """Denoise using the standard LTXPipeline.

        Pre-packs the latent (or swaps to LTXConditionPipeline) before delegating to base.
        """
        device, dtype = self._get_device_and_type()
        source_shape = latent.source_shape
        latents = latent.to_torch(device=device, dtype=dtype)

        media_gen_conditioning_list = kwargs.pop("media_gen_conditioning", None)
        if media_gen_conditioning_list is not None and not isinstance(media_gen_conditioning_list, list):
            media_gen_conditioning_list = [media_gen_conditioning_list]
        original_pipe = self._pipe

        if media_gen_conditioning_list is not None:
            video_condition_list = self._build_video_conditions(media_gen_conditioning_list, source_shape)
            kwargs["conditions"] = video_condition_list
            torch_dtype = self._get_torch_type(self._pipe)
            self._pipe = create_pipe_variant(original_pipe, LTXConditionPipeline, torch_dtype=torch_dtype)
        else:
            latents = self._pack_latents(latents)

        if "num_frames" not in kwargs:
            kwargs["num_frames"] = source_shape[-3]

        # Pre-set so base does not overwrite via prepare_input_latent.
        kwargs["latents"] = latents

        try:
            denoised_artifact = super().denoise_latent(
                latent,
                num_inference_steps=num_inference_steps,
                seed=seed,
                callback=callback,
                start_step=start_step,
                end_step=end_step,
                return_fully_denoised=return_fully_denoised,
                **kwargs,
            )
        finally:
            self._pipe = original_pipe

        return denoised_artifact

    # ------------------------------------------------------------------
    # Packing / Unpacking helpers
    # ------------------------------------------------------------------

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack 5D latent ``[B, C, T, H, W]`` into 2D token format."""
        return self.modular_pipe.pachifier.pack_latents(latents)

    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int, num_frames: int) -> torch.Tensor:
        """Unpack 2D token latents back to 5D ``[B, C, T, H, W]``."""
        latent_num_frames = (num_frames - 1) // self._vae_temporal_compression_ratio + 1
        latent_height = height // self._vae_spatial_compression_ratio
        latent_width = width // self._vae_spatial_compression_ratio
        pachifier = self.modular_pipe.pachifier
        base_token_count = (
            (latent_num_frames // pachifier.config.patch_size_t)
            * (latent_height // pachifier.config.patch_size)
            * (latent_width // pachifier.config.patch_size)
        )
        if latents.shape[1] > base_token_count:
            latents = latents[:, -base_token_count:]
        return pachifier.unpack_latents(latents, latent_num_frames, latent_height, latent_width)

    def _build_video_conditions(
        self,
        media_gen_conditioning_list: list[dict[str, Any]],
        latents_source_shape: tuple[int, ...],
    ) -> list[LTXVideoCondition]:
        """Convert media_gen_conditioning dicts into LTXVideoCondition objects."""
        video_condition_list: list[LTXVideoCondition] = []
        for media_gen_conditioning in media_gen_conditioning_list:
            mode = media_gen_conditioning.get("mode")
            if mode == "image":
                for image_item in media_gen_conditioning.get("images", []):
                    image = resolve_conditioning_image(image_item.get("image"))
                    strength = image_item.get("strength", 1.0)
                    frame_index = self._resolve_frame_index(image_item.get("frame_index", 0), latents_source_shape[-3])
                    video_condition_list.append(
                        LTXVideoCondition(image=image, frame_index=frame_index, strength=strength)
                    )
            elif mode == "video":
                video = resolve_conditioning_video(media_gen_conditioning)
                strength = media_gen_conditioning.get("strength", 1.0)
                frame_index = self._resolve_frame_index(
                    media_gen_conditioning.get("frame_index", 0), latents_source_shape[-3]
                )
                video_condition_list.append(LTXVideoCondition(video=video, frame_index=frame_index, strength=strength))
        return video_condition_list

    @staticmethod
    def _resolve_frame_index(frame_index: int, num_of_frames: int) -> int:
        # No upper bound check: LTX conditioning supports frame_index >= num_frames
        # for future-frame positioning via positional encodings (tokens are concatenated,
        # not indexed into the latent tensor).
        if frame_index < -1:
            raise ValueError(f"Unsupported frame_index {frame_index} for LTX conditioning. Only >= -1 are supported.")
        if frame_index == -1:
            return num_of_frames - 1
        return frame_index
