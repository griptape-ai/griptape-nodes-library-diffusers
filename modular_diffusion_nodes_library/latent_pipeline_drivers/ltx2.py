import logging
from typing import Any, ClassVar, override

import numpy as np
import torch  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.pipelines.ltx2.image_processor import LTX2VideoHDRProcessor  # type: ignore[reportMissingImports]
from diffusers.pipelines.ltx2.pipeline_ltx2 import calculate_shift  # type: ignore[reportMissingImports]
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import (  # type: ignore[reportMissingImports]
    LTX2ConditionPipeline,
    LTX2VideoCondition,
)
from diffusers.pipelines.ltx2.pipeline_ltx2_hdr_lora import (  # type: ignore[reportMissingImports]
    LTX2HDRPipeline,
    LTX2HDRReferenceCondition,
)
from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import (  # type: ignore[reportMissingImports]
    LTX2InContextPipeline,
    LTX2ReferenceCondition,
)
from diffusers.pipelines.ltx2.utils import (  # type: ignore[reportMissingImports]
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from diffusers.utils.torch_utils import randn_tensor  # type: ignore[reportMissingImports]
from griptape_nodes.files.path_utils import canonicalize_for_io
from PIL.Image import Image
from safetensors import safe_open  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import (
    GeneratorState,
    ImageMedia,
    LatentPipelineDriver,
    VideoMedia,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    pixel_frame_index_to_latent_index,
    resize_frames_scale_to_fill,
    resolve_conditioning_image,
    resolve_conditioning_video,
)
from modular_diffusion_nodes_library.utils.pipeline_utils import create_pipe_variant

logger = logging.getLogger("modular_diffusers_nodes_library")


class LTX2PipelineDriver(LatentPipelineDriver):
    produces_video: ClassVar[bool] = True

    _HDR_LORA_ADAPTER_TOKEN: ClassVar[str] = "ic-lora-hdr"
    _GENERIC_LORA_ADAPTER_TOKEN: ClassVar[str] = "lora"
    #: Record concrete ``DiffusionPipeline`` class that produced the latent.
    _PIPELINE_CLASS_META_KEY: ClassVar[str] = "pipeline_class"

    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @property
    def is_hdr_lora_active(self) -> bool:
        """Whether an LTX-2 HDR IC-LoRA adapter is currently loaded on ``self.pipe``.

        Matches any adapter whose name contains ``ic-lora-hdr`` (e.g.
        ``ltx-2.3-22b-ic-lora-hdr-0.9``).
        """
        token = self._HDR_LORA_ADAPTER_TOKEN
        return any(token in name.lower() for name in self._get_loaded_adapter_names())

    def _latent_was_produced_for_hdr(self, latent: LatentArtifact) -> bool:
        """Whether ``latent`` was produced by the HDR LoRA pipeline.

        Prefers the ``pipeline_class`` stamp written by ``denoise_latent``; falls
        back to ``is_hdr_lora_active`` for previews (which decode mid-generation
        before any artifact has been stamped).
        """
        driver_meta = latent.meta.get(self.driver_namespace) if isinstance(latent.meta, dict) else None
        if isinstance(driver_meta, dict):
            stamped = driver_meta.get(self._PIPELINE_CLASS_META_KEY)
            if stamped is not None:
                return stamped == LTX2HDRPipeline.__name__
        return self.is_hdr_lora_active

    @property
    def is_ic_lora_active(self) -> bool:
        """Whether any LoRA adapter is currently loaded on the pipeline."""
        token = self._GENERIC_LORA_ADAPTER_TOKEN
        return any(token in name.lower() for name in self._get_loaded_adapter_names())

    def _get_loaded_adapter_names(self) -> list[str]:
        get_list_adapters = getattr(self.pipe, "get_list_adapters", None)
        if get_list_adapters is None:
            return []
        adapters_by_component = get_list_adapters()
        return [name for names in adapters_by_component.values() for name in names]

    def _infer_reference_downscale_factor_from_loras(self) -> int | None:
        """Read `reference_downscale_factor` from any loaded IC-LoRA's safetensors metadata.

        The LoRA loaders stash safetensors header metadata on the pipe as
        ``pipe._gtn_lora_metadata[adapter_name] = {"path": ..., "metadata": {...}}``.
        LTX-2 IC-LoRA control adapters (e.g. ``...-ref0.5.safetensors``) record their
        training-time reference downscale factor under the ``reference_downscale_factor``
        key. Returns the first valid value found, or ``None``.
        """
        lora_metadata = getattr(self.pipe, "_gtn_lora_metadata", None)
        loaded_adapter_names = set(self._get_loaded_adapter_names())
        if not lora_metadata or not loaded_adapter_names:
            return None

        for adapter_name, entry in lora_metadata.items():
            if adapter_name in loaded_adapter_names:
                metadata = entry.get("metadata") or {}
                raw_value = metadata.get("reference_downscale_factor")
                if raw_value is not None:
                    try:
                        factor = float(raw_value)
                    except (TypeError, ValueError):
                        logger.warning(
                            "LTX2: ignoring non-numeric reference_downscale_factor=%r in %s",
                            raw_value,
                            entry.get("path"),
                        )
                    else:
                        if factor < 1:
                            logger.warning(
                                "LTX2: ignoring invalid reference_downscale_factor=%f in %s",
                                factor,
                                entry.get("path"),
                            )
                        else:
                            return int(factor)
        return None

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        raise NotImplementedError("LTX2 pipelines do not support modular pipelines. This method should not be called.")

    def _get_num_channels_latents(self) -> int:
        transformer = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "transformer_2", None)
        if transformer is None:
            raise AttributeError(
                f"Pipeline '{self.pipe.__class__.__name__}' has no 'transformer' or 'transformer_2' attribute."
            )
        return transformer.config.in_channels

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        """Return 5-D pure-noise latent [B, C, T, H, W] in VAE latent space."""
        device, _ = self._get_device_and_type()
        dtype = torch.float32

        num_frames, height, width = source_shape[-3], source_shape[-2], source_shape[-1]

        vae_spatial_ratio = getattr(self.pipe.vae, "spatial_compression_ratio", 32)
        vae_temporal_ratio = getattr(self.pipe.vae, "temporal_compression_ratio", 8)

        latent_height = height // vae_spatial_ratio
        latent_width = width // vae_spatial_ratio
        latent_frames = (num_frames - 1) // vae_temporal_ratio + 1
        num_channels_latents = self._get_num_channels_latents()

        shape = (1, num_channels_latents, latent_frames, latent_height, latent_width)
        generator = generator_state.to_generator()
        latent = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return self._make_latent_artifact(latent, source_shape=source_shape,
            meta=GeneratorState.from_generator(generator).as_meta(),
        )

    @override
    def add_noise_to_latent(self, latent: LatentArtifact, generator_state: GeneratorState, num_inference_steps: int, strength: float) -> LatentArtifact:
        """Return latent that has been noised, shape should match create_noise_latent. Latent should be unpacked so it may be processed further e.g. with mask."""
        device, _ = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=torch.float32)
        scheduler = self.pipe.scheduler
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        set_timesteps_kwargs: dict[str, Any] = {"sigmas": sigmas, "device": latents.device}
        if scheduler.config.get("use_dynamic_shifting", False):
            set_timesteps_kwargs["mu"] = calculate_shift(
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_image_seq_len", 1024),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.95),
                scheduler.config.get("max_shift", 2.05),
            )
        scheduler.set_timesteps(num_inference_steps, **set_timesteps_kwargs)
        timesteps = scheduler.timesteps

        if timesteps is None or len(timesteps) == 0:
            raise ValueError("Scheduler timesteps are not set. Cannot add noise to latent.")

        generator = generator_state.to_generator()
        noise = randn_tensor(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)

        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        latent_timestep = timesteps[t_start * scheduler.order :][:1]

        noisy = scheduler.scale_noise(latents, latent_timestep, noise)
        return self._make_latent_artifact(noisy, source_shape=latent.source_shape,
            upstream=latent, meta=GeneratorState.from_generator(generator).as_meta(),
        )

    @override
    def prepare_input_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> torch.Tensor:
        device, _ = self._get_device_and_type()
        latents = latents.to(device=device, dtype=torch.float32)
        return self.pipe._denormalize_latents(
            latents,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        )

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Pass through: the pipeline already unpacks (3-D→5-D) and denormalises before returning output latents.
        Also handles 3-D callbacks latents from mid-loop callbacks for preview.
        """
        device, _ = self._get_device_and_type()
        latents_from_pipe = latents_from_pipe.to(device=device, dtype=torch.float32)

        if latents_from_pipe.ndim == 3:
            return self._unpack_callback_latents(latents_from_pipe, latents_source_shape)

        return self.pipe._normalize_latents(
            latents_from_pipe,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        )

    def _unpack_callback_latents(
        self, packed_latents: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Unpack mid-loop callback latents from ``[B, S, D]`` to ``[B, C, F, H, W]``.

        The LTX2 pipeline keeps latents packed and normalised throughout the denoise loo.
        """
        pixel_num_frames = latents_source_shape[-3]
        pixel_height = latents_source_shape[-2]
        pixel_width = latents_source_shape[-1]

        vae_spatial_ratio = getattr(self.pipe.vae, "spatial_compression_ratio", 32)
        vae_temporal_ratio = getattr(self.pipe.vae, "temporal_compression_ratio", 8)

        latent_num_frames = (pixel_num_frames - 1) // vae_temporal_ratio + 1
        latent_height = pixel_height // vae_spatial_ratio
        latent_width = pixel_width // vae_spatial_ratio

        patch_size = self.pipe.transformer_spatial_patch_size
        patch_size_t = self.pipe.transformer_temporal_patch_size
        base_token_count = (
            (latent_num_frames // patch_size_t) * (latent_height // patch_size) * (latent_width // patch_size)
        )
        packed_latents = packed_latents[:, :base_token_count]

        return self.pipe._unpack_latents(
            packed_latents,
            latent_num_frames,
            latent_height,
            latent_width,
            patch_size,
            patch_size_t,
        )

    @override
    def _extract_latents_from_output(self, pipe_output: Any) -> torch.Tensor:
        """LTX2 pipelines return video frames under ``.frames`` instead of ``.images``."""
        return pipe_output.frames

    def _decode_hdr_to_linear_np(self, video: torch.Tensor) -> np.ndarray:
        """Convert decoded LogC3-compressed VAE output to linear HDR np.ndarray.

        Returns shape ``(B, F, H, W, 3)`` float32 with linear HDR values in ``[0, ∞)``.
        """
        vae_spatial_ratio = getattr(self.pipe.vae, "spatial_compression_ratio", 32)
        hdr_processor = LTX2VideoHDRProcessor(vae_scale_factor=vae_spatial_ratio, hdr_transform="logc3")
        hdr = hdr_processor.postprocess_hdr_video(video, output_type="np")

        logger.info(
            "LTX2PipelineDriver._decode_hdr_to_linear_np: linear HDR stats — "
            "min=%.4f max=%.4f mean=%.4f p50=%.4f p95=%.4f p99=%.4f gt1=%.2f%%",
            float(hdr[0].min()),
            float(hdr[0].max()),
            float(hdr[0].mean()),
            float(np.percentile(hdr[0], 50)),
            float(np.percentile(hdr[0], 95)),
            float(np.percentile(hdr[0], 99)),
            float((hdr[0] > 1.0).mean() * 100.0),
        )
        return hdr

    @override
    def decode_latent(self, latent: LatentArtifact) -> list[Image] | np.ndarray:
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=torch.float32)

        if self.pipe.vae.config.timestep_conditioning:
            timestep = torch.zeros(latents.shape[0], device=device, dtype=dtype)
        else:
            timestep = None

        latents = self.pipe._denormalize_latents(
            latents,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        )
        latents = latents.to(device=device, dtype=dtype)

        with torch.no_grad():
            video = self.pipe.vae.decode(latents, timestep, return_dict=False)[0]

        if self._latent_was_produced_for_hdr(latent):
            # HDR IC-LoRA path: return raw linear HDR for encode_hdr_tensor_to_mp4 in vae_decoder.
            return self._decode_hdr_to_linear_np(video)

        frames = self.pipe.video_processor.postprocess_video(video, output_type="pil")[0]
        return frames

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        """Encode an image or video as an LTX2 video latent (5-D tensor [B, C, T, H, W])."""
        if isinstance(media, ImageMedia):
            if not isinstance(media.image, Image):
                raise TypeError(f"{self.driver_namespace}: Expected a PIL Image, got {type(media.image).__name__}.")
            frames = [media.image]
        else:
            frames = media.frames
        device, dtype = self._get_device_and_type()

        video_tensor = self.pipe.video_processor.preprocess_video(frames)
        video_tensor = video_tensor.to(device=device, dtype=self.pipe.vae.dtype)

        generator = generator_state.to_generator()
        with torch.no_grad():
            frame_latents = []
            for vid in video_tensor:
                encoded = self.pipe.vae.encode(vid.unsqueeze(0))
                if hasattr(encoded, "latent_dist"):
                    latent = encoded.latent_dist.sample(generator=generator)
                elif hasattr(encoded, "latents"):
                    latent = encoded.latents
                else:
                    raise ValueError(
                        f"VAE encode output for '{self.pipe.__class__.__name__}' has no latent_dist or latents attribute."
                    )
                frame_latents.append(latent)

        latents = torch.cat(frame_latents, dim=0).to(dtype)
        latents = self.pipe._normalize_latents(
            latents,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        )
        return self._make_latent_artifact(latents, source_shape=media.source_shape)

    @override
    def denoise_latent(self, latent: LatentArtifact | InpaintMaskArtifact, num_inference_steps: int, generator_state: GeneratorState, callback: Any = None, start_step: int = 0, end_step: int = -1, return_fully_denoised: bool = False, **kwargs: Any) -> LatentArtifact:
        kwargs = self._update_args_for_distilled_pipeline(kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", num_inference_steps)
        kwargs["num_frames"] = latent.source_shape[-3]

        if self.is_hdr_lora_active:
            return self._denoise_with_hdr_lora(latent, num_inference_steps, generator_state=generator_state, callback=callback, start_step=start_step, end_step=end_step, return_fully_denoised=return_fully_denoised, **kwargs)

        if self.is_ic_lora_active:
            return self._denoise_with_ic_lora(latent, num_inference_steps, generator_state=generator_state, callback=callback, start_step=start_step, end_step=end_step, return_fully_denoised=return_fully_denoised, **kwargs)

        kwargs.pop("text_embeddings_path", None)
        kwargs = self._set_default_kwargs(kwargs)

        if "media_gen_conditioning" in kwargs:
            return self._denoise_with_video_gen_conditioning(latent, num_inference_steps, generator_state=generator_state, callback=callback, start_step=start_step, end_step=end_step, return_fully_denoised=return_fully_denoised, **kwargs)

        result = super().denoise_latent(latent, num_inference_steps, generator_state=generator_state, callback=callback, start_step=start_step, end_step=end_step, return_fully_denoised=return_fully_denoised, **kwargs)
        return self._stamp_pipeline_class(result, self.pipe.__class__.__name__)

    def _stamp_pipeline_class(self, artifact: LatentArtifact, pipeline_class_name: str) -> LatentArtifact:
        """Return a copy of ``artifact`` whose driver-namespaced meta records ``pipeline_class_name``."""
        return self._make_latent_artifact(
            artifact.to_torch(),
            source_shape=artifact.source_shape,
            upstream=artifact,
            meta={self._PIPELINE_CLASS_META_KEY: pipeline_class_name},
        )

    # ------------------------------------------------------------------
    # Helpers for various denoise paths (base, HDR-LoRA, IC-LoRA, video-gen-conditioning)
    # ------------------------------------------------------------------

    @staticmethod
    def _set_default_kwargs(original_kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = original_kwargs.copy()
        ## set some default values if not in kwargs
        kwargs.setdefault("spatio_temporal_guidance_blocks", [28])
        kwargs.setdefault("guidance_rescale", 0.7)
        kwargs.setdefault("audio_guidance_rescale", 0.7)

        kwargs.setdefault("modality_scale", 3.0)
        kwargs.setdefault("audio_modality_scale", 3.0)

        kwargs.setdefault("stg_scale", 1.0)
        kwargs.setdefault("audio_stg_scale", 1.0)

        kwargs.setdefault("guidance_scale", 3.0)
        kwargs.setdefault("audio_guidance_scale", 3.0)
        kwargs.setdefault("use_cross_timestep", True)
        return kwargs

    @staticmethod
    def _update_args_for_distilled_pipeline(original_kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = original_kwargs.copy()
        if "use_stage_2" in kwargs:
            use_stage_2 = kwargs.pop("use_stage_2")
            if not use_stage_2:
                kwargs["sigmas"] = DISTILLED_SIGMA_VALUES
            else:
                kwargs["sigmas"] = STAGE_2_DISTILLED_SIGMA_VALUES
                kwargs["noise_scale"] = STAGE_2_DISTILLED_SIGMA_VALUES[0]
            kwargs["num_inference_steps"] = len(kwargs["sigmas"])
        return kwargs

    def _run_denoise_with_pipe_variant(
        self,
        latent: LatentArtifact | InpaintMaskArtifact,
        num_inference_steps: int,
        target_class: type[DiffusionPipeline],
        *,
        generator_state: GeneratorState,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        call_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any] | None = None,
    ) -> LatentArtifact:
        torch_dtype = LTX2PipelineDriver._get_torch_type(self.pipe)
        variant_pipe = create_pipe_variant(self.pipe, target_class, torch_dtype=torch_dtype)

        kwargs = call_kwargs.copy()
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        original_pipe = self._pipe
        try:
            self._pipe = variant_pipe
            result = super().denoise_latent(
                latent,
                num_inference_steps,
                generator_state=generator_state,
                callback=callback,
                start_step=start_step,
                end_step=end_step,
                return_fully_denoised=return_fully_denoised,
                **kwargs,
            )
        finally:
            self._pipe = original_pipe
        return self._stamp_pipeline_class(result, target_class.__name__)

    # ------------------------------------------------------------------
    # Generic video gen conditioning support (for pipelines without LoRA adapters, or with fused LoRAs)
    # ------------------------------------------------------------------

    def _denoise_with_video_gen_conditioning(self, latent: LatentArtifact | InpaintMaskArtifact, num_inference_steps: int, generator_state: GeneratorState, callback: Any = None, start_step: int = 0, end_step: int = -1, return_fully_denoised: bool = False, **kwargs: Any) -> LatentArtifact:
        media_gen_conditioning_list = kwargs.pop("media_gen_conditioning")
        if media_gen_conditioning_list is not None and not isinstance(media_gen_conditioning_list, list):
            media_gen_conditioning_list = [media_gen_conditioning_list]
        conditions: list[LTX2VideoCondition] = []

        if media_gen_conditioning_list is not None:
            temporal_ratio = getattr(self.pipe.vae, "temporal_compression_ratio", 8)
            pixel_num_frames = latent.source_shape[-3]
            for media_gen_conditioning in media_gen_conditioning_list:
                mode = media_gen_conditioning.get("mode", "image")
                if mode == "video":
                    frames = resolve_conditioning_video(media_gen_conditioning)
                    strength = float(media_gen_conditioning.get("strength", 1.0))
                    pixel_frame_index = int(media_gen_conditioning.get("frame_index", 0))
                    index = pixel_frame_index_to_latent_index(pixel_frame_index, temporal_ratio, pixel_num_frames)
                    conditions.append(LTX2VideoCondition(frames=frames, index=index, strength=strength))
                else:
                    images = media_gen_conditioning.get("images", [])
                    if not images:
                        msg = "Attempted to build video gen conditioning. Failed because 'images' list is empty."
                        raise ValueError(msg)
                    for i, image_cond in enumerate(images):
                        pixel_frame_index = int(image_cond.get("frame_index", i))
                        latent_index = pixel_frame_index_to_latent_index(
                            pixel_frame_index, temporal_ratio, pixel_num_frames
                        )
                        strength = float(image_cond["strength"])
                        image = resolve_conditioning_image(image_cond.get("image"))
                        conditions.append(
                            LTX2VideoCondition(
                                frames=image,
                                index=latent_index,
                                strength=strength,
                            )
                        )

        if not conditions:
            msg = "Attempted to denoise with media conditioning. Failed because no valid media_gen_conditioning entries were provided."
            raise ValueError(msg)

        return self._run_denoise_with_pipe_variant(
            latent,
            num_inference_steps,
            LTX2ConditionPipeline,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            call_kwargs=kwargs,
            extra_kwargs={"conditions": conditions},
        )

    # ------------------------------------------------------------------
    # HDR IC-LoRA support
    # ------------------------------------------------------------------

    def _denoise_with_hdr_lora(self, latent: LatentArtifact | InpaintMaskArtifact, num_inference_steps: int, generator_state: GeneratorState, callback: Any = None, start_step: int = 0, end_step: int = -1, return_fully_denoised: bool = False, **kwargs: Any) -> LatentArtifact:
        logger.info("LTX2: HDR IC-LoRA path active — denoising with LTX2HDRPipeline.")
        media_gen_conditioning_list = kwargs.pop("media_gen_conditioning", None)
        text_embeddings_path = kwargs.pop("text_embeddings_path", "")
        target_height = latent.source_shape[-2]
        target_width = latent.source_shape[-1]
        reference_conditions = self._build_hdr_reference_conditions(media_gen_conditioning_list, target_height, target_width)
        kwargs = self._set_default_kwargs_hdr(kwargs)

        if text_embeddings_path:
            logger.info(f"LTX2: loading HDR text embeddings from '{text_embeddings_path}' for conditioning.")
            video_context, audio_context = self._load_hdr_text_embeddings(str(text_embeddings_path))
            kwargs["prompt"] = None
            kwargs["negative_prompt"] = None
            kwargs["connector_video_embeds"] = video_context
            kwargs["connector_audio_embeds"] = audio_context

        logger.info("LTX2: calling HDR pipeline: LTX2HDRPipeline")
        return self._run_denoise_with_pipe_variant(
            latent,
            num_inference_steps,
            LTX2HDRPipeline,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            call_kwargs=kwargs,
            extra_kwargs={"reference_conditions": reference_conditions},
        )

    @staticmethod
    def _set_default_kwargs_hdr(original_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Defaults for the `LTX2HDRPipeline` code path.

        The HDR pipeline has no audio inputs and its example uses `guidance_scale=1.0` and
        `stg_scale=0.0`. We also strip any audio-only kwargs that may have been set by the
        caller because `LTX2HDRPipeline.__call__` does not accept them.
        """
        kwargs = {k: v for k, v in original_kwargs.items() if not k.startswith("audio_")}
        kwargs.setdefault("guidance_scale", 1.0)
        kwargs.setdefault("stg_scale", 0.0)
        kwargs.setdefault("spatio_temporal_guidance_blocks", [28])
        kwargs.setdefault("modality_scale", 1.0)
        kwargs.setdefault("guidance_rescale", 0.0)
        return kwargs

    def _load_hdr_text_embeddings(self, text_embeddings_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        canonical_path = canonicalize_for_io(text_embeddings_path)
        if not canonical_path.exists():
            raise ValueError(
                "Attempted to load HDR text embeddings. Failed because the embedding file "
                f"'{canonical_path}' does not exist."
            )

        device, _ = self._get_device_and_type()
        try:
            with safe_open(canonical_path, framework="pt", device=str(device)) as embedding_file:
                video_context = embedding_file.get_tensor("video_context")
                audio_context = embedding_file.get_tensor("audio_context")
        except KeyError as err:
            raise ValueError(
                "Attempted to load HDR text embeddings. Failed because the safetensors file "
                f"'{canonical_path}' is missing required tensor '{err.args[0]}'. Expected 'video_context' and 'audio_context'."
            ) from err

        video_context = video_context.to(device=device)
        audio_context = audio_context.to(device=device)

        return video_context, audio_context

    def _build_hdr_reference_conditions(
        self,
        media_gen_conditioning_list: list[dict[str, Any]] | None,
        target_height: int,
        target_width: int,
    ) -> list[LTX2HDRReferenceCondition]:
        if not media_gen_conditioning_list:
            raise ValueError(
                "Attempted to denoise with HDR IC-LoRA. Failed because no media_gen_conditioning "
                "was provided; HDR LoRA requires a reference video."
            )
        if media_gen_conditioning_list is not None and not isinstance(media_gen_conditioning_list, list):
            media_gen_conditioning_list = [media_gen_conditioning_list]

        reference_conditions: list[LTX2HDRReferenceCondition] = []
        for media_gen_conditioning in media_gen_conditioning_list:
            mode = media_gen_conditioning.get("mode", "image")
            if mode != "video":
                raise ValueError(
                    "Attempted to denoise with HDR IC-LoRA. Failed because media_gen_conditioning mode "
                    f"is '{mode}'; HDR LoRA only supports a 'video' reference."
                )

            frames = resolve_conditioning_video(media_gen_conditioning)
            frames = resize_frames_scale_to_fill(frames, target_height, target_width)
            strength = float(media_gen_conditioning.get("strength", 1.0))
            reference_conditions.append(LTX2HDRReferenceCondition(frames=frames, strength=strength))

        return reference_conditions

    # ------------------------------------------------------------------
    # IC-LoRA support (non-HDR)
    # ------------------------------------------------------------------

    def _denoise_with_ic_lora(self, latent: LatentArtifact | InpaintMaskArtifact, num_inference_steps: int, generator_state: GeneratorState, callback: Any = None, start_step: int = 0, end_step: int = -1, return_fully_denoised: bool = False, **kwargs: Any) -> LatentArtifact:
        logger.info("LTX2: IC-LoRA path active - denoising with LTX2InContextPipeline.")
        media_gen_conditioning_list = kwargs.pop("media_gen_conditioning", None)
        reference_conditions = self._build_ic_reference_conditions(media_gen_conditioning_list)
        kwargs = self._set_default_kwargs(kwargs)

        extra_kwargs: dict[str, Any] = {}
        if reference_conditions:
            extra_kwargs["reference_conditions"] = reference_conditions

        if "reference_downscale_factor" not in kwargs:
            inferred = self._infer_reference_downscale_factor_from_loras()
            if inferred is not None:
                logger.info(
                    "LTX2: reference_downscale_factor not provided; using value %d from LoRA safetensors metadata.",
                    inferred,
                )
                extra_kwargs["reference_downscale_factor"] = inferred

        logger.info("LTX2: calling IC-LoRA pipeline: LTX2InContextPipeline")
        return self._run_denoise_with_pipe_variant(
            latent,
            num_inference_steps,
            LTX2InContextPipeline,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            call_kwargs=kwargs,
            extra_kwargs=extra_kwargs,
        )

    def _build_ic_reference_conditions(
        self,
        media_gen_conditioning_list: list[dict[str, Any]] | None,
    ) -> list[LTX2ReferenceCondition]:
        if media_gen_conditioning_list is None:
            return []
        if not isinstance(media_gen_conditioning_list, list):
            media_gen_conditioning_list = [media_gen_conditioning_list]

        reference_conditions: list[LTX2ReferenceCondition] = []
        for media_gen_conditioning in media_gen_conditioning_list:
            mode = media_gen_conditioning.get("mode", "image")
            if mode != "video":
                raise ValueError(
                    "Attempted to denoise with IC-LoRA. Failed because media_gen_conditioning mode "
                    f"is '{mode}'; IC-LoRA only supports a 'video' reference."
                )

            frames = resolve_conditioning_video(media_gen_conditioning)
            strength = float(media_gen_conditioning.get("strength", 1.0))
            reference_conditions.append(LTX2ReferenceCondition(frames=frames, strength=strength))

        return reference_conditions
