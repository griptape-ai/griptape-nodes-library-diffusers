import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers.models.controlnets.controlnet import ControlNetModel  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.stable_diffusion_xl.before_denoise import (  # type: ignore[reportMissingImports]
    StableDiffusionXLImg2ImgPrepareLatentsStep,
    StableDiffusionXLImg2ImgSetTimestepsStep,
    StableDiffusionXLPrepareLatentsStep,
    StableDiffusionXLSetTimestepsStep,
)
from diffusers.modular_pipelines.stable_diffusion_xl.modular_blocks_stable_diffusion_xl import (  # type: ignore[reportMissingImports]
    StableDiffusionXLAutoBlocks,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint_sd_xl import (  # type: ignore[reportMissingImports]
    StableDiffusionXLControlNetInpaintPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl_img2img import (  # type: ignore[reportMissingImports]
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (  # type: ignore[reportMissingImports]
    StableDiffusionXLInpaintPipeline,
)
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    GeneratorState,
    ImageMedia,
    VideoMedia,
    read_driver_meta,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


# ------------------------------------------------------------------
# Driver-namespaced meta keys (SDXL-only routing state)
# ------------------------------------------------------------------
#: Tag indicating how a latent should be consumed by ``denoise_latent``.
#: Values: ``"noise"``, ``"image_latents"``, ``"noised_image_latents"``.
#: Missing is treated as empty noise.
_KIND_META_KEY = "kind"


# ------------------------------------------------------------------
# Composite blocks for standalone noise / add-noise operations
# ------------------------------------------------------------------


class _SDXLPrepareNoiseLatentStep(SequentialPipelineBlocks):
    """``set_timesteps`` → ``prepare_latents`` so ``init_noise_sigma`` is valid."""

    model_name = "stable-diffusion-xl"  # type: ignore[reportIncompatibleMethodOverride]
    block_classes = [StableDiffusionXLSetTimestepsStep, StableDiffusionXLPrepareLatentsStep]
    block_names = ["set_timesteps", "prepare_latents"]


class _SDXLAddNoiseStep(SequentialPipelineBlocks):
    """``Img2ImgSetTimesteps`` → ``Img2ImgPrepareLatents`` for img2img noise."""

    model_name = "stable-diffusion-xl"  # type: ignore[reportIncompatibleMethodOverride]
    block_classes = [
        StableDiffusionXLImg2ImgSetTimestepsStep,
        StableDiffusionXLImg2ImgPrepareLatentsStep,
    ]
    block_names = ["set_timesteps", "prepare_latents"]


class StableDiffusionXLLatentPipelineDriver(LatentPipelineDriver):
    """Hybrid driver for classic StableDiffusionXLPipeline.

    Uses modular blocks for encode, decode, noise latent creation, and
    add-noise, but delegates denoise to the classic
    ``DiffusionPipeline.__call__`` (via the base class ``denoise_latent``).
    """

    _inpaint_pipeline_class = StableDiffusionXLInpaintPipeline
    _inpaint_controlnet_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = (
        StableDiffusionXLControlNetInpaintPipeline
    )

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return StableDiffusionXLAutoBlocks().init_pipeline()

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return True

    @classmethod
    @override
    def control_pipe_from_standard(
        cls,
        pipe: ModularPipeline | DiffusionPipeline,
        control_net_model_lists: list[str] | str | None,
    ) -> ModularPipeline | DiffusionPipeline:
        if not control_net_model_lists:
            return pipe

        if not isinstance(control_net_model_lists, list):
            control_net_model_lists = [control_net_model_lists]

        controlnet_torch_dtype = cls._get_torch_type(pipe)
        from_pretrained_kwargs: dict[str, Any] = {}
        if controlnet_torch_dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = controlnet_torch_dtype

        control_net_models = [
            ControlNetModel.from_pretrained(cn, **from_pretrained_kwargs) for cn in control_net_model_lists
        ]
        controlnet = control_net_models[0] if len(control_net_models) == 1 else control_net_models

        return StableDiffusionXLControlNetImg2ImgPipeline(controlnet=controlnet, **pipe.components)

    # ------------------------------------------------------------------
    # Modular blocks for encode / decode / noise latent
    # ------------------------------------------------------------------

    @override
    def create_noise_latent(
        self,
        source_shape: tuple[int, ...],
        generator_state: GeneratorState,
    ) -> LatentArtifact:
        """Return pure noise latent tagged ``kind="noise"``."""
        generator = generator_state.to_generator()
        output_state = self._call_block(
            _SDXLPrepareNoiseLatentStep(),
            height=source_shape[-2],
            width=source_shape[-1],
            batch_size=1,
            num_images_per_prompt=1,
            num_inference_steps=self._DEFAULT_NUM_INFERENCE_STEPS,
            generator=generator,
        )
        latents = output_state.get("latents")
        latents = latents / self.pipe.scheduler.init_noise_sigma

        meta = {_KIND_META_KEY: "noise", **GeneratorState.from_generator(generator).as_meta()}
        return self._make_latent_artifact(latents, source_shape=source_shape, meta=meta)

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        """Add noise to image latents via modular pipeline blocks.

        1. ``strength <= 0.0`` — no-op, return input unchanged.
        2. ``strength >= 1.0`` — delegate to ``create_noise_latent``; output
           tagged ``"noise"``.
        3. Input is degenerate (``kind in {"", "noise"}``) with partial strength —
           run the block on the (zero-or-noise) input, normalise by
           ``init_noise_sigma`` & tag as ``"noise"``. This is an unexpected call
           shape; the output is consistent but not statistically meaningful.
        4. Input is image latents (``kind in {"image_latents",
           "noised_image_latents"}`` or any other tag) — run the block, tag
           ``"noised_image_latents"``.
        """
        if strength <= 0.0:
            return latent
        if strength >= 1.0:
            noise_artifact = self.create_noise_latent(latent.source_shape, generator_state)
            return self._make_latent_artifact(
                noise_artifact.to_torch(),
                source_shape=latent.source_shape,
                upstream=latent,
                meta=noise_artifact.meta,
            )

        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        generator = generator_state.to_generator()
        output_state = self._call_block(
            _SDXLAddNoiseStep(),
            image_latents=latents,
            num_inference_steps=num_inference_steps,
            strength=strength,
            batch_size=latents.shape[0],
            num_images_per_prompt=1,
            generator=generator,
            dtype=dtype,
        )
        noised = self._get_required(output_state, "latents", torch.Tensor)
        generator_meta = GeneratorState.from_generator(generator).as_meta()

        kind = read_driver_meta(latent, _KIND_META_KEY, self.driver_namespace, "")
        if kind in ("", "noise"):
            noised = noised / self.pipe.scheduler.init_noise_sigma
            meta = {
                _KIND_META_KEY: "noise",
                **generator_meta,
            }
        else:
            meta = {
                _KIND_META_KEY: "noised_image_latents",
                **generator_meta,
            }
        return self._make_latent_artifact(
            noised,
            source_shape=latent.source_shape,
            upstream=latent,
            meta=meta,
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
        """Switch on ``kind`` to pick the correct entry path.

        - "noise" → scale by init_noise_sigma(num_inference_steps), then img2img at strength=1.0
          (img2img with `latents` provided short-circuits to the txt2img-equivalent path).
        - "image_latents" / "noised_image_latents" / "" / unknown → img2img at strength=1.0.
        - InpaintMaskArtifact → existing inpaint routing in the base class.
        """
        is_inpaint = isinstance(latent, InpaintMaskArtifact)

        if not is_inpaint:
            device, dtype = self._get_device_and_type()
            latents_on_device = latent.to_torch(device=device, dtype=dtype)
            kind = read_driver_meta(latent, _KIND_META_KEY, self.driver_namespace, "")
            if kind == "noise":
                self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
                latents_on_device = latents_on_device * self.pipe.scheduler.init_noise_sigma
                kwargs.setdefault("latents", latents_on_device)
            kwargs.setdefault("image", latents_on_device)
            kwargs.setdefault("strength", 1.0)

        result_artifact = super().denoise_latent(
            latent,
            num_inference_steps,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **kwargs,
        )

        return self._make_latent_artifact(
            result_artifact.to_torch(),
            source_shape=result_artifact.source_shape,
            upstream=result_artifact,
            meta={_KIND_META_KEY: "image_latents"},
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> Image:
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        decode_block = self.modular_pipe.blocks.sub_blocks["decode"]
        output_state = self._call_block(decode_block, latents=latents, output_type="pil")
        return self._get_required(output_state, "images", list)[0]

    @override
    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """SDXL ControlNet Inpaint Pipeline always VAE-encodes image (no latent passthrough)."""
        if self._is_controlnet_pipe():
            source_pil = artifact.source_image_pil()
            if source_pil is None:
                raise ValueError(
                    f"{type(self).__name__} ControlNet+Inpaint requires source_image in the InpaintMaskArtifact."
                )
            return {
                "image": source_pil,
                "mask_image": artifact.mask_image,
                "strength": artifact.strength,
            }
        return super()._get_inpaint_kwargs(artifact)

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        if isinstance(media, VideoMedia):
            raise NotImplementedError(f"'{self.pipe.__class__.__name__}' does not support video.")
        generator = generator_state.to_generator()
        encode_block = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        output_state = self._call_block(encode_block, image=media.image, generator=generator)
        result = self._get_required(output_state, "image_latents", torch.Tensor)
        meta = {_KIND_META_KEY: "image_latents"}
        return self._make_latent_artifact(result, source_shape=media.source_shape, meta=meta)
