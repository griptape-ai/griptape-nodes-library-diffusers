import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.z_image.before_denoise import (  # type: ignore[reportMissingImports]
    ZImagePrepareLatentsStep,
    ZImagePrepareLatentswithImageStep,
    ZImageSetTimestepsStep,
    ZImageSetTimestepsWithStrengthStep,
)
from diffusers.modular_pipelines.z_image.modular_blocks_z_image import (
    ZImageAutoBlocks,  # type: ignore[reportMissingImports]
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.z_image.pipeline_z_image_controlnet import (  # type: ignore[reportMissingImports]
    ZImageControlNetPipeline,
)
from diffusers.pipelines.z_image.pipeline_z_image_controlnet_inpaint import (  # type: ignore[reportMissingImports]
    ZImageControlNetInpaintPipeline,
)
from diffusers.pipelines.z_image.pipeline_z_image_inpaint import (  # type: ignore[reportMissingImports]
    ZImageInpaintPipeline,
)
from huggingface_hub import hf_hub_download  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    GeneratorState,
    ImageMedia,
    VideoMedia,
)
from modular_diffusion_nodes_library.utils.pipeline_utils import detect_offload_method

logger = logging.getLogger("modular_diffusers_nodes_library")

Z_IMAGE_CONTROLNET_REPO_TO_FILENAME = {
    "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
    "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1": "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
}


class _ZImageAddNoiseStep(SequentialPipelineBlocks):
    """Add noise to image latents for img2img workflows.

    Composes SetTimesteps -> SetTimestepsWithStrength -> PrepareLatentsWithImage
    so the scheduler computes the correct timestep from strength and then
    calls scheduler.scale_noise internally.
    """

    model_name = "z-image"  # type: ignore[reportIncompatibleMethodOverride]
    block_classes = [
        ZImageSetTimestepsStep,
        ZImageSetTimestepsWithStrengthStep,
        ZImagePrepareLatentswithImageStep,
    ]
    block_names = ["set_timesteps", "set_timesteps_with_strength", "prepare_latents_with_image"]


class ZImageLatentPipelineDriver(LatentPipelineDriver):
    """
    Driver for ZImagePipeline.
    """

    _inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = ZImageInpaintPipeline
    _inpaint_controlnet_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = ZImageControlNetInpaintPipeline

    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return ZImageAutoBlocks().init_pipeline()

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        if not control_net_model_lists:
            return True

        if not isinstance(control_net_model_lists, list):
            control_net_model_lists = [control_net_model_lists]

        if len(control_net_model_lists) > 1 or control_net_model_lists[0] not in Z_IMAGE_CONTROLNET_REPO_TO_FILENAME:
            return False
        return True

    @classmethod
    @override
    def control_pipe_from_standard(
        cls, pipe: ModularPipeline | DiffusionPipeline, control_net_model_lists: list[str] | str | None
    ):
        if not control_net_model_lists:
            return pipe
        offload_method = detect_offload_method(pipe)  # type: ignore[reportArgumentType]
        if offload_method is not None:
            raise RuntimeError(
                f"Failed to build Z-Image ControlNet pipeline. "
                f"Base pipeline has '{offload_method}' CPU offload enabled. "
                "Z-Image ControlNet does not currently support reusing a CPU-offloaded base; "
                "rebuild the base pipeline with cpu_offload_strategy='None' before adding ControlNet."
            )

        if not isinstance(control_net_model_lists, list):
            control_net_model_lists = [control_net_model_lists]

        if len(control_net_model_lists) > 1:
            raise ValueError("Z-Image ControlNet pipeline only supports a single control net model.")

        first_control_net = control_net_model_lists[0]
        controlnet_torch_dtype = cls._get_torch_type(pipe)
        from_single_file_kwargs: dict[str, Any] = {}
        if controlnet_torch_dtype is not None:
            from_single_file_kwargs["torch_dtype"] = controlnet_torch_dtype

        from_single_file_kwargs["low_cpu_mem_usage"] = False

        if first_control_net in Z_IMAGE_CONTROLNET_REPO_TO_FILENAME:
            controlnet_file_path = hf_hub_download(
                repo_id=first_control_net,
                filename=Z_IMAGE_CONTROLNET_REPO_TO_FILENAME[first_control_net],
            )
        else:
            controlnet_file_path = first_control_net

        control_net_model = ZImageControlNetModel.from_single_file(
            controlnet_file_path,
            **from_single_file_kwargs,
        )

        return ZImageControlNetPipeline(
            controlnet=control_net_model,
            **pipe.components,
        )

    # ------------------------------------------------------------------
    # Modular blocks for encode / decode / noise latent
    # ------------------------------------------------------------------

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        """Create a raw noise latent via modular pipeline block."""
        prepare_latents = ZImagePrepareLatentsStep()
        generator = generator_state.to_generator()
        output_state = self._call_block(
            prepare_latents,
            height=source_shape[-2],
            width=source_shape[-1],
            batch_size=1,
            num_images_per_prompt=1,
            generator=generator,
        )
        latents = self._get_required(output_state, "latents", torch.Tensor)
        return self._make_latent_artifact(
            latents,
            source_shape=source_shape,
            meta=GeneratorState.from_generator(generator).as_meta(),
        )

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        """Add noise to image latents via modular pipeline blocks."""
        device, dtype = self._get_device_and_type()
        source_shape = latent.source_shape
        latents = latent.to_torch(device=device, dtype=dtype)
        noise_artifact = self.create_noise_latent(source_shape, generator_state)
        noise = noise_artifact.to_torch(device=device, dtype=dtype)
        noise_generator_state = GeneratorState.from_artifact(noise_artifact) or generator_state

        output_state = self._call_block(
            _ZImageAddNoiseStep(),
            latents=noise,
            image_latents=latents,
            num_inference_steps=num_inference_steps,
            strength=strength,
        )
        result = self._get_required(output_state, "latents", torch.Tensor)
        return self._make_latent_artifact(
            result,
            source_shape=source_shape,
            upstream=latent,
            meta=noise_generator_state.as_meta(),
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> Image:
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        decode_block = self.modular_pipe.blocks.sub_blocks["decode"]
        output_state = self._call_block(decode_block, latents=latents, output_type="pil")
        return self._get_required(output_state, "images", list)[0]

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        if isinstance(media, VideoMedia):
            raise NotImplementedError(f"'{self.pipe.__class__.__name__}' does not support video.")
        generator = generator_state.to_generator()
        encode_block = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        output_state = self._call_block(encode_block, image=media.image, generator=generator)
        result = self._get_required(output_state, "image_latents", torch.Tensor)
        return self._make_latent_artifact(result, source_shape=media.source_shape)

    @override
    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """Z-Image inpaint always VAE-encodes image internally."""
        source_pil = artifact.source_image_pil()
        if source_pil is None:
            raise ValueError(f"{self.driver_namespace} inpainting requires source_image in the InpaintMaskArtifact.")

        kwargs: dict[str, Any] = {
            "image": source_pil,
            "mask_image": artifact.mask_image,
        }
        if not self._is_controlnet_pipe():
            kwargs["strength"] = artifact.strength
        return kwargs
