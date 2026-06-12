import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import (  # type: ignore[reportMissingImports]
    ZImageControlNetInpaintPipeline,
    ZImageControlNetModel,
    ZImageControlNetPipeline,
    ZImageInpaintPipeline,
)
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
from huggingface_hub import hf_hub_download  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
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

    model_name = "z-image"
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

        offload_method = detect_offload_method(pipe)
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
    def create_noise_latent(self, latents_source_shape: tuple[int, ...], seed: int) -> torch.Tensor:  # noqa: ARG002
        """Create a raw noise latent via modular pipeline block."""
        prepare_latents = ZImagePrepareLatentsStep()
        output_state = self._call_block(
            prepare_latents,
            height=latents_source_shape[-2],
            width=latents_source_shape[-1],
            batch_size=1,
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(seed),
        )
        latents = output_state.get("latents")
        return latents

    @override
    def add_noise_to_latent(
        self,
        latents: torch.Tensor,
        latents_source_shape: tuple[int, ...],
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> torch.Tensor:
        """Add noise to image latents via modular pipeline blocks."""
        device, dtype = self._get_device_and_type()
        noise = self.create_noise_latent(latents_source_shape, seed)
        output_state = self._call_block(
            _ZImageAddNoiseStep(),
            latents=noise.to(device=device, dtype=dtype),
            image_latents=latents.to(device=device, dtype=dtype),
            num_inference_steps=num_inference_steps,
            strength=strength,
        )
        result = output_state.get("latents")
        return result

    @override
    def decode_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> Image:
        decode_block = self.modular_pipe.blocks.sub_blocks["decode"]
        output_state = self._call_block(decode_block, latents=latents, output_type="pil")
        images = output_state.get("images")
        return images[0]

    @override
    def encode_image(self, image: Image | torch.Tensor) -> torch.Tensor:
        encode_block = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        output_state = self._call_block(encode_block, image=image)
        result = output_state.get("image_latents")
        return result

    @override
    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """Z-Image inpaint always VAE-encodes image internally."""
        source_pil = artifact.source_image_pil()
        if source_pil is None:
            raise ValueError(f"{type(self).__name__} inpainting requires source_image in the InpaintMaskArtifact.")

        kwargs: dict[str, Any] = {
            "image": source_pil,
            "mask_image": artifact.mask_image,
        }
        if not self._is_controlnet_pipe():
            kwargs["strength"] = artifact.strength
        return kwargs
