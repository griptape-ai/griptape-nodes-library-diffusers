import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import (
    QwenImageControlNetInpaintPipeline,  # type: ignore[reportMissingImports]
    QwenImageControlNetModel,  # type: ignore[reportMissingImports]
    QwenImageControlNetPipeline,  # type: ignore[reportMissingImports]
    QwenImageInpaintPipeline,  # type: ignore[reportMissingImports]
    QwenImageMultiControlNetModel,  # type: ignore[reportMissingImports]
)
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.qwenimage.before_denoise import (  # type: ignore[reportMissingImports]
    QwenImagePrepareLatentsStep,
    QwenImagePrepareLatentsWithStrengthStep,
    QwenImageSetTimestepsWithStrengthStep,
)
from diffusers.modular_pipelines.qwenimage.modular_blocks_qwenimage import (
    QwenImageAutoBlocks,  # type: ignore[reportMissingImports]
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.param_components.huggingface.huggingface_model_parameter import HuggingFaceModelParameter
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
from modular_diffusion_nodes_library.parameters.controlnet_node_parameter_types import (
    QwenImageControlNetNodesParameterType,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class _QwenImageNoiseWithStrengthSequence(SequentialPipelineBlocks):
    """Add noise to the image latents according to the strength value."""

    block_classes = [
        QwenImagePrepareLatentsStep,
        QwenImageSetTimestepsWithStrengthStep,
        QwenImagePrepareLatentsWithStrengthStep,
    ]
    block_names = ["prepare_latents", "set_timesteps", "prepare_img2img_latents"]


class QwenLatentPipelineDriver(LatentPipelineDriver):
    _inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = QwenImageInpaintPipeline
    _inpaint_controlnet_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = QwenImageControlNetInpaintPipeline

    def __init__(self, pipe: DiffusionPipeline) -> None:
        super().__init__(pipe)

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return True

    @classmethod
    @override
    def control_pipe_from_standard(
        cls, pipe: ModularPipeline | DiffusionPipeline, control_net_model_lists: list[str] | str | None
    ):

        if not control_net_model_lists:
            return pipe

        if not isinstance(control_net_model_lists, list):
            control_net_model_lists = [control_net_model_lists]

        controlnet_torch_dtype = cls._get_torch_type(pipe)
        from_pretrained_kwargs: dict[str, Any] = {}
        if controlnet_torch_dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = controlnet_torch_dtype

        control_net_models = [
            QwenImageControlNetModel.from_pretrained(model, **from_pretrained_kwargs)
            for model in control_net_model_lists
        ]

        if len(control_net_models) == 1:
            controlnet = control_net_models[0]
        else:
            controlnet = QwenImageMultiControlNetModel(control_net_models)

        return QwenImageControlNetPipeline(controlnet=controlnet, **pipe.components)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return QwenImageAutoBlocks().init_pipeline()

    def _get_unpacked_image_latents(self, output_state: dict[str, Any], height: int, width: int) -> torch.Tensor:
        latents = output_state.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise ValueError(f"Expected latents to be a torch.Tensor but got {type(latents)}")
        return self.prepare_output_latent(latents, (1, -1, height, width))

    @override
    def prepare_input_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> torch.Tensor:
        """Return latents ready to be passed into the pipeline, which may involve packing or other preprocessing."""
        packed_latents = self.modular_pipe.pachifier.pack_latents(latents)
        return packed_latents

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Return latents ready to be process further e.g. with mask, which may involve unpacking or other postprocessing."""
        unpack_latents = self.modular_pipe.pachifier.unpack_latents(
            latents_from_pipe, latents_source_shape[-2], latents_source_shape[-1], self.modular_pipe.vae_scale_factor
        )
        image_latents = unpack_latents.squeeze(2)  # [B,z,1,H',W'] → [B,z,H',W']
        return image_latents

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], seed: int) -> LatentArtifact:
        _, dtype = self._get_device_and_type()
        prepare_latents_pipeline = QwenImagePrepareLatentsStep()
        height, width = source_shape[-2], source_shape[-1]

        output_state = self._call_block(
            prepare_latents_pipeline,
            height=height,
            width=width,
            batch_size=1,
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(seed),
            dtype=dtype,
        )

        output_latents = self._get_unpacked_image_latents(output_state, height, width)
        return self._make_latent_artifact(output_latents, source_shape=source_shape)

    @override
    def decode_latent(self, latent: LatentArtifact) -> Image:
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        latents = latents.unsqueeze(2)

        decode_pipeline = self.modular_pipe.blocks.sub_blocks["decode"]
        output_state = self._call_block(decode_pipeline, latents=latents, output_type="pil")

        output_image = self._get_required(output_state, "images", list)
        return output_image[0]

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        noise_with_strength_pipeline = _QwenImageNoiseWithStrengthSequence()
        device, dtype = self._get_device_and_type()
        source_shape = latent.source_shape
        latents = latent.to_torch(device=device, dtype=dtype)
        height, width = source_shape[-2], source_shape[-1]

        noisy_state = self._call_block(
            noise_with_strength_pipeline,
            height=height,
            width=width,
            batch_size=latents.shape[0],
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(seed),
            dtype=dtype,
            latents=None,
            image_latents=self.modular_pipe.pachifier.pack_latents(latents),
            num_inference_steps=num_inference_steps,
            strength=strength,
        )

        output_latents = self._get_unpacked_image_latents(noisy_state, height, width)
        return self._make_latent_artifact(output_latents, source_shape=source_shape, upstream=latent)

    @override
    def encode_image(self, image: Image | torch.Tensor, source_shape: tuple[int, ...]) -> LatentArtifact:
        encode_pipeline = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        if isinstance(image, torch.Tensor):
            height = image.shape[-2]
            width = image.shape[-1]
        else:
            height = image.height
            width = image.width
        output_state = self._call_block(encode_pipeline, image=image, height=height, width=width)
        latents = self._get_required(output_state, "image_latents", torch.Tensor)
        # [B,z,1,H',W'] → [B,z,H',W'] - remove temporal dimension (the same VAE is shared between video and image pipelines)
        return self._make_latent_artifact(latents.squeeze(2), source_shape=source_shape)

    @override
    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """Map an InpaintMaskArtifact to pipeline call kwargs.

        For ``QwenImageControlNetInpaintPipeline`` the call signature is different:
        the source image enters via the ControlNet's ``control_image`` input, the
        mask is passed as ``control_mask``, and the pipeline does not accept
        ``image`` / ``mask_image`` / ``masked_image_latents`` / ``strength``.
        """
        if self._is_controlnet_pipe():
            source_pil = artifact.source_image_pil()
            if source_pil is None:
                raise ValueError(
                    f"{type(self).__name__} ControlNet+Inpaint requires source_image in the InpaintMaskArtifact."
                )
            return {"control_image": source_pil, "control_mask": artifact.mask_image}
        return super()._get_inpaint_kwargs(artifact)

    @classmethod
    @override
    def validate_run_configuration(
        cls,
        pipeline_artifact: DiffusionPipelineArtifact,
        input_latent: LatentArtifact | None,
    ) -> list[Exception] | None:
        errors = super().validate_run_configuration(pipeline_artifact, input_latent)
        if errors is not None:
            return errors

        is_inpaint = isinstance(input_latent, InpaintMaskArtifact)
        is_controlnet = isinstance(pipeline_artifact, ControlNetDiffusionPipelineArtifact)
        if not (is_inpaint and is_controlnet):
            return None

        # Qwen's combined ControlNet+Inpaint pipeline only works with the dedicated inpainting ControlNet checkpoint.
        expected_repo_id = QwenImageControlNetNodesParameterType.INPAINTING_REPO_ID
        incompatible_repo_ids: list[str] = []
        for model in pipeline_artifact.metadata["controlnet_models"]:
            repo_id, _ = HuggingFaceModelParameter._key_to_repo_revision(model)  # noqa: SLF001
            if repo_id != expected_repo_id:
                incompatible_repo_ids.append(repo_id)
        if incompatible_repo_ids:
            return [
                ValueError(
                    f"Qwen ControlNet + Inpaint requires '{expected_repo_id}', "
                    f"but got incompatible model(s): {incompatible_repo_ids}."
                )
            ]
        return None
