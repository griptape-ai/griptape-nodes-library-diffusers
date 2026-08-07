import logging
from typing import Any, ClassVar, override

from diffusers import FluxKontextInpaintPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState
from modular_diffusion_nodes_library.latent_pipeline_drivers.flux import FluxLatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


class FluxKontextLatentPipelineDriver(FluxLatentPipelineDriver):
    _inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = FluxKontextInpaintPipeline
    _inpaint_controlnet_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = None

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @override
    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """FluxKontextPipeline expects a PIL image — it handles its own VAE encoding."""
        source_pil = artifact.source_image_pil()
        if source_pil is None:
            raise ValueError(
                "Attempted to run FluxKontext inpainting failed because no source image was provided. "
                "Connect an 'Encode Masked Media' node with a latent input."
            )
        return {
            "image": source_pil,
            "mask_image": artifact.mask_image,
            "strength": artifact.strength,
        }

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
        # Pop the user-provided context/reference image from runtime kwargs.
        context_image = kwargs.pop("image", None)

        if isinstance(latent, InpaintMaskArtifact):
            # Prevent _auto_resize from snapping image to PREFERRED_KONTEXT_RESOLUTIONS, which overwrites
            # height/width and causes _unpack_latents to mismatch the pre-computed latent patch count.
            kwargs["_auto_resize"] = False
            if context_image is not None:
                kwargs["image_reference"] = context_image
        else:
            # max_area (default 1024²) unconditionally rescales height/width making latent_ids
            # diverge from the pre-packed latent's patch count → RoPE crash; H*W makes the rescaling a no-op.
            kwargs["max_area"] = latent.source_shape[-2] * latent.source_shape[-1]
            if context_image is not None:
                kwargs["image"] = context_image

        return super().denoise_latent(
            latent,
            num_inference_steps,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **kwargs,
        )
