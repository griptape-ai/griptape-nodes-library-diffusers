from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import FluxFillPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.flux import FluxLatentPipelineDriver


class FluxFillLatentPipelineDriver(FluxLatentPipelineDriver):
    """Driver for FluxFillPipeline (FLUX.1-Fill-dev).
    This pipeline only supports inpainting.
    """

    _inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = FluxFillPipeline

    @override
    def _get_num_channels_latents(self) -> int:
        return self.pipe.vae.config.latent_channels

    @override
    def _get_inpaint_pipe(self) -> DiffusionPipeline | None:
        return self.pipe

    @override
    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """FluxFillPipeline expects a PIL image — it handles its own VAE encoding."""
        return {
            "image": artifact.source_image_pil(),
            "mask_image": artifact.mask_image,
        }

    @override
    def denoise_latent(
        self, latents: torch.Tensor, latents_source_shape: tuple[int, ...], **kwargs: Any
    ) -> torch.Tensor:
        if kwargs.get("inpaint_mask_artifact") is None:
            raise NotImplementedError(
                "FluxFillPipeline only supports inpainting. "
                "Connect a 'Encode Inpaint Latent' node to the input_latent input."
            )
        # Derive source shape from the actual source image so that
        # height/width passed to the pipeline and used for unpack match
        # the image being inpainted.
        artifact = kwargs["inpaint_mask_artifact"]
        source_pil = artifact.source_image_pil()
        if source_pil is not None:
            w, h = source_pil.size
            latents_source_shape = (1, 3, h, w)
        return super().denoise_latent(latents, latents_source_shape, **kwargs)
