from typing import Any, ClassVar, override

from diffusers import FluxFillPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
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
        self,
        latent: LatentArtifact | InpaintMaskArtifact,
        **kwargs: Any,
    ) -> LatentArtifact:
        if not isinstance(latent, InpaintMaskArtifact):
            raise NotImplementedError(
                "FluxFillPipeline only supports inpainting. "
                "Connect a 'Encode Inpaint Latent' node to the input_latent input."
            )
        # Override height/width using the actual source image dims so that
        # the pipeline and unpack/decode use the inpainted image size rather than
        # whatever source_shape the artifact carries (which may not match).
        source_pil = latent.source_image_pil()
        if source_pil is not None:
            w, h = source_pil.size
            kwargs.setdefault("height", h)
            kwargs.setdefault("width", w)
        return super().denoise_latent(latent, **kwargs)
