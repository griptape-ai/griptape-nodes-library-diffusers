import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import (  # type: ignore[reportMissingImports]
    FluxControlNetInpaintPipeline,
    FluxControlNetModel,
    FluxControlNetPipeline,
    FluxInpaintPipeline,
)
from diffusers.models import FluxTransformer2DModel  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux.before_denoise import (  # type: ignore[reportMissingImports]
    FluxImg2ImgPrepareLatentsStep,
    FluxImg2ImgSetTimestepsStep,
    FluxPrepareLatentsStep,
)
from diffusers.modular_pipelines.flux.decoders import _unpack_latents  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux.modular_blocks_flux import FluxAutoBlocks  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux.modular_pipeline import FluxModularPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec  # type: ignore[reportMissingImports]
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


# TO DO: raise issue with diffusers library.
class _RegisterTransformerStep(ModularPipelineBlocks):
    """No-op block that registers the transformer component so downstream blocks can access it.
    FluxImg2ImgSetTimestepsStep needs the transformer but isnt registered in expected_components, so we register it here."""

    model_name = "flux"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", FluxTransformer2DModel)]

    @property
    def inputs(self) -> list:
        return []

    @property
    def intermediate_outputs(self) -> list:
        return []

    @torch.no_grad()
    def __call__(
        self, components: FluxModularPipeline, state: PipelineState
    ) -> tuple[FluxModularPipeline, PipelineState]:
        return components, state


class FluxLatentPipelineDriver(LatentPipelineDriver):
    _inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = FluxInpaintPipeline
    _inpaint_controlnet_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = FluxControlNetInpaintPipeline

    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return FluxAutoBlocks().init_pipeline()

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return True

    @classmethod
    @override
    def control_pipe_from_standard(
        cls, pipe: ModularPipeline | DiffusionPipeline, control_net_model_lists: list[str] | str | None
    ):
        """Given a standard pipeline, return a version of the pipeline that can be used for control net generation."""
        # Ensure pipeline is a FluxPipeline.
        if pipe.__class__.__name__ != "FluxPipeline":
            raise ValueError(f"Expected a FluxPipeline, but got {pipe.__class__.__name__}")

        configured_pipe = pipe
        if control_net_model_lists:
            if not isinstance(control_net_model_lists, list):
                control_net_model_lists = [control_net_model_lists]

            controlnet_torch_dtype = cls._get_torch_type(pipe)
            from_pretrained_kwargs = {}
            if controlnet_torch_dtype is not None:
                from_pretrained_kwargs["torch_dtype"] = controlnet_torch_dtype

            control_net_models = []
            for control_net in control_net_model_lists:
                model = FluxControlNetModel.from_pretrained(control_net, **from_pretrained_kwargs)
                control_net_models.append(model)

            if len(control_net_models) == 1:
                controlnet = control_net_models[0]
            else:
                controlnet = control_net_models

            configured_pipe = FluxControlNetPipeline(controlnet=controlnet, **pipe.components)
        return configured_pipe

    def _get_num_channels_latents(self) -> int:
        if hasattr(self.pipe, "transformer") and self.pipe.transformer is not None:
            if hasattr(self.pipe.transformer, "config") and hasattr(self.pipe.transformer.config, "in_channels"):
                return self.pipe.transformer.config.in_channels // 4

        raise RuntimeError(
            "Attempted to infer num_channels_latents. Failed with pipeline="
            f"{self.pipe.__class__.__name__} because the pipeline does not expose transformer.config.in_channels."
        )

    @override
    def prepare_input_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> torch.Tensor:
        """Return latents ready to be passed into the pipeline, which may involve packing or other preprocessing."""
        packed_latents = self.pack_latents(latents, height=latents_source_shape[-2], width=latents_source_shape[-1])
        return packed_latents

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Return latents ready to be process further e.g. with mask, which may involve unpacking or other postprocessing."""
        unpack_latents = self.unpack_latents(
            latents_from_pipe, height=latents_source_shape[-2], width=latents_source_shape[-1]
        )
        return unpack_latents

    # Pack latents from [B, C, H, W] to [B, seq, C]
    def pack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_channels_latents = self._get_num_channels_latents()
        return FluxPipeline._pack_latents(latents, 1, num_channels_latents, latents.shape[-2], latents.shape[-1])

    def unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return _unpack_latents(latents, height, width, self.pipe.vae_scale_factor)

    @override
    def create_noise_latent(self, latents_source_shape: tuple[int, ...], seed: int) -> torch.Tensor:  # noqa: ARG002
        """Return pure noise latent with shape [B, C, H, W]."""
        height, width = latents_source_shape[-2], latents_source_shape[-1]
        device, dtype = self._get_device_and_type()
        generator = torch.Generator().manual_seed(seed)
        prepare_latents = FluxPrepareLatentsStep()
        output_state = self._call_block(
            prepare_latents,
            height=height,
            width=width,
            num_images_per_prompt=1,
            generator=generator,
            batch_size=1,
            dtype=dtype,
        )
        latents = output_state.get("latents")
        latents = latents.to(device)
        latents = self.unpack_latents(latents, height, width)
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
        """Return noised latent with shape [B, C, H, W]."""
        device, dtype = self._get_device_and_type()
        packed_image_latents = self.pack_latents(latents, latents.shape[-2], latents.shape[-1])
        packed_image_latents = packed_image_latents.to(device=device, dtype=dtype)
        height, width = latents_source_shape[-2], latents_source_shape[-1]

        noise_with_strength_pipeline = SequentialPipelineBlocks.from_blocks_dict(
            {
                "register_transformer": _RegisterTransformerStep,
                "set_timesteps": FluxImg2ImgSetTimestepsStep,
                "prepare_latents": FluxPrepareLatentsStep,
                "img2img_prepare": FluxImg2ImgPrepareLatentsStep,
            }
        )

        output_state = self._call_block(
            noise_with_strength_pipeline,
            num_inference_steps=num_inference_steps,
            strength=strength,
            height=height,
            width=width,
            batch_size=1,
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(seed),
            dtype=dtype,
            image_latents=packed_image_latents,
        )

        noisy_latents = output_state.get("latents")
        return self.unpack_latents(noisy_latents, height, width)

    @override
    def decode_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> Image:
        device, dtype = self._get_device_and_type()
        height, width = latents_source_shape[-2], latents_source_shape[-1]
        latents = latents.to(device=device, dtype=dtype)

        packed = self.pack_latents(latents, latents.shape[-2], latents.shape[-1])

        decode_pipeline = self.modular_pipe.blocks.sub_blocks["decode"]
        output_state = self._call_block(decode_pipeline, latents=packed, output_type="pil", width=width, height=height)
        return output_state.get("images")[0]

    @override
    def encode_image(self, image: Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            height = image.shape[-2]
            width = image.shape[-1]
        else:
            height = image.height
            width = image.width
        encode_pipeline = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        output_state = self._call_block(encode_pipeline, image=image, height=height, width=width)
        latents = output_state.get("image_latents")
        return latents

    @override
    def denoise_latent(
        self,
        latents: torch.Tensor,
        latents_source_shape: tuple[int, ...],
        num_inference_steps: int,
        seed: int = 0,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        # FluxControlNetInpaintPipeline does not accept negative_prompt.
        inpaint_mask_artifact = kwargs.get("inpaint_mask_artifact")
        if inpaint_mask_artifact is not None and self._is_controlnet_pipe():
            kwargs.pop("negative_prompt", None)
            kwargs.pop("true_cfg_scale", None)
        return super().denoise_latent(
            latents,
            latents_source_shape,
            num_inference_steps,
            seed=seed,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **kwargs,
        )

    @override
    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """FluxControlNetInpaintPipeline always VAE-encodes image (no latent passthrough)."""
        if not self._is_controlnet_pipe():
            return super()._get_inpaint_kwargs(artifact)
        source_pil = artifact.source_image_pil()
        if source_pil is None:
            raise ValueError("Flux ControlNet inpainting requires the source PIL image in the InpaintMaskArtifact.")
        return {
            "image": source_pil,
            "mask_image": artifact.mask_image,
            "strength": artifact.strength,
        }
