"""VAE Mask Encode node.

Takes an image and a binary mask, encodes both the source image and the
masked image through the pipeline's VAE, and outputs an
``InpaintMaskArtifact``.
"""

import logging
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from PIL import Image as PILImage

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import create_driver, get_driver_class
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState, ImageMedia, MaskMedia
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.pipeline_parameters import ModularDiffusionPipelineParameters
from modular_diffusion_nodes_library.utils.image_utils import load_image_from_url_artifact
from modular_diffusion_nodes_library.utils.pillow_utils import image_artifact_to_pil

logger = logging.getLogger("modular_diffusers_nodes_library")


class VaeMaskEncodeNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Encode a masked image into an InpaintMaskArtifact (mask + masked_latent)."""

    def __init__(self, **kwargs) -> None:
        self._initializing = True
        super().__init__(**kwargs)

        self.pipe_params = ModularDiffusionPipelineParameters(self)
        self.pipe_params.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Source image to encode.",
                allowed_modes={ParameterMode.INPUT},
                user_defined=True,
            )
        )
        mask_param = Parameter(
            name="mask",
            input_types=["ImageArtifact", "ImageUrlArtifact"],
            type="ImageArtifact",
            tooltip="Binary mask (white = inpaint region).",
            allowed_modes={ParameterMode.INPUT},
            user_defined=True,
        )
        mask_param.set_badge(
            variant="help",
            title="Mask convention",
            message=(
                "- ***White*** — region to regenerate (inpaint area)\n"
                "- ***Black*** — region to preserve\n"
                "- ***Grey*** — blends proportionally between the two\n\n"
                "The mask is automatically resized to match the source image."
            ),
        )
        self.add_parameter(mask_param)
        self.add_parameter(
            Parameter(
                name="strength",
                input_types=["float"],
                type="float",
                default_value=0.95,
                tooltip="Inpaint denoising strength (0.0–1.0). 1.0 fully replaces the masked region; lower values blend with the original.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                user_defined=True,
            )
        )
        latents_param = Parameter(
            name="latents",
            output_type="InpaintMaskArtifact",
            tooltip="Inpaint artifact bundling the mask plus source and masked-image latents. Connect to a Generate Latents input_latent for inpainting.",
            allowed_modes={ParameterMode.OUTPUT},
            serializable=False,
        )
        latents_param.set_badge(
            variant="help",
            title="Where to connect this",
            message=(
                "Connect to the `input_latent` of a ***Generate Media Latents*** node — not to a VAE Decode node.\n\n"
                "The Generate node detects the `InpaintMaskArtifact` type and automatically switches to inpainting mode, "
                "regenerating only the masked region while preserving the rest of the image."
            ),
        )
        self.add_parameter(latents_param)
        self._initializing = False
        self._create_status_parameters()

        # Seed a typed placeholder so downstream nodes can react to the InpaintMaskArtifact type.
        placeholder = InpaintMaskArtifact(mask_image=PILImage.new("L", (1, 1), 0))
        self.set_parameter_value("latents", placeholder, initial_setup=True)
        self.parameter_output_values["latents"] = placeholder

    def add_parameter(self, param: Parameter) -> None:
        """Only allow parameters during init — block runtime parameters like prompt."""
        if not self._initializing:
            return
        if not self.does_name_exist(param.name):
            super().add_parameter(param)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []
        if not self.get_parameter_value("pipeline"):
            errors.append(ValueError("Missing required 'pipeline' input."))
        else:
            pipeline_value = self.get_parameter_value("pipeline")
            pipeline_class_name = getattr(pipeline_value, "pipeline_name", None)
            if pipeline_class_name is not None:
                driver_cls = get_driver_class(pipeline_class_name)
                if driver_cls is None or driver_cls._inpaint_pipeline_class is None:
                    errors.append(ValueError(f"Pipeline '{pipeline_class_name}' does not support inpainting."))
        if self.get_parameter_value("image") is None:
            errors.append(ValueError("Missing required 'image' input."))
        if self.get_parameter_value("mask") is None:
            errors.append(ValueError("Missing required 'mask' input."))
        return errors or None

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        parameter = self.get_parameter_by_name(param_name)
        if parameter is None:
            return
        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )
        self.pipe_params.after_value_set(parameter, value)
        self.pipe_params.runtime_parameters.after_value_set(parameter, value)

    def process(self) -> AsyncResult:
        self._clear_execution_status()
        yield lambda: self._run_with_status(
            self._encode, success_msg="Mask encoded successfully.", failure_log="VAE mask encode failed", logger=logger
        )

    def _encode(self) -> None:
        pipe = self.pipe_params.get_pipeline()
        pipeline_class = self.pipe_params.get_pipeline_class()
        driver = create_driver(pipe, pipeline_class)

        image_pil = self._get_image()
        mask_pil = self._get_mask()

        # Ensure mask is same size as image
        if mask_pil.size != image_pil.size:
            mask_pil = mask_pil.resize(image_pil.size, PILImage.NEAREST)

        source_shape = (1, 3, image_pil.height, image_pil.width)
        image = ImageMedia(image=image_pil, source_shape=source_shape)
        mask = MaskMedia(mask=mask_pil, source_shape=source_shape)
        generator_state = GeneratorState.from_seed(42)
        source_encode = driver.encode_media(image, generator_state)
        masked_encode = driver.encode_masked_image(image, mask, generator_state)
        strength = float(self.get_parameter_value("strength") or 1.0)

        artifact = InpaintMaskArtifact(
            mask_image=mask_pil,
            source_image=self.get_parameter_value("image"),
            source_latent=source_encode.to_torch(),
            masked_latent=masked_encode.to_torch(),
            source_shape=source_shape,
            strength=strength,
            meta=dict(source_encode.meta or {}),
        )
        self.set_parameter_value("latents", artifact)
        self.parameter_output_values["latents"] = artifact

    def _get_image(self) -> PILImage.Image:
        artifact = self.get_parameter_value("image")
        if isinstance(artifact, ImageUrlArtifact):
            artifact = load_image_from_url_artifact(artifact)
        return image_artifact_to_pil(artifact).convert("RGB")

    def _get_mask(self) -> PILImage.Image:
        artifact = self.get_parameter_value("mask")
        if isinstance(artifact, ImageUrlArtifact):
            artifact = load_image_from_url_artifact(artifact)
        return image_artifact_to_pil(artifact).convert("L")
