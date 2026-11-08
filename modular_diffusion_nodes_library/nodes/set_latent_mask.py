from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.masked_latent_artifact import MaskedLatentArtifact
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.utils.image_utils import (
    apply_mask_transformations,
    extract_channel_from_image,
    load_image_from_url_artifact,
)
from modular_diffusion_nodes_library.utils.pillow_utils import image_artifact_to_pil

if TYPE_CHECKING:
    from PIL import Image as PILImage  # type: ignore[reportMissingImports]

logger = logging.getLogger("modular_diffusers_nodes_library")


class SetLatentMaskNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Bundle an encoded latent with a mask image."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        latent_param = Parameter(
            name="input_latent",
            input_types=["LatentArtifact"],
            type="LatentArtifact",
            tooltip="Image or video latent e.g. produced by Encode Media Latent.",
            allowed_modes={ParameterMode.INPUT},
            hide_property=True,
        )
        latent_param.set_badge(
            variant="help",
            title="Use an encoded latent",
            message="Connect the `latent_tensor` output of ***Encode Media Latent***.",
        )
        self.add_parameter(latent_param)
        mask_param = Parameter(
            name="mask_image",
            input_types=["ImageUrlArtifact"],
            type="ImageUrlArtifact",
            tooltip="Mask image. White marks the selected region, black marks the rest, and grey is partial.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            hide_property=True,
        )
        mask_param.set_badge(
            variant="help",
            title="Mask convention",
            message=(
                "- ***White*** — fully selected\n"
                "- ***Black*** — not selected\n"
                "- ***Grey*** — partially selected\n\n"
                "The mask is resized to match the input latent when it is converted to a tensor."
            ),
        )
        self.add_parameter(mask_param)

        with ParameterGroup(name="mask_options") as mask_options:
            channel_param = ParameterString(
                name="channel",
                tooltip="Channel to extract from the mask image as blend weight.",
                default_value="red",
                ui_options={"expander": True, "edit_mask": True, "edit_mask_paint_mask": True},
            )
            channel_param.add_trait(Options(choices=["red", "green", "blue", "alpha"]))

            ParameterBool(name="invert_mask", default_value=False)
            ParameterFloat(name="grow_shrink", default_value=0, slider=True, min_val=-100, max_val=100)
            ParameterFloat(name="blur_mask", default_value=0, slider=True, min_val=0, max_val=100)

        self.add_node_element(mask_options)

        latents_param = Parameter(
            name="output_latent",
            output_type="MaskedLatentArtifact",
            tooltip="The input latent bundled with the processed mask.",
            allowed_modes={ParameterMode.OUTPUT},
            serializable=False,
        )
        self.add_parameter(latents_param)
        self._create_status_parameters()

        placeholder = MaskedLatentArtifact()
        self.set_parameter_value("output_latent", placeholder, initial_setup=True)
        self.parameter_output_values["output_latent"] = placeholder

    def validate_before_node_run(self) -> list[Exception] | None:
        return None

    def validate_in_execution_environment(self) -> list[Exception] | None:
        errors: list[Exception] = []
        try:
            self._get_input_latent()
        except ValueError as err:
            errors.append(err)
        try:
            self._get_mask_artifact()
        except ValueError as err:
            errors.append(err)
        if errors:
            return errors
        return None

    def process(self) -> AsyncResult:
        self._clear_execution_status()
        yield lambda: self._run_with_status(
            self._encode,
            success_msg="Set Latent Mask completed successfully.",
            failure_log="Set Latent Mask failed",
            logger=logger,
        )

    def _encode(self) -> None:
        source_latent = self._get_input_latent()
        mask_pil = self._get_mask()
        artifact = MaskedLatentArtifact(
            mask_image=mask_pil,
            source_latent=source_latent,
        )
        self.set_parameter_value("output_latent", artifact)
        self.parameter_output_values["output_latent"] = artifact

    def _get_input_latent(self) -> LatentArtifact:
        latent = self.get_parameter_value("input_latent")
        if latent is None:
            msg = f"Attempted to read 'input_latent' on '{self.name}'. Failed because no latent is connected."
            raise ValueError(msg)
        if not isinstance(latent, LatentArtifact):
            msg = (
                f"Attempted to read 'input_latent' on '{self.name}'. Failed because it is a "
                f"{type(latent).__name__}, not a LatentArtifact from Encode Media Latent."
            )
            raise ValueError(msg)

        latent_tensor_shape = tuple(latent.shape)
        if len(latent_tensor_shape) not in (4, 5) or len(latent.source_shape) != len(latent_tensor_shape):
            msg = (
                f"Attempted to read 'input_latent' on '{self.name}'. Failed with latent shape {latent_tensor_shape} "
                f"and source_shape {latent.source_shape} because it must be a 4-D image or 5-D video latent "
                "whose source_shape has the same number of dimensions."
            )
            raise ValueError(msg)

        return latent

    def _get_mask_artifact(self) -> ImageUrlArtifact:
        artifact = self.get_parameter_value("mask_image")
        if artifact is None:
            msg = f"Attempted to read 'mask_image' on '{self.name}'. Failed because no mask image is connected."
            raise ValueError(msg)
        if not isinstance(artifact, ImageUrlArtifact):
            msg = (
                f"Attempted to read 'mask_image' on '{self.name}'. Failed because it is a "
                f"{type(artifact).__name__}, not an ImageUrlArtifact."
            )
            raise ValueError(msg)

        return artifact

    def _get_mask(self) -> PILImage.Image:
        image_artifact = load_image_from_url_artifact(self._get_mask_artifact())
        mask_image = image_artifact_to_pil(image_artifact)
        mask_image = extract_channel_from_image(mask_image, self.get_parameter_value("channel"))
        return apply_mask_transformations(
            mask_image,
            invert=self.get_parameter_value("invert_mask"),
            grow_shrink=self.get_parameter_value("grow_shrink"),
            blur_radius=self.get_parameter_value("blur_mask"),
        )
