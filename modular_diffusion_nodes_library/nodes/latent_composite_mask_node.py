import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from PIL import Image

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.utils.image_utils import (
    apply_mask_transformations,
    extract_channel_from_image,
    load_image_from_url_artifact,
)
from modular_diffusion_nodes_library.utils.pillow_utils import image_artifact_to_pil

logger = logging.getLogger("modular_diffusers_nodes_library")

MAX_IMAGE_DIMENSION = 2000
MAX_WIDTH = MAX_IMAGE_DIMENSION
MAX_HEIGHT = MAX_IMAGE_DIMENSION

LATENT_SCALE_FACTOR = 8  # Latent space is 1/8 of pixel space


class LatentCompositeMaskNode(ControlNode):
    """Composite a source latent onto a destination latent with an optional mask blend.

    - Place *source_latent* onto *destination_latent* at pixel offset (x_offset, y_offset).
    - If a *mask_image* is supplied, use the chosen channel as a blend weight
      (white = source, black = destination).
    - With *resize_source* enabled the source is rescaled to the destination latent size
      before compositing.
    """

    def __init__(self, **kwargs) -> None:
        self._initializing = True
        super().__init__(**kwargs)

        self.MAX_WIDTH = MAX_WIDTH
        self.MAX_HEIGHT = MAX_HEIGHT

        self.add_parameter(
            Parameter(
                name="destination_latent",
                input_types=["LatentArtifact"],
                type="LatentArtifact",
                tooltip="Base latent that the source will be composited onto.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="source_latent",
                input_types=["LatentArtifact"],
                type="LatentArtifact",
                tooltip="Latent to paste onto the destination.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="x_offset",
                default_value=0,
                tooltip="Horizontal offset in pixel space (divided by 8 internally for latent space).",
                traits={Slider(min_val=0, max_val=self.MAX_WIDTH)},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="y_offset",
                default_value=0,
                tooltip="Vertical offset in pixel space (divided by 8 internally for latent space).",
                traits={Slider(min_val=0, max_val=self.MAX_HEIGHT)},
            )
        )

        self.add_parameter(
            Parameter(
                name="resize_source",
                default_value=False,
                type="bool",
                tooltip="Resize the source latent to the destination latent's spatial size before compositing.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="mask_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                tooltip=(
                    "Optional mask image controlling the blend. "
                    "White pixels use the source; black pixels keep the destination. "
                    "The mask is resized to match the source latent automatically."
                ),
                allowed_modes={ParameterMode.INPUT},
            )
        )

        with ParameterGroup(name="mask_options", ui_options={"collapsed": True}) as mask_options:
            channel_param = ParameterString(
                name="channel",
                tooltip="Channel to extract from the mask image as blend weight.",
                default_value="alpha",
                ui_options={"expander": True, "edit_mask": True, "edit_mask_paint_mask": True},
            )
            channel_param.add_trait(Options(choices=["red", "green", "blue", "alpha"]))

            ParameterBool(name="invert_mask", default_value=False)
            ParameterFloat(name="grow_shrink", default_value=0, slider=True, min_val=-100, max_val=100)
            ParameterFloat(name="blur_mask", default_value=0, slider=True, min_val=-0, max_val=100)

        self.add_node_element(mask_options)

        self.add_parameter(
            Parameter(
                name="output_latent",
                output_type="LatentArtifact",
                tooltip="Resulting latent with the source composited onto the destination.",
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )
        self._initializing = False

    def add_parameter(self, parameter: Parameter) -> None:
        if not self._initializing:
            return
        super().add_parameter(parameter)

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

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []

        dest = self.get_parameter_value("destination_latent")
        if dest is None:
            errors.append(ValueError("Missing required 'destination_latent' input."))
        elif not isinstance(dest, LatentArtifact):
            errors.append(TypeError(f"'destination_latent' must be a LatentArtifact, got {type(dest).__name__}."))

        src = self.get_parameter_value("source_latent")
        if src is None:
            errors.append(ValueError("Missing required 'source_latent' input."))
        elif not isinstance(src, LatentArtifact):
            errors.append(TypeError(f"'source_latent' must be a LatentArtifact, got {type(src).__name__}."))

        return errors or None

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:

        dest_artifact = self.get_parameter_value("destination_latent")
        src_artifact = self.get_parameter_value("source_latent")
        resize_source = self.get_parameter_value("resize_source")
        x_offset = self.get_parameter_value("x_offset")
        y_offset = self.get_parameter_value("y_offset")

        destination = dest_artifact.to_torch().clone()
        device = destination.device
        source = src_artifact.to_torch().clone()
        source = source.to(device)

        dest_h, dest_w = destination.shape[-2], destination.shape[-1]
        src_h, src_w = source.shape[-2], source.shape[-1]

        if resize_source:
            source = F.interpolate(source, size=(dest_h, dest_w), mode="bilinear", align_corners=False)
            src_h, src_w = dest_h, dest_w

        mask_image_artifact = self.get_parameter_value("mask_image")
        channel = self.get_parameter_value("channel")

        if mask_image_artifact is not None:
            if isinstance(mask_image_artifact, ImageUrlArtifact):
                mask_image_artifact = load_image_from_url_artifact(mask_image_artifact)

            pil_image = image_artifact_to_pil(mask_image_artifact)
            mask_image = extract_channel_from_image(pil_image, channel)
        else:
            mask_image = Image.new("L", (src_w, src_h), 255)

        mask_image = apply_mask_transformations(
            mask_image,
            invert=self.get_parameter_value("invert_mask"),
            grow_shrink=self.get_parameter_value("grow_shrink"),
            blur_radius=self.get_parameter_value("blur_mask"),
        )

        channel_array = np.array(mask_image).astype("float32") / 255.0
        mask_tensor = torch.from_numpy(channel_array).clone().unsqueeze(0).unsqueeze(0)  # (1, 1, H_img, W_img)

        mask_tensor = F.interpolate(mask_tensor, size=(src_h, src_w), mode="bilinear", align_corners=False)
        mask_tensor = mask_tensor.to(device)

        # Convert pixel offsets to latent offsets (clamp to valid range)
        x = max(-src_w * LATENT_SCALE_FACTOR, min(x_offset, dest_w * LATENT_SCALE_FACTOR))
        y = max(-src_h * LATENT_SCALE_FACTOR, min(y_offset, dest_h * LATENT_SCALE_FACTOR))
        left = x // LATENT_SCALE_FACTOR
        top = y // LATENT_SCALE_FACTOR
        right = left + src_w
        bottom = top + src_h

        # Calculate intersection of paste region with destination bounds
        left_a = max(0, left)
        top_a = max(0, top)
        right_a = min(right, dest_w)
        bottom_a = min(bottom, dest_h)

        # if Source is entirely outside the destination; return unchanged
        if right_a <= left_a or bottom_a <= top_a:
            return destination

        # Corresponding crop within source / mask
        il = left_a - left
        it = top_a - top
        ir = right_a - left
        ib = bottom_a - top

        # Use ellipsis (...) to index last two spatial dims, supporting both 4D and 5D latents.
        dest_region = destination[..., top_a:bottom_a, left_a:right_a]
        src_region = source[..., it:ib, il:ir]
        # Expand mask to match latent rank (adds extra dims for video's temporal dimension).
        mask_region = mask_tensor[:, :, it:ib, il:ir]
        while mask_region.dim() < destination.dim():
            mask_region = mask_region.unsqueeze(2)

        destination[..., top_a:bottom_a, left_a:right_a] = dest_region * (1.0 - mask_region) + src_region * mask_region

        output_artifact = LatentArtifact.from_torch(destination, source_shape=dest_artifact.source_shape)
        self.set_parameter_value("output_latent", output_artifact)
        self.parameter_output_values["output_latent"] = output_artifact
