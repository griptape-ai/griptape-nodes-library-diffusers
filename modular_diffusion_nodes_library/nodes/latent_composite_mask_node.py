from __future__ import annotations

import logging
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from griptape_nodes.traits.widget import Widget

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
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

_PLACEMENT_WIDGET_LIBRARY = "Griptape Modular Diffusion Nodes Library"
_PLACEMENT_WIDGET_NAME = "LatentPlacementEditor"
# Parameters the placement editor mirrors; either side can set them.
_PLACEMENT_PARAMS = ("x_offset", "y_offset", "resize_source")
# Pixel sizes of both latents. Only a run can read the latents, so only a run reports these.
_SIZE_KEYS = ("dest_width", "dest_height", "source_width", "source_height")


class LatentCompositeMaskNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Composite a source latent onto a destination latent with an optional mask blend.

    - Place *source_latent* onto *destination_latent* at pixel offset (x_offset, y_offset).
    - If a *mask_image* is supplied, use the chosen channel as a blend weight
      (white = source, black = destination).
    - With *resize_source* enabled the source is rescaled to the destination latent size
      before compositing.
    - The output inherits the *destination_latent*'s ``source_shape``. Metadata is merged;
      on key conflicts the **destination** wins.
    """

    def __init__(self, **kwargs) -> None:
        self._initializing = True
        super().__init__(**kwargs)
        self._syncing_to_widget = False  # Prevent loop: param → widget → param
        self._syncing_to_params = False  # Prevent loop: widget → param → widget

        self.MAX_WIDTH = MAX_WIDTH
        self.MAX_HEIGHT = MAX_HEIGHT

        self.add_parameter(
            Parameter(
                name="destination_latent",
                input_types=["LatentArtifact"],
                type="LatentArtifact",
                tooltip=(
                    "Base latent that the source will be composited onto. "
                    "The output inherits this latent's shape. "
                    "Metadata is merged; on key conflicts this latent wins."
                ),
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
            ParameterDict(
                name="placement_editor",
                default_value={
                    "dest_width": 0,
                    "dest_height": 0,
                    "source_width": 0,
                    "source_height": 0,
                    "x_offset": 0,
                    "y_offset": 0,
                    "resize_source": False,
                },
                tooltip=(
                    "Drag the source over the destination to set x_offset / y_offset. "
                    "Latent sizes appear after the node has run once."
                ),
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name=_PLACEMENT_WIDGET_NAME, library=_PLACEMENT_WIDGET_LIBRARY)},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="x_offset",
                default_value=0,
                tooltip="Horizontal offset in pixel space (divided by 8 internally for latent space).",
                traits={Slider(min_val=-self.MAX_WIDTH, max_val=self.MAX_WIDTH, soft_limits=True)},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="y_offset",
                default_value=0,
                tooltip="Vertical offset in pixel space (divided by 8 internally for latent space).",
                traits={Slider(min_val=-self.MAX_HEIGHT, max_val=self.MAX_HEIGHT, soft_limits=True)},
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
                default_value="red",
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
        self._create_status_parameters()

    def add_parameter(self, parameter: Parameter) -> None:  # type: ignore[reportIncompatibleMethodOverride]
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

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "placement_editor" and not self._syncing_to_widget and isinstance(value, dict):
            self._sync_params_from_widget(value)
        elif parameter.name in _PLACEMENT_PARAMS and not self._syncing_to_params:
            self._update_widget_placement()
        return super().after_value_set(parameter, value)

    def after_incoming_connection(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name in _PLACEMENT_PARAMS:
            self._refresh_locked_in_widget()
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name in _PLACEMENT_PARAMS:
            self._refresh_locked_in_widget()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    # ── Widget ↔ parameter sync ────────────────────────────────────────────────

    def _get_locked_params(self) -> list[str]:
        """Placement params with an incoming connection, which the widget must not overwrite."""
        result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
        if not isinstance(result, ListConnectionsForNodeResultSuccess):
            return []
        return [
            conn.target_parameter_name
            for conn in result.incoming_connections
            if conn.target_parameter_name in _PLACEMENT_PARAMS
        ]

    def _current_widget_value(self) -> dict:
        """The stored widget value, with the latent sizes from the latest run laid over it.

        A run reports sizes through `publish_update_to_parameter`, which reaches the editor and this
        node's `parameter_output_values` but not its stored value, so the sizes are read from there.
        """
        stored = self.get_parameter_value("placement_editor") or {}
        produced = self.parameter_output_values.get("placement_editor") or {}
        sizes = {key: produced[key] for key in _SIZE_KEYS if key in produced}
        return {**stored, **sizes}

    def _push_widget(self, new_dict: dict) -> None:
        self._syncing_to_widget = True
        try:
            self.set_parameter_value("placement_editor", new_dict)
            self.publish_update_to_parameter("placement_editor", new_dict)
        finally:
            self._syncing_to_widget = False

    def _refresh_locked_in_widget(self) -> None:
        existing = self._current_widget_value()
        new_locked = self._get_locked_params()
        if set(existing.get("locked") or []) == set(new_locked):
            return
        self._push_widget({**existing, "locked": new_locked})

    def _update_widget_placement(self) -> None:
        """Mirror the placement params into the widget, skipping the push when nothing changed."""
        existing = self._current_widget_value()
        placement = {
            "x_offset": self.get_parameter_value("x_offset") or 0,
            "y_offset": self.get_parameter_value("y_offset") or 0,
            "resize_source": bool(self.get_parameter_value("resize_source")),
        }
        if all(existing.get(key) == placement[key] for key in _PLACEMENT_PARAMS):
            return
        self._push_widget({**existing, **placement, "locked": self._get_locked_params()})

    def _sync_params_from_widget(self, widget_dict: dict) -> None:
        """Copy the widget's placement into every placement param that has no incoming connection.

        The widget value is published first so the editor already holds it when the param updates
        re-render the node; otherwise the widget would redraw from the old value and snap back.
        """
        locked = set(widget_dict.get("locked") or [])
        self.publish_update_to_parameter("placement_editor", widget_dict)
        self._syncing_to_params = True
        try:
            for name in _PLACEMENT_PARAMS:
                if name in widget_dict and name not in locked:
                    self.set_parameter_value(name, widget_dict[name])
                    self.publish_update_to_parameter(name, widget_dict[name])
        finally:
            self._syncing_to_params = False

    def _publish_latent_sizes(self, dest_shape: tuple[int, ...], src_shape: tuple[int, ...]) -> None:
        """Report both latents' pixel sizes to the widget. Runs where the node runs."""
        existing = self.get_parameter_value("placement_editor") or {}
        widget_dict = {
            **existing,
            "dest_width": dest_shape[-1] * LATENT_SCALE_FACTOR,
            "dest_height": dest_shape[-2] * LATENT_SCALE_FACTOR,
            "source_width": src_shape[-1] * LATENT_SCALE_FACTOR,
            "source_height": src_shape[-2] * LATENT_SCALE_FACTOR,
            "x_offset": self.get_parameter_value("x_offset") or 0,
            "y_offset": self.get_parameter_value("y_offset") or 0,
            "resize_source": bool(self.get_parameter_value("resize_source")),
        }
        self.publish_update_to_parameter("placement_editor", widget_dict)

    def validate_in_execution_environment(self) -> list[Exception] | None:
        """Both inputs are held latents, so only the process running the node can read them."""
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
        self._clear_execution_status()
        yield lambda: self._run_with_status(
            self._composite,
            success_msg="Composite completed successfully.",
            failure_log="Latent composite failed",
            logger=logger,
        )

    def _composite(self) -> None:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

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
        self._publish_latent_sizes(tuple(destination.shape), tuple(source.shape))

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
            return

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

        # Destination is the first input: it wins meta on conflicts.
        combined_meta = dict(src_artifact.meta)
        combined_meta.update(dest_artifact.meta)
        output_artifact = LatentArtifact.from_torch(
            destination,
            source_shape=dest_artifact.source_shape,
            meta=combined_meta,
        )
        self.set_parameter_value("output_latent", output_artifact)
        self.parameter_output_values["output_latent"] = output_artifact
