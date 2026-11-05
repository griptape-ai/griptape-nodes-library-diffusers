from __future__ import annotations

import logging
from typing import ClassVar, cast

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    FlexibleImageConfig,
    MediaGenConditioningConfig,
    VideoConditioningConfig,
)
from modular_diffusion_nodes_library.runtime_parameters.conditioning_runtime_parameter import (
    MediaGenConditioningRuntimeParameter,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode
from modular_diffusion_nodes_library.utils.minimax_h3_ref2va_utils import (
    get_reference_video_duration_seconds,
    plan_reference_video_trim,
)

logger = logging.getLogger("diffusers_nodes_library")


class MiniMaxH3Ref2VAPipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    """Runtime surface for MiniMax-H3 ref2va image and video references."""

    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=FlexibleImageConfig(min_count=1, max_count=9, expose_strength=False, expose_frame_index=False),
        video=VideoConditioningConfig(expose_strength=False, expose_frame_index=False),
        default_mode=ConditioningMode.IMAGE,
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._references_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="references",
            multiple=True,
            max_count=4,
            output_key="references",
            accepted_modes=(ConditioningMode.IMAGE, ConditioningMode.VIDEO),
            tooltip="Image and video references for MiniMax-H3 ref2va generation.",
            badge_title="References",
            badge_message=(
                "Connect one image-mode **Media Gen Conditioning** node with up to 9 images, and up to three "
                "video-mode nodes (one clip per node). At least one reference is required. Each video must be "
                "at least 2 seconds; all videos together are limited to 15 seconds. Video audio is discarded.\n\n"
                "In your prompt, refer to images as `<Picture i>` and videos as `<Video k>`."
            ),
        )
        self._duration_warning = ParameterMessage(
            name="ref2va_duration_warning",
            variant="warning",
            title="Reference Video Duration",
            value="",
            hide=True,
        )

    def _add_input_parameters(self) -> None:
        self._node.add_node_element(self._duration_warning)
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip=(
                    "The prompt to guide generation. Refer to image references as `<Picture i>` and video "
                    "references as `<Video k>`."
                ),
            )
        )
        self._references_param.add_input_parameters()

    def _remove_input_parameters(self) -> None:
        self._references_param.remove_input_parameters()
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("ref2va_duration_warning")

    def on_incoming_connection_added(self, param_name: str) -> None:
        self._references_param.on_incoming_connection_added(param_name)

    def on_incoming_connection_removed(self, param_name: str) -> None:
        self._references_param.on_incoming_connection_removed(param_name)

    def _get_pipe_kwargs(self) -> dict:
        return {
            "prompt": self._node.get_parameter_value("prompt"),
            **self._references_param.get_pipe_kwargs(),
        }

    def _set_duration_warning(self, message: str | None) -> None:
        if message:
            self._duration_warning.value = message
            self._duration_warning.hide = False
        else:
            self._duration_warning.value = ""
            self._duration_warning.hide = True

    def validate_in_execution_environment(self) -> list[Exception] | None:
        errors = super().validate_in_execution_environment() or []
        conditioning_errors = self._references_param.validate_in_execution_environment()
        if conditioning_errors:
            errors.extend(conditioning_errors)

        self._set_duration_warning(None)
        if errors:
            return errors

        payloads = self._references_param.get_payloads()
        if not payloads:
            return [
                ValueError(
                    f"Attempted to validate MiniMax-H3 ref2va references on '{self._node.name}'. "
                    "Failed because at least one image or video reference is required."
                )
            ]

        image_count = sum(len(payload.entries) for payload in payloads if payload.mode is ConditioningMode.IMAGE)
        video_entries = [
            entry for payload in payloads if payload.mode is ConditioningMode.VIDEO for entry in payload.entries
        ]
        if image_count > 9:
            return [
                ValueError(
                    f"Attempted to validate MiniMax-H3 ref2va references on '{self._node.name}'. "
                    f"Failed because {image_count} images exceed the limit of 9."
                )
            ]
        if len(video_entries) > 3:
            return [
                ValueError(
                    f"Attempted to validate MiniMax-H3 ref2va references on '{self._node.name}'. "
                    f"Failed because {len(video_entries)} videos exceed the limit of 3."
                )
            ]
        if image_count + len(video_entries) == 0:
            return [
                ValueError(
                    f"Attempted to validate MiniMax-H3 ref2va references on '{self._node.name}'. "
                    "Failed because at least one image or video reference is required."
                )
            ]

        durations: list[float] = []
        for entry in video_entries:
            video_artifact = cast(VideoUrlArtifact, entry.artifact)
            durations.append(get_reference_video_duration_seconds(video_artifact))
        try:
            trim_plan = plan_reference_video_trim(durations)
        except ValueError as error:
            return [error]
        self._set_duration_warning(trim_plan.warning_message)
        if trim_plan.warning_message:
            logger.warning("%s: %s", self._node.name, trim_plan.warning_message)
        return None
