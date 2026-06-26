"""Shared parameter class for runtime-parameter classes that consume media-gen conditioning."""

from __future__ import annotations

from typing import Any, ClassVar

from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    ConditioningInputValue,
    MediaGenConditioningPayload,
    normalize_to_payloads,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode, MediaGenConditioningKey
from modular_diffusion_nodes_library.utils.connection_utils import delete_parameter_list_child_connections

# Wire type shared by the producing and consuming nodes; must equal MediaGenConditioningKey.OUTPUT on both sides.
MEDIA_GEN_CONDITIONING_TYPE = MediaGenConditioningKey.OUTPUT


class MediaGenConditioningRuntimeParameter:
    """Conditioning input parameter for a runtime-parameters class.

    `param_name`: user-visible parameter name on the consuming node.
    `accepted_modes`: accepted ConditioningMode values.
    `multiple=True`: expose a ParameterList for multiple connections.
    `badge_message`: optional help badge (`badge_title` defaults to "Conditioning input").
    """

    _DEFAULT_BADGE_TITLE = "Conditioning input"

    _MODE_TO_URL_TYPE: ClassVar[dict[ConditioningMode, str]] = {
        ConditioningMode.IMAGE: "ImageUrlArtifact",
        ConditioningMode.VIDEO: "VideoUrlArtifact",
    }

    def __init__(
        self,
        node: BaseNode,
        *,
        param_name: str,
        output_key: str | None = None,
        accepted_modes: tuple[ConditioningMode, ...] = (ConditioningMode.IMAGE, ConditioningMode.VIDEO),
        multiple: bool = False,
        tooltip: str | None = None,
        badge_title: str | None = None,
        badge_message: str | None = None,
    ):
        self._node = node
        self._param_name = param_name
        self._param_type = MEDIA_GEN_CONDITIONING_TYPE
        self._output_key = output_key or MediaGenConditioningKey.OUTPUT
        self._accepted_modes = accepted_modes
        self._accepted_url_types: tuple[str, ...] = tuple(
            self._MODE_TO_URL_TYPE[mode] for mode in self._accepted_modes if mode in self._MODE_TO_URL_TYPE
        )
        self._multiple = multiple
        self._tooltip = tooltip or "Media generation conditioning from a Media Gen Conditioning node."
        self._badge_title = badge_title or self._DEFAULT_BADGE_TITLE
        self._badge_message = badge_message

    def add_input_parameters(self) -> None:
        input_types = [self._param_type, *self._accepted_url_types]
        if self._multiple:
            param: Parameter = ParameterList(
                name=self._param_name,
                input_types=input_types,
                type=self._param_type,
                default_value=[],
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                tooltip=self._tooltip,
            )
        else:
            param = Parameter(
                name=self._param_name,
                input_types=input_types,
                type=self._param_type,
                default_value=None,
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                tooltip=self._tooltip,
            )
        if self._badge_message is not None:
            param.set_badge(variant="help", title=self._badge_title, message=self._badge_message)
        self._node.add_parameter(param)

    def remove_input_parameters(self) -> None:
        # Removing a ParameterList does not delete its child connections, so
        # delete them first to avoid dangling connections.
        existing = self._node.get_parameter_by_name(self._param_name)
        if existing is None:
            return
        if isinstance(existing, ParameterList):
            delete_parameter_list_child_connections(self._node, [existing])
        self._node.remove_parameter_element_by_name(self._param_name)

    def get_pipe_kwargs(self) -> dict[str, Any]:
        value = self._node.get_parameter_value(self._param_name)
        if value is None or (self._multiple and not value):
            return {}

        payloads = self._coerce_to_payloads(value)
        if not payloads:
            return {}

        mode_errors = self._validate_payload_modes(payloads)
        if mode_errors:
            raise mode_errors[0]

        if self._multiple:
            return {self._output_key: payloads}
        return {self._output_key: payloads[0]}

    def validate_before_node_run(self) -> list[Exception] | None:
        value = self._node.get_parameter_value(self._param_name)
        if value is None or (self._multiple and not value):
            return None
        try:
            payloads = self._coerce_to_payloads(value)
        except ValueError as err:
            return [err]
        return self._validate_payload_modes(payloads)

    def _coerce_to_payloads(self, value: Any) -> list[MediaGenConditioningPayload]:
        items = value if isinstance(value, list) else [value]
        payloads: list[MediaGenConditioningPayload] = []
        for item in items:
            if isinstance(item, ImageUrlArtifact) and ConditioningMode.IMAGE in self._accepted_modes:
                payloads.append(self._payload_from_image_url(item))
                continue
            if isinstance(item, VideoUrlArtifact) and ConditioningMode.VIDEO in self._accepted_modes:
                payloads.append(self._payload_from_video_url(item))
                continue
            normalized = normalize_to_payloads(item)
            if normalized is None:
                continue
            payloads.extend(normalized)
        return payloads

    @staticmethod
    def _payload_from_image_url(image: ImageUrlArtifact) -> MediaGenConditioningPayload:
        return MediaGenConditioningPayload(
            mode=ConditioningMode.IMAGE,
            entries=(
                ConditioningInputValue(
                    artifact=image,
                    frame_index=0,
                    strength=1.0,
                    kind="image",
                ),
            ),
        )

    @staticmethod
    def _payload_from_video_url(video: VideoUrlArtifact) -> MediaGenConditioningPayload:
        return MediaGenConditioningPayload(
            mode=ConditioningMode.VIDEO,
            entries=(
                ConditioningInputValue(
                    artifact=video,
                    frame_index=0,
                    strength=1.0,
                    kind="video",
                ),
            ),
        )

    def _validate_payload_modes(self, payloads: list[MediaGenConditioningPayload]) -> list[Exception] | None:
        errors: list[Exception] = []
        accepted = ", ".join(m.value for m in self._accepted_modes)
        for payload in payloads:
            if payload.mode in self._accepted_modes:
                continue
            errors.append(
                ValueError(
                    f"Attempted to use media-gen conditioning on '{self._node.name}'. "
                    f"Failed with payload mode '{payload.mode.value}' because only "
                    f"'{accepted}' conditioning is accepted by this pipeline."
                )
            )
        return errors or None
