"""Coordinator for media-gen conditioning parameters. Drives LayoutComposer from control values."""

from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    CONTROL_PARAM_NAMES,
    EMPTY_LAYOUT,
    PARAM_MODE,
    ConditioningInput,
    MediaGenConditioningConfig,
)

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    ConditioningInputValue,
    MediaGenConditioningPayload,
)
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.node_layout_composer import (
    LayoutComposer,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
)

PARAM_CONDITIONING = "conditioning"
CONDITIONING_OUTPUT_TYPE = "media_gen_conditioning"


class MediaGenConditioningParameter:
    """Owns the conditioning parameter surface on a host node."""

    def __init__(self, node: BaseNode, config: MediaGenConditioningConfig) -> None:
        self._node = node
        self._config = config
        self._composer = LayoutComposer(node)

    def add_output_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name=PARAM_CONDITIONING,
                output_type=CONDITIONING_OUTPUT_TYPE,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Media generation conditioning output.",
                serializable=False,
                hide_property=True,
                ui_options={"display_name": "Conditioning Output"},
            )
        )

    def remove_output_parameters(self) -> None:
        self._node.remove_parameter_element_by_name(PARAM_CONDITIONING)

    def add_input_parameters(self) -> None:
        target = self._config.derive_layout(self._current_control_values())
        self._composer.arrange(target)

    def remove_input_parameters(self) -> None:
        self._composer.arrange(EMPTY_LAYOUT)
        # Drop control-param values so the next add_input_parameters() starts clean.
        for name in CONTROL_PARAM_NAMES:
            self._node.parameter_values.pop(name, None)

    def on_parameter_value_change(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
        *,
        initial_setup: bool,  # noqa: ARG002
    ) -> None:
        """Re-arrange the layout when a control-param value changes."""
        if param_name not in CONTROL_PARAM_NAMES:
            return
        if old_value == new_value:
            return
        self._rebuild_layout()

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []
        for ci in self._composer.current.cond_inputs:
            if self._node.get_parameter_value(ci.media_param) is None:
                errors.append(ValueError(f"{self._node.name}: missing required '{ci.media_param}' conditioning input."))
        return errors or None

    def build_conditioning_payload(self) -> MediaGenConditioningPayload:
        """Build typed conditioning payload."""
        mode = self._active_mode()
        entries = tuple(self._value_for_input(ci) for ci in self._composer.current.cond_inputs)
        return MediaGenConditioningPayload(mode=mode, entries=entries)

    def _rebuild_layout(self) -> None:
        control_vals = self._current_control_values()
        target = self._config.derive_layout(control_vals)
        self._composer.arrange(target)

    def _current_control_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name in CONTROL_PARAM_NAMES:
            if self._node.get_parameter_by_name(name) is None:
                continue
            values[name] = self._node.get_parameter_value(name)
        return values

    def _active_mode(self) -> ConditioningMode:
        if self._config.image is None:
            return ConditioningMode.VIDEO
        if self._config.video is None:
            return ConditioningMode.IMAGE
        if self._node.get_parameter_by_name(PARAM_MODE) is None:
            return self._config.default_mode
        raw = self._node.get_parameter_value(PARAM_MODE)
        if raw is None:
            return self._config.default_mode
        try:
            return ConditioningMode(str(raw))
        except ValueError:
            return self._config.default_mode

    def _value_for_input(self, ci: ConditioningInput) -> ConditioningInputValue:
        artifact = self._node.get_parameter_value(ci.media_param)
        if artifact is None:
            msg = f"{self._node.name}: missing required '{ci.media_param}' conditioning input."
            raise ValueError(msg)

        frame_index: int | str
        if ci.kind == "video":
            raw = self._node.get_parameter_value(ci.frame_index_param)
            frame_index = 0 if raw is None else int(raw)
        elif ci.fixed_position is not None:
            frame_index = ci.fixed_position.value
        elif ci.expose_frame_index:
            raw = self._node.get_parameter_value(ci.frame_index_param)
            frame_index = 0 if raw is None else int(raw)
        else:
            frame_index = 0

        if ci.expose_strength:
            raw_strength = self._node.get_parameter_value(ci.strength_param)
            strength = ci.default_strength if raw_strength is None else float(raw_strength)
        else:
            strength = ci.default_strength

        return ConditioningInputValue(artifact=artifact, frame_index=frame_index, strength=strength, kind=ci.kind)
