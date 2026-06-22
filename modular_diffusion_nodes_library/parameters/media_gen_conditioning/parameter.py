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
    Layout,
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
        self._needs_sync = False

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

    def update_config(self, config: MediaGenConditioningConfig) -> None:
        # Syncs config without touching the node. Used on workflow load where parameters
        # are already restored from serialized state. Rebuilds are deferred to _ensure_synced()
        # so that initial_setup rebuilds cannot remove parameters that have connections.
        self._config = config
        self._rebuild_controls()
        self._needs_sync = True

    def _rebuild_controls(self) -> None:
        """Arrange controls only so stale controls are removed and existing ones are updated.
        Cond inputs are not touched — they have live connections.
        """
        target = self._config.derive_layout(self._current_control_values())
        old_controls_only = Layout(control_params=self._composer.current.control_params, cond_inputs=())
        new_controls_only = Layout(control_params=target.control_params, cond_inputs=())
        self._composer.set_current(old_controls_only)
        self._composer.arrange(new_controls_only)

    def add_input_parameters(self) -> None:
        target = self._config.derive_layout(self._current_control_values())
        self._composer.arrange(target)

    def remove_input_parameters(self) -> None:
        if self._needs_sync:
            # Composer's current is stale, update it.
            current_layout = self._config.derive_layout(self._current_control_values())
            self._composer.set_current(current_layout)
        self._composer.arrange(EMPTY_LAYOUT)
        # Drop control-param values so the next add_input_parameters() starts clean.
        for name in CONTROL_PARAM_NAMES:
            self._node.parameter_values.pop(name, None)
        self._needs_sync = False

    def on_parameter_value_change(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
        *,
        initial_setup: bool,
    ) -> None:
        """Re-arrange the layout when a control-param value changes."""
        if param_name not in CONTROL_PARAM_NAMES:
            return
        if initial_setup:
            # Suppress mid-load rebuilds: a partial rebuild (e.g. mode fires before num_images
            # is fully restored) would remove params that already have connections. Defer to
            # _ensure_synced() which runs once at validate/build time with all values settled.
            self._rebuild_controls()
            self._needs_sync = True
            return
        if old_value == new_value:
            return
        if self._needs_sync:
            old_control_vals = {**self._current_control_values(), param_name: old_value}
            old_layout = self._config.derive_layout(old_control_vals)
            self._composer.set_current(old_layout)
        self._rebuild_layout()

    def validate_before_node_run(self) -> list[Exception] | None:
        self._ensure_synced()
        errors: list[Exception] = []
        for ci in self._composer.current.cond_inputs:
            if self._node.get_parameter_value(ci.media_param) is None:
                errors.append(ValueError(f"{self._node.name}: missing required '{ci.media_param}' conditioning input."))
        return errors or None

    def build_conditioning_payload(self) -> MediaGenConditioningPayload:
        """Build typed conditioning payload."""
        self._ensure_synced()
        mode = self._active_mode()
        entries = tuple(self._value_for_input(ci) for ci in self._composer.current.cond_inputs)
        return MediaGenConditioningPayload(mode=mode, entries=entries)

    def _ensure_synced(self) -> None:
        if not self._needs_sync:
            return
        self._composer.set_current(EMPTY_LAYOUT)
        self._rebuild_layout()

    def _rebuild_layout(self) -> None:
        self._needs_sync = False
        control_vals = self._current_control_values()
        target = self._config.derive_layout(control_vals)
        self._composer.arrange(target)

    def _current_control_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name in CONTROL_PARAM_NAMES:
            if self._node.get_parameter_by_name(name) is None:
                continue
            val = self._node.get_parameter_value(name)
            if val is not None:
                values[name] = val
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
            if raw is None:
                frame_index = 0
            else:
                frame_index = int(raw)
        elif ci.fixed_position is not None:
            frame_index = ci.fixed_position.value
        elif ci.expose_frame_index:
            raw = self._node.get_parameter_value(ci.frame_index_param)
            if raw is None:
                frame_index = 0
            else:
                frame_index = int(raw)
        else:
            frame_index = 0

        if ci.expose_strength:
            raw_strength = self._node.get_parameter_value(ci.strength_param)
            if raw_strength is None:
                strength = ci.default_strength
            else:
                strength = float(raw_strength)
        else:
            strength = ci.default_strength

        return ConditioningInputValue(artifact=artifact, frame_index=frame_index, strength=strength, kind=ci.kind)
