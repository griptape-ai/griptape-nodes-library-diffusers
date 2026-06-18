"""Stateful composer that drives the host node from a target `Layout`.

`LayoutComposer.arrange(target)` diffs `target` against the current snapshot
and minimally mutates the host node. Sole module that calls
`node.add_parameter` / `node.add_node_element` / `remove_dynamic_param`
for the conditioning surface.
"""

from __future__ import annotations

import logging

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    EMPTY_LAYOUT,
    ConditioningInput,
    ControlKind,
    ControlParam,
    Layout,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


def remove_dynamic_param(node: BaseNode, name: str) -> None:
    """Remove a parameter (or group) and drop its stored value.

    User-defined params route through `RemoveParameterFromNodeRequest` so the
    retained-mode bus sees the change; others are removed directly.
    """
    param = node.get_parameter_by_name(name)
    if param and getattr(param, "user_defined", False):
        GriptapeNodes.handle_request(RemoveParameterFromNodeRequest(parameter_name=name, node_name=node.name))
    else:
        node.remove_parameter_element_by_name(name)
    node.parameter_values.pop(name, None)


class LayoutComposer:
    """Apply target `Layout`s onto a host node, holding the current snapshot."""

    def __init__(self, node: BaseNode) -> None:
        self._node = node
        self._current: Layout = EMPTY_LAYOUT

    @property
    def current(self) -> Layout:
        return self._current

    def arrange(self, target: Layout) -> None:
        """Drive the host node from `self.current` to `target`."""
        if target == self._current:
            return

        old = self._current
        old_input_keys = {ci.key: ci for ci in old.cond_inputs}
        new_input_keys = {ci.key: ci for ci in target.cond_inputs}
        old_controls = {c.name: c for c in old.control_params}
        new_controls = {c.name: c for c in target.control_params}

        for key, conditioning_input in old_input_keys.items():
            if key not in new_input_keys:
                self._remove_input(conditioning_input)

        for name in old_controls:
            if name not in new_controls:
                remove_dynamic_param(self._node, name)

        for name, control in new_controls.items():
            if name not in old_controls:
                self._add_control(control)

        for key, conditioning_input in new_input_keys.items():
            if key not in old_input_keys:
                self._add_input(conditioning_input)

        self._reorder(target)

        self._current = target

    def _add_control(self, control: ControlParam) -> None:
        if self._node.get_parameter_by_name(control.name) is not None:
            return
        if control.kind is ControlKind.OPTION:
            return self._node.add_parameter(
                Parameter(
                    name=control.name,
                    default_value=control.default,
                    type="str",
                    traits={Options(choices=list(control.choices))},
                    tooltip=control.tooltip,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": control.display_name},
                )
            )
        if control.kind is ControlKind.SLIDER:
            return self._node.add_parameter(
                Parameter(
                    name=control.name,
                    default_value=control.default,
                    type="int",
                    tooltip=control.tooltip,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={
                        "display_name": control.display_name,
                        "slider": {"min_val": control.slider_min, "max_val": control.slider_max},
                        "step": 1,
                    },
                )
            )
        msg = f"_add_control: unknown ControlKind '{control.kind}'."
        raise ValueError(msg)

    def _add_input(self, conditioning_input: ConditioningInput) -> None:
        if conditioning_input.kind == "video":
            self._add_video_input(conditioning_input)
            return
        self._add_image_input(conditioning_input)

    def _add_image_input(self, conditioning_input: ConditioningInput) -> None:
        group_name = conditioning_input.group_param
        if group_name is not None and self._node.get_group_by_name_or_element_id(group_name) is None:
            self._node.add_node_element(
                ParameterGroup(name=group_name, ui_options={"display_name": conditioning_input.group_label})
            )
        if self._node.get_parameter_by_name(conditioning_input.media_param) is None:
            self._node.add_parameter(
                Parameter(
                    name=conditioning_input.media_param,
                    input_types=["ImageArtifact", "ImageUrlArtifact"],
                    type="ImageUrlArtifact",
                    tooltip=conditioning_input.image_tooltip,
                    allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                    ui_options={"display_name": conditioning_input.label, "hide_property": True},
                    user_defined=True,
                    parent_element_name=group_name,
                )
            )
        if (
            conditioning_input.expose_frame_index
            and self._node.get_parameter_by_name(conditioning_input.frame_index_param) is None
        ):
            self._node.add_parameter(
                Parameter(
                    name=conditioning_input.frame_index_param,
                    default_value=0,
                    input_types=["int"],
                    type="int",
                    tooltip=conditioning_input.frame_index_tooltip,
                    ui_options={"display_name": "Frame Index"},
                    user_defined=True,
                    parent_element_name=group_name,
                )
            )
        if (
            conditioning_input.expose_strength
            and self._node.get_parameter_by_name(conditioning_input.strength_param) is None
        ):
            self._node.add_parameter(
                Parameter(
                    name=conditioning_input.strength_param,
                    default_value=conditioning_input.default_strength,
                    input_types=["float"],
                    type="float",
                    tooltip=conditioning_input.strength_tooltip,
                    ui_options={
                        "display_name": "Strength",
                        "slider": {"min_val": 0.0, "max_val": 1.0},
                        "step": 0.01,
                    },
                    user_defined=True,
                    parent_element_name=group_name,
                )
            )

    def _add_video_input(self, conditioning_input: ConditioningInput) -> None:
        if self._node.get_parameter_by_name(conditioning_input.media_param) is None:
            self._node.add_parameter(
                Parameter(
                    name=conditioning_input.media_param,
                    input_types=["VideoArtifact", "VideoUrlArtifact"],
                    type="VideoUrlArtifact",
                    tooltip=conditioning_input.image_tooltip,
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": conditioning_input.label, "hide_property": True},
                    user_defined=True,
                )
            )
        if (
            conditioning_input.expose_frame_index
            and self._node.get_parameter_by_name(conditioning_input.frame_index_param) is None
        ):
            self._node.add_parameter(
                Parameter(
                    name=conditioning_input.frame_index_param,
                    default_value=0,
                    input_types=["int"],
                    type="int",
                    tooltip=conditioning_input.frame_index_tooltip,
                    ui_options={"display_name": "Video Frame Index"},
                    user_defined=True,
                )
            )
        if (
            conditioning_input.expose_strength
            and self._node.get_parameter_by_name(conditioning_input.strength_param) is None
        ):
            self._node.add_parameter(
                Parameter(
                    name=conditioning_input.strength_param,
                    default_value=conditioning_input.default_strength,
                    input_types=["float"],
                    type="float",
                    tooltip=conditioning_input.strength_tooltip,
                    ui_options={
                        "display_name": "Video Strength",
                        "slider": {"min_val": 0.0, "max_val": 1.0},
                        "step": 0.01,
                    },
                    user_defined=True,
                )
            )

    def _remove_input(self, conditioning_input: ConditioningInput) -> None:
        # Remove children before the group.
        remove_dynamic_param(self._node, conditioning_input.media_param)
        remove_dynamic_param(self._node, conditioning_input.frame_index_param)
        remove_dynamic_param(self._node, conditioning_input.strength_param)
        if conditioning_input.group_param is not None:
            self._node.remove_parameter_element_by_name(conditioning_input.group_param)

    def _reorder(self, target: Layout) -> None:
        """Reorder composer-owned top-level children to match `target.all_param_names`.

        Only top-level children of `root_ui_element` participate; group children
        are ordered by the group. Host-owned slots stay put — we only shuffle
        the composer's own slots among themselves.
        """
        current_names = [element.name for element in self._node.root_ui_element._children]  # noqa: SLF001
        current_set = set(current_names)
        desired = [name for name in target.all_param_names if name in current_set]
        if not desired:
            return

        desired_set = set(desired)
        desired_iter = iter(desired)
        order = [next(desired_iter) if name in desired_set else name for name in current_names]
        if order == current_names:
            return
        self._node.reorder_elements(order)
