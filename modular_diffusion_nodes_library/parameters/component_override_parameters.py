"""Delegate that manages per-component override input ports on the builder node."""

from __future__ import annotations

from typing import TYPE_CHECKING

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentArtifact
from modular_diffusion_nodes_library.component_loading.component_slots import (
    ALLOWED_COMPONENT_SLOTS,
    SLOT_DISPLAY_NAMES,
    slot_artifact_type_name,
)

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
        LatentDiffusionPipelineBuilderNode,
    )


class ComponentOverrideParameters:
    """Delegate that manages per-component override input ports on the builder node."""

    GROUP_NAME = "component_overrides"

    def __init__(self, node: LatentDiffusionPipelineBuilderNode) -> None:
        self._node = node
        # A load can swap slots more than once, so a port removed by one swap must survive to be restored by a
        # later one, where a saved connection still targets it.
        self._removed_while_loading: dict[str, Parameter] = {}

    @property
    def current_slots(self) -> list[str]:
        """Slot names currently exposed as override ports"""
        group = self._node.get_group_by_name_or_element_id(self.GROUP_NAME)
        if group is None:
            return []
        slots: list[str] = []
        for param in group.find_elements_by_type(Parameter, find_recursively=False):
            if param.name.startswith("component_"):
                slots.append(param.name[len("component_") :])
        return slots

    def update_slots(self, slots: list[str], *, initial_setup: bool = False) -> None:
        """Replace the current override ports with ports for the given slot names."""
        existing_slots = self.current_slots
        if slots == existing_slots:
            return

        invalid = [s for s in slots if s not in ALLOWED_COMPONENT_SLOTS]
        if invalid:
            msg = (
                f"Attempted to set component override slots. "
                f"Failed because these slot names are not in ALLOWED_COMPONENT_SLOTS: {invalid}."
            )
            raise ValueError(msg)

        self._node.save_parameter_properties()
        if initial_setup:
            self._replace_slots_locally(existing_slots, slots)
        else:
            self._removed_while_loading.clear()
            with self._node.preserve_connections():
                for slot in existing_slots:
                    self._node.remove_parameter_with_connection_cleanup(f"component_{slot}")

                if not slots:
                    self._remove_override_group()
                else:
                    self._add_override_group(slots, kept_parameters={})

        self._node.reorder_parameters_by_groups()
        self._node.clear_parameter_cache()

    def _replace_slots_locally(self, existing_slots: list[str], slots: list[str]) -> None:
        """Swap override ports without engine requests, as loading requires.

        Loading connects to the ports that exist before values are set, so a port re-added during the load is
        re-attached as the same Parameter instead of being recreated.
        """
        for slot in existing_slots:
            parameter = self._node.get_parameter_by_name(f"component_{slot}")
            if parameter is None:
                continue
            self._removed_while_loading[slot] = parameter
            self._node.remove_parameter_element(parameter)

        kept_parameters: dict[str, Parameter] = {}
        for slot in slots:
            if slot in self._removed_while_loading:
                kept_parameters[slot] = self._removed_while_loading.pop(slot)

        if not slots:
            self._node.remove_parameter_element_by_name(self.GROUP_NAME)
        else:
            self._add_override_group(slots, kept_parameters=kept_parameters)

    def _add_override_group(self, slots: list[str], *, kept_parameters: dict[str, Parameter]) -> None:
        """Create the override group (if missing) and add component params for all given slots."""
        if self._node.get_group_by_name_or_element_id(self.GROUP_NAME) is None:
            self._node.add_node_element(
                ParameterGroup(
                    name=self.GROUP_NAME,
                    ui_options={"display_name": "Component Overrides", "collapsed": False},
                    user_defined=False,
                )
            )

        for slot in slots:
            if slot in kept_parameters:
                kept_parameter = kept_parameters[slot]
                # Removal from the group cleared this.
                kept_parameter.parent_element_name = self.GROUP_NAME
                self._node.add_parameter(kept_parameter)
                continue
            artifact_type = slot_artifact_type_name(slot)
            self._node.add_parameter(
                Parameter(
                    name=f"component_{slot}",
                    type=artifact_type,
                    input_types=[artifact_type],
                    default_value=None,
                    tooltip=(
                        f"Optional override for the {SLOT_DISPLAY_NAMES[slot]} component."
                        " Connect a Load Component node to replace the default."
                    ),
                    allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                    ui_options={"display_name": SLOT_DISPLAY_NAMES[slot]},
                    parent_element_name=self.GROUP_NAME,
                )
            )

    def _remove_override_group(self) -> None:
        """Remove the override group if present."""
        if self._node.get_group_by_name_or_element_id(self.GROUP_NAME) is None:
            return
        self._node.remove_parameter_with_connection_cleanup(self.GROUP_NAME)

    def get_component_overrides(self) -> dict[str, ComponentArtifact]:
        """Return connected ComponentArtifact instances keyed by slot name."""
        overrides: dict[str, ComponentArtifact] = {}
        for slot in self.current_slots:
            value = self._node.get_parameter_value(f"component_{slot}")
            if isinstance(value, ComponentArtifact):
                overrides[slot] = value
        return overrides

    @property
    def has_quantized_overrides(self) -> bool:
        """True if any override component uses quantized weights (e.g. GGUF)."""
        overrides = self.get_component_overrides()
        return bool(overrides and any(artifact.is_quantized for artifact in overrides.values()))

    def get_override_config_kwargs(self) -> dict[str, str]:
        """Return cache-key entries for connected component overrides."""
        return {
            f"component_override_{slot}": artifact.load_id for slot, artifact in self.get_component_overrides().items()
        }
