"""FixedSizeParameterList — grow-by-one, shrink-on-disconnect multi-input parameter group.

Replaces ParameterList for runtime parameters where stable slot names are needed for
reliable save/load and reset. Unlike ParameterList, slot names are fixed and predictable
({param_name}_0 … {param_name}_{max_count-1}), so no UUID churn occurs.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class FixedSizeParameterList:
    """Manages a fixed-max set of indexed input slots that grow/shrink via hide/show.

    All max_count slots are added as regular Parameters with predictable names.
    Slot 0 is always visible. When slot N is connected, slot N+1 is revealed.
    When connections are removed, trailing empty slots are re-hidden, keeping exactly
    one empty slot visible after the last connected one.

    All slots live inside a collapsible ParameterGroup; `group_display_name` is required.
    The group and badge are created and destroyed as part of add/remove_input_parameters.
    """

    def __init__(
        self,
        node: BaseNode,
        *,
        param_name: str,
        max_count: int,
        input_types: list[str],
        type: str,
        group_display_name: str,
        tooltip: str = "",
        badge_title: str | None = None,
        badge_message: str | None = None,
        collapsed: bool = False,
    ) -> None:
        self._node = node
        self._param_name = param_name
        self._max_count = max_count
        self._input_types = input_types
        self._type = type
        self._tooltip = tooltip
        self._badge_title = badge_title
        self._badge_message = badge_message
        self._group_display_name = group_display_name
        self._group_name = f"{param_name}_group"
        self._collapsed = collapsed

    # ------------------------------------------------------------------
    # Slot naming
    # ------------------------------------------------------------------

    def slot_name(self, index: int) -> str:
        return f"{self._param_name}_{index}"

    @property
    def slot_names(self) -> list[str]:
        return [self.slot_name(i) for i in range(self._max_count)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def add_input_parameters(self) -> None:
        """Add the group container and all slots; slot 0 is visible, slots 1+ start hidden."""
        if self._node.get_group_by_name_or_element_id(self._group_name) is None:
            self._node.add_node_element(
                ParameterGroup(
                    name=self._group_name,
                    ui_options={"display_name": self._group_display_name, "collapsed": self._collapsed},
                    user_defined=True,
                )
            )
        if self._badge_message is not None:
            group = self._node.get_group_by_name_or_element_id(self._group_name)
            if group is not None:
                group.set_badge(
                    variant="help",
                    title=self._badge_title or "Info",
                    message=self._badge_message,
                )
        for i in range(self._max_count):
            param = Parameter(
                name=self.slot_name(i),
                input_types=self._input_types,
                type=self._type,
                default_value=None,
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                tooltip=self._tooltip,
                parent_element_name=self._group_name,
            )
            param.ui_options = {"hide": i > 0}
            self._node.add_parameter(param)

    def remove_input_parameters(self) -> None:
        """Remove all slots then the group container (children before parent)."""
        for name in self.slot_names:
            if self._node.does_name_exist(name):
                self._node.remove_parameter_element_by_name(name)
        if self._node.get_group_by_name_or_element_id(self._group_name) is not None:
            self._node.remove_parameter_element_by_name(self._group_name)

    # ------------------------------------------------------------------
    # Value collection
    # ------------------------------------------------------------------

    def get_values(self) -> list[Any]:
        """Return non-None values from connected slots, in index order."""
        values = []
        for name in self.slot_names:
            val = self._node.get_parameter_value(name)
            if val is not None:
                values.append(val)
        return values

    # ------------------------------------------------------------------
    # Visibility management
    # ------------------------------------------------------------------

    def on_connection_added(self, param_name: str) -> None:
        """Call when any incoming connection is added to this node."""
        if param_name not in self.slot_names:
            return
        self._update_visibility()

    def on_connection_removed(self, param_name: str) -> None:
        """Call when any incoming connection is removed from this node."""
        if param_name not in self.slot_names:
            return
        self._update_visibility()

    def _update_visibility(self) -> None:
        result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self._node.name))
        if not isinstance(result, ListConnectionsForNodeResultSuccess):
            return
        connected = {c.target_parameter_name for c in result.incoming_connections}

        last_connected = -1
        for i in range(self._max_count):
            if self.slot_name(i) in connected:
                last_connected = i

        # Keep exactly one empty slot visible after the last connected one.
        show_up_to = min(last_connected + 1, self._max_count - 1)
        for i in range(self._max_count):
            name = self.slot_name(i)
            param = self._node.get_parameter_by_name(name)
            if param is None:
                continue
            if i > show_up_to:
                param.allowed_modes = {ParameterMode.OUTPUT}
                self._node.hide_parameter_by_name(name)
            else:
                param.allowed_modes = {ParameterMode.INPUT, ParameterMode.OUTPUT}
                self._node.show_parameter_by_name(name)
