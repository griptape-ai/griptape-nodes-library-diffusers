"""Shared helpers for managing parameter connections via the engine connection registry."""

from __future__ import annotations

from collections.abc import Iterable

from griptape_nodes.exe_types.core_types import Parameter, ParameterList
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.connection_events import (
    DeleteConnectionRequest,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


def delete_parameter_list_child_connections(node: BaseNode, parameter_lists: Iterable[ParameterList]) -> None:
    """Delete connections targeting children of the given ParameterLists.

    RemoveParameterFromNodeRequest only deletes connections matching the parent name;
    ParameterList children use UUID-suffixed names and are orphaned on parent removal.
    Call this BEFORE removing a ParameterList parent.
    """
    child_names: set[str] = set()
    for parameter_list in parameter_lists:
        child_names.update(
            child.name for child in parameter_list.find_elements_by_type(Parameter, find_recursively=False)
        )
    if not child_names:
        return

    list_result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=node.name))
    if not isinstance(list_result, ListConnectionsForNodeResultSuccess):
        return

    for incoming in list_result.incoming_connections:
        if incoming.target_parameter_name not in child_names:
            continue
        GriptapeNodes.handle_request(
            DeleteConnectionRequest(
                source_node_name=incoming.source_node_name,
                source_parameter_name=incoming.source_parameter_name,
                target_node_name=node.name,
                target_parameter_name=incoming.target_parameter_name,
            )
        )

    for outgoing in list_result.outgoing_connections:
        if outgoing.source_parameter_name not in child_names:
            continue
        GriptapeNodes.handle_request(
            DeleteConnectionRequest(
                source_node_name=node.name,
                source_parameter_name=outgoing.source_parameter_name,
                target_node_name=outgoing.target_node_name,
                target_parameter_name=outgoing.target_parameter_name,
            )
        )
