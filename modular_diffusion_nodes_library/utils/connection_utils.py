"""Shared helpers for managing parameter connections via the engine connection registry."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.connection_events import DeleteConnectionRequest
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("modular_diffusers_nodes_library")


def delete_container_parameter_childs(node: BaseNode, containers: Iterable[Any]) -> None:
    """Remove child parameters and their connections from ParameterList or ParameterGroup containers."""
    for container in containers:
        for child in list(container.find_elements_by_type(Parameter, find_recursively=False)):
            GriptapeNodes.handle_request(RemoveParameterFromNodeRequest(parameter_name=child.name, node_name=node.name))


def drop_outgoing_connections(node: BaseNode, parameter_name: str, *, reason: str | None = None) -> int:
    """Drop all outgoing connections from a specific output parameter on a node.

    Args:
        node: The node whose outgoing connections should be dropped.
        parameter_name: The name of the output parameter to drop connections from.
        reason: Optional human-readable reason for the drop, included in log messages.

    Returns:
        The number of connections dropped.
    """
    connections = GriptapeNodes.FlowManager().get_connections()
    outgoing_for_node = connections.outgoing_index.get(node.name, {})
    connection_ids = list(outgoing_for_node.get(parameter_name, []))

    for connection_id in connection_ids:
        connection = connections.connections[connection_id]
        target_node_name = connection.target_node.name
        target_parameter_name = connection.target_parameter.name

        GriptapeNodes.handle_request(
            DeleteConnectionRequest(
                source_node_name=node.name,
                source_parameter_name=parameter_name,
                target_node_name=target_node_name,
                target_parameter_name=target_parameter_name,
            )
        )
        if reason:
            logger.info(
                "%s: Dropped connection %s.%s -> %s.%s (%s)",
                node.name,
                node.name,
                parameter_name,
                target_node_name,
                target_parameter_name,
                reason,
            )
        else:
            logger.info(
                "%s: Dropped connection %s.%s -> %s.%s",
                node.name,
                node.name,
                parameter_name,
                target_node_name,
                target_parameter_name,
            )

    return len(connection_ids)
