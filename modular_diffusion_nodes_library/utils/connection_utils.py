"""Shared helpers for managing parameter connections via the engine connection registry."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.connection_events import (
    DeleteConnectionRequest,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
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
    # Asked as a request rather than off `FlowManager`: connections belong to the orchestrator, and a
    # worker reaching for that manager raises. This runs during hydration, which is worker-side, so the
    # manager route would fail whenever a component value arrives there.
    result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=node.name))
    if not isinstance(result, ListConnectionsForNodeResultSuccess):
        logger.warning(
            "%s: could not list connections for '%s', so none were dropped. The engine reported: %s",
            node.name,
            parameter_name,
            result,
        )
        return 0

    outgoing = [
        connection for connection in result.outgoing_connections if connection.source_parameter_name == parameter_name
    ]

    for connection in outgoing:
        target_node_name = connection.target_node_name
        target_parameter_name = connection.target_parameter_name

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

    return len(outgoing)
