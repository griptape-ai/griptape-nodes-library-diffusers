"""Shared helpers for managing parameter connections via the engine connection registry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


def delete_container_parameter_childs(node: BaseNode, containers: Iterable[Any]) -> None:
    """Remove child parameters and their connections from ParameterList or ParameterGroup containers."""
    for container in containers:
        for child in list(container.find_elements_by_type(Parameter, find_recursively=False)):
            GriptapeNodes.handle_request(RemoveParameterFromNodeRequest(parameter_name=child.name, node_name=node.name))
