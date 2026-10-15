"""Read engine configuration in a way that works wherever a node runs.

`GriptapeNodes.ConfigManager()` raises during node execution in a worker, deliberately: a worker holds
its own copy of that state, and a read answered locally would be silently wrong rather than loudly
absent. A request routes to whichever process owns the state, so it is correct in both places.

Node code should reach for these rather than the manager. The same applies to any other manager: the
guard covers all of them except `StaticFilesManager`, whose answers a worker can give because it shares
the workspace on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from griptape_nodes.retained_mode.events.config_events import (
    GetConfigValueRequest,
    GetConfigValueResultSuccess,
    GetWorkspaceRequest,
    GetWorkspaceResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("modular_diffusers_nodes_library")


def get_config_value(category_and_key: str, *, default: Any = None) -> Any:
    """The configured value for `category_and_key`, or `default` if it is not set.

    A missing key is not an error: these are optional settings, and a node asking whether a feature is
    enabled wants an answer rather than an exception.
    """
    result = GriptapeNodes.handle_request(GetConfigValueRequest(category_and_key=category_and_key))
    if isinstance(result, GetConfigValueResultSuccess):
        return result.value
    logger.debug("Config key '%s' is not set; using %r.", category_and_key, default)
    return default


def get_workspace_path() -> Path:
    """The workspace directory.

    Raises:
        RuntimeError: if the engine cannot report a workspace. Unlike a missing setting, there is no
            sensible stand-in -- a path guessed here would write a node's output somewhere the user
            will not find it.
    """
    result = GriptapeNodes.handle_request(GetWorkspaceRequest())
    if isinstance(result, GetWorkspaceResultSuccess):
        return Path(result.workspace_path)
    msg = f"Attempted to read the workspace directory. Failed because the engine reported: {result}."
    raise RuntimeError(msg)
