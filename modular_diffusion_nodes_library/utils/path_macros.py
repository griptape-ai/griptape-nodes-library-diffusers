from __future__ import annotations

import logging

from griptape_nodes.common.macro_parser import MacroSyntaxError, ParsedMacro
from griptape_nodes.retained_mode.events.project_events import (
    GetPathForMacroRequest,
    GetPathForMacroResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("modular_diffusers_nodes_library")


def expand_path_macros(value: str) -> str:
    """Expand workflow macros (e.g. `{project_dir}`) in a raw path string.

    Returns `value` unchanged if it contains no `{` (no macro to resolve), if it
    cannot be parsed, or if the engine cannot resolve it against the current
    project. Callers surface the resulting filesystem error themselves.
    """
    if not value or "{" not in value:
        return value

    try:
        parsed = ParsedMacro(value)
    except MacroSyntaxError as err:
        logger.warning(
            "Attempted to parse macro template '%s'. Failed because '%s'. Falling back to raw value.",
            value,
            err,
        )
        return value

    result = GriptapeNodes.handle_request(GetPathForMacroRequest(parsed_macro=parsed, variables={}))

    if isinstance(result, GetPathForMacroResultSuccess):
        return str(result.absolute_path)

    logger.warning(
        "Attempted to expand path macros for value '%s'. Failed because '%s'. Falling back to raw value.",
        value,
        result.result_details,
    )
    return value
