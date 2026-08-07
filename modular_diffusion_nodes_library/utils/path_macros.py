from __future__ import annotations

import logging
from pathlib import Path

from griptape_nodes.common.macro_parser import MacroSyntaxError, ParsedMacro
from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
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


def resolve_path_to_macro(value: str) -> str:
    """Resolve an absolute path to its project-relative macro form (e.g. `{project_dir}/...`).

    Returns `value` unchanged if it already contains a macro (`{`), cannot be resolved
    on disk, or the engine cannot map it to a project directory.
    """
    if not value or "{" in value:
        return value

    resolved = Path(value).resolve()
    result = GriptapeNodes.handle_request(AttemptMapAbsolutePathToProjectRequest(absolute_path=resolved))

    if isinstance(result, AttemptMapAbsolutePathToProjectResultSuccess) and result.mapped_path is not None:
        return result.mapped_path

    return value
