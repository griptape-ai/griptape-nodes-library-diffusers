"""Shared, dependency-free codec helpers for pipeline recipes.

This module owns the cross-cutting concerns of recipe (de)serialisation — the
schema version, value normalization/macro conversion, the component-override registry,
and the small value validators — so that the per-class ``to_recipe_dict`` /
``from_recipe_dict`` methods on the artifact classes can stay colocated with
their fields without importing each other. It must not import any artifact
module, to keep the dependency graph acyclic.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros, resolve_path_to_macro

PIPELINE_RECIPE_SCHEMA_VERSION = 1

# Populated by ComponentArtifact subclasses at import time (see __init_subclass__).
COMPONENT_DESERIALIZERS: dict[str, Callable[[dict[str, Any]], Any]] = {}


@runtime_checkable
class RecipeSerializable(Protocol):
    """Any object that knows how to render itself as a recipe-safe dict."""

    def to_recipe_dict(self) -> dict[str, Any]: ...


def recipe_safe(value: Any) -> Any:
    """Recursively convert a value into recipe-safe primitives.

    Objects implementing ``to_recipe_dict`` (component artifacts, repo refs) are
    delegated to; absolute string paths and ``Path`` values are mapped to project
    macros where possible.
    """
    if isinstance(value, RecipeSerializable):
        return value.to_recipe_dict()
    if value is None or isinstance(value, str | int | float | bool):
        if isinstance(value, str) and Path(value).is_absolute():
            return resolve_path_to_macro(value)
        return value
    if isinstance(value, Path):
        return resolve_path_to_macro(str(value))
    if isinstance(value, list | tuple):
        return [recipe_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): recipe_safe(item) for key, item in value.items()}

    msg = f"Pipeline recipe contains an unsupported value: {type(value).__name__}"
    raise TypeError(msg)


def resolve_recipe_value(value: Any) -> Any:
    """Recursively reverse :func:`recipe_safe`: expand macros and rebuild overrides."""
    if isinstance(value, dict) and value.get("type") == "component_override":
        artifact_type = require_string(value.get("artifact_type"), "Component override artifact type")
        deserializer = COMPONENT_DESERIALIZERS.get(artifact_type)
        if deserializer is None:
            msg = f"Unsupported component override artifact type: {artifact_type}"
            raise ValueError(msg)
        return deserializer(value)
    if isinstance(value, str):
        return expand_path_macros(value)
    if isinstance(value, list):
        return [resolve_recipe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): resolve_recipe_value(item) for key, item in value.items()}
    return value


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    msg = f"{label} must be an object."
    raise ValueError(msg)


def require_string(value: Any, label: str) -> str:
    if isinstance(value, str) and value:
        return value
    msg = f"{label} must be a non-empty string."
    raise ValueError(msg)


def optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"{label} must be a string or null."
    raise ValueError(msg)


def require_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    msg = f"{label} must be a boolean."
    raise ValueError(msg)


def require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"{label} must be a list of strings."
        raise ValueError(msg)
    return value


def require_loras(value: Any) -> dict[str, float]:
    loras = require_dict(value, "Pipeline LoRAs")
    if not all(isinstance(path, str) and isinstance(weight, int | float) for path, weight in loras.items()):
        msg = "Pipeline LoRAs must map paths to numeric weights."
        raise ValueError(msg)
    return {path: float(weight) for path, weight in loras.items()}


# Generic recipe conversion only transforms dictionary values, while fused
# LoRA paths are stored as dictionary keys and must be handled explicitly.
def _map_lora_paths(
    loras: dict[str, float],
    map_path: Callable[[str], str],
    *,
    collision_description: str,
) -> dict[str, float]:
    resolved: dict[str, float] = {}
    source_paths: dict[str, str] = {}
    for path, weight in loras.items():
        mapped_path = map_path(path)
        if mapped_path in resolved:
            msg = (
                f"Pipeline LoRA paths {source_paths[mapped_path]!r} and {path!r} "
                f"both {collision_description} {mapped_path!r}."
            )
            raise ValueError(msg)
        resolved[mapped_path] = float(weight)
        source_paths[mapped_path] = path
    return resolved


def recipe_safe_loras(value: dict[str, float]) -> dict[str, float]:
    """Map fused LoRA path keys to portable recipe macros."""
    return _map_lora_paths(
        value,
        resolve_path_to_macro,
        collision_description="map to recipe path",
    )


def resolve_recipe_loras(value: Any) -> dict[str, float]:
    """Validate fused LoRAs and expand path macros stored as dictionary keys."""
    return _map_lora_paths(
        require_loras(value),
        expand_path_macros,
        collision_description="expand to",
    )


def require_runtime_loras(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        msg = "Pipeline runtime LoRAs must be a list."
        raise ValueError(msg)
    for entry in value:
        if not isinstance(entry, dict):
            msg = "Pipeline runtime LoRAs must be objects with 'path' and 'weight'."
            raise ValueError(msg)
        if not isinstance(entry.get("path"), str) or not isinstance(entry.get("weight"), int | float):
            msg = "Pipeline runtime LoRAs must map a string 'path' to a numeric 'weight'."
            raise ValueError(msg)
    return value


def optional_dict(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return require_dict(value, label)
