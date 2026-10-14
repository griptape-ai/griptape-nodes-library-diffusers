"""Extract memory-estimation inputs from workflow source without executing it."""

from __future__ import annotations

import ast
import io
import pickle
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentSourceType,
    HFRepoRef,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep
from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec

_ALLOWED_GLOBALS = {
    "modular_diffusion_nodes_library.artifact_utils.pipeline_artifact": {
        "DiffusionPipelineArtifact": DiffusionPipelineArtifact,
        "ControlNetDiffusionPipelineArtifact": ControlNetDiffusionPipelineArtifact,
    },
    "modular_diffusion_nodes_library.artifact_utils.component_artifact": {
        "ModelComponentArtifact": ModelComponentArtifact,
        "ComponentSourceType": ComponentSourceType,
        "HFRepoRef": HFRepoRef,
    },
    "modular_diffusion_nodes_library.utils.lora_apply_utils": {
        "LoraPipelineRuntimeAdapterStep": LoraPipelineRuntimeAdapterStep,
    },
    "modular_diffusion_nodes_library.utils.lora_spec": {
        "LoraSpec": LoraSpec,
    },
    "enum": {"Enum": Enum, "StrEnum": StrEnum},
}


class WorkflowExtractionError(ValueError):
    """Raised when a workflow source does not contain supported static data."""


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        module_globals = _ALLOWED_GLOBALS.get(module)
        if module_globals is None or name not in module_globals:
            msg = f"Unsupported pickle global '{module}.{name}' in workflow source."
            raise WorkflowExtractionError(msg)
        return module_globals[name]


def _restricted_loads(payload: bytes) -> Any:
    try:
        return _RestrictedUnpickler(io.BytesIO(payload)).load()
    except WorkflowExtractionError:
        raise
    except (EOFError, pickle.PickleError, ValueError, TypeError, ImportError, AttributeError) as error:
        msg = f"Could not decode a workflow pickle payload: {error}"
        raise WorkflowExtractionError(msg) from error


def _literal_pickle_bytes(node: ast.AST) -> bytes | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != "loads" or len(node.args) != 1:
        return None
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "pickle":
        return None
    payload = node.args[0]
    if not isinstance(payload, ast.Constant) or not isinstance(payload.value, bytes):
        return None
    return payload.value


def _keyword_value(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _string_or_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _request_name(request: ast.Call) -> str | None:
    if isinstance(request.func, ast.Name):
        return request.func.id
    if isinstance(request.func, ast.Attribute):
        return request.func.attr
    return None


def _node_type_from_create_call(call: ast.Call) -> str | None:
    return _string_or_name(_keyword_value(call, "node_type"))


def _unique_key_from_subscript(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    if node.value.id != "top_level_unique_values_dict":
        return None
    index = node.slice
    if not isinstance(index, ast.Constant) or not isinstance(index.value, str):
        return None
    return index.value


def _relevant_unique_keys(tree: ast.Module) -> set[str]:
    keys: set[str] = set()
    for statement in ast.walk(tree):
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Await):
            continue
        call = statement.value.value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "ahandle_request"
        ):
            continue
        request = call.args[0] if len(call.args) == 1 else None
        if not isinstance(request, ast.Call) or _request_name(request) != "SetParameterValueRequest":
            continue
        parameter_name = _string_or_name(_keyword_value(request, "parameter_name"))
        value_key = _unique_key_from_subscript(_keyword_value(request, "value"))
        if value_key is None:
            continue
        if parameter_name == "pipeline":
            keys.add(value_key)
            continue
        if parameter_name in {"width", "height", "num_frames"}:
            keys.add(value_key)
            continue
        if parameter_name in {"lora_pipeline", "loras", "pipeline"}:
            keys.add(value_key)
    return keys


def _decode_unique_values(tree: ast.Module, relevant_keys: set[str]) -> dict[str, Any]:
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "top_level_unique_values_dict" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise WorkflowExtractionError("Expected exactly one top_level_unique_values_dict assignment.")

    value = assignments[0].value
    if not isinstance(value, ast.Dict):
        raise WorkflowExtractionError("top_level_unique_values_dict must be a dictionary literal.")

    decoded: dict[str, Any] = {}
    for key_node, value_node in zip(value.keys, value.values, strict=True):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise WorkflowExtractionError("Workflow unique-value keys must be string literals.")
        payload = _literal_pickle_bytes(value_node)
        if payload is None:
            if key_node.value in relevant_keys:
                raise WorkflowExtractionError(f"Workflow value '{key_node.value}' is not a literal pickle payload.")
            continue
        if key_node.value in relevant_keys:
            decoded[key_node.value] = _restricted_loads(payload)
    return decoded


def _node_name_from_create_assignment(statement: ast.Assign) -> tuple[str, str] | None:
    request_result: ast.AST = statement.value
    if isinstance(request_result, ast.Attribute) and request_result.attr == "node_name":
        request_result = request_result.value
    if not isinstance(request_result, ast.Await):
        return None
    call = request_result.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return None
    if call.func.attr != "ahandle_request" or len(call.args) != 1:
        return None
    request = call.args[0]
    if not isinstance(request, ast.Call):
        return None
    node_type = _node_type_from_create_call(request)
    targets = [target.id for target in statement.targets if isinstance(target, ast.Name)]
    if node_type is None or len(targets) != 1:
        return None
    return targets[0], node_type


@dataclass(frozen=True)
class WorkflowDimensionAssignment:
    node_name: str
    width: int | None
    height: int | None
    num_frames: int | None


@dataclass(frozen=True)
class WorkflowBuilder:
    """One pipeline and raw-dimension estimate target extracted from a workflow."""

    artifact: DiffusionPipelineArtifact
    dimensions: WorkflowDimensionAssignment


@dataclass(frozen=True)
class WorkflowMemoryInput:
    """All source-only inputs needed to perform one memory estimate."""

    workflow_path: Path
    builders: list[WorkflowBuilder]


def _integer_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _extract_request_values(
    tree: ast.Module,
    values: dict[str, Any],
    node_types: dict[str, str],
) -> tuple[list[WorkflowDimensionAssignment], dict[str, DiffusionPipelineArtifact]]:
    dimensions: dict[str, dict[str, int | None]] = {}
    pipelines: dict[str, DiffusionPipelineArtifact] = {}
    for statement in ast.walk(tree):
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Await):
            continue
        call = statement.value.value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "ahandle_request"
        ):
            continue
        request = call.args[0] if len(call.args) == 1 else None
        if not isinstance(request, ast.Call) or _request_name(request) != "SetParameterValueRequest":
            continue
        node_name = _string_or_name(_keyword_value(request, "node_name"))
        parameter_name = _string_or_name(_keyword_value(request, "parameter_name"))
        value_key = _unique_key_from_subscript(_keyword_value(request, "value"))
        if node_name is None or parameter_name is None or value_key is None:
            continue

        is_noise_dimension = node_types.get(node_name) in {"NoiseLatentNode", "EmptyLatentNode"} and parameter_name in {
            "width",
            "height",
            "num_frames",
        }
        is_generation_pipeline = parameter_name == "pipeline" and node_types.get(node_name) in {
            "NoiseLatentNode",
            "EmptyLatentNode",
            "DiffusionPipelineGenerateLatentNode",
            "PipelineMemoryEstimateNode",
        }
        if not is_noise_dimension and not is_generation_pipeline:
            continue
        if value_key not in values:
            raise WorkflowExtractionError(f"Workflow value '{value_key}' was not decoded.")

        if is_noise_dimension:
            assignment = dimensions.setdefault(node_name, {"width": None, "height": None, "num_frames": None})
            assignment[parameter_name] = _integer_value(values[value_key])
        if is_generation_pipeline:
            value = values[value_key]
            if isinstance(value, DiffusionPipelineArtifact):
                pipelines[node_name] = value

    dimension_values = [
        WorkflowDimensionAssignment(node_name, item["width"], item["height"], item["num_frames"])
        for node_name, item in dimensions.items()
    ]
    return dimension_values, pipelines


def extract_workflow(path: Path) -> WorkflowMemoryInput:
    """Extract memory-estimation inputs from workflow source without executing it."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        msg = f"Attempted to read workflow '{path}'. Failed because {error}."
        raise WorkflowExtractionError(msg) from error

    values = _decode_unique_values(tree, _relevant_unique_keys(tree))
    node_types: dict[str, str] = {}
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Assign):
            created = _node_name_from_create_assignment(statement)
            if created is not None:
                node_types[created[0]] = created[1]

    dimensions, pipelines = _extract_request_values(tree, values, node_types)
    builders: list[WorkflowBuilder] = []
    for node_name, artifact in pipelines.items():
        matches = [item for item in dimensions if item.node_name == node_name]
        if len(matches) == 1:
            builders.append(WorkflowBuilder(artifact, matches[0]))

    if not builders:
        raise WorkflowExtractionError(
            f"Workflow '{path}' has no unambiguous pipeline and NoiseLatentNode dimension assignments."
        )
    return WorkflowMemoryInput(path, _deduplicate_builders(builders))


def _deduplicate_builders(builders: list[WorkflowBuilder]) -> list[WorkflowBuilder]:
    unique: dict[tuple[str, str | None, int | None, int | None, int | None], WorkflowBuilder] = {}
    for builder in builders:
        dimensions = builder.dimensions
        key = (
            builder.artifact.pipeline_name,
            builder.artifact.config_hash,
            dimensions.width,
            dimensions.height,
            dimensions.num_frames,
        )
        unique.setdefault(key, builder)
    return list(unique.values())
