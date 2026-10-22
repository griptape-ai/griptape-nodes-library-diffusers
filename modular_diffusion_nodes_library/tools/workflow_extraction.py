"""Extract memory-estimation inputs from workflow source without executing it."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import pickle
import re
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
    BasePipelineIdentity,
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.scheduler_component_artifact import SchedulerComponentArtifact
from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import TextEncoderComponentArtifact
from modular_diffusion_nodes_library.artifact_utils.tokenizer_component_artifact import TokenizerComponentArtifact
from modular_diffusion_nodes_library.component_loading.component_slots import SLOT_DISPLAY_NAMES
from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import MODULAR_PIPELINE_TYPE_PROVIDER_MAP
from modular_diffusion_nodes_library.parameters.providers import Provider
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep
from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec, normalize_loras
from modular_diffusion_nodes_library.utils.pipeline_runtime_adapter_step import PipelineRuntimeAdapterStep

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
    "modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact": {
        "TextEncoderComponentArtifact": TextEncoderComponentArtifact,
    },
    "modular_diffusion_nodes_library.artifact_utils.tokenizer_component_artifact": {
        "TokenizerComponentArtifact": TokenizerComponentArtifact,
    },
    "modular_diffusion_nodes_library.artifact_utils.scheduler_component_artifact": {
        "SchedulerComponentArtifact": SchedulerComponentArtifact,
    },
    "modular_diffusion_nodes_library.utils.lora_apply_utils": {
        "LoraPipelineRuntimeAdapterStep": LoraPipelineRuntimeAdapterStep,
    },
    "modular_diffusion_nodes_library.utils.lora_spec": {
        "LoraSpec": LoraSpec,
    },
    "enum": {"Enum": Enum, "StrEnum": StrEnum},
}

NEW_WORKFLOW_SCHEMA_VERSION = (0, 21, 0)
PIPELINE_SETTING_NAMES = frozenset(
    {
        "provider",
        "pipeline_type",
        "model",
        "pipeline",
        "lora_pipeline",
        "memory_optimization_strategy",
        "attention_slicing",
        "vae_slicing",
        "vae_tiling",
        "transformer_layerwise_casting",
        "cpu_offload_strategy",
        "quantization_mode",
        "loras",
        "width",
        "height",
        "num_frames",
        "component_overrides",
        "component_output",
        "component",
        "source_type",
        "file_path",
        "config_source",
        "folder_path",
        "repo_id",
        "revision",
        "subfolder",
        "scheduler_class",
        "text_config",
        "control_nets",
        "controlnet_model",
        "weight",
        "trigger_phrase",
        "control_net",
    }
)
DIMENSION_PARAMETER_NAMES = frozenset({"width", "height", "num_frames"})
COMPONENT_ARTIFACT_TYPES: dict[str, type] = {
    "ModelComponentArtifact": ModelComponentArtifact,
    "TextEncoderComponentArtifact": TextEncoderComponentArtifact,
    "TokenizerComponentArtifact": TokenizerComponentArtifact,
    "SchedulerComponentArtifact": SchedulerComponentArtifact,
}

PIPELINE_NODE_TYPES = frozenset(
    {
        "LatentDiffusionPipelineBuilderNode",
        "NoiseLatentNode",
        "EmptyLatentNode",
        "DiffusionPipelineGenerateLatentNode",
        "PipelineMemoryEstimateNode",
    }
)


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
    node_metadata = _keyword_value(request, "node_metadata")
    if node_type is None and isinstance(node_metadata, ast.Dict):
        for key_node, value_node in zip(node_metadata.keys, node_metadata.values, strict=True):
            if (
                isinstance(key_node, ast.Constant)
                and key_node.value == "node_type"
                and isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, str)
            ):
                node_type = value_node.value
                break
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

    schema_version = _workflow_schema_version(source)
    if schema_version < NEW_WORKFLOW_SCHEMA_VERSION:
        return _extract_legacy_workflow(path, tree)
    return _extract_new_workflow(path, tree)


def _extract_legacy_workflow(path: Path, tree: ast.Module) -> WorkflowMemoryInput:
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


_SCHEMA_VERSION_PATTERN = re.compile(r'^# schema_version = "([0-9]+(?:\.[0-9]+){1,2})"$', re.MULTILINE)


def _workflow_schema_version(source: str) -> tuple[int, int, int]:
    match = _SCHEMA_VERSION_PATTERN.search(source)
    if match is None:
        raise WorkflowExtractionError("Workflow metadata is missing a valid schema_version.")
    parts = [int(part) for part in match.group(1).split(".")]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _extract_new_workflow(path: Path, tree: ast.Module) -> WorkflowMemoryInput:
    node_types: dict[str, str] = {}
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Assign):
            created = _node_name_from_create_assignment(statement)
            if created is not None:
                node_types[created[0]] = created[1]

    values_table = _literal_value_table(tree)
    parameter_values, edges = _collect_new_workflow_data(tree, values_table)
    node_types = _collect_node_types(tree)
    component_artifacts = _component_artifacts_by_node(parameter_values, node_types)
    controlnet_models = _controlnet_values_by_node(parameter_values, node_types)
    lora_values = _lora_values_by_node(parameter_values, node_types)
    builders: list[WorkflowBuilder] = []
    builder_nodes = [
        name for name, node_type in node_types.items() if node_type == "LatentDiffusionPipelineBuilderNode"
    ]
    for builder_node in builder_nodes:
        values = parameter_values.get(builder_node, {})
        if not {"provider", "pipeline_type"} <= values.keys():
            continue
        pipeline_type = values["pipeline_type"]
        provider = values["provider"]
        if pipeline_type == "FluxFillPipeline" and provider == "Flux":
            values["model"] = "black-forest-labs/FLUX.1-Fill-dev"
        builder_values = dict(values)
        builder_values["loras"] = _resolve_fused_loras(builder_node, edges, lora_values, values.get("loras", []))
        runtime_loras = _resolve_runtime_loras(builder_node, edges, lora_values, node_types, parameter_values)
        component_overrides = _resolve_component_overrides(builder_node, edges, component_artifacts)
        artifact = _build_new_workflow_artifact(builder_values, component_overrides)
        controlnet_nodes = _connected_controlnet_nodes(builder_node, node_types, edges)
        connected_controlnets = _connected_controlnet_models(controlnet_nodes, edges, controlnet_models)
        if connected_controlnets:
            artifact = _add_controlnet_artifact(artifact, connected_controlnets)
        artifact = _add_runtime_loras(artifact, runtime_loras)

        dimension_nodes = _connected_dimension_nodes(builder_node, node_types, edges)
        if not dimension_nodes:
            pipeline_nodes = _connected_pipeline_nodes(builder_node, node_types, edges)
            dimensions = WorkflowDimensionAssignment(builder_node, None, None, None)
            if any(
                node_types.get(target_node) in {"DiffusionPipelineGenerateLatentNode", "PipelineMemoryEstimateNode"}
                for target_node in pipeline_nodes
            ):
                builders.append(WorkflowBuilder(artifact, dimensions))
        else:
            for dimension_node in dimension_nodes:
                dimensions = parameter_values.get(dimension_node, {})
                assignment = WorkflowDimensionAssignment(
                    dimension_node,
                    _integer_value(dimensions.get("width")),
                    _integer_value(dimensions.get("height")),
                    _integer_value(dimensions.get("num_frames")),
                )
                builders.append(WorkflowBuilder(artifact, assignment))

    if not builders:
        raise WorkflowExtractionError(
            f"Workflow '{path}' has no supported pipeline builder connected to a NoiseLatentNode or EmptyLatentNode."
        )
    return WorkflowMemoryInput(path, _deduplicate_builders(builders))


def _literal_value_table(tree: ast.Module) -> dict[str, ast.expr]:
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "top_level_unique_values_dict" for target in node.targets)
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise WorkflowExtractionError("Expected one literal top_level_unique_values_dict in new-schema workflow.")

    table: dict[str, ast.expr] = {}
    for key_node, value_node in zip(assignments[0].value.keys, assignments[0].value.values, strict=True):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise WorkflowExtractionError("Workflow unique-value keys must be string literals.")
        if value_node is not None:
            table[key_node.value] = value_node
    return table


def _resolve_literal_expression(node: ast.expr, values_table: dict[str, ast.expr]) -> Any:
    if (
        isinstance(node, ast.Call)
        and _request_name(node) == "decode_value"
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]

    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id != "top_level_unique_values_dict":
            raise WorkflowExtractionError("Workflow value references an unsupported table.")
        key_node = node.slice
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise WorkflowExtractionError("Workflow unique-value reference must use a string literal key.")
        value_node = values_table.get(key_node.value)
        if value_node is None:
            raise WorkflowExtractionError(f"Workflow value '{key_node.value}' was not found in its value table.")
        try:
            return ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
            raise WorkflowExtractionError(
                f"Workflow value '{key_node.value}' is not a supported literal value."
            ) from error

    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
        raise WorkflowExtractionError("Workflow parameter value is not a supported literal expression.") from error


def _node_context_regions(tree: ast.Module) -> list[tuple[int, int, str | None]]:
    regions: list[tuple[int, int, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        context = node.items[0].context_expr
        if (
            not isinstance(context, ast.Call)
            or not isinstance(context.func, ast.Attribute)
            or context.func.attr != "node"
        ):
            continue
        if len(context.args) != 1:
            continue
        node_var = context.args[0].id if isinstance(context.args[0], ast.Name) else None
        regions.append((node.lineno, node.end_lineno or node.lineno, node_var))
    return regions


def _node_context_for_line(regions: list[tuple[int, int, str | None]], line: int) -> str | None:
    matches = [region for region in regions if region[0] <= line <= region[1]]
    if not matches:
        return None
    return max(matches, key=lambda region: region[0])[2]


def _request_calls(tree: ast.Module, request_name: str) -> list[ast.Call]:
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _request_name(node) == request_name]
    return sorted(calls, key=lambda call: (call.lineno, call.col_offset))


def _collect_new_workflow_data(
    tree: ast.Module, values_table: dict[str, ast.expr]
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, str, str]]]:
    regions = _node_context_regions(tree)
    parameter_values: dict[str, dict[str, Any]] = {}
    for request_name in ("AddParameterToNodeRequest", "SetParameterValueRequest"):
        for request in _request_calls(tree, request_name):
            parameter_name = _string_or_name(_keyword_value(request, "parameter_name"))
            if parameter_name not in PIPELINE_SETTING_NAMES:
                continue
            if request_name == "AddParameterToNodeRequest":
                node_name = _node_context_for_line(regions, request.lineno)
                value_node = _keyword_value(request, "default_value")
            else:
                if _string_or_name(_keyword_value(request, "is_output")) == "True":
                    continue
                node_name = _string_or_name(_keyword_value(request, "node_name"))
                output_node = _keyword_value(request, "is_output")
                if isinstance(output_node, ast.Constant) and output_node.value is True:
                    continue
                value_node = _keyword_value(request, "value")
            if node_name is None or not isinstance(value_node, ast.expr):
                continue
            try:
                value = _resolve_literal_expression(value_node, values_table)
            except WorkflowExtractionError:
                if parameter_name in DIMENSION_PARAMETER_NAMES:
                    continue
                raise
            parameter_values.setdefault(node_name, {})[parameter_name] = value

    edges: list[tuple[str, str, str, str]] = []
    for request in _request_calls(tree, "CreateConnectionRequest"):
        source_node = _string_or_name(_keyword_value(request, "source_node_name"))
        source_parameter = _string_or_name(_keyword_value(request, "source_parameter_name"))
        target_node = _string_or_name(_keyword_value(request, "target_node_name"))
        target_parameter = _string_or_name(_keyword_value(request, "target_parameter_name"))
        if source_node is None or source_parameter is None or target_node is None or target_parameter is None:
            continue
        edges.append((source_node, source_parameter, target_node, target_parameter))
    return parameter_values, edges


def _resolve_fused_loras(
    builder_node: str,
    edges: list[tuple[str, str, str, str]],
    lora_values: dict[str, dict[str, Any]],
    base_values: Any,
) -> list[Any]:
    result: list[Any] = []
    if isinstance(base_values, list):
        result.extend(value for value in base_values if value not in ([], {}))
    connected_nodes = [
        source_node
        for source_node, source_parameter, target_node, target_parameter in edges
        if target_node == builder_node
        and target_parameter == "loras"
        and source_parameter == "loras"
        and source_node in lora_values
    ]
    result.extend(lora_values[node_name] for node_name in dict.fromkeys(connected_nodes))
    return result


def _pipeline_path_nodes(
    source_node: str,
    node_types: dict[str, str],
    edges: list[tuple[str, str, str, str]],
) -> set[str]:
    reachable = {source_node}
    pending = [source_node]
    while pending:
        current = pending.pop()
        for edge_source, source_parameter, target_node, target_parameter in edges:
            if (
                edge_source == current
                and source_parameter in {"pipeline", "controlnet_pipeline", "lora_pipeline"}
                and target_parameter == "pipeline"
                and target_node not in reachable
            ):
                reachable.add(target_node)
                pending.append(target_node)
    return reachable


def _resolve_runtime_loras(
    builder_node: str,
    edges: list[tuple[str, str, str, str]],
    lora_values: dict[str, dict[str, Any]],
    node_types: dict[str, str],
    parameter_values: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    runtime: list[tuple[str, dict[str, Any]]] = []
    reachable = _pipeline_path_nodes(builder_node, node_types, edges)
    runtime_nodes = {node_name for node_name in reachable if node_types.get(node_name) == "LoraActivationPipelineNode"}
    for runtime_node in runtime_nodes:
        direct_loras = parameter_values.get(runtime_node, {}).get("loras", [])
        if isinstance(direct_loras, list):
            for raw_lora in direct_loras:
                try:
                    normalized = normalize_loras([raw_lora])
                except (TypeError, ValueError) as error:
                    raise WorkflowExtractionError(f"Could not reconstruct runtime LoRA: {error}") from error
                runtime.extend(
                    (runtime_node, {"path": item.path, "weight": item.weight}) for item in normalized.values()
                )
        for source_node, source_parameter, target_node, target_parameter in edges:
            if (
                target_node == runtime_node
                and target_parameter.startswith("loras_")
                and source_parameter == "loras"
                and source_node in lora_values
            ):
                runtime.append((runtime_node, lora_values[source_node]))
    return runtime


def _connected_controlnet_nodes(
    builder_node: str,
    node_types: dict[str, str],
    edges: list[tuple[str, str, str, str]],
) -> list[str]:
    nodes = [
        source_node
        for source_node, source_parameter, target_node, target_parameter in edges
        if target_node == builder_node
        and target_parameter == "pipeline"
        and source_parameter == "pipeline"
        and node_types.get(source_node) == "ControlNetDiffusionPipelineBuilderNode"
    ]
    return list(dict.fromkeys(nodes))


def _connected_pipeline_nodes(
    builder_node: str,
    node_types: dict[str, str],
    edges: list[tuple[str, str, str, str]],
) -> list[str]:
    nodes = [
        target_node
        for source_node, source_parameter, target_node, target_parameter in edges
        if source_node == builder_node
        and source_parameter == "pipeline"
        and target_parameter == "pipeline"
        and node_types.get(target_node)
        in {"NoiseLatentNode", "EmptyLatentNode", "DiffusionPipelineGenerateLatentNode", "PipelineMemoryEstimateNode"}
    ]
    return list(dict.fromkeys(nodes))


def _connected_dimension_nodes(
    builder_node: str,
    node_types: dict[str, str],
    edges: list[tuple[str, str, str, str]],
) -> list[str]:
    supported_dimension_types = {"NoiseLatentNode", "EmptyLatentNode"}
    targets = [
        target_node
        for source_node, source_parameter, target_node, target_parameter in edges
        if source_node == builder_node
        and source_parameter == "pipeline"
        and target_parameter == "pipeline"
        and node_types.get(target_node) in supported_dimension_types
    ]
    return list(dict.fromkeys(targets))


def _collect_node_types(tree: ast.Module) -> dict[str, str]:
    node_types: dict[str, str] = {}
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Assign):
            created = _node_name_from_create_assignment(statement)
            if created is not None:
                node_types[created[0]] = created[1]
    return node_types


def _controlnet_values_by_node(
    parameter_values: dict[str, dict[str, Any]], node_types: dict[str, str]
) -> dict[str, list[str]]:
    controlnet_models: dict[str, list[str]] = {}
    for node_name, node_type in node_types.items():
        if node_type != "ControlNetNode":
            continue
        model = parameter_values.get(node_name, {}).get("controlnet_model")
        if isinstance(model, str) and model:
            controlnet_models[node_name] = [_parse_model_selection(model)[0]]
    return controlnet_models


def _lora_values_by_node(
    parameter_values: dict[str, dict[str, Any]], node_types: dict[str, str]
) -> dict[str, dict[str, Any]]:
    loras: dict[str, dict[str, Any]] = {}
    for node_name, node_type in node_types.items():
        if node_type != "LoadLora":
            continue
        values = parameter_values.get(node_name, {})
        file_path = values.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue
        loras[node_name] = {
            "path": _expand_component_path(file_path),
            "weight": values.get("weight", 1.0),
            "trigger_phrase": values.get("trigger_phrase") or None,
        }
    return loras


def _component_artifacts_by_node(
    parameter_values: dict[str, dict[str, Any]], node_types: dict[str, str]
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for node_name, values in parameter_values.items():
        node_type = node_types.get(node_name)
        if node_type == "LoadComponent":
            artifact = _model_component_from_values(values)
        elif node_type == "LoadSchedulerComponent":
            artifact = _scheduler_component_from_values(values)
        else:
            continue
        if artifact is not None:
            artifacts[node_name] = artifact
    return artifacts


def _component_slot(values: dict[str, Any]) -> str | None:
    component_display_name = values.get("component")
    if not isinstance(component_display_name, str):
        return None
    for slot, display_name in SLOT_DISPLAY_NAMES.items():
        if component_display_name == display_name:
            return slot
    return None


def _model_component_from_values(values: dict[str, Any]) -> ModelComponentArtifact | None:
    slot = _component_slot(values)
    source_type = values.get("source_type")
    if slot is None:
        return None

    kind = re.sub(r"_\d+$", "", slot)
    if kind == "tokenizer":
        artifact_class = TokenizerComponentArtifact
    elif kind == "text_encoder":
        artifact_class = TextEncoderComponentArtifact
    else:
        artifact_class = ModelComponentArtifact

    if source_type == "Local Folder":
        folder_path = values.get("folder_path")
        if not isinstance(folder_path, str) or not folder_path:
            return None
        folder_path = _expand_component_path(folder_path)
        return artifact_class(
            load_id=_component_load_id("local_folder", slot, folder_path, "", "", ""),
            source_type=ComponentSourceType.LOCAL_DIR,
            component=slot,
            file_path=_expand_component_path(folder_path),
        )

    if source_type == "HuggingFace Repo":
        repo_id = values.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            return None
        revision = values.get("revision") or "main"
        subfolder = values.get("subfolder") or ""
        if not isinstance(revision, str) or not isinstance(subfolder, str):
            return None
        return artifact_class(
            load_id=_component_load_id("hf_repo", slot, "", "", repo_id, revision, subfolder),
            source_type=ComponentSourceType.HF_REPO,
            component=slot,
            repo_ref=HFRepoRef(repo_id=repo_id, revision=revision, subfolder=subfolder),
        )

    if source_type == "Single File" and kind != "tokenizer":
        file_path = values.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            return None
        config_source = values.get("config_source")
        if not isinstance(config_source, str) or not config_source:
            config_source = None
        file_path = _expand_component_path(file_path)
        if config_source is not None:
            config_source = _expand_component_path(config_source)
        return artifact_class(
            load_id=_component_load_id("single_file", slot, file_path, config_source or "", "", ""),
            source_type=ComponentSourceType.SINGLE_FILE,
            component=slot,
            file_path=file_path,
            config_source=config_source,
        )
    return None


def _scheduler_component_from_values(values: dict[str, Any]) -> SchedulerComponentArtifact | None:
    scheduler_class = values.get("scheduler_class")
    source_type = values.get("source_type")
    if not isinstance(scheduler_class, str) or not scheduler_class:
        return None

    if source_type == "Text Config":
        raw_config = values.get("text_config")
        config: dict[str, Any] | None = None
        if isinstance(raw_config, str) and raw_config:
            try:
                config = json.loads(raw_config)
            except json.JSONDecodeError as error:
                raise WorkflowExtractionError(f"Could not parse scheduler text_config: {error}") from error
            if not isinstance(config, dict):
                raise WorkflowExtractionError("Scheduler text_config must be a JSON object.")
        if not config:
            config = None
        text_config = json.dumps(config, sort_keys=True) if config is not None else ""
        return SchedulerComponentArtifact(
            load_id=_component_load_id("text_config", "scheduler", "", "", "", "", text_config),
            source_type=ComponentSourceType.RAW_CONFIG,
            component="scheduler",
            scheduler_class=scheduler_class,
            text_config=config,
        )

    if source_type == "HuggingFace Repo":
        repo_id = values.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            return None
        revision = values.get("revision") or "main"
        subfolder = values.get("subfolder") or "scheduler"
        if not isinstance(revision, str) or not isinstance(subfolder, str):
            return None
        return SchedulerComponentArtifact(
            load_id=_component_load_id("hf_repo", "scheduler", "", "", repo_id, revision, subfolder),
            source_type=ComponentSourceType.HF_REPO,
            component="scheduler",
            scheduler_class=scheduler_class,
            repo_ref=HFRepoRef(repo_id=repo_id, revision=revision, subfolder=subfolder),
        )

    if source_type == "Local Path":
        config_path = values.get("config_path")
        if not isinstance(config_path, str) or not config_path:
            return None
        config_path = _expand_component_path(config_path)
        return SchedulerComponentArtifact(
            load_id=_component_load_id("local_path", "scheduler", config_path, "", "", ""),
            source_type=ComponentSourceType.LOCAL_DIR,
            component="scheduler",
            scheduler_class=scheduler_class,
            config_source=config_path,
        )
    return None


def _component_load_id(
    source_type: str,
    component: str,
    path: str,
    config_source: str,
    repo_id: str,
    revision: str,
    subfolder: str = "",
    text_config: str = "",
) -> str:
    payload = "|".join([source_type, component, path, config_source, repo_id, revision, subfolder, text_config])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _expand_component_path(value: str) -> str:
    if "{" not in value:
        return value
    macro_match = re.match(r"^\{workspace_dir\}(.*)$", value)
    if macro_match is None:
        return value
    workspace = os.environ.get("GRIPTAPE_WORKSPACE_DIR")
    if workspace is None:
        return value
    return str(Path(workspace) / macro_match.group(1).lstrip("/\\"))


def _resolve_component_overrides(
    builder_node: str,
    edges: list[tuple[str, str, str, str]],
    component_artifacts: dict[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for source_node, source_parameter, target_node, target_parameter in edges:
        if target_node != builder_node or source_parameter != "component_output":
            continue
        if not target_parameter.startswith("component_"):
            continue
        artifact = component_artifacts.get(source_node)
        if artifact is not None:
            overrides[target_parameter.removeprefix("component_")] = artifact
    return overrides


def _build_new_workflow_artifact(
    values: dict[str, Any],
    component_overrides: dict[str, Any],
) -> DiffusionPipelineArtifact:
    provider_value = values["provider"]
    pipeline_type = values["pipeline_type"]
    model_value = values.get("model")
    if (
        not isinstance(provider_value, str)
        or not provider_value
        or not isinstance(pipeline_type, str)
        or not pipeline_type
    ):
        raise WorkflowExtractionError("Pipeline builder provider and pipeline_type must be non-empty strings.")

    try:
        provider = Provider(provider_value)
        pipeline_type_parameters = MODULAR_PIPELINE_TYPE_PROVIDER_MAP[provider]
        parameters_class = pipeline_type_parameters.get_pipeline_type_dict()[pipeline_type]
    except (ValueError, KeyError) as error:
        raise WorkflowExtractionError(
            f"Unsupported workflow pipeline selection provider='{provider_value}' pipeline_type='{pipeline_type}'."
        ) from error

    if not isinstance(model_value, str) or not model_value:
        slots = parameters_class.get_component_slots()
        if (
            not parameters_class.supports_build_from_overrides_only()
            or not slots
            or not set(slots) <= set(component_overrides)
        ):
            raise WorkflowExtractionError(
                "Pipeline builder model is required unless all required components are overridden."
            )
        repo_id = ""
        revision = ""
        build_data = {
            "_component_overrides": component_overrides,
            "_pipeline_cls": parameters_class.pipeline_cls(),
            "_all_overrides": True,
        }
        return _build_artifact(
            parameters_class,
            pipeline_type,
            provider_value,
            "",
            repo_id,
            revision,
            values,
            build_data,
            component_overrides,
        )

    repo_id, pinned_revision = _parse_model_selection(model_value)
    revision = pinned_revision or "main"
    build_data = parameters_class.build_data(repo_id, revision)
    return _build_artifact(
        parameters_class,
        pipeline_type,
        provider_value,
        model_value,
        repo_id,
        revision,
        values,
        build_data,
        component_overrides,
    )


def _connected_controlnet_models(
    controlnet_nodes: list[str],
    edges: list[tuple[str, str, str, str]],
    controlnet_models: dict[str, list[str]],
) -> list[str]:
    connected: list[str] = []
    for controlnet_node in controlnet_nodes:
        target_nodes = {
            target_node
            for source_node, source_parameter, target_node, target_parameter in edges
            if source_node == controlnet_node
            and source_parameter == "control_net"
            and target_parameter.startswith("control_nets_")
        }
        if target_nodes:
            connected.extend(controlnet_models.get(controlnet_node, []))
    return connected


def _add_runtime_loras(
    artifact: DiffusionPipelineArtifact, runtime_loras: list[tuple[str, dict[str, Any]]]
) -> DiffusionPipelineArtifact:
    runtime_steps: list[PipelineRuntimeAdapterStep] = []
    for _node_name, raw_lora in runtime_loras:
        try:
            specs = normalize_loras([raw_lora])
        except (TypeError, ValueError) as error:
            raise WorkflowExtractionError(f"Could not reconstruct runtime LoRA: {error}") from error
        runtime_steps.append(LoraPipelineRuntimeAdapterStep(specs))
    return artifact.with_additional_runtime_adapter_steps(runtime_steps)


def _add_controlnet_artifact(
    base_artifact: DiffusionPipelineArtifact, controlnet_models: list[str]
) -> ControlNetDiffusionPipelineArtifact:
    digest = hashlib.sha256(json.dumps({"control_nets": controlnet_models}, sort_keys=True).encode()).hexdigest()
    return ControlNetDiffusionPipelineArtifact(
        base_artifact=base_artifact,
        controlnet_models=controlnet_models,
        config_hash=f"{base_artifact.config_hash}--ControlNet--{digest}",
    )


def _parse_model_selection(value: str) -> tuple[str, str | None]:
    match = re.fullmatch(r"(.+?) \(([a-f0-9]{40})\)", value)
    if match is None:
        return value, None
    return match.group(1), match.group(2)


def _build_artifact(
    parameters_class: type,
    pipeline_type: str,
    provider: str,
    model_value: str,
    repo_id: str,
    revision: str | None,
    values: dict[str, Any],
    build_data: dict[str, Any],
    component_overrides: dict[str, Any],
) -> DiffusionPipelineArtifact:
    lora_values = values.get("loras", [])
    if not isinstance(lora_values, list):
        raise WorkflowExtractionError("Pipeline builder 'loras' value must be a list.")
    try:
        loras = {path: spec.weight for path, spec in normalize_loras(lora_values).items()}
    except (TypeError, ValueError) as error:
        raise WorkflowExtractionError(f"Could not reconstruct pipeline LoRAs: {error}") from error

    optimization_kwargs = {
        "memory_optimization_strategy": values.get("memory_optimization_strategy", "Manual"),
        "attention_slicing": values.get("attention_slicing", False),
        "vae_slicing": values.get("vae_slicing", False),
        "vae_tiling": values.get("vae_tiling", False),
        "transformer_layerwise_casting": values.get("transformer_layerwise_casting", False),
        "cpu_offload_strategy": values.get("cpu_offload_strategy", "None"),
        "quantization_mode": values.get("quantization_mode", "None"),
    }
    parameters_instance = object.__new__(parameters_class)
    pipeline_name = parameters_instance.pipeline_name
    if build_data.get("_all_overrides"):
        config_kwargs = {"provider": provider}
        for slot, artifact in component_overrides.items():
            config_kwargs[f"component_override_{slot}"] = artifact.load_id
        identity = BasePipelineIdentity(
            pipeline_name=pipeline_name,
            config_kwargs=config_kwargs,
            loras=loras,
            optimization_kwargs=optimization_kwargs,
            torch_dtype="bfloat16",
        )
        return DiffusionPipelineArtifact(
            pipeline_name=pipeline_name,
            config_hash=identity.cache_key(),
            builder_module=parameters_class.__module__,
            builder_class_name=parameters_class.__name__,
            build_data=build_data,
            loras=loras,
            optimization_kwargs=optimization_kwargs,
            is_prequantized=any(artifact.is_quantized for artifact in component_overrides.values()),
            supports_layerwise_casting=parameters_class.supports_layerwise_casting(),
            requires_device_map=parameters_class.requires_device_map(),
        )
    config_kwargs = parameters_class.config_kwargs(model_value)
    config_kwargs["provider"] = provider
    for slot, artifact in component_overrides.items():
        config_kwargs[f"component_override_{slot}"] = artifact.load_id
    identity = BasePipelineIdentity(
        pipeline_name=pipeline_name,
        config_kwargs=config_kwargs,
        loras=loras,
        optimization_kwargs=optimization_kwargs,
        torch_dtype="bfloat16",
    )
    prequantized = parameters_class.is_prequantized_repo(repo_id)
    requires_device_map = parameters_class.requires_device_map()

    return DiffusionPipelineArtifact(
        pipeline_name=pipeline_name,
        config_hash=identity.cache_key(),
        builder_module=parameters_class.__module__,
        builder_class_name=parameters_class.__name__,
        build_data={**build_data, "_component_overrides": component_overrides} if component_overrides else build_data,
        loras=loras,
        optimization_kwargs=optimization_kwargs,
        is_prequantized=prequantized,
        supports_layerwise_casting=parameters_class.supports_layerwise_casting(),
        requires_device_map=requires_device_map,
    )


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
