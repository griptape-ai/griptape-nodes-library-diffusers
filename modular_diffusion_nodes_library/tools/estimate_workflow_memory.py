"""Estimate memory for generated workflow source files without executing them."""

from __future__ import annotations

import argparse
import ast
import importlib
import io
import json
import pickle
import sys
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentSourceType,
    HFRepoRef,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import create_driver, get_driver_class
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState
from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import (
    ComponentConfigNotCachedError,
    build_component_on_meta_device,
    resolve_base_component_config,
)
from modular_diffusion_nodes_library.memory_estimation.pipeline_memory_estimator import (
    estimate_pipeline_memory_from_artifact,
)

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
class _DimensionAssignment:
    node_name: str
    width: int | None
    height: int | None
    num_frames: int | None


@dataclass(frozen=True)
class WorkflowBuilder:
    """One pipeline and raw-dimension estimate target extracted from a workflow."""

    artifact: DiffusionPipelineArtifact
    dimensions: _DimensionAssignment


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
) -> tuple[list[_DimensionAssignment], dict[str, DiffusionPipelineArtifact]]:
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

        is_noise_dimension = node_types.get(node_name) == "NoiseLatentNode" and parameter_name in {
            "width",
            "height",
            "num_frames",
        }
        is_generation_pipeline = parameter_name == "pipeline" and node_types.get(node_name) in {
            "NoiseLatentNode",
            "DiffusionPipelineGenerateLatentNode",
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
        _DimensionAssignment(node_name, item["width"], item["height"], item["num_frames"])
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


def _resolve_dimensions(
    dimensions: _DimensionAssignment,
    *,
    width: int | None,
    height: int | None,
    num_frames: int | None,
) -> tuple[int, int, int | None]:
    if (width is None) != (height is None):
        raise WorkflowExtractionError("--width and --height must be supplied together.")
    resolved_width = width if width is not None else dimensions.width
    resolved_height = height if height is not None else dimensions.height
    resolved_frames = num_frames if num_frames is not None else dimensions.num_frames
    if resolved_width is None or resolved_height is None:
        raise WorkflowExtractionError(
            f"Could not determine width and height for NoiseLatentNode '{dimensions.node_name}'. "
            "Supply --width and --height."
        )
    if resolved_width <= 0 or resolved_height <= 0:
        raise WorkflowExtractionError("Width and height must be positive integers.")
    if resolved_frames is not None and resolved_frames <= 0:
        raise WorkflowExtractionError("num_frames must be a positive integer when provided.")
    return resolved_width, resolved_height, resolved_frames


def _is_video_artifact(artifact: DiffusionPipelineArtifact) -> bool:
    driver_class = get_driver_class(artifact.pipeline_name)
    return bool(driver_class and driver_class.produces_video)


def _source_shape(
    artifact: DiffusionPipelineArtifact,
    width: int,
    height: int,
    num_frames: int | None,
) -> tuple[int, ...]:
    if _is_video_artifact(artifact):
        if num_frames is None:
            raise WorkflowExtractionError(
                f"Pipeline '{artifact.pipeline_name}' is a video pipeline and requires num_frames. Supply --num-frames."
            )
        return (1, 3, num_frames, height, width)
    return (1, 3, height, width)


def _pipeline_repository(artifact: DiffusionPipelineArtifact) -> tuple[str, str | None]:
    build_data = artifact.build_data
    repo_id = build_data.get("base_repo_id") or build_data.get("repo_id")
    revision = build_data.get("base_revision") or build_data.get("revision")
    if not isinstance(repo_id, str) or not repo_id:
        raise WorkflowExtractionError(
            f"Could not resolve config for pipeline '{artifact.pipeline_name}'. "
            "The serialized artifact has no base model repository."
        )
    if revision is not None and not isinstance(revision, str):
        revision = None
    return repo_id, revision


def _resolve_pipeline_cls(artifact: DiffusionPipelineArtifact) -> type:
    if not artifact.builder_module or not artifact.builder_class_name:
        raise WorkflowExtractionError(
            f"Could not resolve config for pipeline '{artifact.pipeline_name}'. "
            "The serialized artifact has no builder metadata."
        )
    builder_class = getattr(importlib.import_module(artifact.builder_module), artifact.builder_class_name)
    return builder_class.pipeline_cls()


_SHAPE_COMPONENT_SLOTS = {"vae", "transformer", "transformer_2", "unet", "scheduler"}


def _shape_component_slots(pipe: Any) -> list[str]:
    return [name for name, component in pipe.components.items() if component is None and name in _SHAPE_COMPONENT_SLOTS]


def _component_class(component_spec: Any) -> type | None:
    component_cls = component_spec.type_hint
    if isinstance(component_cls, tuple):
        component_cls = next((item for item in component_cls if isinstance(item, type)), None)
    if isinstance(component_cls, type):
        return component_cls
    return None


def _build_shape_component(component_spec: Any, config: dict[str, Any], torch_dtype: torch.dtype) -> Any:
    component_cls = _component_class(component_spec)
    if component_cls is None:
        raise WorkflowExtractionError("Could not resolve a concrete Diffusers component class from its modular spec.")
    if issubclass(component_cls, torch.nn.Module):
        return build_component_on_meta_device(component_cls, config, torch_dtype, cast_dtype=False)
    return component_spec.create(config)


def _build_shape_only_pipe(artifact: DiffusionPipelineArtifact) -> Any:
    driver_class = get_driver_class(artifact.pipeline_name)
    if driver_class is None:
        raise WorkflowExtractionError(
            f"Could not resolve latent shape for unsupported pipeline '{artifact.pipeline_name}'."
        )

    # Driver _create_modular_pipe() only needs a small placeholder for the
    # driver-specific branch (WAN T2V vs. WAN 2.2, for example). The resulting
    # modular pipeline is then populated with config-only components below.
    driver_shell = driver_class.__new__(driver_class)
    driver_shell._pipe = SimpleNamespace(transformer_2=None)  # noqa: SLF001
    driver_shell._modular_pipe = None  # noqa: SLF001
    try:
        pipe = driver_class._create_modular_pipe(driver_shell)  # noqa: SLF001
    except NotImplementedError as error:
        raise WorkflowExtractionError(
            f"Could not resolve latent shape for pipeline '{artifact.pipeline_name}' because its driver "
            "does not expose a modular prepare-latents pipeline."
        ) from error

    repo_id, revision = _pipeline_repository(artifact)
    pipeline_cls = _resolve_pipeline_cls(artifact)
    components: dict[str, Any] = {}
    for slot in _shape_component_slots(pipe):
        spec = pipe.get_component_spec(slot)
        if _component_class(spec) is None:
            raise WorkflowExtractionError(
                f"Could not resolve latent shape for pipeline '{artifact.pipeline_name}' because component "
                f"'{slot}' has no concrete config class."
            )
        overrides = artifact.build_data.get("_component_overrides", {})
        override = overrides.get(slot) if isinstance(overrides, dict) else None
        if override is not None:
            config_dict = override.try_read_config(pipeline_cls=pipeline_cls)
        else:
            config_dict = resolve_base_component_config(repo_id, slot, revision)
        if config_dict is None:
            raise ComponentConfigNotCachedError(
                f"Attempted to resolve latent geometry for pipeline '{artifact.pipeline_name}'. "
                f"Failed because the '{slot}' config is not available in the local cache."
            )
        components[slot] = _build_shape_component(spec, config_dict, torch.bfloat16)

    pipe.update_components(**components)
    return pipe


def resolve_noise_latent_shape(
    artifact: DiffusionPipelineArtifact,
    *,
    width: int,
    height: int,
    num_frames: int | None = None,
) -> tuple[int, ...]:
    """Resolve the public latent shape through Diffusers' config-only driver path.

    Raw dimensions are passed through without snapping. The driver delegates the
    shape calculation to its pinned Diffusers prepare-latents block and performs
    any model-specific unpacking before the returned shape is inspected.
    """
    source_shape = _source_shape(artifact, width, height, num_frames)
    pipe = _build_shape_only_pipe(artifact)
    driver = create_driver(pipe, artifact.pipeline_name)
    driver._modular_pipe = pipe  # noqa: SLF001
    original_execution_device = getattr(type(pipe), "_execution_device", None)
    if original_execution_device is not None:
        type(pipe)._execution_device = property(lambda _: torch.device("cpu"))
    try:
        latent = driver.create_noise_latent(source_shape, GeneratorState.from_seed(0))
    finally:
        if original_execution_device is not None:
            type(pipe)._execution_device = original_execution_device
    return latent.shape


def _make_latent(
    artifact: DiffusionPipelineArtifact,
    width: int,
    height: int,
    num_frames: int | None,
) -> LatentArtifact:
    source_shape = _source_shape(artifact, width, height, num_frames)
    shape = resolve_noise_latent_shape(
        artifact,
        width=width,
        height=height,
        num_frames=num_frames,
    )
    return LatentArtifact(shape=shape, dtype="torch.bfloat16", source_shape=source_shape)


def estimate_workflow(
    workflow: WorkflowMemoryInput,
    *,
    width: int | None = None,
    height: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    warnings = [
        "Raw workflow dimensions were used without model-specific dimension snapping or validation; "
        "latent shape was resolved through the cached Diffusers prepare-latents path."
    ]
    for builder in workflow.builders:
        resolved_width, resolved_height, resolved_frames = _resolve_dimensions(
            builder.dimensions,
            width=width,
            height=height,
            num_frames=num_frames,
        )
        latent = _make_latent(builder.artifact, resolved_width, resolved_height, resolved_frames)
        estimate = estimate_pipeline_memory_from_artifact(builder.artifact, latent)
        estimate_data = estimate.to_dict()
        estimate_data.pop("pipeline_name", None)
        dimensions: dict[str, int | None] = {"width": resolved_width, "height": resolved_height}
        if _is_video_artifact(builder.artifact):
            dimensions["num_frames"] = resolved_frames
        results.append(
            {
                "pipeline_name": builder.artifact.pipeline_name,
                "dimensions": dimensions,
                "estimate": estimate_data,
            }
        )
    return {"workflow": str(workflow.workflow_path), "warnings": warnings, "builders": results}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate memory for a workflow source file without running it.")
    parser.add_argument("workflow", type=Path, help="Path to a generated workflow Python file.")
    parser.add_argument("--width", type=int, help="Override or provide the raw image/video width.")
    parser.add_argument("--height", type=int, help="Override or provide the raw image/video height.")
    parser.add_argument("--num-frames", type=int, help="Override or provide the raw video frame count.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    return parser


def _format_human(result: dict[str, Any]) -> str:
    lines = [f"Workflow: {result['workflow']}", *[f"Warning: {warning}" for warning in result["warnings"]]]
    for index, builder in enumerate(result["builders"], start=1):
        estimate = builder["estimate"]
        dimensions = builder["dimensions"]
        lines.extend(
            [
                "",
                f"Builder {index}: {builder['pipeline_name']}",
                f"Dimensions: {dimensions}",
                f"Estimated peak: {estimate['estimated_peak_gb']} GB",
            ]
        )
        for component in estimate["components"]:
            lines.append(
                f"  {component['component_name']}: {component['total_gb']} GB "
                f"(weights {component['weight_gb']} GB, activations {component['activation_gb']} GB)"
            )
        for warning in estimate["warnings"]:
            lines.append(f"  Warning: {warning}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = estimate_workflow(
            extract_workflow(args.workflow),
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
        )
    except (WorkflowExtractionError, OSError, ValueError, RuntimeError) as error:
        print(f"Workflow memory estimation failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_format_human(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
