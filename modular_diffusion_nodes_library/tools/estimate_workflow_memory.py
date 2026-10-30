"""Estimate memory for generated workflow source files without executing them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.artifact_utils.scheduler_component_artifact import SchedulerComponentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import create_driver, get_driver_class
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState
from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import (
    ComponentConfigNotCachedError,
    build_component_on_meta_device,
    resolve_base_component_config,
)
from modular_diffusion_nodes_library.memory_estimation.pipeline_memory_estimator import (
    estimate_pipeline_memory_from_artifact,
    resolve_pipeline_cls,
)
from modular_diffusion_nodes_library.tools.workflow_extraction import WorkflowBuilder as WorkflowBuilder
from modular_diffusion_nodes_library.tools.workflow_extraction import (
    WorkflowDimensionAssignment,
    WorkflowExtractionError,
    WorkflowMemoryInput,
    extract_workflow,
)
from modular_diffusion_nodes_library.utils.torch_utils import to_human_readable_size


def _resolve_dimensions(
    dimensions: WorkflowDimensionAssignment,
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


_SHAPE_COMPONENT_SLOTS = {"vae", "transformer", "transformer_2", "unet", "scheduler"}


def _shape_component_slots(pipe: Any, pipeline_cls: type) -> list[str]:
    # The driver's block set can declare slots the pipeline class lacks (Wan22Blocks has
    # transformer_2, WanAnimatePipeline does not), so only request configs for declared slots.
    declared_slots, _ = pipeline_cls._get_signature_keys(pipeline_cls)  # noqa: SLF001
    return [
        name
        for name, component in pipe.components.items()
        if component is None and name in _SHAPE_COMPONENT_SLOTS and name in declared_slots
    ]


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
    driver_shell._pipe = cast(Any, SimpleNamespace(transformer_2=None))  # noqa: SLF001
    driver_shell._modular_pipe = None  # noqa: SLF001
    try:
        pipe = driver_class._create_modular_pipe(driver_shell)  # noqa: SLF001
    except NotImplementedError as error:
        raise WorkflowExtractionError(
            f"Could not resolve latent shape for pipeline '{artifact.pipeline_name}' because its driver "
            "does not expose a modular prepare-latents pipeline."
        ) from error

    pipeline_cls = resolve_pipeline_cls(artifact)
    components: dict[str, Any] = {}
    overrides = artifact.build_data.get("_component_overrides", {})
    for slot in _shape_component_slots(pipe, pipeline_cls):
        spec = pipe.get_component_spec(slot)
        if _component_class(spec) is None:
            raise WorkflowExtractionError(
                f"Could not resolve latent shape for pipeline '{artifact.pipeline_name}' because component "
                f"'{slot}' has no concrete config class."
            )
        override = overrides.get(slot) if isinstance(overrides, dict) else None
        if isinstance(override, SchedulerComponentArtifact):
            components[slot] = override.materialize(pipeline_cls=pipeline_cls, slot=slot)
            continue
        if override is not None:
            config_dict = override.try_read_config()
        else:
            repo_id, revision = _pipeline_repository(artifact)
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
        lora_adapters = estimate.get("lora_adapters", [])
        if lora_adapters:
            lines.append("  LoRA adapters:")
            for adapter in lora_adapters:
                size = to_human_readable_size(adapter.get("weight_bytes", 0))
                lines.append(f"    {adapter['adapter_name']}: {size} ({adapter['path']})")
            total_lora_bytes = sum(adapter.get("weight_bytes", 0) for adapter in lora_adapters)
            lines.append(f"    Total LoRA weights: {to_human_readable_size(total_lora_bytes)}")
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
