from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import scan_cache_dir  # pyright: ignore[reportMissingImports]
from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentArtifact,
    ComponentSourceType,
    HFRepoRef,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.scheduler_component_artifact import SchedulerComponentArtifact
from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import TextEncoderComponentArtifact
from modular_diffusion_nodes_library.artifact_utils.tokenizer_component_artifact import TokenizerComponentArtifact
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros

PIPELINE_RECIPE_SCHEMA_VERSION = 1


def serialize_pipeline_artifact(artifact: DiffusionPipelineArtifact) -> str:
    """Serialize a supported pipeline artifact to a versioned JSON recipe."""
    recipe = {
        "schema_version": PIPELINE_RECIPE_SCHEMA_VERSION,
        "artifact": _serialize_pipeline_artifact(artifact),
    }
    return json.dumps(recipe, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def deserialize_pipeline_artifact(recipe_json: str) -> DiffusionPipelineArtifact:
    """Deserialize a supported versioned pipeline recipe."""
    try:
        document = json.loads(recipe_json)
    except json.JSONDecodeError as error:
        msg = f"Invalid pipeline recipe JSON: {error.msg}"
        raise ValueError(msg) from error

    document_data = _require_dict(document, "Pipeline recipe")
    schema_version = document_data.get("schema_version")
    if schema_version != PIPELINE_RECIPE_SCHEMA_VERSION:
        msg = f"Unsupported pipeline recipe schema version: {schema_version!r}"
        raise ValueError(msg)

    return _deserialize_pipeline_artifact(_require_dict(document_data.get("artifact"), "Pipeline recipe artifact"))


def validate_pipeline_recipe_dependencies(artifact: DiffusionPipelineArtifact) -> list[str]:
    """Return local files and pinned Hugging Face revisions unavailable for an artifact."""
    if isinstance(artifact, ControlNetDiffusionPipelineArtifact):
        issues = []
        for repo_id in artifact.controlnet_models:
            if not _list_cached_repo_revisions(repo_id):
                issues.append(f"ControlNet repository is unavailable locally: {repo_id}")
        return [*issues, *validate_pipeline_recipe_dependencies(artifact.base_artifact)]

    issues = []
    for lora_path in artifact.loras:
        if not Path(lora_path).is_file():
            issues.append(f"LoRA file is unavailable: {lora_path}")

    return [*issues, *_validate_build_data_repositories(artifact.build_data)]


def format_pipeline_artifact_summary(artifact: DiffusionPipelineArtifact) -> str:
    """Format the persisted pipeline configuration for a Load Pipeline status message."""
    base_artifact = artifact.base_artifact if isinstance(artifact, ControlNetDiffusionPipelineArtifact) else artifact
    build_data = base_artifact.build_data
    repo_id = build_data.get("base_repo_id") or build_data.get("repo_id") or "Unknown"
    revision = build_data.get("base_revision") or build_data.get("revision") or "Unpinned"
    lines = ["Loaded successfully", "", "Pipeline", f"  Type: {base_artifact.pipeline_name}", f"  Model: {repo_id}", f"  Revision: {revision}"]

    if isinstance(artifact, ControlNetDiffusionPipelineArtifact):
        lines.extend(["", "ControlNets"])
        lines.extend(f"  {index}. {model}" for index, model in enumerate(artifact.controlnet_models, start=1))

    if base_artifact.loras:
        lines.extend(["", "LoRAs"])
        for index, (path, weight) in enumerate(base_artifact.loras.items(), start=1):
            lines.extend([f"  {index}. {Path(path).name}", f"     Path: {path}", f"     Weight: {weight:g}"])

    component_overrides = build_data.get("_component_overrides", {})
    if isinstance(component_overrides, dict) and component_overrides:
        lines.extend(["", "Component Overrides"])
        for slot, override in component_overrides.items():
            if not isinstance(override, ComponentArtifact):
                lines.append(f"  {slot}: Unsupported override configuration")
                continue
            lines.extend([f"  {slot.title()}", f"    Type: {type(override).__name__}", f"    Source: {override.source_type.value}"])
            if override.repo_ref is not None:
                lines.append(f"    Model: {override.repo_ref.repo_id}")
                if override.repo_ref.revision:
                    lines.append(f"    Revision: {override.repo_ref.revision}")
                if override.repo_ref.subfolder:
                    lines.append(f"    Subfolder: {override.repo_ref.subfolder}")
            else:
                lines.append(f"    Path: {override.file_path or override.config_source or 'Unknown'}")
            if override.torch_dtype:
                lines.append(f"    Dtype: {override.torch_dtype}")

    return "\n".join(lines)


def _validate_build_data_repositories(build_data: dict[str, Any]) -> list[str]:
    issues = []
    for key, value in build_data.items():
        if not key.endswith("repo_id") or not isinstance(value, str):
            continue

        revisions = _list_cached_repo_revisions(value)
        if not revisions:
            issues.append(f"Repository is unavailable locally: {value}")
            continue

        revision_key = key.removesuffix("repo_id") + "revision"
        revision = build_data.get(revision_key)
        if revision is None and key in {"repo_id", "base_repo_id"}:
            revision = build_data.get("revision")
        if isinstance(revision, str) and revision:
            cached_revisions = {cached_revision for _, cached_revision in revisions}
            if revision not in cached_revisions:
                issues.append(f"Repository revision is unavailable locally: {value} ({revision})")
    return issues


def _list_cached_repo_revisions(repo_id: str) -> list[tuple[str, str]]:
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            return [(repo_id, revision.commit_hash) for revision in repo.revisions]
    return []


def _deserialize_pipeline_artifact(recipe: dict[str, Any]) -> DiffusionPipelineArtifact:
    artifact_type = recipe.get("type")
    if artifact_type == "controlnet":
        base_recipe = _require_dict(recipe.get("base"), "ControlNet base artifact")
        base_artifact = _deserialize_pipeline_artifact(base_recipe)
        controlnet_models = _require_string_list(recipe.get("controlnet_models"), "ControlNet models")
        config_hash = _require_string(recipe.get("config_hash"), "ControlNet config hash")
        return ControlNetDiffusionPipelineArtifact(
            base_artifact=base_artifact,
            controlnet_models=controlnet_models,
            config_hash=config_hash,
        )

    if artifact_type != "base":
        msg = f"Unsupported pipeline recipe artifact type: {artifact_type!r}"
        raise ValueError(msg)

    builder = _require_dict(recipe.get("builder"), "Pipeline builder")
    flags = _require_dict(recipe.get("flags"), "Pipeline flags")
    return DiffusionPipelineArtifact(
        pipeline_name=_require_string(recipe.get("pipeline_name"), "Pipeline name"),
        config_hash=_require_string(recipe.get("config_hash"), "Pipeline config hash"),
        builder_module=_optional_string(builder.get("module"), "Pipeline builder module"),
        builder_class_name=_optional_string(builder.get("class_name"), "Pipeline builder class name"),
        build_data=_require_dict(_resolve_recipe_value(recipe.get("build_data")), "Pipeline build data"),
        build_data_error=_optional_string(recipe.get("build_data_error"), "Pipeline build data error"),
        loras=_require_loras(_resolve_recipe_value(recipe.get("loras"))),
        optimization_kwargs=_require_dict(_resolve_recipe_value(recipe.get("optimization")), "Pipeline optimization"),
        is_prequantized=_require_bool(flags.get("is_prequantized"), "Pipeline is_prequantized flag"),
        supports_layerwise_casting=_require_bool(
            flags.get("supports_layerwise_casting"), "Pipeline supports_layerwise_casting flag"
        ),
        requires_device_map=_require_bool(flags.get("requires_device_map"), "Pipeline requires_device_map flag"),
    )


def _resolve_recipe_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("type") == "component_override":
        return _deserialize_component_artifact(value)
    if isinstance(value, str):
        return expand_path_macros(value)
    if isinstance(value, list):
        return [_resolve_recipe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _resolve_recipe_value(item) for key, item in value.items()}
    return value


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    msg = f"{label} must be an object."
    raise ValueError(msg)


def _require_string(value: Any, label: str) -> str:
    if isinstance(value, str) and value:
        return value
    msg = f"{label} must be a non-empty string."
    raise ValueError(msg)


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"{label} must be a string or null."
    raise ValueError(msg)


def _require_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    msg = f"{label} must be a boolean."
    raise ValueError(msg)


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"{label} must be a list of strings."
        raise ValueError(msg)
    return value


def _require_loras(value: Any) -> dict[str, float]:
    loras = _require_dict(value, "Pipeline LoRAs")
    if not all(isinstance(path, str) and isinstance(weight, int | float) for path, weight in loras.items()):
        msg = "Pipeline LoRAs must map paths to numeric weights."
        raise ValueError(msg)
    return {path: float(weight) for path, weight in loras.items()}


def _serialize_component_artifact(artifact: ComponentArtifact) -> dict[str, Any]:
    artifact_type = type(artifact).__name__
    if artifact_type not in {
        "ModelComponentArtifact",
        "TextEncoderComponentArtifact",
        "TokenizerComponentArtifact",
        "SchedulerComponentArtifact",
    }:
        msg = f"Unsupported component override artifact type: {artifact_type}"
        raise ValueError(msg)

    result: dict[str, Any] = {
        "type": "component_override",
        "artifact_type": artifact_type,
        "load_id": artifact.load_id,
        "source_type": artifact.source_type.value,
        "component": artifact.component,
        "torch_dtype": artifact.torch_dtype,
    }
    if isinstance(artifact, ModelComponentArtifact):
        result.update(
            {
                "repo_ref": _serialize_repo_ref(artifact.repo_ref),
                "file_path": _json_safe_value(artifact.file_path),
                "config_source": _json_safe_value(artifact.config_source),
            }
        )
    if isinstance(artifact, SchedulerComponentArtifact):
        result.update(
            {
                "scheduler_class": artifact.scheduler_class,
                "config_source": _json_safe_value(artifact.config_source),
                "repo_ref": _serialize_repo_ref(artifact.repo_ref),
                "text_config": _json_safe_value(artifact.text_config),
            }
        )
    return result


def _deserialize_component_artifact(data: dict[str, Any]) -> ComponentArtifact:
    artifact_type = _require_string(data.get("artifact_type"), "Component override artifact type")
    source_type = ComponentSourceType(_require_string(data.get("source_type"), "Component override source type"))
    kwargs = {
        "load_id": _require_string(data.get("load_id"), "Component override load ID"),
        "source_type": source_type,
        "component": _require_string(data.get("component"), "Component override component"),
        "torch_dtype": _require_string(data.get("torch_dtype"), "Component override torch dtype"),
    }
    if artifact_type == "SchedulerComponentArtifact":
        return SchedulerComponentArtifact(
            **kwargs,
            scheduler_class=_require_string(data.get("scheduler_class"), "Scheduler component class"),
            config_source=_optional_string(_resolve_recipe_value(data.get("config_source")), "Scheduler config source"),
            repo_ref=_deserialize_repo_ref(data.get("repo_ref")),
            text_config=_optional_dict(data.get("text_config"), "Scheduler text config"),
        )

    model_classes = {
        "ModelComponentArtifact": ModelComponentArtifact,
        "TextEncoderComponentArtifact": TextEncoderComponentArtifact,
        "TokenizerComponentArtifact": TokenizerComponentArtifact,
    }
    artifact_class = model_classes.get(artifact_type)
    if artifact_class is None:
        msg = f"Unsupported component override artifact type: {artifact_type}"
        raise ValueError(msg)
    return artifact_class(
        **kwargs,
        repo_ref=_deserialize_repo_ref(data.get("repo_ref")),
        file_path=_optional_string(_resolve_recipe_value(data.get("file_path")), "Component override file path"),
        config_source=_optional_string(_resolve_recipe_value(data.get("config_source")), "Component override config source"),
    )


def _serialize_repo_ref(repo_ref: HFRepoRef | None) -> dict[str, str | None] | None:
    if repo_ref is None:
        return None
    return {
        "repo_id": repo_ref.repo_id,
        "revision": repo_ref.revision,
        "subfolder": repo_ref.subfolder,
    }


def _deserialize_repo_ref(value: Any) -> HFRepoRef | None:
    if value is None:
        return None
    data = _require_dict(value, "Component override repository reference")
    return HFRepoRef(
        repo_id=_require_string(data.get("repo_id"), "Component override repository ID"),
        revision=_optional_string(data.get("revision"), "Component override repository revision"),
        subfolder=_optional_string(data.get("subfolder"), "Component override repository subfolder"),
    )


def _optional_dict(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _require_dict(value, label)


def _serialize_pipeline_artifact(artifact: DiffusionPipelineArtifact) -> dict[str, Any]:
    if artifact.runtime_adapter_steps():
        msg = "Pipelines with runtime adapter steps cannot be saved. Save the base pipeline and reconnect the adapter."
        raise ValueError(msg)

    if isinstance(artifact, ControlNetDiffusionPipelineArtifact):
        return {
            "type": "controlnet",
            "base": _serialize_pipeline_artifact(artifact.base_artifact),
            "config_hash": artifact.config_hash,
            "controlnet_models": _json_safe_value(artifact.controlnet_models),
        }

    if type(artifact) is not DiffusionPipelineArtifact:
        msg = f"Unsupported pipeline artifact type: {type(artifact).__name__}"
        raise ValueError(msg)

    return {
        "type": "base",
        "pipeline_name": artifact.pipeline_name,
        "config_hash": artifact.config_hash,
        "builder": {
            "module": artifact.builder_module,
            "class_name": artifact.builder_class_name,
        },
        "build_data": _json_safe_value(artifact.build_data),
        "build_data_error": artifact.build_data_error,
        "loras": _json_safe_value(artifact.loras),
        "optimization": _json_safe_value(artifact.optimization_kwargs),
        "flags": {
            "is_prequantized": artifact.is_prequantized,
            "supports_layerwise_casting": artifact.supports_layerwise_casting,
            "requires_device_map": artifact.requires_device_map,
        },
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, ComponentArtifact):
        return _serialize_component_artifact(value)
    if value is None or isinstance(value, str | int | float | bool):
        if isinstance(value, str):
            return _path_to_macro(value)
        return value
    if isinstance(value, Path):
        return _path_to_macro(str(value))
    if isinstance(value, list | tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}

    msg = f"Pipeline recipe contains non-JSON value: {type(value).__name__}"
    raise TypeError(msg)


def _path_to_macro(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value

    result = GriptapeNodes.handle_request(AttemptMapAbsolutePathToProjectRequest(absolute_path=path))
    if isinstance(result, AttemptMapAbsolutePathToProjectResultSuccess) and result.mapped_path is not None:
        return result.mapped_path
    return value